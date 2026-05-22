#!/usr/bin/env python3
"""
Privacy Filter BR v2 — Fine-tuning script for Google Colab.

Differences from v1:
- 18 categories (added 6 e-commerce: ORDER_ID, TRACKING_CODE, INVOICE_NUMBER,
  CLIENT_REVENUE, TRANSACTION_ID, CUSTOMER_ID; removed unused DATE)
- 73 output labels (1 + 18×4 BIOES)
- modules_to_save=['score'] (was 'classifier' — that was a v1 bug)
- dtype=float32 to avoid NaN in MoE forward pass (v1 lesson)
- New transformers API: eval_strategy, processing_class
- merge_and_unload + save merged model so pipeline loads correct head

Usage in Colab (A100 recommended):
    1. Upload data/dataset_br_v2.jsonl + data/dataset_br_v2_holdout.jsonl
    2. !pip install -q transformers peft datasets seqeval accelerate
    3. !python finetune_v2.py
    4. Download privacy-filter-br-v2/ directory
"""
import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from peft import LoraConfig, get_peft_model, TaskType
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report


# === LABEL TAXONOMY (v2) ===

CATEGORIES = [
    # Original BR PII
    "PRIVATE_PERSON", "PRIVATE_CPF", "PRIVATE_CNPJ", "PRIVATE_RG",
    "PRIVATE_CNH", "PRIVATE_PIS", "PRIVATE_TITULO_ELEITOR",
    "PRIVATE_CERTIDAO", "PRIVATE_IE", "PRIVATE_EMAIL",
    "PRIVATE_PHONE", "PRIVATE_ADDRESS",
    # E-commerce / B2B (v2 additions)
    "PRIVATE_ORDER_ID", "PRIVATE_TRACKING_CODE", "PRIVATE_INVOICE_NUMBER",
    "PRIVATE_CLIENT_REVENUE", "PRIVATE_TRANSACTION_ID", "PRIVATE_CUSTOMER_ID",
]

# 1 background + 18 cats × 4 BIOES = 73 labels
LABELS = ["O"] + [f"{tag}-{cat}" for cat in CATEGORIES for tag in ["B", "I", "E", "S"]]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

assert len(LABELS) == 73, f"Expected 73 labels, got {len(LABELS)}"


# === DATA PROCESSING ===

def char_spans_to_token_tags(text: str, entities: list, tokenizer, max_length: int = 512):
    """Convert character-level entity spans to token-level BIOES tags."""
    encoding = tokenizer(
        text, max_length=max_length, truncation=True,
        return_offsets_mapping=True, padding=False,
    )
    offsets = encoding["offset_mapping"]
    n_tokens = len(offsets)
    tags = [LABEL2ID["O"]] * n_tokens

    sorted_ents = sorted(entities, key=lambda e: e["start"])
    for ent in sorted_ents:
        ent_start, ent_end, ent_label = ent["start"], ent["end"], ent["label"]
        if ent_label not in CATEGORIES:
            continue  # skip unknown labels (shouldn't happen, but defensive)
        token_idxs = [
            i for i, (s, e) in enumerate(offsets)
            if s < ent_end and e > ent_start and s != e
        ]
        if not token_idxs:
            continue
        if len(token_idxs) == 1:
            tags[token_idxs[0]] = LABEL2ID[f"S-{ent_label}"]
        else:
            tags[token_idxs[0]] = LABEL2ID[f"B-{ent_label}"]
            tags[token_idxs[-1]] = LABEL2ID[f"E-{ent_label}"]
            for i in token_idxs[1:-1]:
                tags[i] = LABEL2ID[f"I-{ent_label}"]

    encoding.pop("offset_mapping")
    encoding["labels"] = tags
    return encoding


def load_dataset(path: str, tokenizer, max_length: int = 512) -> Dataset:
    examples = []
    skipped = 0
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            try:
                tok = char_spans_to_token_tags(ex["text"], ex["entities"], tokenizer, max_length)
                examples.append(tok)
            except Exception as e:
                skipped += 1
    if skipped:
        print(f"  skipped {skipped} examples (tokenization errors)")
    return Dataset.from_list(examples)


# === METRICS ===

def compute_metrics_fn(p):
    preds, labels = p
    preds = np.argmax(preds, axis=2)
    true_labels = [
        [ID2LABEL[l] for l in label if l != -100]
        for label in labels
    ]
    true_preds = [
        [ID2LABEL[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(preds, labels)
    ]
    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
    }


# === TRAINING ===

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="openai/privacy-filter")
    parser.add_argument("--train-file", default="/content/dataset_br_v2.jsonl")
    parser.add_argument("--eval-file", default="/content/dataset_br_v2_holdout.jsonl")
    parser.add_argument("--output-dir", default="/content/privacy-filter-br-v2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Loading base model with new head ({len(LABELS)} classes)...")
    # IMPORTANT: dtype=float32 to avoid NaN logits in MoE forward pass with reinit'd head
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
        dtype=torch.float32,
    )

    # IMPORTANT: .cuda() BEFORE get_peft_model() so PEFT wraps the GPU version
    if torch.cuda.is_available():
        model = model.cuda()

    print("Configuring LoRA adapters...")
    # IMPORTANT: modules_to_save=['score'] — privacy-filter calls its head 'score',
    # not 'classifier' like most HF models. Using wrong name = head not trained.
    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        modules_to_save=["score"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading datasets...")
    train_ds = load_dataset(args.train_file, tokenizer, args.max_length)
    eval_ds = load_dataset(args.eval_file, tokenizer, args.max_length)
    print(f"  Train: {len(train_ds)} examples")
    print(f"  Eval:  {len(eval_ds)} examples")

    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=bf16_supported,
        fp16=not bf16_supported and torch.cuda.is_available(),
        eval_strategy="epoch",   # new HF API (was evaluation_strategy)
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,  # new HF API (was tokenizer=)
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )

    print("\nStarting training...")
    trainer.train()

    print("\nFinal evaluation:")
    metrics = trainer.evaluate()
    print(metrics)

    # IMPORTANT: merge_and_unload before saving so inference pipeline can load
    # a regular HF model without needing PEFT to be installed.
    print("\nMerging LoRA adapters into base model...")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Per-label classification report
    print("\nGenerating per-label classification report...")
    preds = trainer.predict(eval_ds)
    pred_ids = np.argmax(preds.predictions, axis=2)
    true_labels = [
        [ID2LABEL[l] for l in label if l != -100]
        for label in preds.label_ids
    ]
    pred_labels = [
        [ID2LABEL[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(pred_ids, preds.label_ids)
    ]
    report = classification_report(true_labels, pred_labels, digits=4)
    print(report)

    with open(f"{args.output_dir}/benchmark.txt", "w") as f:
        f.write(report)
        f.write(f"\n\nFinal metrics: {metrics}\n")

    print(f"\nDone! Model saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
