#!/usr/bin/env python3
"""
Privacy Filter BR v3 — Hybrid taxonomy fine-tune for Google Colab.

Strategy: REUSE the original openai/privacy-filter head (33 labels) and
EXTEND it with 14 BR-specific categories. The original 33-label head weights
are copied into the first 33 positions of the new 89-label head; positions
33-88 are initialized fresh.

Why this is better than v2 reinit:
- Model retains capability for account_number, private_url, secret, private_date
- Transfer learning: shared head positions for person/email/phone/address
- Smaller fine-tune signal needed since most categories already learned

Categories (22 total):
    ORIGINAL OAI (8, positions 1-32 in head):
        account_number, private_address, private_date, private_email,
        private_person, private_phone, private_url, secret
    BR NEW (14, positions 33-88):
        private_cpf, private_cnpj, private_rg, private_cnh, private_pis,
        private_titulo_eleitor, private_certidao, private_ie,
        private_order_id, private_tracking_code, private_invoice_number,
        private_client_revenue, private_transaction_id, private_customer_id

Total labels: 1 (O) + 22*4 (BIOES) = 89

Usage:
    !python finetune_v3.py \\
        --train-file /content/dataset_br_v3.jsonl \\
        --eval-file  /content/dataset_br_v3_holdout.jsonl \\
        --output-dir /content/privacy-filter-br-v3
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


# === TAXONOMY ===
# IMPORTANT: ORIG_CATEGORIES order MUST match openai/privacy-filter config.json:
# 0=O, 1-4=account_number, 5-8=private_address, 9-12=private_date,
# 13-16=private_email, 17-20=private_person, 21-24=private_phone,
# 25-28=private_url, 29-32=secret
ORIG_CATEGORIES = [
    "account_number", "private_address", "private_date", "private_email",
    "private_person", "private_phone", "private_url", "secret",
]
NEW_CATEGORIES = [
    "private_cpf", "private_cnpj", "private_rg", "private_cnh", "private_pis",
    "private_titulo_eleitor", "private_certidao", "private_ie",
    "private_order_id", "private_tracking_code", "private_invoice_number",
    "private_client_revenue", "private_transaction_id", "private_customer_id",
]
CATEGORIES = ORIG_CATEGORIES + NEW_CATEGORIES

LABELS = ["O"] + [f"{tag}-{cat}" for cat in CATEGORIES for tag in ["B", "I", "E", "S"]]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

NUM_ORIG_LABELS = 1 + len(ORIG_CATEGORIES) * 4  # 33
assert NUM_ORIG_LABELS == 33, "ORIG_CATEGORIES must match openai/privacy-filter 33-label head"
assert len(LABELS) == 89, f"Expected 89 total labels, got {len(LABELS)}"


# === DATA PROCESSING ===

def char_spans_to_token_tags(text: str, entities: list, tokenizer, max_length: int = 512):
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
            continue
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
            except Exception:
                skipped += 1
    if skipped:
        print(f"  skipped {skipped} examples (tokenization errors)")
    return Dataset.from_list(examples)


def compute_metrics_fn(p):
    preds, labels = p
    preds = np.argmax(preds, axis=2)
    true_labels = [[ID2LABEL[l] for l in lab if l != -100] for lab in labels]
    true_preds = [[ID2LABEL[p_] for (p_, l) in zip(pred, lab) if l != -100]
                  for pred, lab in zip(preds, labels)]
    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
    }


def find_score_layer(model):
    """The head layer might be called .score (Privacy Filter) or .classifier."""
    for name in ["score", "classifier"]:
        if hasattr(model, name):
            return getattr(model, name)
    raise AttributeError("Could not find score/classifier layer on model")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="openai/privacy-filter")
    parser.add_argument("--train-file", default="/content/dataset_br_v3.jsonl")
    parser.add_argument("--eval-file", default="/content/dataset_br_v3_holdout.jsonl")
    parser.add_argument("--output-dir", default="/content/privacy-filter-br-v3")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Step 1: Load original model (33 labels) to grab head weights
    print("Loading ORIGINAL head (33 labels) to copy weights from...")
    orig_model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        dtype=torch.float32,
    )
    orig_score = find_score_layer(orig_model)
    orig_W = orig_score.weight.data.clone()
    orig_b = orig_score.bias.data.clone() if orig_score.bias is not None else None
    assert orig_W.shape[0] == NUM_ORIG_LABELS, (
        f"Original head has {orig_W.shape[0]} labels, expected {NUM_ORIG_LABELS}"
    )
    del orig_model

    # Step 2: Load with extended head (89 labels), random init
    print(f"Loading base with EXTENDED head ({len(LABELS)} labels, ignore_mismatched_sizes)...")
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
        dtype=torch.float32,
    )

    # Step 3: Copy original weights into first NUM_ORIG_LABELS positions
    print(f"Copying original head weights into positions [0..{NUM_ORIG_LABELS - 1}]...")
    new_score = find_score_layer(model)
    with torch.no_grad():
        new_score.weight.data[:NUM_ORIG_LABELS] = orig_W
        if orig_b is not None and new_score.bias is not None:
            new_score.bias.data[:NUM_ORIG_LABELS] = orig_b
    # Positions [NUM_ORIG_LABELS:] keep their fresh init (small Gaussian by default)

    # Step 4: Move to GPU BEFORE wrapping with PEFT
    if torch.cuda.is_available():
        model = model.cuda()

    # Step 5: LoRA
    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        modules_to_save=["score"],  # privacy-filter uses 'score', not 'classifier'
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
        eval_strategy="epoch",
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
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )

    print("\nStarting training...")
    trainer.train()

    print("\nFinal evaluation:")
    metrics = trainer.evaluate()
    print(metrics)

    print("\nMerging LoRA adapters and saving merged model...")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\nPer-label classification report:")
    preds = trainer.predict(eval_ds)
    pred_ids = np.argmax(preds.predictions, axis=2)
    true_labels = [[ID2LABEL[l] for l in lab if l != -100] for lab in preds.label_ids]
    pred_labels = [[ID2LABEL[p_] for (p_, l) in zip(pred, lab) if l != -100]
                   for pred, lab in zip(pred_ids, preds.label_ids)]
    report = classification_report(true_labels, pred_labels, digits=4)
    print(report)

    with open(f"{args.output_dir}/benchmark.txt", "w") as f:
        f.write(report)
        f.write(f"\n\nFinal metrics: {metrics}\n")

    print(f"\nDone! Model saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
