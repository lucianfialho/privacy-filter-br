#!/usr/bin/env python3
"""
Privacy Filter BR v3 — Local fine-tune (small GPU: 8GB, Turing+).

Adapted from finetune_v3.py for resource-constrained setups:
- Base model: neuralmind/bert-base-portuguese-cased (BERTimbau, 110M)
  instead of openai/privacy-filter (1.5B). Fits comfortably in 8GB.
- Reinit head from scratch (no OAI transfer) — BERTimbau has no PII head
  to reuse, so we initialize all 89 BIOES labels fresh.
- fp16 instead of bf16 (Turing doesn't support bf16).
- Smaller max_length default (256) to save memory.

Same 22-category hybrid taxonomy as v3 (8 OAI + 14 BR).

Usage on local machine:
    python finetune_v3_local.py \\
        --train-file data/dataset_br_v3.jsonl \\
        --eval-file  data/dataset_br_v3_holdout.jsonl \\
        --output-dir checkpoints/v3-local
"""
import os
import json
import argparse

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
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report


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
assert len(LABELS) == 89


def char_spans_to_token_tags(text, entities, tokenizer, max_length):
    encoding = tokenizer(
        text, max_length=max_length, truncation=True,
        return_offsets_mapping=True, padding=False,
    )
    offsets = encoding["offset_mapping"]
    tags = [LABEL2ID["O"]] * len(offsets)
    for ent in sorted(entities, key=lambda e: e["start"]):
        if ent["label"] not in CATEGORIES:
            continue
        idxs = [i for i, (s, e) in enumerate(offsets)
                if s < ent["end"] and e > ent["start"] and s != e]
        if not idxs:
            continue
        if len(idxs) == 1:
            tags[idxs[0]] = LABEL2ID[f"S-{ent['label']}"]
        else:
            tags[idxs[0]] = LABEL2ID[f"B-{ent['label']}"]
            tags[idxs[-1]] = LABEL2ID[f"E-{ent['label']}"]
            for i in idxs[1:-1]:
                tags[i] = LABEL2ID[f"I-{ent['label']}"]
    encoding.pop("offset_mapping")
    encoding["labels"] = tags
    return encoding


def load_dataset(path, tokenizer, max_length):
    examples = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            try:
                examples.append(char_spans_to_token_tags(
                    ex["text"], ex["entities"], tokenizer, max_length))
            except Exception:
                pass
    return Dataset.from_list(examples)


def compute_metrics_fn(p):
    preds = np.argmax(p.predictions, axis=2)
    labels = p.label_ids
    true_labels = [[ID2LABEL[l] for l in lab if l != -100] for lab in labels]
    true_preds = [[ID2LABEL[p_] for (p_, l) in zip(pred, lab) if l != -100]
                  for pred, lab in zip(preds, labels)]
    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="neuralmind/bert-base-portuguese-cased")
    parser.add_argument("--train-file", default="data/dataset_br_v3.jsonl")
    parser.add_argument("--eval-file", default="data/dataset_br_v3_holdout.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/v3-local")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Loading base with {len(LABELS)}-label head (fresh init)...")
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  bf16 supported: {torch.cuda.is_bf16_supported()}")
    elif torch.backends.mps.is_available():
        model = model.to("mps")
        print("  GPU: Apple Silicon (MPS)")

    print("Loading datasets...")
    train_ds = load_dataset(args.train_file, tokenizer, args.max_length)
    eval_ds = load_dataset(args.eval_file, tokenizer, args.max_length)
    print(f"  Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    fp16 = torch.cuda.is_available() and not bf16
    use_mps = (not torch.cuda.is_available()) and torch.backends.mps.is_available()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=bf16,
        fp16=fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0 if use_mps else 2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics_fn,
    )

    print("\nStarting training...")
    trainer.train()

    print("\nFinal evaluation:")
    print(trainer.evaluate())

    print("\nSaving model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    preds = trainer.predict(eval_ds)
    pred_ids = np.argmax(preds.predictions, axis=2)
    true_labels = [[ID2LABEL[l] for l in lab if l != -100] for lab in preds.label_ids]
    pred_labels = [[ID2LABEL[p_] for (p_, l) in zip(pred, lab) if l != -100]
                   for pred, lab in zip(pred_ids, preds.label_ids)]
    report = classification_report(true_labels, pred_labels, digits=4)
    print(report)
    with open(f"{args.output_dir}/benchmark.txt", "w") as f:
        f.write(report)
    print(f"\nDone! Model saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
