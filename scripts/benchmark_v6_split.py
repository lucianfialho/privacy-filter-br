"""v6 cross-style benchmark: 4 models × 3 styles = 12 cells.

Models: v3, v4, v5, v6
Styles:
  - v3_style: data/dataset_br_v3_holdout.jsonl (Haiku-rewritten, structured templates)
  - v4_style: data/dataset_br_v4_holdout.jsonl (gpt5nano-rewritten, structured templates)
  - v6_narrative: data/dataset_br_v6_new_holdout.jsonl (gpt5nano-rewritten, narrative templates)

Goal: confirm v6 generalizes all three styles without regression.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

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


def char_spans_to_token_tags(text, entities, tokenizer, max_length=256):
    encoding = tokenizer(text, max_length=max_length, truncation=True,
                        return_offsets_mapping=True, padding=False)
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


def evaluate(model_dir: Path, examples: list) -> dict:
    from datasets import Dataset
    from transformers import (
        AutoModelForTokenClassification, AutoTokenizer,
        DataCollatorForTokenClassification, Trainer, TrainingArguments,
    )
    from seqeval.metrics import f1_score, precision_score, recall_score

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), model_max_length=512)
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))

    rows = []
    for ex in examples:
        try:
            rows.append(char_spans_to_token_tags(ex["text"], ex["entities"], tokenizer))
        except Exception:
            pass
    eval_ds = Dataset.from_list(rows)

    args = TrainingArguments(
        output_dir="/tmp/bench_ignored",
        per_device_eval_batch_size=16,
        report_to="none",
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=args, eval_dataset=eval_ds,
                     processing_class=tokenizer,
                     data_collator=DataCollatorForTokenClassification(tokenizer))
    preds = trainer.predict(eval_ds)
    pred_ids = np.argmax(preds.predictions, axis=2)
    true_labels = [[ID2LABEL[l] for l in lab if l != -100] for lab in preds.label_ids]
    pred_labels = [[ID2LABEL[p_] for (p_, l) in zip(pred, lab) if l != -100]
                   for pred, lab in zip(pred_ids, preds.label_ids)]
    return {
        "f1": f1_score(true_labels, pred_labels),
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
        "n": len(rows),
    }


def main() -> None:
    print("Loading holdouts...")
    holdouts = {
        "v3_style (Haiku struct)": json.loads(line) for line in open(ROOT / "data/dataset_br_v3_holdout.jsonl")
    }
    # Re-do with correct dict comprehension
    def load(p):
        with open(p) as f:
            return [json.loads(l) for l in f]
    holdouts = {
        "v3_style (Haiku struct)": load(ROOT / "data/dataset_br_v3_holdout.jsonl"),
        "v4_style (gpt5nano struct)": load(ROOT / "data/dataset_br_v4_holdout.jsonl"),
        "v6_narrative (gpt5nano prose)": load(ROOT / "data/dataset_br_v6_new_holdout.jsonl"),
    }
    for name, ex in holdouts.items():
        print(f"  {name}: {len(ex)} examples")

    models = {
        "v3": ROOT / "checkpoints/v3-local",
        "v4": ROOT / "checkpoints/v4-local",
        "v5": ROOT / "checkpoints/v5-local",
        "v6": ROOT / "checkpoints/v6-local",
    }

    results = {}
    for m_name, m_dir in models.items():
        if not (m_dir / "model.safetensors").exists():
            print(f"\nSKIPPING {m_name}: no model file")
            continue
        results[m_name] = {}
        for h_name, examples in holdouts.items():
            print(f"\n=== {m_name} on {h_name} ===")
            r = evaluate(m_dir, examples)
            print(f"  precision={r['precision']:.4f} recall={r['recall']:.4f} f1={r['f1']:.4f}")
            results[m_name][h_name] = r

    print()
    print("=" * 90)
    print("F1 MATRIX")
    print("=" * 90)
    h_names = list(holdouts.keys())
    print(f"{'Model':<8} " + " ".join(f"{h:>30}" for h in h_names))
    for m in results:
        row = " ".join(f"{results[m][h]['f1']:>30.4f}" for h in h_names)
        print(f"{m:<8} {row}")

    print()
    print("Spread (max-min across styles, lower = better generalization):")
    for m in results:
        f1s = [results[m][h]["f1"] for h in h_names]
        spread = max(f1s) - min(f1s)
        print(f"  {m}: {spread:.4f}")

    out = ROOT / "data/benchmark_v6_split.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nFull results: {out}")


if __name__ == "__main__":
    main()
