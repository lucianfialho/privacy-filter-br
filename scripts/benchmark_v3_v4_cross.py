"""Cross-holdout benchmark: v3 and v4 evaluated on BOTH holdouts.

The 2x2 we still need to be honest about v4's improvement:

|                | holdout v3 (buggy) | holdout v4 (clean) |
|----------------|--------------------|--------------------|
| v3             | already published  | needs computing    |
| v4             | needs computing    | already published  |

If v4 wins both, the improvement is real. If v4 only wins on its own holdout,
the gap is partly an artifact of label-quality differences in the eval set.

Uses seqeval BIOES-level metrics to match the training-time benchmark.txt numbers.
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


def evaluate(model_dir: Path, holdout_path: Path) -> dict:
    from datasets import Dataset
    from transformers import (
        AutoModelForTokenClassification, AutoTokenizer,
        DataCollatorForTokenClassification, Trainer, TrainingArguments,
    )
    from seqeval.metrics import f1_score, precision_score, recall_score

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), model_max_length=512)
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))

    examples = []
    with holdout_path.open() as f:
        for line in f:
            ex = json.loads(line)
            try:
                examples.append(char_spans_to_token_tags(ex["text"], ex["entities"], tokenizer))
            except Exception:
                pass
    eval_ds = Dataset.from_list(examples)

    args = TrainingArguments(
        output_dir="/tmp/bench_ignored",
        per_device_eval_batch_size=16,
        report_to="none",
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model, args=args, eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )
    preds = trainer.predict(eval_ds)
    pred_ids = np.argmax(preds.predictions, axis=2)
    true_labels = [[ID2LABEL[l] for l in lab if l != -100] for lab in preds.label_ids]
    pred_labels = [[ID2LABEL[p_] for (p_, l) in zip(pred, lab) if l != -100]
                   for pred, lab in zip(pred_ids, preds.label_ids)]

    return {
        "f1": f1_score(true_labels, pred_labels),
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
        "n_examples": len(examples),
    }


def main() -> None:
    cells = [
        ("v3", "holdout_v3", ROOT / "checkpoints/v3-local", ROOT / "data/dataset_br_v3_holdout.jsonl"),
        ("v3", "holdout_v4", ROOT / "checkpoints/v3-local", ROOT / "data/dataset_br_v4_holdout.jsonl"),
        ("v4", "holdout_v3", ROOT / "checkpoints/v4-local", ROOT / "data/dataset_br_v3_holdout.jsonl"),
        ("v4", "holdout_v4", ROOT / "checkpoints/v4-local", ROOT / "data/dataset_br_v4_holdout.jsonl"),
    ]
    results = {}
    for model_name, holdout_name, model_dir, holdout_path in cells:
        print(f"\n=== {model_name} on {holdout_name} ===")
        print(f"  model: {model_dir}")
        print(f"  data:  {holdout_path}")
        r = evaluate(model_dir, holdout_path)
        results[(model_name, holdout_name)] = r
        print(f"  precision={r['precision']:.4f} recall={r['recall']:.4f} f1={r['f1']:.4f} (n={r['n_examples']})")

    print("\n=== 2x2 RESULTS (seqeval BIOES F1) ===\n")
    print(f"{'':<10} {'holdout_v3':>14} {'holdout_v4':>14}")
    for m in ("v3", "v4"):
        h3 = results[(m, "holdout_v3")]["f1"]
        h4 = results[(m, "holdout_v4")]["f1"]
        print(f"{m:<10} {h3:>14.4f} {h4:>14.4f}")

    print("\nKey deltas:")
    v3_h3 = results[("v3", "holdout_v3")]["f1"]
    v3_h4 = results[("v3", "holdout_v4")]["f1"]
    v4_h3 = results[("v4", "holdout_v3")]["f1"]
    v4_h4 = results[("v4", "holdout_v4")]["f1"]
    print(f"  v4 vs v3 on holdout_v3: {v4_h3 - v3_h3:+.4f}  (if positive, v4 wins on the buggy holdout — strong signal)")
    print(f"  v4 vs v3 on holdout_v4: {v4_h4 - v3_h4:+.4f}  (if positive, v4 wins on clean holdout)")
    print(f"  v3 on rel vs orig:      {v3_h4 - v3_h3:+.4f}  (v3 regression on clean labels — confirms it learned the bug)")
    print(f"  v4 on rel vs orig:      {v4_h4 - v4_h3:+.4f}  (v4 should be roughly equal on both)")

    out = ROOT / "data/benchmark_v3_v4_cross.json"
    with out.open("w") as f:
        json.dump({f"{m}_{h}": v for (m, h), v in results.items()}, f, indent=2)
    print(f"\nFull results saved to {out}")


if __name__ == "__main__":
    main()
