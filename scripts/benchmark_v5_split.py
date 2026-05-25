"""v5 split benchmark: evaluate v5 separately on the v3-style and v4-style halves
of dataset_br_v5_holdout.jsonl.

dataset_br_v5_holdout = dataset_br_v3_holdout + dataset_br_v4_holdout (concat in that order).
So:
  - First 4929 lines  → v3-style (Haiku-rewritten)
  - Last  4964 lines  → v4-style (gpt-5-nano-rewritten)

If v5 truly generalizes both styles, both subsets should be >= 0.99 F1.
If one drops significantly, v5 still has style-bias.

Also evaluates v3 and v4 on the same two subsets for direct comparison.

Uses seqeval BIOES F1 (same metric as training).
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
        "n_examples": len(rows),
    }


def main() -> None:
    print("Loading holdouts...")
    with open(ROOT / "data/dataset_br_v3_holdout.jsonl") as f:
        v3_examples = [json.loads(line) for line in f]
    with open(ROOT / "data/dataset_br_v4_holdout.jsonl") as f:
        v4_examples = [json.loads(line) for line in f]
    print(f"  v3-style (Haiku):     {len(v3_examples)}")
    print(f"  v4-style (gpt5nano):  {len(v4_examples)}")
    print(f"  v5 combined holdout:  {len(v3_examples) + len(v4_examples)}")

    cells = [
        ("v3", ROOT / "checkpoints/v3-local"),
        ("v4", ROOT / "checkpoints/v4-local"),
        ("v5", ROOT / "checkpoints/v5-local"),
    ]

    results = {}
    for model_name, model_dir in cells:
        if not (model_dir / "model.safetensors").exists():
            print(f"\nSKIPPING {model_name} — {model_dir}/model.safetensors not found")
            continue
        print(f"\n=== {model_name} on v3-style (Haiku) holdout ===")
        r_v3 = evaluate(model_dir, v3_examples)
        print(f"  precision={r_v3['precision']:.4f} recall={r_v3['recall']:.4f} f1={r_v3['f1']:.4f}")
        print(f"=== {model_name} on v4-style (gpt5nano) holdout ===")
        r_v4 = evaluate(model_dir, v4_examples)
        print(f"  precision={r_v4['precision']:.4f} recall={r_v4['recall']:.4f} f1={r_v4['f1']:.4f}")
        results[model_name] = {"v3_style": r_v3, "v4_style": r_v4}

    print("\n" + "=" * 70)
    print("F1 MATRIX (model × holdout-style):")
    print("=" * 70)
    print(f"{'':<10} {'v3-style (Haiku)':>20} {'v4-style (gpt5nano)':>22} {'Δ':>10}")
    for model_name in results:
        r3 = results[model_name]["v3_style"]["f1"]
        r4 = results[model_name]["v4_style"]["f1"]
        delta = r4 - r3
        print(f"{model_name:<10} {r3:>20.4f} {r4:>22.4f} {delta:>+10.4f}")

    print("\nVerdicts:")
    if "v5" in results:
        v5_v3 = results["v5"]["v3_style"]["f1"]
        v5_v4 = results["v5"]["v4_style"]["f1"]
        spread = abs(v5_v4 - v5_v3)
        print(f"  v5 spread (|v4-v3|):  {spread:.4f}")
        if spread < 0.005 and min(v5_v3, v5_v4) > 0.99:
            print("  ✅ v5 generalizes both styles. Candidate to replace v3 as default.")
        elif spread < 0.02:
            print("  ⚠️  v5 has minor style bias but both >= acceptable. Inspect per-category.")
        else:
            print("  ❌ v5 still style-biased. Mixing wasn't enough.")

    out = ROOT / "data/benchmark_v5_split.json"
    out.write_text(json.dumps({m: {k: v for k, v in r.items()} for m, r in results.items()},
                              indent=2))
    print(f"\nFull results saved to {out}")


if __name__ == "__main__":
    main()
