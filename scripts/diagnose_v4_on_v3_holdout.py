"""Diagnose v4's catastrophic drop on holdout_v3 (F1 0.55 vs 0.99 on own holdout).

Runs v4 on the v3 holdout and reports per-category precision/recall/F1, plus
sample examples of (a) v4 over-predicts (false positives) and (b) v4 misses
gold entities (false negatives).
"""
from __future__ import annotations

import json
from collections import defaultdict
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


def main() -> None:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

    print("Loading v4 from checkpoints/v4-local...")
    tok = AutoTokenizer.from_pretrained("checkpoints/v4-local", model_max_length=512)
    model = AutoModelForTokenClassification.from_pretrained("checkpoints/v4-local").eval()
    ner = pipeline("token-classification", model=model, tokenizer=tok,
                   aggregation_strategy="simple", device=-1)

    print("Loading holdout_v3...")
    with open(ROOT / "data/dataset_br_v3_holdout.jsonl") as f:
        examples = [json.loads(line) for line in f]
    print(f"  {len(examples)} examples")

    # Per-category TP/FP/FN
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)

    # Sample failures
    sample_fp: dict[str, list] = defaultdict(list)  # v4 predicts but not in gold
    sample_fn: dict[str, list] = defaultdict(list)  # gold has but v4 misses
    sample_relabel: list = []  # same span, different category

    max_chars = 1500

    for i, ex in enumerate(examples):
        if i % 500 == 0 and i > 0:
            print(f"  {i}/{len(examples)}...")
        text = ex["text"]
        if len(text) > max_chars:
            text = text[:max_chars]
        gold = {(e["start"], e["end"], e["label"]) for e in ex["entities"]
                if e["end"] <= len(text)}
        preds = ner(text)
        pred_set = {(p["start"], p["end"], p["entity_group"]) for p in preds}

        # Position-based comparison
        gold_by_pos = {(s, e): l for s, e, l in gold}
        pred_by_pos = {(s, e): l for s, e, l in pred_set}

        for sp in gold:
            label = sp[2]
            if sp in pred_set:
                tp[label] += 1
            else:
                fn[label] += 1
                if (sp[0], sp[1]) in pred_by_pos:
                    other = pred_by_pos[(sp[0], sp[1])]
                    if len(sample_relabel) < 30:
                        sample_relabel.append((i, sp, other, text[max(0, sp[0]-15):min(len(text), sp[1]+15)]))
                else:
                    if len(sample_fn[label]) < 4:
                        sample_fn[label].append((i, sp, text[max(0, sp[0]-15):min(len(text), sp[1]+15)]))

        for sp in pred_set:
            label = sp[2]
            if sp not in gold:
                fp[label] += 1
                if len(sample_fp[label]) < 4:
                    sample_fp[label].append((i, sp, text[max(0, sp[0]-15):min(len(text), sp[1]+15)]))

    print()
    print("Per-category metrics (v4 on holdout_v3):")
    print(f"{'category':<30} {'TP':>6} {'FP':>6} {'FN':>6} {'prec':>8} {'rec':>8} {'F1':>8}")
    print("-" * 78)
    all_labels = set(tp) | set(fp) | set(fn)
    rows = []
    for label in sorted(all_labels):
        t = tp[label]
        f_ = fp[label]
        n = fn[label]
        p = t / (t + f_) if (t + f_) > 0 else 0.0
        r = t / (t + n) if (t + n) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        rows.append((label, t, f_, n, p, r, f1))
    rows.sort(key=lambda x: x[6])  # ascending by F1 (worst first)
    for label, t, f_, n, p, r, f1 in rows:
        print(f"{label:<30} {t:>6} {f_:>6} {n:>6} {p:>8.4f} {r:>8.4f} {f1:>8.4f}")

    print()
    print("=" * 78)
    print("Worst 5 categories — sample failures")
    print("=" * 78)
    for label, t, f_, n, p, r, f1 in rows[:5]:
        print(f"\n### {label} (F1={f1:.4f}, FP={f_}, FN={n})")
        if sample_fp[label]:
            print("  False positives (v4 predicts, gold doesn't have):")
            for _, sp, ctx in sample_fp[label][:3]:
                print(f"    span={sp}, ctx=...{ctx!r}...")
        if sample_fn[label]:
            print("  False negatives (gold has, v4 misses):")
            for _, sp, ctx in sample_fn[label][:3]:
                print(f"    span={sp}, ctx=...{ctx!r}...")

    print()
    print("=" * 78)
    print(f"Re-labels (same span, different category): {len(sample_relabel)} examples")
    print("=" * 78)
    confusion = defaultdict(int)
    for _, gold_sp, v4_pred, _ in sample_relabel:
        confusion[(gold_sp[2], v4_pred)] += 1
    for (gold, pred), count in sorted(confusion.items(), key=lambda x: -x[1])[:10]:
        print(f"  gold={gold} → v4={pred}: {count}")

    print()
    print("Sample re-labels (top 10):")
    for line_no, gold_sp, v4_pred, ctx in sample_relabel[:10]:
        print(f"  line {line_no}: gold={gold_sp[2]:<25} v4={v4_pred:<25} text={ctx!r}")


if __name__ == "__main__":
    main()
