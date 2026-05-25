"""Phase 1 B2B benchmark: model on real CVM FRE board-member docs.

Loads data/phase1_b2b.jsonl and scores per category with both exact and
overlap matching, plus delta vs synthetic-holdout F1 — quantifies the
synthetic→real gap. Boundary merger (from br_pii_guardrail) applied by
default — fixes BIOES sub-token fragmentation that broke exact-match.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "br-pii-guardrail/src"))
from br_pii_guardrail.ner import _merge_adjacent_spans  # noqa: E402

SYNTHETIC_F1 = {
    "private_cnpj": 0.9970,
    "private_person": 0.9955,
    "private_cpf": 0.9964,
    "private_date": 1.0000,
}


def overlaps(a_s, a_e, b_s, b_e):
    return a_s < b_e and a_e > b_s


def main() -> None:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/v7-local")
    ap.add_argument("--input", default=str(ROOT / "data/phase1_b2b.jsonl"))
    ap.add_argument("--no-merge", action="store_true", help="disable boundary merger")
    args = ap.parse_args()

    with open(args.input) as f:
        docs = [json.loads(line) for line in f]
    print(f"Loaded {len(docs)} B2B docs from {args.input}")
    print(f"Loading {args.model} (boundary merger {'OFF' if args.no_merge else 'ON'})...")
    tok = AutoTokenizer.from_pretrained(args.model, model_max_length=512)
    model = AutoModelForTokenClassification.from_pretrained(args.model).eval()
    ner = pipeline("token-classification", model=model, tokenizer=tok,
                   aggregation_strategy="simple", device=-1)

    # Track exact and overlap separately
    exact_tp = defaultdict(int)
    exact_fp = defaultdict(int)
    exact_fn = defaultdict(int)
    over_tp = defaultdict(int)
    over_fp = defaultdict(int)
    over_fn = defaultdict(int)

    gold_categories = {"private_cnpj", "private_person", "private_cpf", "private_date"}

    sample_frag = defaultdict(list)
    sample_miss = defaultdict(list)

    for doc in docs:
        text = doc["text"][:1500]
        gold_in_range = [e for e in doc["entities"] if e["end"] <= 1500]
        text_short = text
        preds = list(ner(text_short))
        if not args.no_merge:
            preds = _merge_adjacent_spans(preds, text_short, max_gap=3)

        gold_in_scope = [e for e in gold_in_range if e["label"] in gold_categories]
        pred_in_scope = [p for p in preds if p["entity_group"] in gold_categories]

        # === EXACT match scoring ===
        gold_keys = {(e["start"], e["end"], e["label"]) for e in gold_in_scope}
        pred_keys = {(p["start"], p["end"], p["entity_group"]) for p in pred_in_scope}
        for k in gold_keys:
            if k in pred_keys:
                exact_tp[k[2]] += 1
            else:
                exact_fn[k[2]] += 1
        for k in pred_keys:
            if k not in gold_keys:
                exact_fp[k[2]] += 1

        gold_matched = [False] * len(gold_in_scope)
        pred_matched = [False] * len(pred_in_scope)
        for gi, g in enumerate(gold_in_scope):
            for pi, p in enumerate(pred_in_scope):
                if g["label"] == p["entity_group"] and overlaps(g["start"], g["end"], p["start"], p["end"]):
                    gold_matched[gi] = True
                    pred_matched[pi] = True
                    g_text = text_short[g["start"]:g["end"]]
                    p_text = text_short[p["start"]:p["end"]]
                    if g_text != p_text and len(sample_frag[g["label"]]) < 4:
                        sample_frag[g["label"]].append((g_text, p_text))
        for gi, g in enumerate(gold_in_scope):
            if gold_matched[gi]:
                over_tp[g["label"]] += 1
            else:
                over_fn[g["label"]] += 1
                if len(sample_miss[g["label"]]) < 5:
                    sample_miss[g["label"]].append(text_short[g["start"]:g["end"]])
        for pi, p in enumerate(pred_in_scope):
            if not pred_matched[pi]:
                over_fp[p["entity_group"]] += 1

    def f1(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return p, r, (2 * p * r / (p + r) if (p + r) > 0 else 0.0)

    print()
    print("=" * 80)
    print(f"{args.model} on REAL CVM B2B (merger={'off' if args.no_merge else 'on'}, n={len(docs)})")
    print("=" * 80)
    print(f"{'category':<22} {'mode':<8} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>8} {'R':>8} {'F1':>8} {'synth':>8} {'Δ':>8}")
    print("-" * 80)
    for label in sorted(gold_categories):
        p, r, f = f1(exact_tp[label], exact_fp[label], exact_fn[label])
        synth = SYNTHETIC_F1.get(label, 0)
        print(f"{label:<22} {'exact':<8} {exact_tp[label]:>4} {exact_fp[label]:>4} {exact_fn[label]:>4} {p:>8.4f} {r:>8.4f} {f:>8.4f} {synth:>8.4f} {f-synth:>+8.4f}")
        p, r, f = f1(over_tp[label], over_fp[label], over_fn[label])
        print(f"{label:<22} {'overlap':<8} {over_tp[label]:>4} {over_fp[label]:>4} {over_fn[label]:>4} {p:>8.4f} {r:>8.4f} {f:>8.4f} {synth:>8.4f} {f-synth:>+8.4f}")
        print()

    print("\nSample fragmented matches (boundary mismatch):")
    for label, examples in sample_frag.items():
        for gold_text, pred_text in examples:
            print(f"  [{label}] gold={gold_text!r}  pred={pred_text!r}")
    print("\nSample misses (no overlapping span):")
    for label, examples in sample_miss.items():
        for gold_text in examples[:3]:
            print(f"  [{label}] {gold_text!r}")


if __name__ == "__main__":
    main()
