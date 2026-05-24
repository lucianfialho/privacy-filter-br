"""A/B comparison: v3 (original) vs v3.1 (trained on relabeled) on the SAME sample.

Tests the hypothesis that v3.1 actually improved (despite F1 drop on holdout).
Mechanics:
  1. Sample N examples from dataset_br_v3.jsonl (same seed as audit_v3_disagreement.py).
  2. Run both models on those texts.
  3. For each example, compute:
     - v3 disagreement with ORIGINAL gold (string-match labels)
     - v3.1 disagreement with ORIGINAL gold
     - v3.1 disagreement with RELABELED gold (regex-augmented labels)
     - v3 vs v3.1 agreement (do they predict the same spans?)
  4. Report aggregate stats. Key claim: v3.1 disagreement with original gold should
     stay similar (still finding "extras" the labeler missed), BUT v3.1 disagreement
     with relabeled gold should be much LOWER if it learned the relabel.

Also reports per-category extras/misses for both models.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_ORIG = ROOT / "data" / "dataset_br_v3.jsonl"
DATASET_RELABELED = ROOT / "data" / "dataset_br_v3_relabeled.jsonl"
MODEL_V3 = ROOT / "checkpoints" / "v3-local"
MODEL_V31 = ROOT / "checkpoints" / "v3.1-local"


def spans_to_set(entities: list[dict]) -> set[tuple]:
    return {(e["start"], e["end"], e["label"]) for e in entities}


def load_pipeline(model_dir: Path):
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
    tok = AutoTokenizer.from_pretrained(str(model_dir), model_max_length=512)
    mdl = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    return pipeline(
        "token-classification", model=mdl, tokenizer=tok,
        aggregation_strategy="first", device=-1,
    )


def predictions(pipe, text: str, max_chars: int = 1500) -> set[tuple]:
    if len(text) > max_chars:
        text = text[:max_chars]
    preds = pipe(text)
    return {(p["start"], p["end"], p["entity_group"]) for p in preds}, text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load both versions of the dataset, aligned by line index
    print("Loading datasets (original + relabeled)...")
    with DATASET_ORIG.open() as f_orig, DATASET_RELABELED.open() as f_rel:
        orig_rows = [json.loads(line) for line in f_orig]
        rel_rows = [json.loads(line) for line in f_rel]
    assert len(orig_rows) == len(rel_rows), "datasets must be aligned"

    indices = random.sample(range(len(orig_rows)), min(args.sample, len(orig_rows)))
    print(f"Sample: {len(indices)} examples")

    print(f"Loading v3 from {MODEL_V3}...")
    pipe_v3 = load_pipeline(MODEL_V3)
    print(f"Loading v3.1 from {MODEL_V31}...")
    pipe_v31 = load_pipeline(MODEL_V31)

    n = 0
    # Disagreement counters: predicted span NOT in gold
    v3_extras_vs_orig = 0
    v3_extras_vs_rel = 0
    v31_extras_vs_orig = 0
    v31_extras_vs_rel = 0

    # Per-category extras
    v3_extras_cat: Counter[str] = Counter()
    v31_extras_cat_orig: Counter[str] = Counter()
    v31_extras_cat_rel: Counter[str] = Counter()

    # Cross-model agreement
    v3_v31_overlap = 0
    v3_only_count = 0  # span in v3 prediction but not in v3.1
    v31_only_count = 0  # span in v3.1 prediction but not in v3

    for idx_pos, idx in enumerate(indices):
        if idx_pos % 50 == 0 and idx_pos > 0:
            print(f"  {idx_pos}/{len(indices)}...")
        orig_ex = orig_rows[idx]
        rel_ex = rel_rows[idx]
        text = orig_ex["text"]

        preds_v3, text_truncated = predictions(pipe_v3, text)
        preds_v31, _ = predictions(pipe_v31, text)

        # Restrict gold labels to within truncated range
        gold_orig = {(e["start"], e["end"], e["label"]) for e in orig_ex["entities"]
                     if e["end"] <= len(text_truncated)}
        gold_rel = {(e["start"], e["end"], e["label"]) for e in rel_ex["entities"]
                    if e["end"] <= len(text_truncated)}

        # Predicted spans NOT in gold (disagreement / extras)
        for sp in preds_v3 - gold_orig:
            v3_extras_vs_orig += 1
            v3_extras_cat[sp[2]] += 1
        for sp in preds_v3 - gold_rel:
            v3_extras_vs_rel += 1
        for sp in preds_v31 - gold_orig:
            v31_extras_vs_orig += 1
            v31_extras_cat_orig[sp[2]] += 1
        for sp in preds_v31 - gold_rel:
            v31_extras_vs_rel += 1
            v31_extras_cat_rel[sp[2]] += 1

        # Cross-model
        v3_v31_overlap += len(preds_v3 & preds_v31)
        v3_only_count += len(preds_v3 - preds_v31)
        v31_only_count += len(preds_v31 - preds_v3)

        n += 1

    print()
    print(f"=== Summary over {n} examples ===\n")
    print("Predicted spans NOT in gold (disagreement count, lower is better):")
    print(f"  v3   vs original gold:   {v3_extras_vs_orig}")
    print(f"  v3   vs relabeled gold:  {v3_extras_vs_rel}  (Δ={v3_extras_vs_rel - v3_extras_vs_orig:+})")
    print(f"  v3.1 vs original gold:   {v31_extras_vs_orig}")
    print(f"  v3.1 vs relabeled gold:  {v31_extras_vs_rel}  (Δ={v31_extras_vs_rel - v31_extras_vs_orig:+})")
    print()
    if v3_extras_vs_orig > 0:
        print(f"v3.1 reduces disagreement vs ORIGINAL gold by: "
              f"{(v3_extras_vs_orig - v31_extras_vs_orig)*100/v3_extras_vs_orig:.1f}%")
    if v3_extras_vs_rel > 0:
        print(f"v3.1 reduces disagreement vs RELABELED gold by: "
              f"{(v3_extras_vs_rel - v31_extras_vs_rel)*100/v3_extras_vs_rel:.1f}%")

    print()
    print("Cross-model:")
    total_v3 = v3_v31_overlap + v3_only_count
    total_v31 = v3_v31_overlap + v31_only_count
    print(f"  v3 total predictions:    {total_v3}")
    print(f"  v3.1 total predictions:  {total_v31}")
    print(f"  v3 ∩ v3.1 (agree):       {v3_v31_overlap}")
    print(f"  v3 only:                 {v3_only_count}")
    print(f"  v3.1 only (new finds):   {v31_only_count}")

    print()
    print("Top categories where v3 finds extras vs original gold:")
    for lbl, c in v3_extras_cat.most_common(8):
        v31_c = v31_extras_cat_orig.get(lbl, 0)
        delta = v31_c - c
        print(f"  {lbl:<26} v3={c:>5}  v3.1={v31_c:>5}  Δ={delta:+}")

    print()
    print("Same categories vs RELABELED gold (fairer comparison for v3.1):")
    for lbl in [k for k, _ in v3_extras_cat.most_common(8)]:
        v3_rel = v3_extras_cat.get(lbl, 0)  # approximation; v3 uses orig
        v31_rel = v31_extras_cat_rel.get(lbl, 0)
        print(f"  {lbl:<26} v3.1_vs_rel={v31_rel:>5}")


if __name__ == "__main__":
    main()
