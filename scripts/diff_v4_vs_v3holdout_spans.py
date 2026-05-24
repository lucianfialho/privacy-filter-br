"""Side-by-side comparison of v4 predictions vs v3-holdout gold spans.

For each example with mismatches, show:
  GOLD:  text[gold_start:gold_end] (start, end, label)
  V4:    text[v4_start:v4_end] (start, end, label)

Filter to overlapping spans (same approximate position, different boundary or label)
to focus on the boundary-mismatch hypothesis.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and a[1] > b[0]


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

    max_chars = 1500
    shown = 0
    boundary_off_by_n: dict[int, int] = {}  # delta → count

    for i, ex in enumerate(examples):
        if shown >= 15:
            break
        text = ex["text"]
        if len(text) > max_chars:
            text = text[:max_chars]
        gold = [e for e in ex["entities"] if e["end"] <= len(text)]
        preds = ner(text)
        v4_spans = [{"start": p["start"], "end": p["end"], "label": p["entity_group"]} for p in preds]

        # For each gold, find best-overlapping v4 span
        for g in gold:
            for v in v4_spans:
                if not overlaps((g["start"], g["end"]), (v["start"], v["end"])):
                    continue
                # Found overlap
                start_delta = v["start"] - g["start"]
                end_delta = v["end"] - g["end"]
                total_delta = abs(start_delta) + abs(end_delta)
                boundary_off_by_n[total_delta] = boundary_off_by_n.get(total_delta, 0) + 1

                # Skip exact matches (no need to show)
                if total_delta == 0 and g["label"] == v["label"]:
                    continue

                if shown < 15:
                    gold_text = text[g["start"]:g["end"]]
                    v4_text = text[v["start"]:v["end"]]
                    ctx_s = max(0, min(g["start"], v["start"]) - 10)
                    ctx_e = min(len(text), max(g["end"], v["end"]) + 10)
                    ctx = text[ctx_s:ctx_e]
                    print(f"\n--- Example {i} ---")
                    print(f"  Context: ...{ctx!r}...")
                    print(f"  GOLD: [{g['start']}:{g['end']}] {gold_text!r} ({g['label']})")
                    print(f"  V4:   [{v['start']}:{v['end']}] {v4_text!r}  ({v['label']})")
                    print(f"  delta: start={start_delta:+}, end={end_delta:+}, label_match={g['label']==v['label']}")
                    shown += 1
                break  # one v4 match per gold is enough

    print()
    print("=" * 70)
    print("Boundary delta distribution (start_diff + end_diff abs):")
    print("=" * 70)
    total = sum(boundary_off_by_n.values())
    for delta in sorted(boundary_off_by_n):
        count = boundary_off_by_n[delta]
        pct = 100 * count / total if total else 0
        bar = "█" * int(pct / 2)
        print(f"  Δ={delta:>3}  {count:>5} ({pct:>5.1f}%)  {bar}")


if __name__ == "__main__":
    main()
