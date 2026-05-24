"""Audit 3: v3 disagreement audit on a sample of the training set.

Runs v3 inference on N random examples from dataset_br_v3.jsonl and compares
predicted spans with labeled spans. Reports:
- example-level disagreement rate
- per-category precision/recall as if labels were ground truth
- confidence distribution for predicted spans
- examples where v3 finds MORE entities than labeled (suggests labeler miss)
- examples where v3 finds FEWER entities than labeled (suggests label noise)
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "dataset_br_v3.jsonl"
MODEL_DIR = ROOT / "checkpoints" / "v3-local"


def spans_to_set(entities: list[dict]) -> set[tuple]:
    return {(e["start"], e["end"], e["label"]) for e in entities}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=2000,
                        help="number of training examples to audit")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading dataset from {DATASET}...")
    with DATASET.open() as f:
        all_examples = [json.loads(line) for line in f]
    print(f"  total: {len(all_examples)}")

    sample = random.sample(all_examples, min(args.sample, len(all_examples)))
    print(f"  sampling: {len(sample)}")

    print(f"Loading model from {MODEL_DIR}...")
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), model_max_length=512)
    model = AutoModelForTokenClassification.from_pretrained(str(MODEL_DIR))
    pipe = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="first",
        device=-1,
    )
    max_chars = 1500  # truncate input texts to stay under 512 tokens
    n_truncated = 0

    n_examples = len(sample)
    fully_agree = 0
    examples_v3_finds_more = 0  # v3 finds entities the labeler missed
    examples_v3_finds_less = 0  # v3 misses entities the labeler caught
    examples_v3_relabels = 0    # v3 predicts same span with different label

    extra_per_label: Counter[str] = Counter()    # v3 finds, labeler missed
    missed_per_label: Counter[str] = Counter()   # labeler has, v3 doesn't
    confusion: Counter[tuple[str, str]] = Counter()  # (gold_label, pred_label)

    confidences: list[float] = []
    sample_extras: list[tuple[int, dict, str]] = []

    for idx, ex in enumerate(sample):
        if idx % 200 == 0 and idx > 0:
            print(f"  {idx}/{n_examples}...")
        text = ex["text"]
        if len(text) > max_chars:
            text = text[:max_chars]
            n_truncated += 1
        # Filter gold entities to ones within truncated range
        gold_entities_in_range = [e for e in ex["entities"] if e["end"] <= len(text)]
        gold_set = spans_to_set(gold_entities_in_range)
        gold_by_position = {(e["start"], e["end"]): e["label"] for e in gold_entities_in_range}

        preds = pipe(text)
        pred_set = {(p["start"], p["end"], p["entity_group"]) for p in preds}
        pred_by_position = {(p["start"], p["end"]): p["entity_group"] for p in preds}

        for p in preds:
            confidences.append(float(p["score"]))

        if pred_set == gold_set:
            fully_agree += 1
            continue

        v3_extras = pred_set - gold_set
        v3_missed = gold_set - pred_set
        relabeled = False

        # Find overlapping positions with different labels
        for pos, pred_lbl in pred_by_position.items():
            if pos in gold_by_position and gold_by_position[pos] != pred_lbl:
                relabeled = True
                confusion[(gold_by_position[pos], pred_lbl)] += 1
                # also record under extras since it's a v3 disagreement
                extra_per_label[pred_lbl] += 1
                missed_per_label[gold_by_position[pos]] += 1

        # entities v3 found that gold doesn't have at all
        for s, e, lbl in v3_extras:
            if (s, e) not in gold_by_position:
                extra_per_label[lbl] += 1
                if len(sample_extras) < 12:
                    ctx_start = max(0, s - 20)
                    ctx_end = min(len(text), e + 20)
                    sample_extras.append((idx, {"start": s, "end": e, "label": lbl},
                                          text[ctx_start:ctx_end]))

        # entities gold has that v3 didn't find at all
        for s, e, lbl in v3_missed:
            if (s, e) not in pred_by_position:
                missed_per_label[lbl] += 1

        if v3_extras and not v3_missed:
            examples_v3_finds_more += 1
        elif v3_missed and not v3_extras:
            examples_v3_finds_less += 1
        elif relabeled and not v3_extras and not v3_missed:
            examples_v3_relabels += 1
        else:
            # mixed — count as "finds more" for clarity
            examples_v3_finds_more += 1

    print()
    print(f"Sample: {n_examples} examples from dataset_br_v3.jsonl ({n_truncated} truncated to {max_chars} chars)")
    print(f"Fully agree (v3 prediction == gold labels): {fully_agree} ({100*fully_agree/n_examples:.2f}%)")
    print(f"v3 finds MORE than labeled: {examples_v3_finds_more} ({100*examples_v3_finds_more/n_examples:.2f}%) — likely labeler miss")
    print(f"v3 finds LESS than labeled: {examples_v3_finds_less} ({100*examples_v3_finds_less/n_examples:.2f}%) — likely label noise")
    print(f"v3 relabels same span: {examples_v3_relabels} ({100*examples_v3_relabels/n_examples:.2f}%)")
    print()
    print(f"Confidence distribution over {len(confidences)} predicted spans:")
    if confidences:
        print(f"  mean: {statistics.mean(confidences):.3f}")
        print(f"  median: {statistics.median(confidences):.3f}")
        print(f"  stdev: {statistics.stdev(confidences):.3f}")
        print(f"  min: {min(confidences):.3f}, max: {max(confidences):.3f}")
        below_70 = sum(1 for c in confidences if c < 0.7)
        below_90 = sum(1 for c in confidences if c < 0.9)
        print(f"  spans with conf < 0.70: {below_70} ({100*below_70/len(confidences):.2f}%)")
        print(f"  spans with conf < 0.90: {below_90} ({100*below_90/len(confidences):.2f}%)")
    print()
    print("Top categories where v3 finds entities NOT in gold labels (suggesting labeler miss):")
    for lbl, count in extra_per_label.most_common(20):
        print(f"  {lbl}: {count}")
    print()
    print("Top categories where gold labels has entity v3 doesn't predict (suggesting label noise or low recall):")
    for lbl, count in missed_per_label.most_common(20):
        print(f"  {lbl}: {count}")
    print()
    print("Top confusion pairs (gold_label → v3_label):")
    for (gold, pred), count in confusion.most_common(15):
        print(f"  {gold} → {pred}: {count}")

    if sample_extras:
        print()
        print("Sample 'v3 found extra' cases (line_no, predicted span, context):")
        for line_no, span, ctx in sample_extras:
            print(f"  line {line_no}: {span['label']} [{span['start']}:{span['end']}]   ...{ctx!r}...")


if __name__ == "__main__":
    main()
