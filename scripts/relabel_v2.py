"""Apply the new format-aware labeler to existing dataset texts.

Why: regenerating the full dataset means calling the LLM rewriter 50k times again
($$+hours). Instead, we keep the existing texts and re-derive labels using the new
labeler. The trick is reconstructing the `inserted` dict (which the generator threw
away) from the existing labels.

Approach: each existing labeled span tells us "this string at this position is a
PII of label L". We collect all unique (text_substring, label) pairs as our
inserted dict, then run the new labeler on the text. The new labeler will:
  1. For format-aware labels, match all separator-variant occurrences in text.
  2. For exact labels (person, email, address, etc), match only literal occurrences.

Output: data/dataset_br_v4.jsonl with augmented labels.

This is strictly more permissive than the regex-based scripts/relabel_dataset.py
because it uses the actual entity values from the dataset (not just generic regex
patterns), so we benefit from the specific values gpt-5-nano produced.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.labeler import find_spans  # noqa: E402

INPUT = ROOT / "data" / "dataset_br_v3.jsonl"
OUTPUT = ROOT / "data" / "dataset_br_v4.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    examples_with_new_labels = 0
    total_added: Counter[str] = Counter()
    total_examples = 0
    total_original_spans = 0
    total_new_spans = 0

    with args.input.open() as f_in, args.output.open("w") as f_out:
        for idx, line in enumerate(f_in):
            if args.limit is not None and idx >= args.limit:
                break
            row = json.loads(line)
            total_examples += 1
            text = row["text"]
            original_entities = row["entities"]
            total_original_spans += len(original_entities)

            # Reconstruct inserted dict from existing labels.
            # Format: {value: LABEL_UPPER} (labeler expects upper-case PRIVATE_*).
            inserted: dict[str, str] = {}
            for ent in original_entities:
                value = text[ent["start"]:ent["end"]]
                if not value:
                    continue
                label_upper = ent["label"].upper()
                # If the same value appears with different labels (rare),
                # keep the first occurrence; ties shouldn't happen for synthetic data.
                inserted.setdefault(value, label_upper)

            new_spans_upper = find_spans(text, inserted)
            # Convert back to lowercase label format (matching dataset convention)
            new_spans = [
                {"start": s["start"], "end": s["end"], "label": s["label"].lower()}
                for s in new_spans_upper
            ]

            # Diff: which spans are new?
            original_keys = {(e["start"], e["end"], e["label"]) for e in original_entities}
            new_keys = {(s["start"], s["end"], s["label"]) for s in new_spans}

            added = new_keys - original_keys
            if added:
                examples_with_new_labels += 1
                for _, _, lbl in added:
                    total_added[lbl] += 1
                total_new_spans += len(added)

            row["entities"] = new_spans
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Total examples processed: {total_examples}")
    print(f"Original label count: {total_original_spans}")
    print(f"New labels added: {total_new_spans}")
    print(f"Examples modified: {examples_with_new_labels} ({100*examples_with_new_labels/total_examples:.2f}%)")
    print()
    print("Per category (top adds):")
    for lbl, c in total_added.most_common(20):
        print(f"  {lbl}: +{c}")
    print()
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
