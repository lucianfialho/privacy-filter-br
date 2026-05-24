"""Audit 2: per-category entity count for dataset_br_v3.jsonl.

Counts entities per label, per example, and identifies under-represented categories.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

DATASET = Path(__file__).resolve().parent.parent / "data" / "dataset_br_v3.jsonl"


def main() -> None:
    entity_counts: Counter[str] = Counter()
    examples_with_label: Counter[str] = Counter()
    per_template_labels: dict[str, Counter[str]] = defaultdict(Counter)
    total_examples = 0

    with DATASET.open() as f:
        for line in f:
            row = json.loads(line)
            total_examples += 1
            labels_in_example = {e["label"] for e in row["entities"]}
            for label in labels_in_example:
                examples_with_label[label] += 1
            for ent in row["entities"]:
                entity_counts[ent["label"]] += 1
                per_template_labels[row.get("template", "unknown")][ent["label"]] += 1

    print(f"Total examples: {total_examples}")
    print(f"Total entity spans: {sum(entity_counts.values())}")
    print(f"Unique categories: {len(entity_counts)}")
    print()
    print(f"{'category':<32} {'spans':>8} {'examples':>10} {'%examples':>10}")
    print("-" * 64)
    for label, count in sorted(entity_counts.items(), key=lambda x: -x[1]):
        ex_count = examples_with_label[label]
        pct = 100.0 * ex_count / total_examples
        print(f"{label:<32} {count:>8} {ex_count:>10} {pct:>9.2f}%")

    print()
    print("Categories appearing in <5% of examples (under-represented candidates):")
    under = [(label, examples_with_label[label]) for label in entity_counts
             if examples_with_label[label] < total_examples * 0.05]
    for label, ex_count in sorted(under, key=lambda x: x[1]):
        print(f"  {label}: {ex_count} examples ({100*ex_count/total_examples:.2f}%)")

    print()
    print(f"Templates: {len(per_template_labels)}")
    for tmpl in sorted(per_template_labels):
        cats = per_template_labels[tmpl]
        print(f"  {tmpl}: {sum(cats.values())} spans across {len(cats)} categories")


if __name__ == "__main__":
    main()
