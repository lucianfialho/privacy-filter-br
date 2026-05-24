"""Audit 1 (rewritten): detect PII patterns visible in text but missing from labels.

The original "round-trip" idea was: did gpt-5-nano drop inserted PII during rewriting?
Our dataset doesn't preserve the "intended PII" separately from the final labels — if
gpt-5-nano altered an entity, the string-matcher in step 4 just wouldn't label it. The
example then looks "clean" but has unlabeled PII in text.

This audit catches the symptom directly: run regex for canonical BR PII patterns and
report mentions in text that are NOT covered by any label span.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

DATASET = Path(__file__).resolve().parent.parent / "data" / "dataset_br_v3.jsonl"


# Conservative regexes for canonical BR PII formats. False positives possible but rare
# for these tight patterns; false negatives are expected (e.g. masked CPFs ***.***).
PII_PATTERNS = {
    "private_cpf": re.compile(r"\b\d{3}[.\s]\d{3}[.\s]\d{3}[-.\s]\d{2}\b"),
    "private_cnpj": re.compile(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b"),
    "private_cep": re.compile(r"\b\d{5}[-.]?\d{3}\b"),
    "private_phone": re.compile(r"\(\d{2}\)\s?9?\d{4}-\d{4}\b"),
    "private_email": re.compile(r"\b[\w.+_-]+@[\w.-]+\.[a-zA-Z]{2,}\b"),
    "private_rg": re.compile(r"\b\d{1,2}\.\d{3}\.\d{3}-[\dXx]\b"),
    "private_pis": re.compile(r"\b\d{3}\.\d{5}\.\d{2}-\d\b"),
    "private_cnh": re.compile(r"\b\d{11}\b"),  # 11 digits — could also be CPF without separators
    "private_titulo_eleitor": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    "private_url": re.compile(r"https?://\S+"),
}


def covered_by_label(start: int, end: int, label_spans: list[tuple[int, int]]) -> bool:
    """Check if (start, end) overlaps with any labeled span."""
    for ls, le in label_spans:
        if start < le and end > ls:
            return True
    return False


def main() -> None:
    total_examples = 0
    examples_with_unlabeled = 0
    unlabeled_per_pattern: Counter[str] = Counter()
    pattern_total: Counter[str] = Counter()
    sample_issues: dict[str, list[tuple[int, str, str]]] = defaultdict(list)

    with DATASET.open() as f:
        for idx, line in enumerate(f):
            row = json.loads(line)
            total_examples += 1
            text = row["text"]
            label_spans = [(e["start"], e["end"]) for e in row["entities"]]

            example_has_unlabeled = False
            for pattern_label, regex in PII_PATTERNS.items():
                for match in regex.finditer(text):
                    pattern_total[pattern_label] += 1
                    if not covered_by_label(match.start(), match.end(), label_spans):
                        unlabeled_per_pattern[pattern_label] += 1
                        example_has_unlabeled = True
                        if len(sample_issues[pattern_label]) < 3:
                            ctx_start = max(0, match.start() - 20)
                            ctx_end = min(len(text), match.end() + 20)
                            sample_issues[pattern_label].append(
                                (idx, match.group(), text[ctx_start:ctx_end])
                            )
            if example_has_unlabeled:
                examples_with_unlabeled += 1

    print(f"Total examples: {total_examples}")
    print(f"Examples with at least one unlabeled regex match: {examples_with_unlabeled} ({100*examples_with_unlabeled/total_examples:.2f}%)")
    print()
    print("Per-pattern: unlabeled regex matches / total matches found")
    print(f"{'pattern':<28} {'unlabeled':>10} {'total':>10} {'miss-rate':>10}")
    print("-" * 64)
    for pattern_label in sorted(pattern_total, key=lambda x: -unlabeled_per_pattern.get(x, 0)):
        unlabeled = unlabeled_per_pattern.get(pattern_label, 0)
        total = pattern_total[pattern_label]
        rate = 100.0 * unlabeled / total if total else 0.0
        print(f"{pattern_label:<28} {unlabeled:>10} {total:>10} {rate:>9.2f}%")

    print()
    print("Sample unlabeled matches (line_no, match, context):")
    for pattern_label, samples in sample_issues.items():
        if samples:
            print(f"\n  {pattern_label}:")
            for line_no, match_text, ctx in samples:
                print(f"    line {line_no}: '{match_text}'   ...{ctx!r}...")


if __name__ == "__main__":
    main()
