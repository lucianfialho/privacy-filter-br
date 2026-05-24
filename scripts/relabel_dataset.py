"""Audit 4 prep: produce dataset_br_v3_relabeled.jsonl.

Root cause from audits 1-3: the original labeler in src/labeler.py does exact-string
matching of inserted PII values. When gpt-5-nano rewriter alters formatting
(`-` → `.`, adds/removes spaces, masks digits), the exact match fails and the PII
goes unlabeled. Audit 1 found 16.76% of examples with unlabeled regex-matchable PII.

This script augments existing labels with regex-detected PII patterns. Strategy:
  1. Keep all original labels (they're correct for the spans they cover).
  2. Run conservative BR PII regexes on text.
  3. For each regex match NOT covered by an existing label, add a new label.
  4. Apply tight regexes to avoid false positives (CPF needs canonical separator
     structure; not just 11 digits which collides with CNH/IE).
  5. Skip regex categories where false-positive risk dominates (cnh, cep when
     surrounded by alphanumeric noise).

Output: data/dataset_br_v3_relabeled.jsonl with same structure as input but
augmented entities. Plus a stats summary to stderr.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "dataset_br_v3.jsonl"
OUTPUT = ROOT / "data" / "dataset_br_v3_relabeled.jsonl"

# Tight regexes for canonical BR PII. False positives are worse than misses here —
# adding a wrong label is more damaging than leaving a real PII unlabeled, since
# the latter just preserves the status quo while the former actively corrupts.
PII_PATTERNS = {
    # CPF: 3 digits + sep + 3 digits + sep + 3 digits + sep + 2 digits.
    # Separators can be . - / or space, but must be present (digits-only matches IE/CNH).
    "private_cpf": re.compile(r"\b\d{3}[.\s\-/]\d{3}[.\s\-/]\d{3}[.\s\-/]\d{2}\b"),
    "private_cnpj": re.compile(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b"),
    "private_email": re.compile(r"\b[\w.+_-]+@[\w.-]+\.[a-zA-Z]{2,}\b"),
    "private_phone": re.compile(r"\(\d{2}\)\s?9?\d{4}[-\s]?\d{4}\b"),
    "private_pis": re.compile(r"\b\d{3}\.\d{5}\.\d{2}-\d\b"),
    "private_url": re.compile(r"https?://[^\s<>\"'`]+"),
    # RG with format: must have either dot+dash or dash with check digit
    "private_rg": re.compile(r"\b\d{1,2}\.\d{3}\.\d{3}-[\dXx]\b"),
    # Titulo eleitor: 12 digits with spaces (4-4-4)
    "private_titulo_eleitor": re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b"),
}

# Categories where regex risks collision and we should NOT add automatic labels:
# - cep: collides with order IDs and product codes (any 8-digit chunk with dash)
# - cnh: collides with IE and any 11-digit ID
# - account_number, customer_id, etc: too varied; skip
SKIP_REGEX = {"private_cep", "private_cnh", "account_number", "private_customer_id",
              "private_ie", "private_invoice_number", "private_order_id",
              "private_tracking_code", "private_transaction_id", "secret",
              "private_date", "private_certidao", "private_client_revenue"}


def covered_by_label(start: int, end: int, label_spans: list[dict]) -> bool:
    """True if (start, end) overlaps with any labeled span."""
    for ls in label_spans:
        if start < ls["end"] and end > ls["start"]:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--limit", type=int, default=None,
                        help="process only first N examples (for testing)")
    args = parser.parse_args()

    added_per_label: Counter[str] = Counter()
    examples_modified = 0
    total_examples = 0

    with args.input.open() as f_in, args.output.open("w") as f_out:
        for idx, line in enumerate(f_in):
            if args.limit is not None and idx >= args.limit:
                break
            row = json.loads(line)
            total_examples += 1
            text = row["text"]
            existing = list(row["entities"])
            new_spans = []

            for pattern_label, regex in PII_PATTERNS.items():
                if pattern_label in SKIP_REGEX:
                    continue
                for m in regex.finditer(text):
                    if not covered_by_label(m.start(), m.end(), existing + new_spans):
                        new_spans.append({
                            "start": m.start(),
                            "end": m.end(),
                            "label": pattern_label,
                        })
                        added_per_label[pattern_label] += 1

            if new_spans:
                examples_modified += 1
                # Merge and sort by start position
                merged = sorted(existing + new_spans, key=lambda s: s["start"])
                row["entities"] = merged

            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Total examples processed: {total_examples}", file=sys.stderr)
    print(f"Examples modified (≥1 new label added): {examples_modified} ({100*examples_modified/total_examples:.2f}%)", file=sys.stderr)
    print(f"Total new spans added: {sum(added_per_label.values())}", file=sys.stderr)
    print(f"  per category:", file=sys.stderr)
    for label, count in sorted(added_per_label.items(), key=lambda x: -x[1]):
        print(f"    {label}: {count}", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
