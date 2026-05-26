"""Pre-flight audit: catch instrumentation bugs BEFORE training.

Lesson from v7: trained 5h GPU + paid $1.50 batch on 140k examples with ZERO
private_date labels because the labeler never tagged dates. A 30-second
Counter on labels would have caught it.

Run this on any dataset.jsonl BEFORE submitting batch / training.
Exit code 1 if any expected label has 0 occurrences.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

EXPECTED_LABELS = {
    "private_person", "private_cpf", "private_cnpj", "private_rg", "private_pis",
    "private_cnh", "private_titulo_eleitor", "private_ie", "private_certidao",
    "private_email", "private_phone", "private_address", "private_date",
    "private_url", "account_number", "private_customer_id", "private_order_id",
    "private_invoice_number", "private_tracking_code", "private_transaction_id",
    "private_client_revenue", "secret",
}


def audit(path: str) -> int:
    counts: Counter[str] = Counter()
    total = 0
    with open(path) as f:
        for line in f:
            total += 1
            d = json.loads(line)
            for e in d.get("entities", []):
                counts[e["label"]] += 1

    print(f"=== {path}: {total} docs ===")
    print(f"{'label':<26} {'count':>10} {'docs%':>8} status")
    print("-" * 60)
    zero = []
    rare = []
    for label in sorted(EXPECTED_LABELS):
        c = counts.get(label, 0)
        pct = 100 * c / total if total else 0
        if c == 0:
            status, marker = "MISSING ❌", "zero"
            zero.append(label)
        elif pct < 1.0:
            status = "rare ⚠️"
            rare.append(label)
        else:
            status = "ok ✓"
        print(f"{label:<26} {c:>10} {pct:>7.1f}% {status}")

    extra = set(counts.keys()) - EXPECTED_LABELS
    if extra:
        print(f"\nUnexpected labels found: {sorted(extra)}")

    print()
    if zero:
        print(f"❌ FAIL — {len(zero)} required labels MISSING: {zero}")
        print("   Likely cause: field rendered in templates but not registered in `inserted` dict.")
        print("   Fix: scripts/openai_batch.py build_prompt_and_metadata — add inserted[value] = LABEL")
        return 1
    if rare:
        print(f"⚠️  WARN — {len(rare)} labels under 1% coverage: {rare}")
    print(f"✓ PASS — all {len(EXPECTED_LABELS)} expected labels present")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: audit_label_distribution.py <dataset.jsonl>", file=sys.stderr)
        sys.exit(2)
    sys.exit(audit(sys.argv[1]))
