"""
Migrate v2 dataset (PRIVATE_FOO labels) to v3 hybrid taxonomy (private_foo lowercase).

Why: v3 reuses original openai/privacy-filter head which uses lowercase labels
like 'private_person', 'private_email'. We normalize our dataset to match.

Mapping:
    PRIVATE_X -> private_x (lowercase, keeps prefix for BR-specific labels)

Usage:
    python3 scripts/migrate_v3_labels.py \\
        --input data/dataset_br_v2.jsonl \\
        --output data/dataset_br_v3.jsonl
"""
import argparse
import json
import sys
from pathlib import Path


# Labels that already exist in original openai/privacy-filter head
# (we just normalize case to match: PRIVATE_PERSON -> private_person)
ORIG_LABELS = {
    "PRIVATE_PERSON": "private_person",
    "PRIVATE_EMAIL": "private_email",
    "PRIVATE_PHONE": "private_phone",
    "PRIVATE_ADDRESS": "private_address",
}

# BR-specific new labels — lowercase, keep "private_" prefix for consistency
BR_LABEL_MAP = {
    "PRIVATE_CPF": "private_cpf",
    "PRIVATE_CNPJ": "private_cnpj",
    "PRIVATE_RG": "private_rg",
    "PRIVATE_CNH": "private_cnh",
    "PRIVATE_PIS": "private_pis",
    "PRIVATE_TITULO_ELEITOR": "private_titulo_eleitor",
    "PRIVATE_CERTIDAO": "private_certidao",
    "PRIVATE_IE": "private_ie",
    "PRIVATE_ORDER_ID": "private_order_id",
    "PRIVATE_TRACKING_CODE": "private_tracking_code",
    "PRIVATE_INVOICE_NUMBER": "private_invoice_number",
    "PRIVATE_CLIENT_REVENUE": "private_client_revenue",
    "PRIVATE_TRANSACTION_ID": "private_transaction_id",
    "PRIVATE_CUSTOMER_ID": "private_customer_id",
}

ALL_MAP = {**ORIG_LABELS, **BR_LABEL_MAP}


def migrate(input_path: Path, output_path: Path) -> dict:
    stats = {"total": 0, "entities": 0, "unknown": 0}
    unknown_labels = set()

    with input_path.open() as f_in, output_path.open("w") as f_out:
        for line in f_in:
            ex = json.loads(line)
            new_entities = []
            for ent in ex.get("entities", []):
                old = ent["label"]
                if old in ALL_MAP:
                    ent = {**ent, "label": ALL_MAP[old]}
                else:
                    stats["unknown"] += 1
                    unknown_labels.add(old)
                    continue  # skip unknown labels
                new_entities.append(ent)
                stats["entities"] += 1
            ex["entities"] = new_entities
            f_out.write(json.dumps(ex, ensure_ascii=False) + "\n")
            stats["total"] += 1

    stats["unknown_labels"] = sorted(unknown_labels)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Migrating {in_path} -> {out_path}")
    stats = migrate(in_path, out_path)
    print(f"  total examples: {stats['total']}")
    print(f"  total entities: {stats['entities']}")
    if stats["unknown"]:
        print(f"  UNKNOWN labels dropped: {stats['unknown']}", file=sys.stderr)
        for lbl in stats["unknown_labels"]:
            print(f"    - {lbl}", file=sys.stderr)


if __name__ == "__main__":
    main()
