"""Structure-aware scanners for JSON, CSV, PDF."""
from __future__ import annotations

import io
import json
from typing import Optional

from br_pii_guardrail.core import Match, _dedupe


# Mapping: key/header name (lowercase) -> label.
# When schema match wins, we ALSO run regex to validate format.
SCHEMA_KEYS = {
    # Brazilian IDs
    "cpf": "private_cpf",
    "cnpj": "private_cnpj",
    "rg": "private_rg",
    "cnh": "private_cnh",
    "pis": "private_pis",
    "pasep": "private_pis",
    "titulo_eleitor": "private_titulo_eleitor",
    "ie": "private_ie",
    "inscricao_estadual": "private_ie",
    # Personal
    "email": "private_email",
    "e_mail": "private_email",
    "telefone": "private_phone",
    "celular": "private_phone",
    "phone": "private_phone",
    "endereco": "private_address",
    "address": "private_address",
    "cep": "private_address",
    "nome": "private_person",
    "name": "private_person",
    "person": "private_person",
    # Finance / B2B
    "cartao": "account_number",
    "card": "account_number",
    "conta": "account_number",
    "agencia": "account_number",
    "tracking": "private_tracking_code",
    "rastreio": "private_tracking_code",
    "nota_fiscal": "private_invoice_number",
    "nf": "private_invoice_number",
    "invoice": "private_invoice_number",
    "fatura": "private_invoice_number",
    "order_id": "private_order_id",
    "pedido_id": "private_order_id",
    "customer_id": "private_customer_id",
    "cliente_id": "private_customer_id",
    "transaction_id": "private_transaction_id",
    "revenue": "private_client_revenue",
    "faturamento": "private_client_revenue",
    # Secrets
    "api_key": "secret",
    "token": "secret",
    "password": "secret",
    "senha": "secret",
}


def _schema_label_for(key: str) -> Optional[str]:
    """Normalize key name and return matching label, if any."""
    k = key.lower().strip().replace("-", "_").replace(" ", "_")
    return SCHEMA_KEYS.get(k)


def _stringify(value) -> str:
    """Convert leaf to str for regex scanning."""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    return ""


def _scan_leaf(text: str, label_from_key, recognizers) -> list[Match]:
    """Dedupe per-leaf: schema + regex matches against ONE text segment."""
    leaf_matches: list[Match] = []
    if label_from_key:
        leaf_matches.append(Match(
            start=0, end=len(text), label=label_from_key,
            value=text, source="schema", confidence=0.95,
        ))
    for rec in recognizers:
        leaf_matches.extend(rec.find(text))
    return _dedupe(leaf_matches)


def scan_json_obj(obj, recognizers: list, ner=None, _path: str = "$") -> list[Match]:
    """Walk JSON-like recursively. Each leaf gets its OWN dedupe scope
    (because all leaves start at offset 0 in their own string)."""
    matches: list[Match] = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            label_from_key = _schema_label_for(str(k))
            child_path = f"{_path}.{k}"
            if isinstance(v, (dict, list)):
                matches.extend(scan_json_obj(v, recognizers, ner, child_path))
            else:
                text = _stringify(v)
                if not text:
                    continue
                matches.extend(_scan_leaf(text, label_from_key, recognizers))

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            matches.extend(scan_json_obj(item, recognizers, ner, f"{_path}[{i}]"))

    else:
        text = _stringify(obj)
        if text:
            matches.extend(_scan_leaf(text, None, recognizers))

    return matches


def scan_csv_text(csv_text: str, recognizers: list, ner=None,
                  delimiter: str = ",") -> dict[str, list[Match]]:
    """Scan CSV. Returns {column_name: [matches in that column's cells]}.

    Uses header (first row) to infer label per column via SCHEMA_KEYS.
    """
    import csv

    reader = csv.reader(io.StringIO(csv_text), delimiter=delimiter)
    try:
        headers = next(reader)
    except StopIteration:
        return {}

    col_labels = [_schema_label_for(h) for h in headers]
    result: dict[str, list[Match]] = {h: [] for h in headers}

    for row in reader:
        for col_idx, cell in enumerate(row):
            if col_idx >= len(headers):
                break
            text = cell.strip()
            if not text:
                continue
            col_name = headers[col_idx]
            # Schema-based labeling
            if col_labels[col_idx]:
                result[col_name].append(Match(
                    0, len(text), col_labels[col_idx], text,
                    source="schema", confidence=0.95,
                ))
            # Regex validation
            for rec in recognizers:
                for m in rec.find(text):
                    result[col_name].append(m)
    return result
