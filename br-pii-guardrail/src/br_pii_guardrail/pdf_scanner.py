"""PDF scanner: extracts text + tables and runs PII recognizers.

Optional dependency: pdfplumber. Install with `pip install br-pii-guardrail[pdf]`.
"""
from __future__ import annotations

from typing import Iterable

from br_pii_guardrail.core import Match
from br_pii_guardrail.scanners import _scan_leaf, _schema_label_for


def scan_pdf_bytes(pdf_bytes: bytes, recognizers: list) -> dict[str, list[Match]]:
    """Scan a PDF byte string. Returns matches grouped by region:
        {"page_1_text": [...], "page_1_table_0": [...], ...}

    Tables get schema-aware treatment per column (first row = header).
    """
    try:
        import pdfplumber
    except ImportError as e:
        raise ImportError(
            "pdfplumber required. Install with `pip install br-pii-guardrail[pdf]`"
        ) from e

    import io
    result: dict[str, list[Match]] = {}

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            # 1) Free text on the page (intro paragraphs, footers, etc)
            page_text = page.extract_text() or ""
            if page_text:
                key = f"page_{page_idx}_text"
                result[key] = _scan_leaf(page_text, None, recognizers)

            # 2) Tables on the page — apply schema-aware column labeling
            for table_idx, table in enumerate(page.extract_tables() or []):
                if not table or len(table) < 1:
                    continue
                headers = [h or "" for h in table[0]]
                col_labels = [_schema_label_for(h) for h in headers]
                key = f"page_{page_idx}_table_{table_idx}"
                table_matches: list[Match] = []
                for row in table[1:]:
                    for col_idx, cell in enumerate(row or []):
                        if not cell or col_idx >= len(headers):
                            continue
                        text = str(cell).strip()
                        if not text:
                            continue
                        table_matches.extend(
                            _scan_leaf(text, col_labels[col_idx], recognizers)
                        )
                if table_matches:
                    result[key] = table_matches

    return result


def scan_pdf_file(path: str, recognizers: list) -> dict[str, list[Match]]:
    """Convenience wrapper. Reads file then calls scan_pdf_bytes."""
    with open(path, "rb") as f:
        return scan_pdf_bytes(f.read(), recognizers)


def flatten_pdf_matches(grouped: dict[str, list[Match]]) -> Iterable[Match]:
    """Flatten {region: [matches]} into a single iterable, useful for downstream
    tokenization that doesn't care about region grouping."""
    for region, matches in grouped.items():
        yield from matches
