"""Format-aware character-level labeler for synthetic NER training data.

The original labeler used `re.escape(value)` for exact-string matching of inserted
PIIs in the generated text. When the LLM rewriter altered formatting (e.g., changed
`-` to `.` in a CPF check-digit position), the exact match silently failed and the
PII went unlabeled. The 2026-05-23 audits found ~16.76% of dataset_br_v3.jsonl
examples had at least one regex-detectable unlabeled PII.

This version applies **format-aware matching** to structured PII categories
(CPF, CNPJ, RG, PIS, CNH, titulo eleitor, IE, phone, certidao): the inserted value
is normalized to its alphanumeric skeleton, then a regex is built that tolerates
up to 2 separator chars (`.- /` plus whitespace variants) between consecutive
skeleton characters, anchored at alnum word boundaries.

Free-text PII categories (PRIVATE_PERSON, PRIVATE_EMAIL, PRIVATE_ADDRESS,
PRIVATE_URL, PRIVATE_CLIENT_REVENUE, PRIVATE_DATE, PRIVATE_ORDER_ID,
PRIVATE_TRACKING_CODE, PRIVATE_INVOICE_NUMBER, PRIVATE_TRANSACTION_ID,
PRIVATE_CUSTOMER_ID, ACCOUNT_NUMBER, SECRET) keep exact-string matching: they
don't have canonical numeric formats, and format-aware matching risks false
positives on substrings.
"""
from __future__ import annotations

import re

_FORMAT_AWARE_LABELS = {
    "PRIVATE_CPF", "PRIVATE_CNPJ", "PRIVATE_RG", "PRIVATE_PIS",
    "PRIVATE_CNH", "PRIVATE_TITULO_ELEITOR", "PRIVATE_IE",
    "PRIVATE_PHONE", "PRIVATE_CERTIDAO",
}

# Separator characters allowed between alnum chars in structured PII.
# Includes ASCII space, dot, dash, slash, tab, narrow no-break space ( ),
# and no-break space ( ).
_SEP_CLASS = r"[.\-/()\s  ]"


def _alnum_skeleton(s: str) -> str:
    """Extract just the alphanumeric characters of s, preserving order."""
    return "".join(c for c in s if c.isalnum())


def _build_format_aware_pattern(value: str, max_sep_per_gap: int = 2) -> re.Pattern | None:
    """Build a regex that matches value tolerating optional separators between alnum chars.

    Anchors at alnum word boundaries (no alnum before/after the match).
    Returns None if the skeleton is too short (< 4 chars) to match safely.
    """
    skeleton = _alnum_skeleton(value)
    if len(skeleton) < 4:
        return None
    parts = []
    for i, c in enumerate(skeleton):
        parts.append(re.escape(c))
        if i < len(skeleton) - 1:
            parts.append(f"{_SEP_CLASS}{{0,{max_sep_per_gap}}}")
    pattern = r"(?<![A-Za-z0-9])" + "".join(parts) + r"(?![A-Za-z0-9])"
    return re.compile(pattern, flags=re.IGNORECASE)


def find_spans(text: str, inserted: dict[str, str]) -> list[dict]:
    """Find character-level spans for each inserted PII value in text.

    For labels in _FORMAT_AWARE_LABELS, uses format-aware matching that tolerates
    separator variants. For other labels, uses exact-string matching.
    Handles overlapping by keeping the longest match (left-to-right scan).
    Returns list of {"start": int, "end": int, "label": str}.
    """
    raw_spans = []
    # For format-aware labels, dedupe by (skeleton, label) to avoid running
    # equivalent patterns multiple times. Exact match runs unconditionally so we
    # still capture the original value byte-for-byte (including mask chars like '*').
    seen_format_aware: set[tuple[str, str]] = set()

    for value, label in inserted.items():
        if not value:
            continue
        # ALWAYS do exact match: captures the literal value as-inserted, including
        # mask chars ('*') and other non-alnum/non-separator content that
        # format-aware skeleton matching would strip.
        for m in re.finditer(re.escape(value), text):
            raw_spans.append({"start": m.start(), "end": m.end(), "label": label})
        # ADDITIONALLY, for structured PII labels, do format-aware matching to
        # catch separator variants the LLM rewriter may have produced.
        if label in _FORMAT_AWARE_LABELS:
            skeleton = _alnum_skeleton(value)
            key = (skeleton, label)
            if key in seen_format_aware:
                continue
            seen_format_aware.add(key)
            pattern = _build_format_aware_pattern(value)
            if pattern is not None:
                for m in pattern.finditer(text):
                    raw_spans.append({"start": m.start(), "end": m.end(), "label": label})

    raw_spans.sort(key=lambda s: (s["start"], -(s["end"] - s["start"])))
    result = []
    last_end = -1
    for span in raw_spans:
        if span["start"] >= last_end:
            result.append(span)
            last_end = span["end"]
    return sorted(result, key=lambda s: s["start"])


def to_entity_format(text: str, spans: list[dict]) -> dict:
    """Returns the NER JSONL format: {text, entities}."""
    return {"text": text, "entities": spans}


def label_text(text: str, inserted: dict[str, str]) -> dict:
    """Full pipeline: text + inserted PII dict → labeled example."""
    spans = find_spans(text, inserted)
    return to_entity_format(text, spans)
