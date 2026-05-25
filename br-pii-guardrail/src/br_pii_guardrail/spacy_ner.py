"""spaCy NER recognizer — complements BERTimbau v6 for person/location detection.

spaCy's pt_core_news_lg is trained on a much larger PT corpus (CONLL-style PER/LOC/ORG)
and tends to outperform our fine-tuned model on names in free text.

Optional dependency. Install:
    pip install spacy
    python -m spacy download pt_core_news_lg
"""
from __future__ import annotations

import re

from br_pii_guardrail.core import Match


# Map spaCy entity types -> our taxonomy. ORG excluded (too noisy);
# MISC included because pt_core_news_lg often labels names-with-titles as MISC
# (e.g. "Cliente João Silva", "Dr. Pedro Santos").
SPACY_LABEL_MAP = {
    "PER":    "private_person",
    "PERS":   "private_person",
    "PERSON": "private_person",
    "LOC":    "private_address",
    "GPE":    "private_address",
    "MISC":   "private_person",  # filtered heavily by _looks_like_real_person
}


# Prefix words that spaCy often glues to person spans. Strip before emitting.
_PERSON_PREFIXES = {
    "cliente", "clientes", "sr", "sra", "sr.", "sra.", "senhor", "senhora",
    "dr", "dra", "dr.", "dra.", "doutor", "doutora",
    "prof", "prof.", "professor", "professora",
    "funcionário", "funcionario", "funcionária", "funcionaria",
    "colaborador", "colaboradora", "usuário", "usuario", "usuária", "usuaria",
    "paciente", "comprador", "compradora", "vendedor", "vendedora",
}


# Patterns that look like false positives from spaCy (acronyms, IDs, etc)
_HAS_DIGIT = re.compile(r"\d")
_ALL_UPPER = re.compile(r"^[A-Z]{2,5}$")
_ID_LIKE = re.compile(r"^[A-Z]+[-_]?\d", re.IGNORECASE)


def _looks_like_real_person(text: str) -> bool:
    """Heuristic to reject obvious non-person spans from spaCy PER."""
    t = text.strip()
    if not t:
        return False
    if _HAS_DIGIT.search(t):
        return False  # 'ML-2024-789456' is not a person
    if _ALL_UPPER.match(t):
        return False  # 'CTO', 'CPF', 'CNPJ', 'RG' — acronyms
    if _ID_LIKE.match(t):
        return False  # 'X-123', 'AB_456'
    # Must have at least one space (most BR person names are at least 2 tokens)
    # but allow single-token names if they start uppercase and are long enough
    if " " not in t and len(t) < 4:
        return False
    return True


def _looks_like_real_location(text: str) -> bool:
    t = text.strip()
    if not t or _HAS_DIGIT.search(t):
        return False
    if _ALL_UPPER.match(t):  # 'SP', 'RJ' alone are too short to be useful
        return False
    return True


class SpacyRecognizer:
    """Wrapper around a spaCy model. Lazy-loads on first use."""

    def __init__(self, model_name: str = "pt_core_news_lg", confidence: float = 0.85):
        self.model_name = model_name
        self.confidence = confidence
        self._nlp = None

    def _ensure_loaded(self):
        if self._nlp is None:
            try:
                import spacy
            except ImportError as e:
                raise ImportError(
                    "spacy required. Install with `pip install br-pii-guardrail[spacy]`"
                ) from e
            try:
                self._nlp = spacy.load(self.model_name)
            except OSError as e:
                raise OSError(
                    f"spaCy model {self.model_name!r} not found. Download with:\n"
                    f"  python -m spacy download {self.model_name}"
                ) from e

    @staticmethod
    def _strip_prefix(text: str, start: int) -> tuple[str, int]:
        """If text starts with a known title/prefix (e.g. 'Cliente João'),
        return the rest plus adjusted start offset."""
        parts = text.split(maxsplit=1)
        if len(parts) == 2 and parts[0].lower().rstrip(".,:;") in _PERSON_PREFIXES:
            # parts[1] starts at index = position of parts[1] in text
            try:
                idx = text.index(parts[1])
                return parts[1], start + idx
            except ValueError:
                pass
        return text, start

    def find(self, text: str) -> list[Match]:
        if not text or not text.strip():
            return []
        self._ensure_loaded()
        doc = self._nlp(text)
        out: list[Match] = []
        for ent in doc.ents:
            label = SPACY_LABEL_MAP.get(ent.label_)
            if not label:
                continue

            value = ent.text
            start = ent.start_char

            # Strip prefix words like "Cliente", "Dr.", etc
            if label == "private_person":
                value, start = self._strip_prefix(value, start)

            # Apply FP filters
            if label == "private_person" and not _looks_like_real_person(value):
                continue
            if label == "private_address" and not _looks_like_real_location(value):
                continue

            out.append(Match(
                start=start,
                end=start + len(value),
                label=label,
                value=value,
                source="ner",
                confidence=self.confidence,
            ))
        return out
