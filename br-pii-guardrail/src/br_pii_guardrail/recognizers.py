"""BR-specific PII recognizers (regex + checksum validators)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from br_pii_guardrail.core import Match


# ============================== CPF ==============================
_CPF_RE = re.compile(r"\b(\d{3})\.?(\d{3})\.?(\d{3})[-\s]?(\d{2})\b")


def _cpf_check(digits: str) -> bool:
    if len(digits) != 11 or digits == digits[0] * 11:
        return False
    for i in (9, 10):
        s = sum(int(digits[j]) * ((i + 1) - j) for j in range(i))
        d = (s * 10) % 11
        if d == 10:
            d = 0
        if d != int(digits[i]):
            return False
    return True


class CpfRecognizer:
    label = "private_cpf"

    def find(self, text: str) -> list[Match]:
        out = []
        for m in _CPF_RE.finditer(text):
            digits = "".join(m.groups())
            if _cpf_check(digits):
                out.append(Match(m.start(), m.end(), self.label, m.group(),
                                 source="checksum", confidence=1.0))
        return out


# ============================== CNPJ ==============================
_CNPJ_RE = re.compile(r"\b(\d{2})\.?(\d{3})\.?(\d{3})/?(\d{4})-?(\d{2})\b")


def _cnpj_check(digits: str) -> bool:
    if len(digits) != 14 or digits == digits[0] * 14:
        return False
    weights1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    weights2 = [6] + weights1
    for weights, i in [(weights1, 12), (weights2, 13)]:
        s = sum(int(digits[j]) * weights[j] for j in range(i))
        d = 11 - (s % 11)
        d = 0 if d >= 10 else d
        if d != int(digits[i]):
            return False
    return True


class CnpjRecognizer:
    label = "private_cnpj"

    def find(self, text: str) -> list[Match]:
        out = []
        for m in _CNPJ_RE.finditer(text):
            digits = "".join(m.groups())
            if _cnpj_check(digits):
                out.append(Match(m.start(), m.end(), self.label, m.group(),
                                 source="checksum", confidence=1.0))
        return out


# ============================== Email ==============================
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)


class EmailRecognizer:
    label = "private_email"

    def find(self, text: str) -> list[Match]:
        return [Match(m.start(), m.end(), self.label, m.group(), source="regex")
                for m in _EMAIL_RE.finditer(text)]


# ============================== Phone (BR) ==============================
# Cobre: (11) 98765-4321, 11 98765-4321, 11987654321, +55 11 98765-4321
_PHONE_RE = re.compile(
    r"(?:\+?55\s?)?(?:\(?(\d{2})\)?\s?)?(9?\d{4})[-\s]?(\d{4})\b"
)


class PhoneRecognizer:
    label = "private_phone"

    def find(self, text: str) -> list[Match]:
        out = []
        for m in _PHONE_RE.finditer(text):
            # Heuristic: filter out matches that look like other IDs (need ≥10 digits total)
            digits = re.sub(r"\D", "", m.group())
            if 10 <= len(digits) <= 13:
                out.append(Match(m.start(), m.end(), self.label, m.group(),
                                 source="regex", confidence=0.85))
        return out


# ============================== Credit card (Luhn) ==============================
_CARD_RE = re.compile(r"\b(?:\d[\s-]?){13,19}\b")


def _luhn(digits: str) -> bool:
    s = 0
    for i, c in enumerate(reversed(digits)):
        n = int(c)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        s += n
    return s % 10 == 0


class CardRecognizer:
    label = "account_number"

    def find(self, text: str) -> list[Match]:
        out = []
        for m in _CARD_RE.finditer(text):
            digits = re.sub(r"\D", "", m.group())
            if 13 <= len(digits) <= 19 and _luhn(digits):
                out.append(Match(m.start(), m.end(), self.label, m.group(),
                                 source="checksum", confidence=1.0))
        return out


# ============================== URL ==============================
_URL_RE = re.compile(r"\bhttps?://[^\s<>\"']+", re.IGNORECASE)


class UrlRecognizer:
    label = "private_url"

    def find(self, text: str) -> list[Match]:
        return [Match(m.start(), m.end(), self.label, m.group(), source="regex")
                for m in _URL_RE.finditer(text)]


# ============================== CEP ==============================
_CEP_RE = re.compile(r"\b(\d{5})-?(\d{3})\b")


class CepRecognizer:
    label = "private_address"

    def find(self, text: str) -> list[Match]:
        return [Match(m.start(), m.end(), self.label, m.group(),
                      source="regex", confidence=0.6)
                for m in _CEP_RE.finditer(text)]


# ============================== Tracking Correios ==============================
_TRACKING_RE = re.compile(r"\b[A-Z]{2}\d{9}[A-Z]{2}\b")


class TrackingRecognizer:
    label = "private_tracking_code"

    def find(self, text: str) -> list[Match]:
        return [Match(m.start(), m.end(), self.label, m.group(), source="regex")
                for m in _TRACKING_RE.finditer(text)]


# ============================== Secret / API keys ==============================
_SECRET_RES = [
    re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}\b"),  # OpenAI/Anthropic
    re.compile(r"\bghp_[A-Za-z0-9]{30,}\b"),                  # GitHub PAT
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                      # AWS Access Key
    re.compile(r"Bearer\s+[A-Za-z0-9._-]{20,}", re.IGNORECASE),  # Bearer tokens
]


class SecretRecognizer:
    label = "secret"

    def find(self, text: str) -> list[Match]:
        out = []
        for pat in _SECRET_RES:
            for m in pat.finditer(text):
                out.append(Match(m.start(), m.end(), self.label, m.group(),
                                 source="regex", confidence=0.95))
        return out


def default_recognizers() -> list:
    """Return list of recognizers in order. Checksum-based come first so they
    win priority resolution in the deduper."""
    return [
        CpfRecognizer(),
        CnpjRecognizer(),
        CardRecognizer(),
        EmailRecognizer(),
        UrlRecognizer(),
        TrackingRecognizer(),
        SecretRecognizer(),
        PhoneRecognizer(),
        CepRecognizer(),
    ]
