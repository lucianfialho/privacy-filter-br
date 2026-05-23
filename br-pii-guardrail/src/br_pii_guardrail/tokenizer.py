"""Per-tenant deterministic PII tokenizer with reversible AES-GCM vault."""
from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


_TOKEN_RE = re.compile(r"\[([A-Z][A-Z0-9_]*)_([A-Za-z0-9_-]{4,16})\]")


def derive_tenant_key(master_key: bytes, tenant_id: str) -> bytes:
    if len(master_key) < 32:
        raise ValueError("master_key must be at least 32 bytes")
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=tenant_id.encode("utf-8"),
        info=b"br-pii-guardrail-v1",
    ).derive(master_key)


class Tokenizer:
    def __init__(self, tenant_key: bytes):
        if len(tenant_key) != 32:
            raise ValueError("tenant_key must be 32 bytes")
        self._key = tenant_key
        self._aead = AESGCM(tenant_key)

    @staticmethod
    def _label_tag(label: str) -> str:
        clean = label.replace("private_", "").replace("PRIVATE_", "").upper()
        return re.sub(r"[^A-Z0-9_]", "_", clean)

    def _token_id(self, value: str, label: str) -> str:
        digest = hmac.new(
            self._key, f"{label}:{value}".encode("utf-8"), hashlib.sha256
        ).digest()
        return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")[:6]

    def token_for(self, value: str, label: str) -> str:
        return f"[{self._label_tag(label)}_{self._token_id(value, label)}]"

    def encrypt(self, value: str) -> str:
        nonce = secrets.token_bytes(12)
        ct = self._aead.encrypt(nonce, value.encode("utf-8"), None)
        return base64.urlsafe_b64encode(nonce + ct).decode("ascii").rstrip("=")

    def decrypt(self, blob: str) -> str:
        pad = "=" * (-len(blob) % 4)
        raw = base64.urlsafe_b64decode(blob + pad)
        return self._aead.decrypt(raw[:12], raw[12:], None).decode("utf-8")

    def tokenize(self, text: str, matches) -> tuple[str, dict[str, str]]:
        """Replace each Match with a deterministic token. Returns (masked, vault).

        matches: iterable of objects with .start, .end, .label attrs (e.g. core.Match).
        """
        sorted_ms = sorted(matches, key=lambda m: m.start, reverse=True)
        masked = text
        vault: dict[str, str] = {}
        for m in sorted_ms:
            value = text[m.start:m.end]
            if not value:
                continue
            tok = self.token_for(value, m.label)
            if tok not in vault:
                vault[tok] = self.encrypt(value)
            masked = masked[:m.start] + tok + masked[m.end:]
        return masked, vault

    def detokenize(self, masked_text: str, vault: dict[str, str]) -> str:
        def _restore(match: re.Match) -> str:
            token = match.group(0)
            blob = vault.get(token)
            if not blob:
                return token
            try:
                return self.decrypt(blob)
            except Exception:
                return token
        return _TOKEN_RE.sub(_restore, masked_text)
