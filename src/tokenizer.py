"""
Privacy guardrail tokenizer — deterministic per-tenant.

Pattern:
    1. NER model finds PII spans in `text`
    2. Tokenizer replaces each span with a typed placeholder: `[CPF_a3f8c2]`
       - Token is deterministic: same (tenant, label, value) -> same token
       - Original value is AES-GCM encrypted into a "vault" dict
    3. Caller sends `masked_text` to external LLM (GPT-4o, Claude, etc)
    4. Caller stores `masked_text` + `vault` in DB (PII never persisted in clear)
    5. To restore: `detokenize(masked_text, vault)` decrypts vault entries

Properties:
    - Deterministic per-tenant: `tokenize(same_cpf)` -> same token within tenant
      enabling `COUNT(DISTINCT)`, joins, ML on tokens.
    - Isolated cross-tenant: salt-per-tenant means same CPF in tenant A != token in B.
    - Reversible: vault holds AES-GCM(nonce || ciphertext) per token.

Usage:
    from src.tokenizer import Tokenizer, derive_tenant_key
    import os

    master_key = os.environ["PII_MASTER_KEY"].encode()  # 32+ bytes, store in KMS
    tenant_key = derive_tenant_key(master_key, "tenant_acme")
    tok = Tokenizer(tenant_key)

    masked, vault = tok.tokenize(text, ner_entities)
    # ... send masked to LLM, get response back ...
    original = tok.detokenize(llm_response, vault)
"""
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
    """HKDF-derived 32-byte AES key for a tenant.

    Same `master_key + tenant_id` always yields the same key.
    """
    if len(master_key) < 32:
        raise ValueError("master_key must be at least 32 bytes")
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=tenant_id.encode("utf-8"),
        info=b"pii-guardrail-v1",
    ).derive(master_key)


class Tokenizer:
    """Per-tenant deterministic tokenizer + reversible vault."""

    def __init__(self, tenant_key: bytes):
        if len(tenant_key) != 32:
            raise ValueError("tenant_key must be 32 bytes (use derive_tenant_key)")
        self._key = tenant_key
        self._aead = AESGCM(tenant_key)

    # ----- tokens (deterministic) -----
    def _token_id(self, value: str, label: str) -> str:
        """6-char base64 hash of HMAC(tenant_key, label:value)."""
        digest = hmac.new(
            self._key, f"{label}:{value}".encode("utf-8"), hashlib.sha256
        ).digest()
        b64 = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
        return b64[:6]

    @staticmethod
    def _label_tag(label: str) -> str:
        """Normalize 'private_cpf' -> 'CPF', 'account_number' -> 'ACCOUNT_NUMBER'."""
        clean = label.replace("private_", "").replace("PRIVATE_", "").upper()
        # Keep alnum + underscore
        return re.sub(r"[^A-Z0-9_]", "_", clean)

    def token_for(self, value: str, label: str) -> str:
        return f"[{self._label_tag(label)}_{self._token_id(value, label)}]"

    # ----- encryption (reversible vault) -----
    def encrypt(self, value: str) -> str:
        nonce = secrets.token_bytes(12)
        ct = self._aead.encrypt(nonce, value.encode("utf-8"), None)
        return base64.urlsafe_b64encode(nonce + ct).decode("ascii").rstrip("=")

    def decrypt(self, blob: str) -> str:
        pad = "=" * (-len(blob) % 4)
        raw = base64.urlsafe_b64decode(blob + pad)
        nonce, ct = raw[:12], raw[12:]
        return self._aead.decrypt(nonce, ct, None).decode("utf-8")

    # ----- public API -----
    def tokenize(self, text: str, entities: list[dict]) -> tuple[str, dict[str, str]]:
        """Replace each entity span with a token. Returns (masked_text, vault).

        `entities` is a list of {"start": int, "end": int, "label": str} —
        same shape as what the NER model outputs.

        `vault` is {token: encrypted_blob}. Persist this in DB next to the
        masked text. To decrypt, pass it back to `detokenize()`.

        Note: the same value within this tenant will resolve to the same
        token; the vault entry is therefore stable too (so it's safe to merge
        vaults across messages of the same conversation).
        """
        # Replace right-to-left so earlier offsets stay valid.
        sorted_ents = sorted(entities, key=lambda e: e["start"], reverse=True)
        masked = text
        vault: dict[str, str] = {}
        for ent in sorted_ents:
            start, end = ent["start"], ent["end"]
            value = text[start:end]
            if not value:
                continue
            token = self.token_for(value, ent["label"])
            if token not in vault:
                vault[token] = self.encrypt(value)
            masked = masked[:start] + token + masked[end:]
        return masked, vault

    def detokenize(self, masked_text: str, vault: dict[str, str]) -> str:
        """Restore original PII from masked text + vault.

        Tokens not present in the vault are left untouched (e.g. tokens that
        the LLM may have hallucinated). This is safe-by-default."""
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
