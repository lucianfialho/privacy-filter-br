"""br-pii-guardrail: PII detection + tokenization for Brazilian Portuguese."""

from br_pii_guardrail.core import Guardrail, Match
from br_pii_guardrail.tokenizer import Tokenizer, derive_tenant_key

__version__ = "0.1.1"
__all__ = ["Guardrail", "Match", "Tokenizer", "derive_tenant_key"]
