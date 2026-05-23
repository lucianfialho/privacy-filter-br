"""Optional NER fallback using a HuggingFace token-classification model.

Use only when regex/checksum recognizers don't cover all the content (free text).
Import lazily so the package works without transformers installed.
"""
from __future__ import annotations

from br_pii_guardrail.core import Match


class NER:
    """Wrapper around a HF token-classification pipeline."""

    def __init__(self, model_path: str, device: int | str = "cpu",
                 aggregation_strategy: str = "simple"):
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
        except ImportError as e:
            raise ImportError(
                "transformers required for NER. Install with `pip install br-pii-guardrail[ner]`"
            ) from e

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path).eval()
        self.pipeline = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy=aggregation_strategy,
            device=device if isinstance(device, int) else -1,
        )

    def find(self, text: str) -> list[Match]:
        """Run NER and return Match list."""
        if not text:
            return []
        out: list[Match] = []
        for ent in self.pipeline(text):
            label = ent["entity_group"]
            start, end = ent["start"], ent["end"]
            out.append(Match(
                start=start, end=end, label=label,
                value=text[start:end], source="ner",
                confidence=float(ent["score"]),
            ))
        return out
