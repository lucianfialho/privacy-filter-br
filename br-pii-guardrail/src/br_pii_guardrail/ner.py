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
        """Run NER and return Match list. Merges adjacent same-label spans."""
        if not text:
            return []
        raw = list(self.pipeline(text))
        merged = _merge_adjacent_spans(raw, text, max_gap=3)
        out: list[Match] = []
        for ent in merged:
            label = ent["entity_group"]
            start, end = ent["start"], ent["end"]
            out.append(Match(
                start=start, end=end, label=label,
                value=text[start:end], source="ner",
                confidence=float(ent["score"]),
            ))
        return out


def _merge_adjacent_spans(predictions: list[dict], text: str, max_gap: int = 3) -> list[dict]:
    """Merge contiguous predictions of the same label that are separated by
    <= max_gap characters of separator-only content (e.g., punctuation, whitespace,
    sub-word boundaries from BERT WordPiece tokenization).

    Fixes the BIOES sub-token fragmentation seen in v6+:
      pred 1: '792.498.927-' (private_cpf, score=0.99)
      pred 2: '72'           (private_cpf, score=0.99)
      ↓ merged
      pred:   '792.498.927-72' (private_cpf, score=0.99)
    """
    if not predictions:
        return predictions
    sorted_preds = sorted(predictions, key=lambda p: p["start"])
    merged = [dict(sorted_preds[0])]
    for p in sorted_preds[1:]:
        last = merged[-1]
        gap_text = text[last["end"]:p["start"]]
        is_separator_gap = all(c in ".-/() \t  \n" for c in gap_text)
        if (p["entity_group"] == last["entity_group"]
                and (p["start"] - last["end"]) <= max_gap
                and is_separator_gap):
            last["end"] = p["end"]
            last["score"] = min(last["score"], p["score"])
            last["word"] = text[last["start"]:last["end"]]
        else:
            merged.append(dict(p))
    return merged
