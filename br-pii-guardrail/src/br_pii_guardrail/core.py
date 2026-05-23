"""Match dataclass + Guardrail orchestrator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional


@dataclass(frozen=True)
class Match:
    """A detected PII span."""
    start: int
    end: int
    label: str          # e.g. 'private_cpf', 'private_email'
    value: str          # the literal substring
    source: str = "regex"  # 'regex' | 'checksum' | 'schema' | 'ner'
    confidence: float = 1.0

    def overlaps(self, other: "Match") -> bool:
        return self.start < other.end and other.start < self.end


def _dedupe(matches: Iterable[Match]) -> list[Match]:
    """Resolve overlaps: keep highest-priority (checksum > schema > regex > ner)
    then highest confidence, then longest span."""
    priority = {"checksum": 4, "schema": 3, "regex": 2, "ner": 1}
    sorted_ms = sorted(
        matches,
        key=lambda m: (priority.get(m.source, 0), m.confidence, m.end - m.start),
        reverse=True,
    )
    kept: list[Match] = []
    for m in sorted_ms:
        if not any(m.overlaps(k) for k in kept):
            kept.append(m)
    return sorted(kept, key=lambda m: m.start)


@dataclass
class Guardrail:
    """Orchestrates PII detection layers. Use scan_* for typed inputs."""

    recognizers: list = field(default_factory=list)
    ner: Optional[object] = None  # optional NER fallback (br_pii_guardrail.ner.NER)
    use_ner_fallback: bool = False  # only run NER if other layers find nothing

    @classmethod
    def default(cls, ner: Optional[object] = None) -> "Guardrail":
        """Build with all default BR recognizers enabled."""
        from br_pii_guardrail.recognizers import default_recognizers
        return cls(recognizers=default_recognizers(), ner=ner)

    def scan_text(self, text: str) -> list[Match]:
        """Run all recognizers + (optional) NER on free text."""
        matches: list[Match] = []
        for rec in self.recognizers:
            matches.extend(rec.find(text))
        if self.ner is not None and (matches if not self.use_ner_fallback else not matches):
            # Run NER either always (default) or only as fallback when nothing matched
            ner_matches = self.ner.find(text)
            matches.extend(ner_matches)
        return _dedupe(matches)

    def scan_json(self, obj) -> list[Match]:
        """Recursively walk JSON-like object. Matches are returned per-leaf via
        a synthetic text representation (key path included for traceability)."""
        from br_pii_guardrail.scanners import scan_json_obj
        return scan_json_obj(obj, recognizers=self.recognizers, ner=self.ner)
