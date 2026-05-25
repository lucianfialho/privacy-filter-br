"""Phase 1 partial: evaluate v6 on LeNER-Br (real Brazilian legal text).

LeNER-Br categories overlapping with ours:
  PESSOA       → private_person
  LOCAL        → private_address (partial — LOCAL is more geographic)
  TEMPO        → private_date (partial — LeNER TEMPO includes years/periods)

For each LeNER-Br test sentence:
  1. Reconstruct text from CoNLL tokens (space-joined)
  2. Reconstruct character-level gold spans for PESSOA / LOCAL / TEMPO
  3. Run v6 inference
  4. Compare v6's predictions for {private_person, private_address, private_date}
  5. Report precision/recall/F1 per overlapping category

Span-level matching: a prediction matches gold if exact (start, end, label).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LENER_DIR = ROOT / "data/lener-br/lener-br-master/leNER-Br/test"

# LeNER-Br → our schema
LENER_TO_OURS = {
    "PESSOA": "private_person",
    "LOCAL": "private_address",
    "TEMPO": "private_date",
}


def read_conll_file(path: Path) -> list[tuple[str, list[dict]]]:
    """Parse a CoNLL file into a list of (text, entities) for each sentence.
    Sentences are separated by blank lines.
    """
    examples = []
    current_tokens = []
    current_tags = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_tokens:
                    examples.append(build_example(current_tokens, current_tags))
                    current_tokens = []
                    current_tags = []
                continue
            parts = line.split()
            if len(parts) >= 2:
                current_tokens.append(parts[0])
                current_tags.append(parts[-1])
    if current_tokens:
        examples.append(build_example(current_tokens, current_tags))
    return examples


def build_example(tokens: list[str], tags: list[str]) -> tuple[str, list[dict]]:
    """Build (text, entities) tuple from aligned tokens+tags."""
    text_parts = []
    entities = []
    current_span_start = None
    current_span_label = None
    pos = 0
    for tok, tag in zip(tokens, tags):
        if pos > 0:
            text_parts.append(" ")
            pos += 1
        tok_start = pos
        text_parts.append(tok)
        pos += len(tok)

        if tag.startswith("B-"):
            # Close any open span
            if current_span_start is not None:
                entities.append({
                    "start": current_span_start,
                    "end": tok_start - 1 if pos > tok_start else tok_start,
                    "label": current_span_label,
                })
            current_span_start = tok_start
            current_span_label = tag[2:]
        elif tag.startswith("I-"):
            if current_span_start is None:
                # Malformed (I- without B-); treat as B-
                current_span_start = tok_start
                current_span_label = tag[2:]
        else:  # O or other
            if current_span_start is not None:
                entities.append({
                    "start": current_span_start,
                    "end": tok_start - 1 if pos > tok_start else tok_start,
                    "label": current_span_label,
                })
                current_span_start = None
                current_span_label = None
    # Close final span if any
    if current_span_start is not None:
        entities.append({
            "start": current_span_start,
            "end": pos,
            "label": current_span_label,
        })

    text = "".join(text_parts)

    # Fix end positions to point right after last token of span
    for e in entities:
        # Walk forward from start; consume tokens until we hit the next non-I or end
        pass

    return text, entities


def main() -> None:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

    print("Loading LeNER-Br test set...")
    all_examples = []
    for conll_file in sorted(LENER_DIR.glob("*.conll")):
        all_examples.extend(read_conll_file(conll_file))
    print(f"  Total sentences: {len(all_examples)}")

    # Filter to sentences with at least one mapped entity
    relevant = []
    for text, ents in all_examples:
        mapped = [e for e in ents if e["label"] in LENER_TO_OURS and e["end"] > e["start"]]
        if mapped:
            relevant.append((text, mapped))
    print(f"  Sentences with PESSOA/LOCAL/TEMPO: {len(relevant)}")

    # Limit to first N for speed
    SAMPLE = 500
    sample = relevant[:SAMPLE]
    print(f"  Sample for v6: {len(sample)}")

    print("\nLoading v6...")
    tok = AutoTokenizer.from_pretrained("checkpoints/v6-local", model_max_length=512)
    model = AutoModelForTokenClassification.from_pretrained("checkpoints/v6-local").eval()
    ner = pipeline("token-classification", model=model, tokenizer=tok,
                   aggregation_strategy="simple", device=-1)

    # Per-category TP/FP/FN. Compare gold-mapped category vs v6 prediction.
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    sample_misses: dict[str, list] = defaultdict(list)

    for i, (text, gold) in enumerate(sample):
        if i % 100 == 0 and i > 0:
            print(f"  {i}/{len(sample)}...")
        # Map gold labels to our schema
        gold_mapped = [
            {"start": e["start"], "end": e["end"], "label": LENER_TO_OURS[e["label"]]}
            for e in gold
        ]
        gold_set = {(e["start"], e["end"], e["label"]) for e in gold_mapped}

        preds = ner(text)
        # Filter v6 predictions to overlapping categories only
        relevant_labels = set(LENER_TO_OURS.values())
        pred_set = {(p["start"], p["end"], p["entity_group"])
                    for p in preds if p["entity_group"] in relevant_labels}

        for sp in gold_set:
            label = sp[2]
            if sp in pred_set:
                tp[label] += 1
            else:
                fn[label] += 1
                if len(sample_misses[label]) < 5:
                    sample_misses[label].append((i, sp, text))
        for sp in pred_set:
            label = sp[2]
            if sp not in gold_set:
                fp[label] += 1

    print()
    print("=" * 70)
    print(f"v6 on LeNER-Br (real legal text), {len(sample)} sentences")
    print("=" * 70)
    print(f"{'category':<22} {'TP':>6} {'FP':>6} {'FN':>6} {'prec':>8} {'rec':>8} {'F1':>8}")
    print("-" * 70)
    for label in sorted(LENER_TO_OURS.values()):
        t = tp[label]
        f_ = fp[label]
        n = fn[label]
        p = t / (t + f_) if (t + f_) > 0 else 0.0
        r = t / (t + n) if (t + n) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        print(f"{label:<22} {t:>6} {f_:>6} {n:>6} {p:>8.4f} {r:>8.4f} {f1:>8.4f}")

    # Total
    tp_total = sum(tp.values())
    fp_total = sum(fp.values())
    fn_total = sum(fn.values())
    p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    print(f"{'TOTAL':<22} {tp_total:>6} {fp_total:>6} {fn_total:>6} {p:>8.4f} {r:>8.4f} {f1:>8.4f}")

    print()
    print("Sample misses per category:")
    for label, examples in sample_misses.items():
        print(f"\n### {label}:")
        for i, sp, text in examples[:3]:
            span_text = text[sp[0]:sp[1]]
            ctx_s = max(0, sp[0] - 30)
            ctx_e = min(len(text), sp[1] + 30)
            print(f"  '{span_text}'  ctx=...{text[ctx_s:ctx_e]!r}...")


if __name__ == "__main__":
    main()
