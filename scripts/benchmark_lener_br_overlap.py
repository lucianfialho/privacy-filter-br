"""Phase 1 partial — v6 on LeNER-Br with OVERLAP-based scoring.

Same as benchmark_lener_br.py but uses overlap-based matching:
A v6 prediction matches a gold span if they overlap (and labels match per our mapping).

This is more honest for cross-schema benchmarks where annotation conventions
differ between datasets.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LENER_DIR = ROOT / "data/lener-br/lener-br-master/leNER-Br/test"

LENER_TO_OURS = {
    "PESSOA": "private_person",
    # LOCAL/TEMPO have schema mismatch — uncomment to include with caveat
    # "LOCAL": "private_address",
    # "TEMPO": "private_date",
}


def read_conll_file(path: Path):
    examples = []
    tokens, tags = [], []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    examples.append(build(tokens, tags))
                    tokens, tags = [], []
                continue
            parts = line.split()
            if len(parts) >= 2:
                tokens.append(parts[0])
                tags.append(parts[-1])
    if tokens:
        examples.append(build(tokens, tags))
    return examples


def build(tokens, tags):
    text_parts = []
    entities = []
    pos = 0
    cur_start = None
    cur_label = None
    cur_end = None
    for tok, tag in zip(tokens, tags):
        if pos > 0:
            text_parts.append(" ")
            pos += 1
        tok_start = pos
        text_parts.append(tok)
        pos += len(tok)
        tok_end = pos

        if tag.startswith("B-"):
            if cur_start is not None:
                entities.append({"start": cur_start, "end": cur_end, "label": cur_label})
            cur_start = tok_start
            cur_end = tok_end
            cur_label = tag[2:]
        elif tag.startswith("I-") and cur_start is not None and tag[2:] == cur_label:
            cur_end = tok_end
        else:
            if cur_start is not None:
                entities.append({"start": cur_start, "end": cur_end, "label": cur_label})
                cur_start = None
                cur_label = None
                cur_end = None
    if cur_start is not None:
        entities.append({"start": cur_start, "end": cur_end, "label": cur_label})
    return "".join(text_parts), entities


def overlaps(a_s, a_e, b_s, b_e):
    return a_s < b_e and a_e > b_s


def main():
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

    print("Loading LeNER-Br...")
    all_examples = []
    for f in sorted(LENER_DIR.glob("*.conll")):
        all_examples.extend(read_conll_file(f))

    relevant = []
    for text, ents in all_examples:
        mapped = [e for e in ents if e["label"] in LENER_TO_OURS]
        if mapped:
            relevant.append((text, mapped))
    print(f"  Sentences with PESSOA: {len(relevant)}")

    print("\nLoading v6...")
    tok = AutoTokenizer.from_pretrained("checkpoints/v6-local", model_max_length=512)
    model = AutoModelForTokenClassification.from_pretrained("checkpoints/v6-local").eval()
    ner = pipeline("token-classification", model=model, tokenizer=tok,
                   aggregation_strategy="simple", device=-1)

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    sample_fn = defaultdict(list)
    sample_fp = defaultdict(list)
    sample_match = defaultdict(list)

    for i, (text, gold) in enumerate(relevant):
        if i % 100 == 0 and i > 0:
            print(f"  {i}/{len(relevant)}...")
        gold_mapped = [
            {"start": e["start"], "end": e["end"], "label": LENER_TO_OURS[e["label"]]}
            for e in gold
        ]
        relevant_labels = set(LENER_TO_OURS.values())
        preds = [p for p in ner(text) if p["entity_group"] in relevant_labels]

        # For each gold span, check if any v6 prediction overlaps AND has same label
        gold_matched = [False] * len(gold_mapped)
        pred_matched = [False] * len(preds)

        for gi, g in enumerate(gold_mapped):
            for pi, p in enumerate(preds):
                if g["label"] == p["entity_group"] and overlaps(g["start"], g["end"], p["start"], p["end"]):
                    gold_matched[gi] = True
                    pred_matched[pi] = True
                    if len(sample_match[g["label"]]) < 5:
                        sample_match[g["label"]].append(
                            (text[g["start"]:g["end"]], text[p["start"]:p["end"]])
                        )

        for gi, g in enumerate(gold_mapped):
            if gold_matched[gi]:
                tp[g["label"]] += 1
            else:
                fn[g["label"]] += 1
                if len(sample_fn[g["label"]]) < 5:
                    sample_fn[g["label"]].append((i, g, text))
        for pi, p in enumerate(preds):
            if not pred_matched[pi]:
                fp[p["entity_group"]] += 1
                if len(sample_fp[p["entity_group"]]) < 5:
                    sample_fp[p["entity_group"]].append((i, p, text))

    print()
    print("=" * 70)
    print(f"v6 on LeNER-Br (OVERLAP scoring), {len(relevant)} sentences")
    print("=" * 70)
    print(f"{'category':<22} {'TP':>6} {'FP':>6} {'FN':>6} {'prec':>8} {'rec':>8} {'F1':>8}")
    print("-" * 70)
    for label in sorted(LENER_TO_OURS.values()):
        t, f_, n = tp[label], fp[label], fn[label]
        p = t / (t + f_) if (t + f_) > 0 else 0.0
        r = t / (t + n) if (t + n) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        print(f"{label:<22} {t:>6} {f_:>6} {n:>6} {p:>8.4f} {r:>8.4f} {f1:>8.4f}")

    print()
    print("Sample MATCHES (v6 found a real LeNER entity, possibly different boundaries):")
    for label, matches in sample_match.items():
        print(f"\n{label}:")
        for gold_text, pred_text in matches:
            print(f"  gold={gold_text!r:<40} v6={pred_text!r}")

    print()
    print("Sample FN (v6 missed entirely):")
    for label, examples in sample_fn.items():
        print(f"\n{label}:")
        for i, g, text in examples[:5]:
            print(f"  '{text[g['start']:g['end']]}'")

    print()
    print("Sample FP (v6 over-predicted, not in gold):")
    for label, examples in sample_fp.items():
        print(f"\n{label}:")
        for i, p, text in examples[:5]:
            print(f"  '{text[p['start']:p['end']]}'")


if __name__ == "__main__":
    main()
