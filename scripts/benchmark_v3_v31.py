"""Benchmark v3 and v3.1 on both original and relabeled holdouts (2x2 matrix).

Output: per-cell precision/recall/F1 (micro + macro), per-category for the most
interesting cells. Tests the hypothesis that v3.1's "F1 drop" was due to the
holdout having the same labeler bug.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def spans_to_set(entities: list[dict]) -> set[tuple]:
    return {(e["start"], e["end"], e["label"]) for e in entities}


def compute_f1(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def load_pipe(model_dir: Path):
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
    tok = AutoTokenizer.from_pretrained(str(model_dir), model_max_length=512)
    mdl = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    return pipeline("token-classification", model=mdl, tokenizer=tok,
                    aggregation_strategy="first", device=-1)


def evaluate(pipe, examples: list[dict], max_chars: int = 1500) -> dict:
    per_label_tp: defaultdict[str, int] = defaultdict(int)
    per_label_fp: defaultdict[str, int] = defaultdict(int)
    per_label_fn: defaultdict[str, int] = defaultdict(int)

    for i, ex in enumerate(examples):
        if i % 500 == 0 and i > 0:
            print(f"    {i}/{len(examples)}")
        text = ex["text"]
        if len(text) > max_chars:
            text = text[:max_chars]
        preds = pipe(text)
        pred_set = {(p["start"], p["end"], p["entity_group"]) for p in preds}
        gold_set = {(e["start"], e["end"], e["label"]) for e in ex["entities"]
                    if e["end"] <= len(text)}

        for sp in gold_set:
            label = sp[2]
            if sp in pred_set:
                per_label_tp[label] += 1
            else:
                per_label_fn[label] += 1
        for sp in pred_set:
            label = sp[2]
            if sp not in gold_set:
                per_label_fp[label] += 1

    all_labels = set(per_label_tp) | set(per_label_fp) | set(per_label_fn)
    per_label = {l: compute_f1(per_label_tp[l], per_label_fp[l], per_label_fn[l])
                 for l in all_labels}

    tp = sum(per_label_tp.values())
    fp = sum(per_label_fp.values())
    fn = sum(per_label_fn.values())
    micro = compute_f1(tp, fp, fn)
    macro_p = sum(v["precision"] for v in per_label.values()) / len(per_label)
    macro_r = sum(v["recall"] for v in per_label.values()) / len(per_label)
    macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r) if (macro_p + macro_r) > 0 else 0.0

    return {"per_label": per_label, "micro": micro,
            "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1}}


def fmt(r: dict) -> str:
    return f"P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-orig", default=str(ROOT / "data" / "dataset_br_v3_holdout.jsonl"))
    parser.add_argument("--holdout-rel", default=str(ROOT / "data" / "dataset_br_v3_holdout_relabeled.jsonl"))
    parser.add_argument("--v3-model", default=str(ROOT / "checkpoints" / "v3-local"))
    parser.add_argument("--v31-model", default=str(ROOT / "checkpoints" / "v3.1-local"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.holdout_orig) as f:
        examples_orig = [json.loads(line) for line in f]
    with open(args.holdout_rel) as f:
        examples_rel = [json.loads(line) for line in f]

    if args.limit:
        examples_orig = examples_orig[:args.limit]
        examples_rel = examples_rel[:args.limit]

    print(f"Holdout: {len(examples_orig)} examples (orig labels + relabeled)")

    print(f"\nLoading v3 from {args.v3_model}...")
    pipe_v3 = load_pipe(Path(args.v3_model))
    print(f"Loading v3.1 from {args.v31_model}...")
    pipe_v31 = load_pipe(Path(args.v31_model))

    print("\nEvaluating v3 on original holdout...")
    res_v3_orig = evaluate(pipe_v3, examples_orig)
    print("Evaluating v3 on relabeled holdout...")
    res_v3_rel = evaluate(pipe_v3, examples_rel)
    print("Evaluating v3.1 on original holdout...")
    res_v31_orig = evaluate(pipe_v31, examples_orig)
    print("Evaluating v3.1 on relabeled holdout...")
    res_v31_rel = evaluate(pipe_v31, examples_rel)

    print("\n=== 2x2 RESULTS (model × holdout-labels) ===\n")
    print(f"v3   vs orig:      micro {fmt(res_v3_orig['micro'])}  macro F1={res_v3_orig['macro']['f1']:.4f}")
    print(f"v3   vs relabeled: micro {fmt(res_v3_rel['micro'])}  macro F1={res_v3_rel['macro']['f1']:.4f}")
    print(f"v3.1 vs orig:      micro {fmt(res_v31_orig['micro'])}  macro F1={res_v31_orig['macro']['f1']:.4f}")
    print(f"v3.1 vs relabeled: micro {fmt(res_v31_rel['micro'])}  macro F1={res_v31_rel['macro']['f1']:.4f}")

    print("\n=== Deltas ===")
    print(f"v3   on relabel vs v3   on orig:  micro F1 Δ = {res_v3_rel['micro']['f1']  - res_v3_orig['micro']['f1']:+.4f}")
    print(f"v3.1 on relabel vs v3.1 on orig:  micro F1 Δ = {res_v31_rel['micro']['f1'] - res_v31_orig['micro']['f1']:+.4f}")
    print(f"v3.1 vs v3 on orig:               micro F1 Δ = {res_v31_orig['micro']['f1'] - res_v3_orig['micro']['f1']:+.4f}")
    print(f"v3.1 vs v3 on relabel:            micro F1 Δ = {res_v31_rel['micro']['f1'] - res_v3_rel['micro']['f1']:+.4f}")

    print("\n=== Per-category F1 (focus: categories with relabel) ===")
    focus = ["private_email", "private_cpf", "private_phone", "private_rg",
             "private_titulo_eleitor", "private_cnpj", "private_pis",
             "private_person", "private_address"]
    print(f"{'category':<28} {'v3/orig':>10} {'v3/rel':>10} {'v3.1/orig':>10} {'v3.1/rel':>10}")
    print("-" * 76)
    for cat in focus:
        a = res_v3_orig['per_label'].get(cat, {}).get('f1', 0)
        b = res_v3_rel['per_label'].get(cat, {}).get('f1', 0)
        c = res_v31_orig['per_label'].get(cat, {}).get('f1', 0)
        d = res_v31_rel['per_label'].get(cat, {}).get('f1', 0)
        print(f"{cat:<28} {a:>10.4f} {b:>10.4f} {c:>10.4f} {d:>10.4f}")

    # Save full results
    out = {
        "v3_orig": res_v3_orig,
        "v3_rel": res_v3_rel,
        "v31_orig": res_v31_orig,
        "v31_rel": res_v31_rel,
    }
    out_path = ROOT / "data" / "benchmark_v3_v31_2x2.json"
    with out_path.open("w") as f:
        # Convert defaults
        def encode(x):
            if isinstance(x, dict):
                return {k: encode(v) for k, v in x.items()}
            return x
        json.dump(encode(out), f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
