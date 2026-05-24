"""Spot-check relabel_v2 output: show examples where new labels were added.

Helps verify the new labeler isn't adding false positives.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V3 = ROOT / "data" / "dataset_br_v3.jsonl"
V4 = ROOT / "data" / "dataset_br_v4.jsonl"


def main() -> None:
    with V3.open() as f_orig, V4.open() as f_new:
        v3_rows = [json.loads(line) for line in f_orig]
        v4_rows = [json.loads(line) for line in f_new]

    shown = 0
    for idx in range(min(len(v3_rows), len(v4_rows))):
        v3 = v3_rows[idx]
        v4 = v4_rows[idx]
        v3_keys = {(e["start"], e["end"], e["label"]) for e in v3["entities"]}
        v4_keys = {(e["start"], e["end"], e["label"]) for e in v4["entities"]}
        added = v4_keys - v3_keys
        if not added:
            continue
        if shown >= 15:
            break
        shown += 1
        print(f"\n=== Example {idx} (template={v4.get('template', '?')}) ===")
        for s, e, lbl in sorted(added):
            ctx_s = max(0, s - 25)
            ctx_e = min(len(v4["text"]), e + 25)
            highlighted = (
                v4["text"][ctx_s:s]
                + f"[[{v4['text'][s:e]}]]"
                + v4["text"][e:ctx_e]
            )
            print(f"  +{lbl} [{s}:{e}]  ...{highlighted!r}...")


if __name__ == "__main__":
    main()
