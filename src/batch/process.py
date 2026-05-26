"""cmd_process: poll/download batch, label rewritten text, split train/holdout."""
from __future__ import annotations

import json
import random
import subprocess
import sys
import time
from pathlib import Path

import openai

from src.labeler import label_text
from src.validator import ValidationResult, validate_example


def _resolve_batch_id(args) -> str:
    if args.batch_id:
        return args.batch_id
    bid_file = Path(args.metadata).with_name("batch_id.txt")
    if bid_file.exists():
        return bid_file.read_text().strip()
    print("Provide --batch-id or have batch_id.txt next to metadata", file=sys.stderr)
    sys.exit(1)


def _wait_for_batch(client, batch_id):
    print(f"Polling batch {batch_id} ...")
    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"  status={batch.status} | counts={batch.request_counts}")
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            return batch
        time.sleep(30)


def cmd_process(args) -> None:
    """Poll/download batch results, validate, label, append to dataset."""
    client = openai.OpenAI()
    batch_id = _resolve_batch_id(args)
    batch = _wait_for_batch(client, batch_id)

    if batch.status != "completed":
        print(f"Batch ended with status {batch.status}", file=sys.stderr)
        sys.exit(1)

    print("Downloading output file ...")
    output_resp = client.files.content(batch.output_file_id)
    raw = output_resp.read()
    raw_path = Path(args.metadata).with_name(f"batch_output_{batch_id}.jsonl")
    raw_path.write_bytes(raw)
    print(f"  saved raw to {raw_path}")

    meta_by_id = {}
    with open(args.metadata) as f:
        for line in f:
            d = json.loads(line)
            meta_by_id[d["custom_id"]] = d
    print(f"Loaded {len(meta_by_id)} metadata records")

    holdout_path = Path(args.output).with_name(Path(args.output).stem + "_holdout.jsonl")
    fp_train = Path(args.output).open("a")
    fp_holdout = holdout_path.open("a")
    stats = {"ok": 0, "invalid": 0, "no_meta": 0, "api_error": 0}

    for line in raw_path.open():
        rec = json.loads(line)
        cid = rec["custom_id"]
        meta = meta_by_id.get(cid)
        if not meta:
            stats["no_meta"] += 1
            continue
        resp = rec.get("response") or {}
        if rec.get("error") or resp.get("status_code") != 200:
            stats["api_error"] += 1
            continue
        try:
            text = resp["body"]["choices"][0]["message"]["content"].strip()
        except Exception:
            stats["api_error"] += 1
            continue
        if not text:
            stats["api_error"] += 1
            continue

        example = label_text(text, meta["inserted"])
        result = validate_example(example)
        if result != ValidationResult.VALID:
            stats["invalid"] += 1
            continue
        # Normalize labels to lowercase to match training/holdout convention
        # (label_text emits UPPERCASE; downstream training expects lowercase).
        for ent in example["entities"]:
            ent["label"] = ent["label"].lower()
        example["template"] = meta["template"]
        out_line = json.dumps(example, ensure_ascii=False) + "\n"
        if random.random() < args.holdout_ratio:
            fp_holdout.write(out_line)
        else:
            fp_train.write(out_line)
        stats["ok"] += 1

    fp_train.close()
    fp_holdout.close()
    print(f"Done. {stats}")

    # Auto-audit: catch the v7 instrumentation-bug class (zero-label categories)
    # BEFORE the dataset is used for training. Non-fatal — dataset is already
    # written (expensive batch results preserved), but warn loudly so the user
    # knows to investigate before submitting to GPU.
    audit_script = Path(__file__).resolve().parent.parent.parent / "scripts/audit_label_distribution.py"
    if audit_script.exists():
        print(f"\nRunning post-process label audit on {args.output} ...")
        # Use sys.executable so we run the same interpreter that runs cmd_process
        # (the venv's python). Hardcoded "python" fails on systems where only
        # python3 is installed, or when running from a venv without symlinks.
        result = subprocess.run([sys.executable, str(audit_script), args.output])
        if result.returncode != 0:
            print(f"\n⚠️  AUDIT FAILED — dataset has missing or zero-coverage labels.")
            print(f"   Inspect {args.output} before training. Bug class: issue #3.")
