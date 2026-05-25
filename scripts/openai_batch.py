"""OpenAI Batch API pipeline — CLI dispatcher.

Usage:
    # 1. Generate 4devs profiles cache (~3h with throttle)
    python3 scripts/openai_batch.py perfis --n 10000 --output data/perfis_cache.jsonl

    # 2. Prepare batch JSONL from cached profiles + templates
    python3 scripts/openai_batch.py prepare --n 47000 \\
        --perfis data/perfis_cache.jsonl \\
        --output data/batch_input.jsonl

    # 3. Submit batch to OpenAI
    python3 scripts/openai_batch.py submit \\
        --input data/batch_input.jsonl

    # 4. Process batch results once completed
    python3 scripts/openai_batch.py process \\
        --batch-id batch_XYZ \\
        --metadata data/batch_metadata.jsonl \\
        --output data/dataset_br_v2.jsonl

Implementations live in src/batch/ — one module per command.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.batch import cmd_extras, cmd_perfis, cmd_prepare, cmd_process, cmd_submit


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_perfis = sub.add_parser("perfis")
    p_perfis.add_argument("--n", type=int, default=10000)
    p_perfis.add_argument("--output", default="data/perfis_cache.jsonl")
    p_perfis.add_argument("--workers", type=int, default=3)

    p_prep = sub.add_parser("prepare")
    p_prep.add_argument("--n", type=int, required=True)
    p_prep.add_argument("--perfis", default="data/perfis_cache.jsonl")
    p_prep.add_argument("--output", default="data/batch_input.jsonl")

    p_ext = sub.add_parser("extras")
    p_ext.add_argument("--n", type=int, default=2000)
    p_ext.add_argument("--output", default="data/batch_extras.jsonl")

    p_sub = sub.add_parser("submit")
    p_sub.add_argument("--input", default="data/batch_input.jsonl")

    p_proc = sub.add_parser("process")
    p_proc.add_argument("--batch-id", default=None)
    p_proc.add_argument("--metadata", default="data/batch_input.metadata.jsonl")
    p_proc.add_argument("--output", default="data/dataset_br_v2.jsonl")
    p_proc.add_argument("--holdout-ratio", type=float, default=0.09)

    args = parser.parse_args()
    {
        "perfis": cmd_perfis,
        "prepare": cmd_prepare,
        "extras": cmd_extras,
        "submit": cmd_submit,
        "process": cmd_process,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
