#!/usr/bin/env python3
"""
Generate the Privacy Filter BR training dataset.

Usage:
    python scripts/generate_dataset.py --n 11000 --output data/dataset_br.jsonl
    python scripts/generate_dataset.py --n 100000 --workers 20 --output data/dataset_br_v2.jsonl

Parallelized via ThreadPoolExecutor. Each worker independently:
  1. Calls 4devs API for a profile
  2. Calls Haiku to rewrite a template
  3. Writes valid examples to disk (thread-safe with lock)

Writes each example to disk immediately — safe to kill and resume at any time.
"""
import argparse
import json
import os
import sys
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generator import generate_example, GeneratorStats
from src.haiku import HaikuGenerator


def count_existing(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def worker_task(_idx: int, haiku: HaikuGenerator):
    """Each worker generates one example. Returns example or None."""
    try:
        return generate_example(haiku)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate BR PII dataset")
    parser.add_argument("--n", type=int, default=11000, help="Target valid examples")
    parser.add_argument("--output", type=str, default="data/dataset_br.jsonl")
    parser.add_argument("--holdout-ratio", type=float, default=0.09)
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (1=serial, 10-20 recommended for Anthropic API)")
    parser.add_argument("--batch-submit", type=int, default=200,
                        help="How many tasks to submit to the pool at once")
    parser.add_argument("--provider", choices=["claude", "minimax"], default=None,
                        help="LLM provider (default: claude or $PROVIDER env)")
    args = parser.parse_args()

    if args.provider:
        os.environ["PROVIDER"] = args.provider

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    holdout_path = args.output.replace(".jsonl", "_holdout.jsonl")

    already_done = count_existing(args.output) + count_existing(holdout_path)
    target = args.n
    remaining = target - already_done

    if already_done > 0:
        print(f"Resuming: {already_done} already on disk, need {remaining} more")
    else:
        print(f"Generating {target} valid examples → {args.output}")
    print(f"Workers: {args.workers}")

    if remaining <= 0:
        print("Already complete!")
        return

    haiku = HaikuGenerator()
    stats = GeneratorStats()
    write_lock = threading.Lock()
    stats_lock = threading.Lock()

    train_f = open(args.output, "a", encoding="utf-8")
    holdout_f = open(holdout_path, "a", encoding="utf-8")

    def record_and_write(example):
        with stats_lock:
            stats.record(example, None if example else "validation_failed")
            current_valid = stats.valid
            current_total = stats.total
            current_discarded = stats.discarded

        if example:
            with write_lock:
                if random.random() < args.holdout_ratio:
                    holdout_f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    holdout_f.flush()
                else:
                    train_f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    train_f.flush()

        status = "✓" if example else "✗"
        rate = current_valid / current_total * 100 if current_total > 0 else 0
        total_so_far = already_done + current_valid
        print(
            f"  [{status}] attempts {current_total:5d} | valid {total_so_far:6d}/{target} "
            f"| {rate:.0f}% accept | discarded {current_discarded}",
            flush=True,
        )

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = set()
            submitted = 0

            # Initial batch
            initial = min(args.batch_submit, args.workers * 4)
            for _ in range(initial):
                futures.add(pool.submit(worker_task, submitted, haiku))
                submitted += 1

            while stats.valid < remaining and futures:
                done, futures = next_done(futures)
                for fut in done:
                    example = fut.result()
                    record_and_write(example)

                    if stats.valid >= remaining:
                        break

                    # Submit one more to keep pool busy
                    futures.add(pool.submit(worker_task, submitted, haiku))
                    submitted += 1

            # Drain remaining (don't lose work)
            for fut in futures:
                try:
                    example = fut.result(timeout=60)
                    record_and_write(example)
                except Exception:
                    pass
    finally:
        train_f.close()
        holdout_f.close()

    final_train = count_existing(args.output)
    final_holdout = count_existing(holdout_path)
    print(f"\nDone!")
    print(f"  Train: {final_train} → {args.output}")
    print(f"  Holdout: {final_holdout} → {holdout_path}")
    print(f"  This run: {stats.total} attempts ({stats.discarded} discarded)")


def next_done(futures):
    """Return (done_set, pending_set) — blocks until at least one is done."""
    from concurrent.futures import wait, FIRST_COMPLETED
    done, pending = wait(futures, return_when=FIRST_COMPLETED)
    return done, pending


if __name__ == "__main__":
    main()
