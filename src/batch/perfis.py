"""cmd_perfis: pre-generate 4devs profiles to JSONL cache (resumable, threaded)."""
from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.pessoa import gerar_perfil_completo


def cmd_perfis(args) -> None:
    """Pre-generate N 4devs profiles to a JSONL cache."""
    target = args.n
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    existing = 0
    if out.exists():
        with out.open() as f:
            existing = sum(1 for _ in f)
        print(f"Resuming: {existing} profiles already cached, need {target - existing} more")
        if existing >= target:
            print("Already have enough.")
            return

    remaining = target - existing
    print(f"Generating {remaining} profiles to {out} ...")

    write_lock = threading.Lock()
    fp = out.open("a")
    completed = {"n": 0}

    def worker(_):
        try:
            perfil = gerar_perfil_completo()
            line = json.dumps(perfil, ensure_ascii=False) + "\n"
            with write_lock:
                fp.write(line)
                fp.flush()
                completed["n"] += 1
                if completed["n"] % 50 == 0:
                    print(f"  [perfis] {completed['n']}/{remaining}", flush=True)
        except Exception as e:
            print(f"  [perfis] error: {type(e).__name__}: {e}", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(worker, range(remaining)))

    fp.close()
    print(f"Done. Total perfis in cache: {existing + completed['n']}")
