"""cmd_prepare: build main batch JSONL from cached perfis + templates."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from src.batch.config import MODEL
from src.batch.instrumentation import build_prompt_and_metadata
from src.batch.templates import TEMPLATES


def cmd_prepare(args) -> None:
    """Build batch JSONL + metadata sidecar."""
    perfis_path = Path(args.perfis)
    perfis = [json.loads(line) for line in perfis_path.open()]
    random.shuffle(perfis)
    print(f"Loaded {len(perfis)} profiles")

    out_input = Path(args.output)
    out_meta = out_input.with_suffix(".metadata.jsonl")
    out_input.parent.mkdir(parents=True, exist_ok=True)

    target = args.n
    fp_in = out_input.open("w")
    fp_meta = out_meta.open("w")

    for i in range(target):
        perfil = perfis[i % len(perfis)]
        template = random.choice(TEMPLATES)
        try:
            prompt, inserted = build_prompt_and_metadata(perfil, template)
        except Exception as e:
            print(f"  [prepare] skip {i}: {type(e).__name__}: {e}", file=sys.stderr)
            continue

        custom_id = f"req-{i:06d}"
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": 1024,
            "reasoning_effort": "minimal",
        }
        line = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        fp_in.write(json.dumps(line, ensure_ascii=False) + "\n")
        fp_meta.write(json.dumps({
            "custom_id": custom_id,
            "template": template,
            "inserted": inserted,
        }, ensure_ascii=False) + "\n")

        if (i + 1) % 5000 == 0:
            print(f"  [prepare] {i+1}/{target}")

    fp_in.close()
    fp_meta.close()
    print(f"Wrote {target} prompts to {out_input}")
    print(f"Wrote metadata to {out_meta}")
