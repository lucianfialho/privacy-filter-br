"""cmd_extras: generate OAI-only PII categories (no 4devs profile needed)."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from src.batch.config import JINJA, MODEL, PROMPT_PREFIX
from src.batch.templates import EXTRAS_TEMPLATES
from src.extras import build_extras_perfil


def build_extras_prompt_and_metadata() -> tuple[str, dict[str, str], str]:
    """Returns (prompt, inserted_map, template_name) for an 'extras' example
    covering OAI categories (date, url, secret, account_number)."""
    perfil = build_extras_perfil()
    inserted = dict(perfil["_inserted_extras"])
    template_name = random.choice(EXTRAS_TEMPLATES)
    tpl = JINJA.get_template(f"{template_name}.jinja2")
    rendered = tpl.render(**perfil)
    return PROMPT_PREFIX + rendered, inserted, template_name


def cmd_extras(args) -> None:
    """Generate N batch entries for OAI-only categories (no 4devs needed)."""
    out_input = Path(args.output)
    out_meta = out_input.with_suffix(".metadata.jsonl")
    out_input.parent.mkdir(parents=True, exist_ok=True)

    fp_in = out_input.open("w")
    fp_meta = out_meta.open("w")
    target = args.n

    for i in range(target):
        try:
            prompt, inserted, template = build_extras_prompt_and_metadata()
        except Exception as e:
            print(f"  [extras] skip {i}: {type(e).__name__}: {e}", file=sys.stderr)
            continue

        custom_id = f"extras-{i:06d}"
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

        if (i + 1) % 500 == 0:
            print(f"  [extras] {i+1}/{target}")

    fp_in.close()
    fp_meta.close()
    print(f"Wrote {target} extras prompts to {out_input}")
    print(f"Wrote metadata to {out_meta}")
