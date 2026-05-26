"""Generate dataset examples via Claude Haiku CLI (sequential, subprocess).

Companion to scripts/openai_batch.py which uses OpenAI Batch API. Haiku via
CLI is sequential (~3s/example) so target small-to-medium batches (1k-10k)
to inject rewriter diversity — addresses gpt-5-nano context-token overfit
exposed in v8 Phase 1 (date 0/30, see v8 milestone plan).

Uses same instrumentation (src/batch/instrumentation.build_prompt_and_metadata)
and labeler (src/labeler.label_text) as the OpenAI pipeline. Output format
is compatible: `data/dataset_br_v<N>_haiku.jsonl` + `_haiku_holdout.jsonl`.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.batch.instrumentation import build_prompt_and_metadata
from src.batch.templates import TEMPLATES
from src.haiku import HaikuGenerator
from src.labeler import label_text
from src.validator import ValidationResult, validate_example


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--perfis", default="data/perfis_full.jsonl")
    ap.add_argument("--output", default="data/dataset_br_v8_haiku.jsonl")
    ap.add_argument("--holdout-ratio", type=float, default=0.09)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    perfis = [json.loads(line) for line in open(args.perfis)]
    random.shuffle(perfis)
    print(f"Loaded {len(perfis)} profiles, targeting {args.n} examples via Haiku")

    gen = HaikuGenerator(provider="claude")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    holdout_path = out.with_name(out.stem + "_holdout.jsonl")

    fp_train = out.open("w")
    fp_holdout = holdout_path.open("w")
    stats = {"ok": 0, "invalid": 0, "haiku_error": 0}

    t0 = time.monotonic()
    for i in range(args.n):
        perfil = perfis[i % len(perfis)]
        template = random.choice(TEMPLATES)
        try:
            prompt, inserted = build_prompt_and_metadata(perfil, template)
            # Haiku-rewritten text — strip PROMPT_PREFIX since we send only the body
            rendered = prompt.split("\n\n", 1)[-1]
            text = gen.generate(template, _ctx_for(perfil, prompt, rendered))
        except Exception as e:
            stats["haiku_error"] += 1
            print(f"  [{i}] haiku error: {type(e).__name__}: {e}", file=sys.stderr)
            continue

        example = label_text(text, inserted)
        if validate_example(example) != ValidationResult.VALID:
            stats["invalid"] += 1
            continue
        for ent in example["entities"]:
            ent["label"] = ent["label"].lower()
        example["template"] = template
        line = json.dumps(example, ensure_ascii=False) + "\n"
        if random.random() < args.holdout_ratio:
            fp_holdout.write(line)
        else:
            fp_train.write(line)
        stats["ok"] += 1

        if (i + 1) % 25 == 0:
            elapsed = time.monotonic() - t0
            rate = (i + 1) / elapsed
            eta = (args.n - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{args.n}] rate={rate:.2f} req/s eta={eta/60:.1f}min stats={stats}", flush=True)

    fp_train.close()
    fp_holdout.close()
    print(f"\nDone. {stats}")
    print(f"  train: {out}")
    print(f"  holdout: {holdout_path}")


def _ctx_for(perfil, prompt, rendered):
    """Build the dict HaikuGenerator.generate expects.

    HaikuGenerator re-renders the template internally using these vars, so
    they must match what instrumentation.py passes to Jinja. Keep this small
    by reading vars used in the rendered text — we pass the perfil unmodified.
    """
    # Mirror what instrumentation.py puts in chosen — same keys.
    nome = perfil["nome"]
    ctx = {
        "nome": nome,
        "nome_upper": nome.upper(),
        "email": perfil["email"],
        "cidade": perfil["cidade"],
        "estado": perfil["estado"],
        "data_nasc": perfil["data_nasc"],
        "endereco": perfil["endereco"],
    }
    # Variant fields used in templates with _valor/_label suffix — best effort:
    # we don't have access to which variant was chosen, so use perfil's
    # canonical value as fallback. instrumentation.py would have chosen one.
    for k in ("cpf", "cnpj", "rg", "cep", "celular", "telefone_fixo",
              "pis", "cnh", "ie", "titulo_eleitor", "certidao_nascimento",
              "order_id", "tracking_code", "invoice_number", "revenue",
              "transaction_id", "customer_id"):
        if k in perfil and perfil[k]:
            ctx[f"{k}_valor"] = perfil[k]
            ctx[f"{k}_label"] = k.replace("_", " ").title()
    return ctx


if __name__ == "__main__":
    main()
