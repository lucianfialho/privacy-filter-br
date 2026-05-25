"""
OpenAI Batch API pipeline for generating PT-BR PII examples.

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
"""
import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import openai
from jinja2 import Environment, FileSystemLoader

from src.pessoa import gerar_perfil_completo
from src.variants import get_variants_for_perfil, pick_variant
from src.labeler import label_text
from src.validator import validate_example, ValidationResult
from src.extras import build_extras_perfil

TEMPLATES = [
    # Structured documents (v1-v5 baseline)
    "email", "nfe", "contrato", "holerite",
    "certidao", "cadastro", "comunicado", "relatorio",
    "nfe_completa", "darf", "boleto",
    "comprovante_pix", "extrato_bancario", "fatura_servico",
    "pedido_marketplace", "dashboard_vendas",
    "comprovante_delivery", "relatorio_faturamento",
    # Narrative / conversational (v6 — fix CUST/order disambig + revenue in prose)
    "artigo_noticia", "email_conversacional", "doc_tecnico",
    "nota_livre", "dialogo_chat", "email_thread",
    "comentario_sistema", "artigo_blog",
    "rh_perfil_narrativo", "incident_report",
]

EXTRAS_TEMPLATES = [
    "extras_devops_log", "extras_notification_email",
    "extras_bank_statement", "extras_api_docs",
]

PROMPT_PREFIX = (
    "Reescreva o texto abaixo em português BR natural e formal, "
    "mantendo TODOS os valores exatamente como estão (CPF, CNPJ, nomes, etc). "
    "NÃO altere nenhum número ou dado pessoal:\n\n"
)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
JINJA = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))

MODEL = os.getenv("OPENAI_BATCH_MODEL", "gpt-5-nano")


# ---------------------------- perfis ----------------------------
def cmd_perfis(args):
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

    import threading
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


# ---------------------------- prepare ----------------------------
def build_prompt_and_metadata(perfil, template_name):
    """Returns (rendered_prompt, inserted_map) for a profile+template combo."""
    variants = get_variants_for_perfil(perfil)
    chosen = {}
    inserted = {}
    for field_name, vs in variants.items():
        if not vs:
            continue
        value, label = pick_variant(vs)
        chosen[f"{field_name}_valor"] = value
        chosen[f"{field_name}_label"] = label.replace("PRIVATE_", "").title()
        for v, lbl in vs:
            if v:
                inserted[v] = lbl

    inserted[perfil["nome"]] = "PRIVATE_PERSON"
    inserted[perfil["email"]] = "PRIVATE_EMAIL"
    chosen["nome"] = perfil["nome"]
    chosen["email"] = perfil["email"]
    chosen["cidade"] = perfil["cidade"]
    chosen["estado"] = perfil["estado"]
    chosen["data_nasc"] = perfil["data_nasc"]
    chosen["endereco"] = perfil["endereco"]

    tpl = JINJA.get_template(f"{template_name}.jinja2")
    rendered = tpl.render(**chosen)
    return PROMPT_PREFIX + rendered, inserted


def build_extras_prompt_and_metadata():
    """Returns (prompt, inserted_map, template_name) for an 'extras' example
    covering OAI categories (date, url, secret, account_number)."""
    perfil = build_extras_perfil()
    inserted = dict(perfil["_inserted_extras"])
    template_name = random.choice(EXTRAS_TEMPLATES)
    tpl = JINJA.get_template(f"{template_name}.jinja2")
    rendered = tpl.render(**perfil)
    return PROMPT_PREFIX + rendered, inserted, template_name


def cmd_extras(args):
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


def cmd_prepare(args):
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


# ---------------------------- submit ----------------------------
def cmd_submit(args):
    """Upload JSONL and create a batch job."""
    client = openai.OpenAI()
    print(f"Uploading {args.input} ...")
    with open(args.input, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    print(f"  file id: {file_obj.id}")

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"Created batch: {batch.id}")
    print(f"  status: {batch.status}")

    out_id = Path(args.input).with_name("batch_id.txt")
    out_id.write_text(batch.id)
    print(f"Saved batch id to {out_id}")


# ---------------------------- process ----------------------------
def cmd_process(args):
    """Poll/download batch results, validate, label, append to dataset."""
    client = openai.OpenAI()
    batch_id = args.batch_id
    if not batch_id:
        bid_file = Path(args.metadata).with_name("batch_id.txt")
        if bid_file.exists():
            batch_id = bid_file.read_text().strip()
        else:
            print("Provide --batch-id or have batch_id.txt next to metadata", file=sys.stderr)
            sys.exit(1)

    print(f"Polling batch {batch_id} ...")
    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"  status={batch.status} | counts={batch.request_counts}")
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(30)

    if batch.status != "completed":
        print(f"Batch ended with status {batch.status}", file=sys.stderr)
        sys.exit(1)

    print("Downloading output file ...")
    output_resp = client.files.content(batch.output_file_id)
    raw = output_resp.read()
    raw_path = Path(args.metadata).with_name(f"batch_output_{batch_id}.jsonl")
    raw_path.write_bytes(raw)
    print(f"  saved raw to {raw_path}")

    # Load metadata
    meta_by_id = {}
    with open(args.metadata) as f:
        for line in f:
            d = json.loads(line)
            meta_by_id[d["custom_id"]] = d
    print(f"Loaded {len(meta_by_id)} metadata records")

    # Process responses
    holdout_path = Path(args.output).with_name(Path(args.output).stem + "_holdout.jsonl")
    holdout_ratio = args.holdout_ratio
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
        example["template"] = meta["template"]
        out_line = json.dumps(example, ensure_ascii=False) + "\n"
        if random.random() < holdout_ratio:
            fp_holdout.write(out_line)
        else:
            fp_train.write(out_line)
        stats["ok"] += 1

    fp_train.close()
    fp_holdout.close()
    print(f"Done. {stats}")


# ---------------------------- main ----------------------------
def main():
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
