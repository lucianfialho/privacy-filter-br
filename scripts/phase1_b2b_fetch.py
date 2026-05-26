"""Phase 1 B2B: build real-world test set from CVM FRE 2024 board-member CSV.

Each row → one doc with gold spans for Nome/CPF/CNPJ/Data_Nascimento. The
free-text Experiencia_Profissional field carries unlabeled PII (other names,
dates, CNPJs) that we treat as ENRICHMENT — useful for measuring v6 recall
on long-form B2B prose.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data/fre_2024/fre_cia_aberta_administrador_membro_conselho_fiscal_2024.csv"
OUT = ROOT / "data/phase1_b2b.jsonl"


def find_span(text: str, value: str) -> tuple[int, int] | None:
    """Find first occurrence of value in text, return (start, end) or None."""
    if not value:
        return None
    idx = text.find(value)
    if idx == -1:
        return None
    return idx, idx + len(value)


def build_doc(row: dict) -> dict | None:
    """Convert a CSV row into a (text, entities) labeled example."""
    nome = (row.get("Nome") or "").strip()
    cpf = (row.get("CPF") or "").strip()
    cnpj = (row.get("CNPJ_Companhia") or "").strip()
    data_nasc = (row.get("Data_Nascimento") or "").strip()
    nome_companhia = (row.get("Nome_Companhia") or "").strip()
    profissao = (row.get("Profissao") or "").strip()
    cargo = (row.get("Cargo_Eletivo_Ocupado") or "").strip()
    experiencia = (row.get("Experiencia_Profissional") or "").strip()
    # Clean weird control chars from CVM data
    experiencia = experiencia.replace("\x07", " ").replace("\t", " ")
    while "  " in experiencia:
        experiencia = experiencia.replace("  ", " ")

    if not (nome and cpf and cnpj):
        return None

    # Reformat date from YYYY-MM-DD to DD/MM/YYYY (more BR-natural)
    if len(data_nasc) == 10 and data_nasc[4] == "-":
        data_nasc_br = f"{data_nasc[8:10]}/{data_nasc[5:7]}/{data_nasc[0:4]}"
    else:
        data_nasc_br = data_nasc

    # Build natural prose document (NOT template-style label:value pairs)
    text = (
        f"Cadastro de administrador — {nome_companhia} (CNPJ {cnpj})\n\n"
        f"O administrador {nome}, portador do CPF {cpf}, "
        f"nascido em {data_nasc_br}, exerce profissão de {profissao}. "
        f"Cargo na empresa: {cargo}.\n\n"
        f"Experiência profissional:\n{experiencia}"
    )

    # Build gold entities for known fields
    entities = []
    for value, label in [
        (nome, "private_person"),
        (cpf, "private_cpf"),
        (cnpj, "private_cnpj"),
        (data_nasc_br, "private_date"),
    ]:
        span = find_span(text, value)
        if span:
            entities.append({"start": span[0], "end": span[1], "label": label})

    # Sort by start
    entities.sort(key=lambda e: e["start"])
    return {"text": text, "entities": entities, "source": "cvm_fre_conselho_2024"}


def main() -> None:
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Did you unzip data/fre_2024.zip?")
        return

    random.seed(42)
    docs = []
    with CSV_PATH.open(encoding="latin1") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = list(reader)

    print(f"Total CSV rows: {len(rows)}")
    random.shuffle(rows)

    # Filter to rows with all key fields populated
    target = 30
    for row in rows:
        if len(docs) >= target:
            break
        doc = build_doc(row)
        if doc is None:
            continue
        # Avoid too-short docs (need richness)
        if len(doc["text"]) < 500:
            continue
        # Avoid too-long (>3000 chars) for v6's 512 token budget
        if len(doc["text"]) > 3000:
            doc["text"] = doc["text"][:3000]
            # Re-filter entities within range
            doc["entities"] = [e for e in doc["entities"] if e["end"] <= 3000]
        docs.append(doc)

    print(f"Built {len(docs)} B2B test docs.")
    print(f"Avg text length: {sum(len(d['text']) for d in docs) / len(docs):.0f} chars")

    # Stats on label categories
    from collections import Counter
    cats = Counter()
    for d in docs:
        for e in d["entities"]:
            cats[e["label"]] += 1
    print("Gold spans per category:")
    for c, n in cats.most_common():
        print(f"  {c}: {n}")

    with OUT.open("w") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(docs)} docs to {OUT}")


if __name__ == "__main__":
    main()
