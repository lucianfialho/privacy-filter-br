"""Builds prompt + inserted-dict for a perfil + template combo.

This is where PII registration happens — every value passed to the Jinja2
template must also be added to `inserted` so the labeler downstream knows
what to mark as PII in the rewritten text. Bugs here cause silent
training-data corruption (see issue #3: data_nasc/endereco never registered).
"""
from __future__ import annotations

import unicodedata

from src.batch.config import JINJA, PROMPT_PREFIX
from src.variants import get_variants_for_perfil, pick_variant


def _strip_accents(s: str) -> str:
    """ASCII-fold by removing combining accents. 'João' → 'Joao'."""
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def build_prompt_and_metadata(perfil: dict, template_name: str) -> tuple[str, dict[str, str]]:
    """Returns (rendered_prompt, inserted_map) for a profile+template combo."""
    variants = get_variants_for_perfil(perfil)
    chosen: dict[str, str] = {}
    inserted: dict[str, str] = {}
    for field_name, vs in variants.items():
        if not vs:
            continue
        value, label = pick_variant(vs)
        chosen[f"{field_name}_valor"] = value
        chosen[f"{field_name}_label"] = label.replace("PRIVATE_", "").title()
        for v, lbl in vs:
            if v:
                inserted[v] = lbl

    # Name variants — real-world docs use multiple cases (Title Case in
    # contracts, ALL-CAPS in formal docs, lowercase in informal/email, and
    # sometimes accent-stripped from OCR or terminal-only systems). Register
    # all combinations (deduped via set) so labeler catches whichever the
    # rewriter or downstream input emits.
    nome = perfil["nome"]
    nome_ascii = _strip_accents(nome)
    for variant in {
        nome, nome.upper(), nome.title(), nome.lower(),
        nome_ascii, nome_ascii.upper(), nome_ascii.title(), nome_ascii.lower(),
    }:
        inserted[variant] = "PRIVATE_PERSON"
    inserted[perfil["email"]] = "PRIVATE_EMAIL"

    # Date — register canonical (DD/MM/YYYY from 4devs) plus common separator
    # variants the rewriter might emit. Not added to _FORMAT_AWARE_LABELS in
    # src/labeler.py — skeleton match on 8-digit dates would FP on unrelated IDs.
    data_nasc = perfil["data_nasc"]
    inserted[data_nasc] = "PRIVATE_DATE"
    inserted[data_nasc.replace("/", "-")] = "PRIVATE_DATE"
    inserted[data_nasc.replace("/", ".")] = "PRIVATE_DATE"

    # Address — full street + structured prefix (street + number). CEPs are
    # still labeled via variantes_cep in the variants loop above.
    endereco = perfil["endereco"]
    inserted[endereco] = "PRIVATE_ADDRESS"
    parts = endereco.split(",")
    if len(parts) >= 2:
        inserted[",".join(parts[:2]).strip()] = "PRIVATE_ADDRESS"

    chosen["nome"] = nome
    chosen["nome_upper"] = nome.upper()
    chosen["email"] = perfil["email"]
    chosen["cidade"] = perfil["cidade"]
    chosen["estado"] = perfil["estado"]
    chosen["data_nasc"] = data_nasc
    chosen["endereco"] = endereco

    tpl = JINJA.get_template(f"{template_name}.jinja2")
    rendered = tpl.render(**chosen)
    return PROMPT_PREFIX + rendered, inserted
