"""Builds prompt + inserted-dict for a perfil + template combo.

This is where PII registration happens — every value passed to the Jinja2
template must also be added to `inserted` so the labeler downstream knows
what to mark as PII in the rewritten text. Bugs here cause silent
training-data corruption (see issue #3: data_nasc/endereco never registered).
"""
from __future__ import annotations

from src.batch.config import JINJA, PROMPT_PREFIX
from src.variants import get_variants_for_perfil, pick_variant


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
