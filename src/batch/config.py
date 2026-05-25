"""Shared config for OpenAI Batch pipeline: env, model, prompt prefix, Jinja env."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

PROMPT_PREFIX = (
    "Reescreva o texto abaixo em português BR natural e formal, "
    "mantendo TODOS os valores exatamente como estão (CPF, CNPJ, nomes, etc). "
    "NÃO altere nenhum número ou dado pessoal:\n\n"
)

TEMPLATES_DIR = _REPO_ROOT / "templates"
JINJA = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))

MODEL = os.getenv("OPENAI_BATCH_MODEL", "gpt-5-nano")
