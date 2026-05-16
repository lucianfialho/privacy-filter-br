import os
import subprocess
from jinja2 import Environment, FileSystemLoader

CLUE_POSITIONS = {
    "cpf": [
        {"position": "before",  "template": "CPF {value}"},
        {"position": "before",  "template": "CPF nº {value}"},
        {"position": "before",  "template": "portador do CPF {value}"},
        {"position": "before",  "template": "inscrito no CPF sob nº {value}"},
        {"position": "after",   "template": "{value} (CPF)"},
        {"position": "after",   "template": "{value}, CPF do interessado"},
        {"position": "omitted", "template": "{value}"},
    ],
    "cnpj": [
        {"position": "before",  "template": "CNPJ {value}"},
        {"position": "before",  "template": "CNPJ nº {value}"},
        {"position": "after",   "template": "{value} (CNPJ)"},
        {"position": "omitted", "template": "{value}"},
    ],
    "rg": [
        {"position": "before",  "template": "RG {value}"},
        {"position": "before",  "template": "RG nº {value}"},
        {"position": "after",   "template": "{value} (RG)"},
        {"position": "omitted", "template": "{value}"},
    ],
    "ie": [
        {"position": "before",  "template": "Inscrição Estadual {value}"},
        {"position": "before",  "template": "IE {value}"},
        {"position": "omitted", "template": "{value}"},
    ],
    "pis": [
        {"position": "before",  "template": "PIS/PASEP {value}"},
        {"position": "before",  "template": "PIS nº {value}"},
        {"position": "omitted", "template": "{value}"},
    ],
    "titulo": [
        {"position": "before",  "template": "Título de Eleitor nº {value}"},
        {"position": "omitted", "template": "{value}"},
    ],
    "certidao": [
        {"position": "before",  "template": "Certidão nº {value}"},
        {"position": "omitted", "template": "{value}"},
    ],
    "cnh": [
        {"position": "before",  "template": "CNH nº {value}"},
        {"position": "omitted", "template": "{value}"},
    ],
}

_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "..", "templates")


class HaikuGenerator:
    """Wraps `claude --print` subprocess. Uses Claude subscription, no API tokens needed.

    Run `claude auth login` once before using.
    """

    def __init__(self, **kwargs):
        self.env = Environment(loader=FileSystemLoader(_TEMPLATES_DIR))

    def generate(self, template_name: str, context: dict) -> str:
        tpl = self.env.get_template(f"{template_name}.jinja2")
        rendered = tpl.render(**context)
        prompt = (
            f"Reescreva o texto abaixo em português BR natural e formal, "
            f"mantendo TODOS os valores exatamente como estão (CPF, CNPJ, nomes, etc). "
            f"NÃO altere nenhum número ou dado pessoal:\n\n{rendered}"
        )
        result = subprocess.run(
            ["claude", "--print"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"claude CLI failed: {result.stderr[:200]}")
        return result.stdout.strip()
