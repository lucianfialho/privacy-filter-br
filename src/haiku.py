import os
import json
import requests
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
    def __init__(self, **kwargs):
        # Reads ANTHROPIC_BASE_URL (e.g. http://lucian-desktop:8082) for local GLM proxy.
        # Falls back to https://api.anthropic.com.
        self.base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
        self.auth_token = os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = os.getenv("HAIKU_MODEL", "claude-haiku-4-5-20251001")
        self.env = Environment(loader=FileSystemLoader(_TEMPLATES_DIR))

    def generate(self, template_name: str, context: dict) -> str:
        tpl = self.env.get_template(f"{template_name}.jinja2")
        rendered = tpl.render(**context)
        prompt = (
            f"Reescreva o texto abaixo em português BR natural e formal, "
            f"mantendo TODOS os valores exatamente como estão (CPF, CNPJ, nomes, etc). "
            f"NÃO altere nenhum número ou dado pessoal:\n\n{rendered}"
        )
        # Raw streaming HTTP — robust against proxies that emit thinking blocks
        resp = requests.post(
            f"{self.base_url}/v1/messages",
            headers={
                "x-api-key": self.auth_token,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 4000,  # GLM consumes lots of tokens in thinking before text
                "stream": True,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=180,  # GLM thinking can take ~1-2min
            stream=True,
        )
        resp.raise_for_status()

        chunks = []
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if event.get("type") == "content_block_delta":
                delta = event.get("delta", {})
                # Only collect text deltas — skip thinking/tool deltas
                if delta.get("type") == "text_delta":
                    chunks.append(delta.get("text", ""))
        return "".join(chunks)
