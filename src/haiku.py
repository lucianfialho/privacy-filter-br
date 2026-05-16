import os
import subprocess
import requests
from jinja2 import Environment, FileSystemLoader

# Auto-load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

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

PROMPT_PREFIX = (
    "Reescreva o texto abaixo em português BR natural e formal, "
    "mantendo TODOS os valores exatamente como estão (CPF, CNPJ, nomes, etc). "
    "NÃO altere nenhum número ou dado pessoal:\n\n"
)


class HaikuGenerator:
    """Provider-aware generator. Two backends:

    - provider='claude' (default): subprocess `claude --print`, uses subscription
    - provider='minimax': HTTP API to MiniMax chat completions, uses MINIMAX_API_KEY

    Provider can also be set via env: PROVIDER=claude|minimax
    """

    def __init__(self, provider: str | None = None, **kwargs):
        self.provider = (provider or os.getenv("PROVIDER", "claude")).lower()
        self.env = Environment(loader=FileSystemLoader(_TEMPLATES_DIR))
        if self.provider == "minimax":
            self.minimax_key = os.getenv("MINIMAX_API_KEY", "")
            self.minimax_url = os.getenv(
                "MINIMAX_URL",
                "https://api.minimax.io/v1/text/chatcompletion_v2",
            )
            self.minimax_model = os.getenv("MINIMAX_MODEL", "MiniMax-Text-01")

    def generate(self, template_name: str, context: dict) -> str:
        tpl = self.env.get_template(f"{template_name}.jinja2")
        rendered = tpl.render(**context)
        prompt = PROMPT_PREFIX + rendered

        if self.provider == "minimax":
            return self._generate_minimax(prompt)
        return self._generate_claude(prompt)

    def _generate_claude(self, prompt: str) -> str:
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

    def _generate_minimax(self, prompt: str) -> str:
        resp = requests.post(
            self.minimax_url,
            headers={
                "Authorization": f"Bearer {self.minimax_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.minimax_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
                "temperature": 0.7,
            },
            timeout=60,
        )
        data = resp.json()
        # Common shapes: OpenAI-style {"choices":[{"message":{"content":...}}]}
        # MiniMax v2 also returns this shape, but errors return base_resp/error fields
        if "choices" in data:
            return data["choices"][0]["message"]["content"].strip()
        # Older MiniMax shape
        if "reply" in data:
            return str(data["reply"]).strip()
        # Surface error details
        raise RuntimeError(f"MiniMax response (HTTP {resp.status_code}): {data}")
