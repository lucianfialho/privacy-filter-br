import os
import re
import subprocess
import threading
import time
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
        if self.provider == "claude":
            self.claude_model = os.getenv("CLAUDE_MODEL", "haiku")
        if self.provider == "minimax":
            self.minimax_key = os.getenv("MINIMAX_API_KEY", "")
            self.minimax_url = os.getenv(
                "MINIMAX_URL",
                "https://api.minimax.io/v1/text/chatcompletion_v2",
            )
            # M2.7 default (reasoning model — emits <think> blocks before answer)
            self.minimax_model = os.getenv("MINIMAX_MODEL", "MiniMax-M2.7")

    def generate(self, template_name: str, context: dict) -> str:
        tpl = self.env.get_template(f"{template_name}.jinja2")
        rendered = tpl.render(**context)
        prompt = PROMPT_PREFIX + rendered

        if self.provider == "minimax":
            return self._generate_minimax(prompt)
        return self._generate_claude(prompt)

    def _generate_claude(self, prompt: str) -> str:
        # Strip ANTHROPIC_API_KEY from subprocess env — claude CLI prefers subscription
        # when no API key is set. The .env file is loaded for MINIMAX_API_KEY but
        # ANTHROPIC_API_KEY might be a placeholder ("your_key_here") that breaks claude.
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        result = subprocess.run(
            ["claude", "--print", "--model", self.claude_model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "(no output)")[:200]
            raise RuntimeError(f"claude CLI failed: {err}")
        return result.stdout.strip()

    # Global rate limiter for MiniMax (Starter plan: 1500 req / 5h ≈ 300 req/h).
    # We aim for 270/h (margin) → 1 request every ~13.3s across ALL workers in this process.
    _minimax_lock = threading.Lock()
    _minimax_last = 0.0
    _minimax_min_interval = float(os.getenv("MINIMAX_MIN_INTERVAL", "13.3"))

    @classmethod
    def _minimax_throttle(cls) -> None:
        with cls._minimax_lock:
            now = time.monotonic()
            wait = cls._minimax_min_interval - (now - cls._minimax_last)
            if wait > 0:
                time.sleep(wait)
            cls._minimax_last = time.monotonic()

    def _generate_minimax(self, prompt: str) -> str:
        self._minimax_throttle()
        resp = requests.post(
            self.minimax_url,
            headers={
                "Authorization": f"Bearer {self.minimax_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.minimax_model,
                "messages": [{"role": "user", "content": prompt}],
                # M2.7 is a reasoning model — emits <think> blocks before final text
                # Give enough budget for both
                "max_tokens": 4096,
                "temperature": 0.7,
            },
            timeout=120,
        )
        data = resp.json()

        # Check for error envelope (MiniMax error responses)
        base_resp = data.get("base_resp", {})
        if base_resp and base_resp.get("status_code") not in (0, None):
            raise RuntimeError(
                f"MiniMax error {base_resp.get('status_code')}: {base_resp.get('status_msg')}"
            )

        choices = data.get("choices")
        if not choices:
            raise RuntimeError(f"MiniMax response missing 'choices'. Full: {data}")

        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if not content:
            # Sometimes content is in finish_reason or top-level reply
            finish = choices[0].get("finish_reason")
            raise RuntimeError(
                f"MiniMax content empty (finish_reason={finish}). Full: {data}"
            )

        text = content if isinstance(content, str) else str(content)
        # Strip reasoning model think blocks
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.strip()
