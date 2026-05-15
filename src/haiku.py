import os
from anthropic import Anthropic
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
        # Uses official Anthropic SDK. Reads ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN).
        # ANTHROPIC_BASE_URL still works for proxies.
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN", "")
        base_url = os.getenv("ANTHROPIC_BASE_URL")
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = Anthropic(**client_kwargs)
        self.model = os.getenv("HAIKU_MODEL", "claude-haiku-4-5")
        self.env = Environment(loader=FileSystemLoader(_TEMPLATES_DIR))

    def generate(self, template_name: str, context: dict) -> str:
        tpl = self.env.get_template(f"{template_name}.jinja2")
        rendered = tpl.render(**context)
        prompt = (
            f"Reescreva o texto abaixo em português BR natural e formal, "
            f"mantendo TODOS os valores exatamente como estão (CPF, CNPJ, nomes, etc). "
            f"NÃO altere nenhum número ou dado pessoal:\n\n{rendered}"
        )
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
