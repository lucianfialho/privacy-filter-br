"""
Generators for OAI Privacy Filter categories that are NOT in our BR pipeline:
- private_date (BR formats)
- private_url (with sensitive data in params)
- secret (API keys, tokens, passwords)
- account_number (bank accounts, card numbers w/ Luhn)

These are used to generate ~2k extras so the v3 hybrid model doesn't "forget"
these original capabilities during fine-tuning.
"""
import random
import string
from datetime import date, timedelta


# -------------------------- DATES (BR) --------------------------
def _rand_date(start_year: int = 1960, end_year: int = 2026) -> date:
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    return start + timedelta(days=random.randint(0, (end - start).days))


MESES_PT = [
    "janeiro", "fevereiro", "março", "abril", "maio", "junho",
    "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
]


def gen_date_variants() -> list[tuple[str, str]]:
    d = _rand_date()
    return [
        (d.strftime("%d/%m/%Y"), "private_date"),
        (d.strftime("%d-%m-%Y"), "private_date"),
        (d.strftime("%d.%m.%Y"), "private_date"),
        (f"{d.day} de {MESES_PT[d.month - 1]} de {d.year}", "private_date"),
        (d.strftime("%Y-%m-%d"), "private_date"),
    ]


# -------------------------- URLS --------------------------
DOMAINS = [
    "empresa-x.com.br", "minha-loja.com", "saas-cliente.io",
    "portal-cliente.net", "sistema-interno.com.br",
]


def gen_url_variants() -> list[tuple[str, str]]:
    domain = random.choice(DOMAINS)
    path = random.choice([
        f"/auth/reset?token={_rand_hex(32)}",
        f"/api/v1/users/{random.randint(1000, 99999)}?email={_rand_email()}",
        f"/pedido?id={random.randint(100000, 999999)}&customer={_rand_hex(8)}",
        f"/share/document/{_rand_hex(16)}",
        f"/invoice?nf={random.randint(10000, 99999)}&cliente={_rand_hex(12)}",
    ])
    url = f"https://{domain}{path}"
    return [(url, "private_url")]


# -------------------------- SECRETS / TOKENS --------------------------
def _rand_hex(n: int) -> str:
    return "".join(random.choice("0123456789abcdef") for _ in range(n))


def _rand_b64(n: int) -> str:
    alphabet = string.ascii_letters + string.digits + "+/"
    return "".join(random.choice(alphabet) for _ in range(n))


def _rand_alnum(n: int) -> str:
    return "".join(random.choice(string.ascii_letters + string.digits) for _ in range(n))


def gen_secret_variants() -> list[tuple[str, str]]:
    kind = random.choice([
        "openai", "anthropic", "github", "aws", "bearer", "generic", "password",
    ])
    if kind == "openai":
        secret = f"sk-proj-{_rand_alnum(48)}"
    elif kind == "anthropic":
        secret = f"sk-ant-api03-{_rand_alnum(95)}"
    elif kind == "github":
        secret = f"ghp_{_rand_alnum(36)}"
    elif kind == "aws":
        secret = f"AKIA{_rand_alnum(16).upper()}"
    elif kind == "bearer":
        secret = f"Bearer {_rand_b64(32)}.{_rand_b64(48)}"
    elif kind == "password":
        secret = "".join(random.choice(string.ascii_letters + string.digits + "!@#$%") for _ in range(16))
    else:
        secret = _rand_alnum(40)
    return [(secret, "secret")]


# -------------------------- ACCOUNT NUMBERS --------------------------
def _luhn_check_digit(num: str) -> int:
    """Compute Luhn check digit for a numeric string."""
    total = 0
    for i, d in enumerate(reversed(num)):
        n = int(d)
        if i % 2 == 0:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return (10 - (total % 10)) % 10


def gen_account_variants() -> list[tuple[str, str]]:
    kind = random.choice(["bank", "card_visa", "card_master"])
    if kind == "bank":
        # Conta-corrente: agência + conta
        ag = f"{random.randint(0, 9999):04d}"
        cc = f"{random.randint(0, 999999):06d}-{random.randint(0, 9)}"
        return [(f"Ag. {ag} / Conta {cc}", "account_number")]
    elif kind == "card_visa":
        # Visa: starts with 4, 16 digits, Luhn-valid
        base = "4" + "".join(random.choice("0123456789") for _ in range(14))
        full = base + str(_luhn_check_digit(base))
        formatted = " ".join(full[i:i + 4] for i in range(0, 16, 4))
        return [(formatted, "account_number"), (full, "account_number")]
    else:
        # Mastercard: 51-55 prefix, 16 digits, Luhn-valid
        base = str(random.randint(51, 55)) + "".join(random.choice("0123456789") for _ in range(13))
        full = base + str(_luhn_check_digit(base))
        formatted = " ".join(full[i:i + 4] for i in range(0, 16, 4))
        return [(formatted, "account_number"), (full, "account_number")]


# -------------------------- helpers --------------------------
def _rand_email() -> str:
    user = "".join(random.choice(string.ascii_lowercase) for _ in range(random.randint(5, 12)))
    return f"{user}@{random.choice(DOMAINS)}"


# -------------------------- profile builder --------------------------
def build_extras_perfil() -> dict:
    """Builds a synthetic 'profile' with extras-only PII for template rendering."""
    dates = gen_date_variants()
    urls = gen_url_variants()
    secrets = gen_secret_variants()
    accounts = gen_account_variants()

    # Pick one variant per category to render
    return {
        "extras_date": dates[0][0],
        "extras_url": urls[0][0],
        "extras_secret": secrets[0][0],
        "extras_account": accounts[0][0],
        # Keep all variants for labeler (so it finds any reformatting)
        "_inserted_extras": {
            **{v: lbl for v, lbl in dates},
            **{v: lbl for v, lbl in urls},
            **{v: lbl for v, lbl in secrets},
            **{v: lbl for v, lbl in accounts},
        },
    }
