import random
import string
import uuid
from src.fodevs import Fodevs4

ESTADOS_BR = ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "GO", "DF", "AM",
               "CE", "PE", "PA", "MT", "MS", "ES", "PB", "RN", "AL", "PI",
               "TO", "SE", "RO", "AC", "AP", "RR", "MA"]


def _gen_order_id() -> str:
    """Realistic Brazilian marketplace order IDs."""
    kind = random.choice(["ml", "magalu", "shopee", "numeric", "alpha"])
    if kind == "ml":  # Mercado Livre
        return f"ML-{random.randint(2020, 2026)}-{random.randint(10000, 99999999):08d}"
    if kind == "magalu":
        return f"MGL{random.randint(100000000000, 999999999999)}"
    if kind == "shopee":
        return f"{random.randint(2400000000000000, 2499999999999999)}"
    if kind == "numeric":
        return f"{random.randint(1000000000, 9999999999):010d}"
    return "PED-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=10))


def _gen_tracking_code() -> str:
    """Correios tracking format: 2 letters + 9 digits + 'BR'."""
    prefix = "".join(random.choices(string.ascii_uppercase, k=2))
    digits = "".join(random.choices(string.digits, k=9))
    return f"{prefix}{digits}BR"


def _gen_invoice_number() -> str:
    """Invoice/NF numbers in various formats."""
    kind = random.choice(["nfe", "fatura", "inv"])
    if kind == "nfe":
        return f"NF-{random.randint(2020, 2026)}-{random.randint(1, 9999):06d}"
    if kind == "fatura":
        return f"FAT-{random.randint(2020, 2026)}-{random.randint(1, 999999):06d}"
    return f"INV-{random.randint(100000, 999999)}"


def _gen_revenue() -> str:
    """Revenue value in BR format: R$ 1.234.567,89"""
    cents = random.randint(0, 99)
    units = random.choice([
        random.randint(100, 999),
        random.randint(1000, 99999),
        random.randint(100000, 9999999),
    ])
    # BR format with thousand separators
    int_str = f"{units:,}".replace(",", ".")
    return f"R$ {int_str},{cents:02d}"


def _gen_transaction_id() -> str:
    """PIX/payment transaction IDs."""
    kind = random.choice(["pix", "card", "boleto"])
    if kind == "pix":
        return f"E{random.randint(10000000, 99999999)}{random.randint(1000000000, 9999999999):010d}"
    if kind == "card":
        return str(uuid.uuid4()).replace("-", "")[:24].upper()
    return f"BOL-{random.randint(100000000, 999999999)}"


def _gen_customer_id() -> str:
    """Internal customer/client IDs."""
    kind = random.choice(["numeric", "alpha"])
    if kind == "numeric":
        return f"{random.randint(100000, 9999999)}"
    return "CUST-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))


def gerar_perfil_completo(estado: str | None = None) -> dict:
    """Generates a complete synthetic BR person profile with all PII types,
    including e-commerce/B2B identifiers."""
    fodevs = Fodevs4()
    estado = estado or random.choice(ESTADOS_BR)

    pessoa = fodevs.gerar_pessoa(estado=estado)

    perfil = {
        "nome": pessoa["nome"],
        "cpf": pessoa["cpf"],
        "rg": pessoa["rg"],
        "email": pessoa["email"],
        "celular": pessoa["celular"],
        "telefone_fixo": pessoa.get("telefone_fixo", ""),
        "cep": pessoa["cep"],
        "endereco": f"{pessoa['endereco']}, {pessoa['numero']}, {pessoa['bairro']}",
        "cidade": pessoa["cidade"],
        "estado": estado,
        "data_nasc": pessoa.get("data_nasc", ""),
        "pis": pessoa.get("pis", fodevs.gerar_pis()),
        "cnh": pessoa.get("cnh", fodevs.gerar_cnh()),
        "ie": fodevs.gerar_ie(estado=estado),
        "titulo_eleitor": fodevs.gerar_titulo_eleitor(estado=estado),
        "certidao_nascimento": fodevs.gerar_certidao(tipo="nascimento"),
        "cnpj": fodevs.gerar_cnpj(),
        # E-commerce / B2B (locally generated, no API call)
        "order_id": _gen_order_id(),
        "tracking_code": _gen_tracking_code(),
        "invoice_number": _gen_invoice_number(),
        "revenue": _gen_revenue(),
        "transaction_id": _gen_transaction_id(),
        "customer_id": _gen_customer_id(),
    }
    return perfil
