import re
import random


def variantes_cpf(cpf: str) -> list[tuple[str, str]]:
    raw = re.sub(r"[.\-]", "", cpf)
    parcial = cpf[:8] + "***-**"
    espacos = cpf.replace(".", " ").replace("-", " ")
    return [(cpf, "PRIVATE_CPF"), (raw, "PRIVATE_CPF"),
            (parcial, "PRIVATE_CPF"), (espacos, "PRIVATE_CPF")]


def variantes_cnpj(cnpj: str) -> list[tuple[str, str]]:
    raw = re.sub(r"[.\-/]", "", cnpj)
    parcial = cnpj[:11] + "****-**"
    espacos = re.sub(r"[.\-/]", " ", cnpj)
    return [(cnpj, "PRIVATE_CNPJ"), (raw, "PRIVATE_CNPJ"),
            (parcial, "PRIVATE_CNPJ"), (espacos, "PRIVATE_CNPJ")]


def variantes_rg(rg: str) -> list[tuple[str, str]]:
    raw = re.sub(r"[.\-]", "", rg)
    parcial = rg[:5] + "***-*"
    espacos = rg.replace(".", " ").replace("-", " ")
    return [(rg, "PRIVATE_RG"), (raw, "PRIVATE_RG"),
            (parcial, "PRIVATE_RG"), (espacos, "PRIVATE_RG")]


def variantes_cep(cep: str) -> list[tuple[str, str]]:
    raw = cep.replace("-", "")
    espacos = cep.replace("-", " ")
    return [(cep, "PRIVATE_ADDRESS"), (raw, "PRIVATE_ADDRESS"),
            (espacos, "PRIVATE_ADDRESS")]


def variantes_telefone(tel: str) -> list[tuple[str, str]]:
    raw = re.sub(r"[\s()\-]", "", tel)
    sem_ddd = re.sub(r"^\(\d{2}\)\s*", "", tel)
    espacos = re.sub(r"[\-()\s]+", " ", tel).strip()
    return [(tel, "PRIVATE_PHONE"), (raw, "PRIVATE_PHONE"),
            (sem_ddd, "PRIVATE_PHONE"), (espacos, "PRIVATE_PHONE")]


def variantes_pis(pis: str) -> list[tuple[str, str]]:
    raw = re.sub(r"[.\-]", "", pis)
    return [(pis, "PRIVATE_PIS"), (raw, "PRIVATE_PIS")]


def variantes_cnh(cnh: str) -> list[tuple[str, str]]:
    return [(cnh, "PRIVATE_CNH"), (cnh[:5] + "******", "PRIVATE_CNH")]


def variantes_titulo(titulo: str) -> list[tuple[str, str]]:
    return [(titulo, "PRIVATE_TITULO_ELEITOR"),
            (titulo[:4] + "****" + titulo[-4:], "PRIVATE_TITULO_ELEITOR")]


def variantes_certidao(cert: str) -> list[tuple[str, str]]:
    return [(cert, "PRIVATE_CERTIDAO"),
            (re.sub(r"\s", "", cert), "PRIVATE_CERTIDAO")]


def variantes_ie(ie: str) -> list[tuple[str, str]]:
    raw = re.sub(r"[.\-/]", "", ie)
    return [(ie, "PRIVATE_IE"), (raw, "PRIVATE_IE")]


# ============= e-commerce / B2B categories =============

def variantes_order_id(order_id: str) -> list[tuple[str, str]]:
    """Order ID formats: ML-2026-00098765, 2000001234567890, PEDIDO 12345"""
    return [(order_id, "PRIVATE_ORDER_ID")]


def variantes_tracking(tracking: str) -> list[tuple[str, str]]:
    """Correios tracking: BR123456789BR (13 chars). Also without 'BR' prefix/suffix."""
    return [(tracking, "PRIVATE_TRACKING_CODE")]


def variantes_invoice(invoice: str) -> list[tuple[str, str]]:
    """Invoice/NF number: 000456789, NF-2026-0123, INV-2026-456"""
    return [(invoice, "PRIVATE_INVOICE_NUMBER")]


def variantes_revenue(revenue: str) -> list[tuple[str, str]]:
    """Revenue/financial value formats:
    R$ 1.234.567,89  | R$1234567,89 | 1.234.567,89 | 1234567.89
    """
    no_currency = revenue.replace("R$ ", "").replace("R$", "").strip()
    raw_us = no_currency.replace(".", "").replace(",", ".")
    return [
        (revenue, "PRIVATE_CLIENT_REVENUE"),
        (no_currency, "PRIVATE_CLIENT_REVENUE"),
        (raw_us, "PRIVATE_CLIENT_REVENUE"),
    ]


def variantes_transaction(tx: str) -> list[tuple[str, str]]:
    """Transaction IDs: PIX E12345...XX, card tx UUIDs, etc."""
    return [(tx, "PRIVATE_TRANSACTION_ID")]


def variantes_customer_id(cid: str) -> list[tuple[str, str]]:
    """Internal customer ID — usually numeric or alphanumeric."""
    return [(cid, "PRIVATE_CUSTOMER_ID")]


def pick_variant(variants: list[tuple[str, str]]) -> tuple[str, str]:
    return random.choice(variants)


def get_variants_for_perfil(perfil: dict) -> dict[str, list[tuple[str, str]]]:
    """Returns all format variants for every PII field in a perfil."""
    out = {
        "cpf": variantes_cpf(perfil["cpf"]),
        "cnpj": variantes_cnpj(perfil["cnpj"]),
        "rg": variantes_rg(perfil["rg"]),
        "cep": variantes_cep(perfil["cep"]),
        "celular": variantes_telefone(perfil["celular"]),
        "telefone_fixo": variantes_telefone(perfil["telefone_fixo"]) if perfil.get("telefone_fixo") else [],
        "pis": variantes_pis(perfil["pis"]),
        "cnh": variantes_cnh(perfil["cnh"]),
        "titulo_eleitor": variantes_titulo(perfil["titulo_eleitor"]),
        "certidao_nascimento": variantes_certidao(perfil["certidao_nascimento"]),
        "ie": variantes_ie(perfil["ie"]),
    }
    # E-commerce / B2B fields (optional — only if present in perfil)
    if perfil.get("order_id"):
        out["order_id"] = variantes_order_id(perfil["order_id"])
    if perfil.get("tracking_code"):
        out["tracking_code"] = variantes_tracking(perfil["tracking_code"])
    if perfil.get("invoice_number"):
        out["invoice_number"] = variantes_invoice(perfil["invoice_number"])
    if perfil.get("revenue"):
        out["revenue"] = variantes_revenue(perfil["revenue"])
    if perfil.get("transaction_id"):
        out["transaction_id"] = variantes_transaction(perfil["transaction_id"])
    if perfil.get("customer_id"):
        out["customer_id"] = variantes_customer_id(perfil["customer_id"])
    return out
