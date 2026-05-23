"""Smoke tests for br-pii-guardrail."""
import secrets

from br_pii_guardrail import Guardrail, Tokenizer, derive_tenant_key
from br_pii_guardrail.recognizers import (
    CpfRecognizer, CnpjRecognizer, EmailRecognizer,
    PhoneRecognizer, CardRecognizer, UrlRecognizer,
)
from br_pii_guardrail.scanners import scan_csv_text


# ---------- recognizers ----------
def test_cpf_checksum_valid():
    text = "CPF 680.075.670-97 do cliente"
    ms = CpfRecognizer().find(text)
    assert len(ms) == 1
    assert ms[0].label == "private_cpf"
    assert ms[0].source == "checksum"


def test_cpf_invalid_checksum_rejected():
    text = "CPF 111.111.111-11 (invalido)"
    assert CpfRecognizer().find(text) == []


def test_cnpj_checksum_valid():
    text = "CNPJ 11.222.333/0001-81"
    ms = CnpjRecognizer().find(text)
    assert len(ms) == 1
    assert ms[0].label == "private_cnpj"


def test_email():
    ms = EmailRecognizer().find("contato joao@example.com")
    assert len(ms) == 1
    assert ms[0].value == "joao@example.com"


def test_phone_br():
    cases = ["(11) 98765-4321", "11 98765-4321", "+55 11 98765-4321"]
    for text in cases:
        ms = PhoneRecognizer().find(text)
        assert len(ms) >= 1, f"failed on {text!r}"


def test_card_luhn():
    ms = CardRecognizer().find("Cartão 4111 1111 1111 1111")
    assert len(ms) == 1
    assert ms[0].label == "account_number"


def test_url():
    ms = UrlRecognizer().find("acesse https://exemplo.com/reset?token=abc")
    assert len(ms) == 1


# ---------- Guardrail orchestration ----------
def test_guardrail_text_mixed():
    g = Guardrail.default()
    text = ("Cliente João Silva, CPF 680.075.670-97, "
            "CNPJ 11.222.333/0001-81, email j@x.com, "
            "tel (11) 98765-4321")
    ms = g.scan_text(text)
    labels = {m.label for m in ms}
    assert "private_cpf" in labels
    assert "private_cnpj" in labels
    assert "private_email" in labels
    assert "private_phone" in labels


def test_guardrail_dedupe_priority():
    """Cartão 16 dígitos não pode virar phone também."""
    g = Guardrail.default()
    ms = g.scan_text("Cartao 4111 1111 1111 1111 do cliente")
    card_ms = [m for m in ms if m.label == "account_number"]
    assert len(card_ms) == 1


# ---------- JSON scanner ----------
def test_scan_json_schema_key():
    g = Guardrail.default()
    data = {"customer": {"cpf": "680.075.670-97", "name": "João"},
            "order_id": "ML-2024-789456"}
    ms = g.scan_json(data)
    labels = {m.label for m in ms}
    assert "private_cpf" in labels


# ---------- CSV scanner ----------
def test_scan_csv():
    g = Guardrail.default()
    csv = "nome,cpf,email\nJoão,680.075.670-97,j@x.com\n"
    result = scan_csv_text(csv, g.recognizers)
    assert result["cpf"]
    assert any(m.label == "private_cpf" for m in result["cpf"])


# ---------- Tokenizer ----------
def test_tokenizer_roundtrip():
    master = secrets.token_bytes(32)
    t = Tokenizer(derive_tenant_key(master, "acme"))
    g = Guardrail.default()
    text = "Cliente CPF 680.075.670-97 confirmado"
    ms = g.scan_text(text)
    masked, vault = t.tokenize(text, ms)
    assert "680.075.670-97" not in masked
    restored = t.detokenize(masked, vault)
    assert restored == text


def test_tokenizer_deterministic():
    master = secrets.token_bytes(32)
    t = Tokenizer(derive_tenant_key(master, "acme"))
    tok1 = t.token_for("680.075.670-97", "private_cpf")
    tok2 = t.token_for("680.075.670-97", "private_cpf")
    assert tok1 == tok2


def test_tokenizer_cross_tenant_isolation():
    master = secrets.token_bytes(32)
    acme = Tokenizer(derive_tenant_key(master, "acme"))
    globex = Tokenizer(derive_tenant_key(master, "globex"))
    a = acme.token_for("680.075.670-97", "private_cpf")
    g = globex.token_for("680.075.670-97", "private_cpf")
    assert a != g
