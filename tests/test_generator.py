import pytest
from unittest.mock import patch, Mock, MagicMock
from src.generator import generate_example, GeneratorStats

MOCK_PERFIL = {
    "nome": "Maria Silva", "cpf": "123.456.789-09",
    "cnpj": "72.682.864/0001-41", "rg": "12.345.678-9",
    "cnh": "12345678901", "pis": "123.45678.90-1",
    "titulo_eleitor": "228214550167",
    "certidao_nascimento": "256757 01 55 2022 1 08962 159 2623902-70",
    "ie": "818.800.070.960", "email": "maria@email.com",
    "celular": "(11) 99999-9999", "telefone_fixo": "(11) 3333-3333",
    "cep": "01310-100",
    "endereco": "Av Paulista, 1000, Bela Vista",
    "cidade": "São Paulo", "estado": "SP", "data_nasc": "01/01/1990",
    # e-commerce / B2B identifiers
    "order_id": "ML-2026-12345678",
    "tracking_code": "BR123456789BR",
    "invoice_number": "NF-2026-000456",
    "revenue": "R$ 1.234.567,89",
    "transaction_id": "E1234567820261015143218",
    "customer_id": "CUST-ABC12345",
}


def test_generate_example_returns_valid_example():
    with patch("src.generator.gerar_perfil_completo", return_value=MOCK_PERFIL), \
         patch("src.generator.HaikuGenerator") as MockHaiku:
        MockHaiku.return_value.generate.return_value = (
            "Contrato com Maria Silva, CPF 123.456.789-09, "
            "CNPJ 72.682.864/0001-41, residente à Av Paulista, 1000, Bela Vista, "
            "CEP 01310-100, São Paulo/SP. E-mail: maria@email.com. "
            "Celular: (11) 99999-9999. RG: 12.345.678-9. "
            "PIS: 123.45678.90-1. Data nasc: 01/01/1990."
        )
        result = generate_example()

    assert result is not None
    assert "text" in result
    assert "entities" in result
    assert len(result["entities"]) >= 3


def test_generate_example_returns_none_on_short_text():
    with patch("src.generator.gerar_perfil_completo", return_value=MOCK_PERFIL), \
         patch("src.generator.HaikuGenerator") as MockHaiku:
        MockHaiku.return_value.generate.return_value = "curto"
        result = generate_example()
    assert result is None


def test_generator_stats_tracks_discards():
    stats = GeneratorStats()
    stats.record(None, "text_too_short")
    stats.record(None, "text_too_short")
    stats.record({"text": "ok", "entities": []}, None)
    assert stats.total == 3
    assert stats.discarded == 2
    assert stats.valid == 1
    assert stats.discard_reasons["text_too_short"] == 2
