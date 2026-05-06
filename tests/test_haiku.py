import pytest
from unittest.mock import patch, Mock
from src.haiku import HaikuGenerator, CLUE_POSITIONS


def test_haiku_generator_returns_text():
    sse_lines = [
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"João Silva"}}',
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":" tem CPF 123.456.789-09"}}',
        'data: {"type":"message_stop"}',
    ]
    mock_resp = Mock()
    mock_resp.iter_lines.return_value = iter(sse_lines)
    mock_resp.raise_for_status = Mock()
    with patch("src.haiku.requests.post", return_value=mock_resp):
        gen = HaikuGenerator()
        result = gen.generate("email", {"nome": "João Silva", "cpf_valor": "123.456.789-09",
                                         "email": "joao@email.com", "cidade": "São Paulo", "estado": "SP"})
    assert "João Silva" in result
    assert "123.456.789-09" in result


def test_clue_positions_has_three_options():
    assert len(CLUE_POSITIONS["cpf"]) >= 3


def test_clue_before_format():
    positions = CLUE_POSITIONS["cpf"]
    clue_before = [p for p in positions if p["position"] == "before"]
    assert len(clue_before) >= 1
    assert "{value}" in clue_before[0]["template"]


def test_clue_after_format():
    positions = CLUE_POSITIONS["cpf"]
    clue_after = [p for p in positions if p["position"] == "after"]
    assert len(clue_after) >= 1


def test_clue_omitted_format():
    positions = CLUE_POSITIONS["cpf"]
    omitted = [p for p in positions if p["position"] == "omitted"]
    assert len(omitted) >= 1
