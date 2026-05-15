import pytest
from unittest.mock import patch, Mock
from src.haiku import HaikuGenerator, CLUE_POSITIONS


def test_haiku_generator_returns_text():
    # Mock Anthropic SDK response
    mock_block = Mock()
    mock_block.text = "João Silva tem CPF 123.456.789-09"
    mock_message = Mock()
    mock_message.content = [mock_block]
    with patch("src.haiku.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = mock_message
        gen = HaikuGenerator()
        result = gen.generate("email", {
            "nome": "João Silva", "cpf_valor": "123.456.789-09",
            "email": "joao@email.com", "cidade": "São Paulo", "estado": "SP"
        })
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
