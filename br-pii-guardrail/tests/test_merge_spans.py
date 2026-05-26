"""Tests for boundary-merger post-process in br_pii_guardrail.ner."""
from br_pii_guardrail.ner import _merge_adjacent_spans


def test_merge_cpf_fragments():
    """Two contiguous fragments of same label → 1 span."""
    text = "CPF 792.498.927-72 da pessoa"
    raw = [
        {"entity_group": "private_cpf", "start": 4, "end": 16, "score": 0.99, "word": "792.498.927-"},
        {"entity_group": "private_cpf", "start": 16, "end": 18, "score": 0.99, "word": "72"},
    ]
    merged = _merge_adjacent_spans(raw, text)
    assert len(merged) == 1
    assert merged[0]["start"] == 4
    assert merged[0]["end"] == 18


def test_merge_with_gap():
    """Fragments separated by punctuation but within max_gap → merge."""
    text = "CNPJ 11.222.333/0001-81 emitente"
    raw = [
        {"entity_group": "private_cnpj", "start": 5, "end": 21, "score": 0.99, "word": "11.222.333/0001-"},
        {"entity_group": "private_cnpj", "start": 21, "end": 23, "score": 0.99, "word": "81"},
    ]
    merged = _merge_adjacent_spans(raw, text)
    assert len(merged) == 1
    assert text[merged[0]["start"]:merged[0]["end"]] == "11.222.333/0001-81"


def test_no_merge_different_labels():
    """Adjacent spans of DIFFERENT labels stay separate."""
    text = "João da Silva CPF 123.456.789-09"
    raw = [
        {"entity_group": "private_person", "start": 0, "end": 13, "score": 0.99, "word": "João da Silva"},
        {"entity_group": "private_cpf", "start": 18, "end": 32, "score": 0.99, "word": "123.456.789-09"},
    ]
    merged = _merge_adjacent_spans(raw, text)
    assert len(merged) == 2


def test_no_merge_far_apart():
    """Same label but separated by > max_gap chars → don't merge."""
    text = "Cliente João, e o outro era CUST-12345 ali"
    raw = [
        {"entity_group": "private_person", "start": 8, "end": 12, "score": 0.99, "word": "João"},
        {"entity_group": "private_person", "start": 28, "end": 38, "score": 0.99, "word": "CUST-12345"},
    ]
    merged = _merge_adjacent_spans(raw, text, max_gap=3)
    assert len(merged) == 2


def test_no_merge_word_gap():
    """Gap contains alphanumeric content → don't merge (different entities)."""
    text = "João da Silva e Maria"
    raw = [
        {"entity_group": "private_person", "start": 0, "end": 13, "score": 0.99, "word": "João da Silva"},
        {"entity_group": "private_person", "start": 16, "end": 21, "score": 0.99, "word": "Maria"},
    ]
    merged = _merge_adjacent_spans(raw, text, max_gap=3)
    # "e" is alphanumeric in gap → don't merge
    assert len(merged) == 2


def test_merge_three_fragments():
    """Three consecutive fragments → 1 span."""
    text = "Nome Maria José Silva Santos"
    raw = [
        {"entity_group": "private_person", "start": 5, "end": 10, "score": 0.99, "word": "Maria"},
        {"entity_group": "private_person", "start": 11, "end": 15, "score": 0.99, "word": "José"},
        {"entity_group": "private_person", "start": 16, "end": 21, "score": 0.99, "word": "Silva"},
    ]
    merged = _merge_adjacent_spans(raw, text)
    assert len(merged) == 1
    assert text[merged[0]["start"]:merged[0]["end"]] == "Maria José Silva"


def test_score_is_min_of_merged():
    """Merged span's score = min of merged fragments (conservative)."""
    text = "CPF 123.456.789-09 ok"
    raw = [
        {"entity_group": "private_cpf", "start": 4, "end": 16, "score": 0.99, "word": "123.456.789-"},
        {"entity_group": "private_cpf", "start": 16, "end": 18, "score": 0.85, "word": "09"},
    ]
    merged = _merge_adjacent_spans(raw, text)
    assert merged[0]["score"] == 0.85


def test_empty_input():
    assert _merge_adjacent_spans([], "any text") == []
