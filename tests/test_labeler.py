from src.labeler import find_spans, to_entity_format, label_text


# -------------------- backwards-compatibility tests --------------------


def test_find_spans_single_match():
    text = "CPF 680.075.670-97 do cliente"
    spans = find_spans(text, {"680.075.670-97": "PRIVATE_CPF"})
    assert spans == [{"start": 4, "end": 18, "label": "PRIVATE_CPF"}]


def test_find_spans_multiple_matches():
    text = "João Silva tem CPF 123.456.789-09 e CNPJ 72.682.864/0001-41"
    spans = find_spans(text, {
        "João Silva": "PRIVATE_PERSON",
        "123.456.789-09": "PRIVATE_CPF",
        "72.682.864/0001-41": "PRIVATE_CNPJ"
    })
    assert len(spans) == 3
    labels = {s["label"] for s in spans}
    assert labels == {"PRIVATE_PERSON", "PRIVATE_CPF", "PRIVATE_CNPJ"}


def test_find_spans_missing_value_returns_empty():
    text = "Texto sem PII"
    spans = find_spans(text, {"680.075.670-97": "PRIVATE_CPF"})
    assert spans == []


def test_find_spans_overlapping_uses_longest():
    text = "João Silva mora aqui"
    spans = find_spans(text, {
        "João Silva": "PRIVATE_PERSON",
        "João": "PRIVATE_PERSON"
    })
    assert len(spans) == 1
    assert spans[0]["end"] - spans[0]["start"] == len("João Silva")


def test_to_entity_format_output_structure():
    text = "CPF 680.075.670-97"
    spans = [{"start": 4, "end": 18, "label": "PRIVATE_CPF"}]
    result = to_entity_format(text, spans)
    assert result["text"] == text
    assert result["entities"] == spans


def test_label_text_full_pipeline():
    text = "Contrato com Maria Silva, CPF 123.456.789-09"
    inserted = {"Maria Silva": "PRIVATE_PERSON", "123.456.789-09": "PRIVATE_CPF"}
    result = label_text(text, inserted)
    assert result["text"] == text
    assert len(result["entities"]) == 2


# -------------------- format-aware tests (the bug we fixed) --------------------


def test_format_aware_cpf_separator_variant():
    """The actual bug from dataset_br_v3.jsonl line 0: dot instead of dash."""
    text = "CPF: 320.575.016.04 do cliente"
    inserted = {"320.575.016-04": "PRIVATE_CPF"}
    spans = find_spans(text, inserted)
    assert len(spans) == 1
    assert spans[0]["label"] == "PRIVATE_CPF"
    assert text[spans[0]["start"]:spans[0]["end"]] == "320.575.016.04"


def test_format_aware_cpf_with_spaces():
    text = "CPF: 320 575 016 04"
    inserted = {"320.575.016-04": "PRIVATE_CPF"}
    spans = find_spans(text, inserted)
    assert len(spans) == 1
    assert text[spans[0]["start"]:spans[0]["end"]] == "320 575 016 04"


def test_format_aware_cpf_no_separators():
    text = "CPF: 32057501604 do cliente"
    inserted = {"320.575.016-04": "PRIVATE_CPF"}
    spans = find_spans(text, inserted)
    assert len(spans) == 1
    assert text[spans[0]["start"]:spans[0]["end"]] == "32057501604"


def test_format_aware_cnpj_separator_variant():
    text = "CNPJ 12.345.678/0001.90 da empresa"
    inserted = {"12.345.678/0001-90": "PRIVATE_CNPJ"}
    spans = find_spans(text, inserted)
    assert len(spans) == 1
    assert text[spans[0]["start"]:spans[0]["end"]] == "12.345.678/0001.90"


def test_format_aware_phone_with_dashes():
    text = "Telefone (11)-98765-4321 para contato"
    inserted = {"(11) 98765-4321": "PRIVATE_PHONE"}
    spans = find_spans(text, inserted)
    assert len(spans) == 1


def test_format_aware_pis_variant():
    text = "PIS: 020 74375 29 2"
    inserted = {"020.74375.29-2": "PRIVATE_PIS"}
    spans = find_spans(text, inserted)
    assert len(spans) == 1


def test_format_aware_word_boundary_respected():
    """A long ID like 'ML-2026-32057501604' should NOT be matched as the embedded CPF."""
    text = "Pedido ML-2026-32057501604XYZ"
    inserted = {"320.575.016-04": "PRIVATE_CPF"}
    spans = find_spans(text, inserted)
    # The embedded sequence is surrounded by alnum on both sides → boundary fails
    assert spans == []


def test_format_aware_too_short_falls_back_to_exact():
    """Skeletons under 4 chars use exact match (no format variants)."""
    text = "Código 123"
    inserted = {"123": "PRIVATE_PHONE"}
    spans = find_spans(text, inserted)
    # 3-char skeleton, falls back to exact match
    assert len(spans) == 1


def test_free_text_label_still_exact():
    """PRIVATE_PERSON / EMAIL / ADDRESS keep exact matching."""
    text = "Maria  Silva mora aqui"  # double space — should NOT match "Maria Silva"
    inserted = {"Maria Silva": "PRIVATE_PERSON"}
    spans = find_spans(text, inserted)
    assert spans == []


def test_variants_in_inserted_deduplicated():
    """Multiple variants of same skeleton+label shouldn't produce duplicate spans."""
    text = "CPF 123.456.789-09 do cliente"
    inserted = {
        "123.456.789-09": "PRIVATE_CPF",
        "12345678909": "PRIVATE_CPF",
        "123.456.***-**": "PRIVATE_CPF",
        "123 456 789 09": "PRIVATE_CPF",
    }
    spans = find_spans(text, inserted)
    # Should find exactly one span (the original CPF), not 4
    assert len(spans) == 1
    assert spans[0]["label"] == "PRIVATE_CPF"
