---
type: concept
tags: [pretraining, mlm, masking, ner-prerequisites]
sources: 1
updated: 2026-05-23
---

# Pretraining objectives and their effect on NER

A summary of how different self-supervised pretraining objectives affect the resulting encoder's usefulness for **span-selection** tasks like NER. This page synthesizes findings from SpanBERT and related papers and frames the question we care about: **what backbone gives us the best starting point for BR PII NER?**

## The objectives

### Masked Language Modeling (MLM, BERT original)

Mask 15% of **sub-word tokens** independently. Predict each from full context.

**Pros:** simple, well-studied. Works fine for sentence-level tasks.

**Cons for NER:** sub-word tokens are masked independently → no incentive to encode multi-token entity content in any particular position. A token in the middle of a long PII span can be unambiguously predicted from immediate neighbors, leaving the boundary representations under-trained for span understanding.

### Whole Word Masking (WWM)

Mask all sub-word tokens belonging to the same word together. Used by BERTimbau, RoBERTa-WWM, Chinese WWM-BERT.

**Pros:** prevents trivial sub-word completion. Forces context-based prediction of full words.

**Cons for NER:** still per-word. Multi-word entities (compound names, addresses) are masked one word at a time. Boundaries between adjacent words within the same entity remain locally predictable.

**SpanBERT Table 6 evidence:** WWM beats raw MLM on SQuAD (+0.5 F1) and TriviaQA (+0.8) but **regresses on coref (-1.1)**. WWM is a partial improvement, not a span-aware one.

### Named Entity Masking

Mask spans selected by an off-the-shelf NER tagger. Used by ERNIE 1.0 (Sun et al, 2019).

**Pros:** forces the model to predict entities from context, in theory directly useful for downstream NER.

**Cons:** quality of NER tagger upstream is a ceiling. Also tested as 50%-entity + 50%-random in SpanBERT.

**SpanBERT Table 6:** NE masking is similar to random spans on most tasks; **slightly worse on coref**. The linguistic prior doesn't help meaningfully.

### Geometric Span Masking (SpanBERT)

Sample span length from `Geo(p=0.2)` clipped at 10. Mean ~3.8 words. Mask all sub-words in the span.

**Pros:** wider range of span lengths than entity-based masking; forces longer-range completion.

**Cons:** alone, only marginally better than WWM. Real gains come when combined with SBO ([[span-boundary-objective]]).

### Span Masking + SBO (SpanBERT full)

Adds a span-boundary objective on top of geometric span masking. See [[span-boundary-objective]].

**Pros:** **the strongest variant for span-selection tasks**. Biggest coref gain (+2.7 over span-masking alone).

**Cons:** more expensive to pretrain; doesn't help non-span tasks much.

### Replaced Token Detection (RTD, DeBERTa-v3)

Train a generator + discriminator setup. Discriminator predicts whether each token was replaced by a small generator.

**Pros:** more sample-efficient than MLM. Uses every token (not just 15%) for the discriminator loss.

**Cons:** not explicitly span-aware. DeBERTa-v3's strength on NER (per GLiNER paper) seems to come from disentangled attention + RTD's overall representation quality, not from any span-specific trick.

## Comparison summary

| Objective | Span-aware | NER F1 gain over MLM (proxy) | Used by |
| --- | --- | --- | --- |
| MLM | No | baseline | BERT, BERTimbau |
| WWM | Partial | +0.5 to +1.0 | BERTimbau (yes), RoBERTa-WWM |
| NE masking | Partial | similar to WWM | ERNIE 1.0 |
| Span masking | Yes | +1.0 to +2.0 | SpanBERT (component) |
| Span + SBO | Yes | +2.0 to +3.0 | SpanBERT (full) |
| RTD | No | not directly comparable | DeBERTa-v3 |

(Numbers are rough; from SpanBERT's ablations, scaled to typical NER tasks.)

## Implications for us

Our current backbone is **BERTimbau** = BERT-base + WWM on PT-BR corpus. That puts us at "partial" span-awareness.

**Cheapest upgrades:**

1. **Switch to DeBERTa-v3** (or mDeBERTa-v3 for PT-BR). RTD gives stronger representations overall, even without explicit span objective.
2. **Continual pretrain BERTimbau with span masking + SBO** on PT-BR corpus. Higher cost but uses our existing tokenizer/vocab.

**Untested:**

- Whether SBO at fine-tune time (instead of pretraining time) gives any of the benefit.
- Whether BR-specific PII patterns benefit more from span objectives than English NER does. CPF/CNPJ tokenize to 5-8 subwords each, which is well within SpanBERT's span distribution.

## Open questions

- Has anyone trained a SpanBERT-PT or SpanBERTimbau? (Search needed.)
- DeBERTa-v3 + SBO continual pretraining — proposed by anyone? (Search needed.)
- For BR PII specifically, is the bottleneck pretraining or fine-tuning data quality ([[../questions/synthetic-data-quality]])?

## Related

- [[../sources/2026-05-23-spanbert]] — primary source
- [[span-boundary-objective]] — the SBO mechanic
- [[span-based-ner]] — fine-tuning architecture (orthogonal)
- [[../entities/spanbert-model]]
- [[bertimbau]] — TBD page
- [[deberta-v3]] — TBD page
