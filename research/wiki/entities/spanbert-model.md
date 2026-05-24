---
type: entity
tags: [model, pretraining, bert-variant, span-representations]
sources: 1
updated: 2026-05-23
kind: model
---

# SpanBERT

A BERT variant with span-aware pretraining. Same architecture and tokenizer as BERT-large, different pretraining objectives.

- **Paper:** [arXiv 1907.10529](https://arxiv.org/abs/1907.10529) (Joshi et al, 2019). See [[../sources/2026-05-23-spanbert]].
- **Code:** [github.com/facebookresearch/SpanBERT](https://github.com/facebookresearch/SpanBERT)
- **HF:** [SpanBERT/spanbert-base-cased](https://huggingface.co/SpanBERT/spanbert-base-cased), [SpanBERT/spanbert-large-cased](https://huggingface.co/SpanBERT/spanbert-large-cased)

## What it is

A BERT-large model pretrained with two modifications:

1. **Span masking** instead of token masking (geometric distribution, mean ~3.8 words, capped at 10).
2. **Span Boundary Objective (SBO)** — an auxiliary loss that predicts each masked token from boundary representations only. See [[../concepts/span-boundary-objective]].

Single-sequence training (no NSP) is a third change, but it's a training procedure decision rather than an architectural one.

## Variants released

| Variant | Params | Corpus | Notes |
| --- | --- | --- | --- |
| spanbert-base-cased | 110M | BooksCorpus + EN Wiki | Same as BERT-base, span pretraining |
| spanbert-large-cased | 340M | BooksCorpus + EN Wiki | Main paper variant |

**No multilingual or PT version is publicly available.**

## Benchmark numbers (from paper)

- **SQuAD 1.1:** 94.6 F1 (above human 91.2)
- **SQuAD 2.0:** 88.7 F1
- **OntoNotes coref:** 79.6 F1 (+6.6 absolute over prior SOTA)
- **TACRED relation extraction:** 70.8 F1
- **GLUE average:** 82.8

Strongest gains on **span-selection** tasks. Modest gains on sentence-level tasks (GLUE).

## NER performance

The paper does NOT include CoNLL-2003 NER as a primary benchmark, surprising given the span-selection framing. Some follow-up work measured SpanBERT on CoNLL-2003 (typical numbers: 92-93 F1) but it isn't a clear win over RoBERTa or DeBERTa.

This is one reason GLiNER chose DeBERTa-v3 over SpanBERT as backbone — by 2023, SpanBERT's specific tricks had been partially absorbed by other variants.

## Strengths

- Strong span representations available at boundary tokens by construction.
- Drop-in replacement for BERT in any downstream task.
- Especially good when fine-tuning data is small (paper shows lower-resource gains).

## Weaknesses

- English only. No public multilingual variant. Anything PT requires either retraining or relying on English transfer.
- 2019 architecture. RoBERTa, DeBERTa-v3, ELECTRA have all caught up or surpassed it on most modern benchmarks via different mechanisms.
- NER specifically isn't a primary benchmark — gains are inferred, not demonstrated.

## Use in our project

**Currently not used.**

Potential v4 paths:

1. **Train SpanBERTimbau.** Take BERTimbau, do continual pretraining with span masking + SBO on a PT-BR corpus (BrWaC or similar). Cost: ~50-200h GPU on A100. Output: PT span-aware backbone, fine-tune as v3 (BIOES) or v4 (span-based).
2. **Test SpanBERT-base-cased as a cross-lingual baseline.** Fine-tune on our BR PII data and see if the span pretraining beats BERTimbau's WWM even with English pretraining. Cheap experiment, ~1h.
3. **Borrow the SBO idea at fine-tune time.** Add auxiliary span-reconstruction head to v3 training. Untested, novel, cheap.

## Related

- [[../sources/2026-05-23-spanbert]] — origin paper
- [[../concepts/span-boundary-objective]] — the SBO mechanic
- [[../concepts/pretraining-objectives-for-ner]] — broader context
- [[../entities/gliner-model]] — chose DeBERTa-v3 instead, partly because SpanBERT's tricks were superseded
- [[bertimbau]] — our current backbone (TBD)
- [[deberta-v3]] — alternative backbone (TBD)
