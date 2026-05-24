---
type: concept
tags: [pretraining, span-representations, sbo, bert-variant]
sources: 1
updated: 2026-05-23
---

# Span Boundary Objective (SBO)

An auxiliary pretraining objective introduced by [[../sources/2026-05-23-spanbert]] that forces a transformer encoder to encode the **content of a masked span** into the **representations of the two boundary tokens** (one position before the span, one position after).

## The mechanic

Given a masked span (x_s, ..., x_e), for each masked token x_i in the span, predict x_i using **only**:

- `h_{s-1}` — encoder hidden state at the token immediately before the span (left boundary)
- `h_{e+1}` — encoder hidden state at the token immediately after the span (right boundary)
- `p_{i-s+1}` — a learned position embedding marking the relative offset of x_i from the left boundary

Pass these through a small FFN to get the SBO prediction:

```
h0 = [h_{s-1}; h_{e+1}; p_{i-s+1}]            # concatenation
h1 = LayerNorm(GeLU(W_1 @ h0))
y_i = LayerNorm(GeLU(W_2 @ h1))
loss_SBO(x_i) = -log P(x_i | y_i)              # cross-entropy via the input embedding
```

Total loss per masked token: `L(x_i) = L_MLM(x_i) + L_SBO(x_i)`.

**Critical:** the prediction uses ONLY the boundary representations — the masked tokens themselves are not visible. The model can't cheat by attending to the masked content from inside the span.

## Why it works

Forces the model to **stuff span content into boundary tokens during pretraining**. At downstream task time, when you need a span representation for QA / coref / NER, you can use the boundary tokens directly and they already encode the span content.

In span selection tasks (extractive QA, coref linking), the dominant computational pattern is "score this candidate span". Span representations are typically `[h_start; h_end]` or some variant. SBO ensures those endpoints are maximally informative.

## What SBO is NOT

- **Not span enumeration.** SBO is a pretraining loss, not an inference-time mechanic. The downstream architecture can be anything.
- **Not specific to NER.** Originally validated on QA, coref, RE, GLUE. NER wasn't even a primary benchmark in the paper.
- **Not the only span-aware objective.** Other approaches: pair2vec (Joshi et al, 2019), ERNIE knowledge spans (Zhang et al, 2019), XLNet's permutation LM on spans.

## Empirical impact (from paper ablations)

| Setup | SQuAD 2.0 | NewsQA | TriviaQA | Coref | MNLI-m | QNLI | GLUE avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Span Masking (2seq, +NSP) | 85.4 | 73.0 | 78.8 | 76.4 | 87.0 | 93.3 | 83.4 |
| Span Masking (1seq, no NSP) | 86.7 | 73.4 | 80.0 | 76.3 | 87.3 | 93.8 | 83.8 |
| Span Masking (1seq) + **SBO** | **86.8** | **74.1** | **80.3** | **79.0** | **87.6** | **93.9** | **84.0** |

SBO gain over span-masking-alone is **modest on QA (+0.1 to +0.7), large on coref (+2.7)**. Coref is the most span-representation-dependent task, hence the biggest SBO gain.

## Relevance to our work

We don't pretrain — we fine-tune BERTimbau (which has whole-word masking but no SBO). So SBO doesn't directly apply.

**But**: there are at least two ways to borrow the idea at fine-tune time:

1. **Auxiliary SBO loss during NER fine-tuning.** Add a small head that, given a predicted entity span, tries to reconstruct the span content from boundary tokens. Joint loss = BIOES + SBO-reconstruction. Untested in literature, but cheap to try.

2. **Predict entity type from boundaries.** Instead of per-token classification, predict the entity type using only `(h_start, h_end)` of each candidate span. This is closer to [[span-based-ner]] but uses the SBO insight that boundaries should be informative.

## Open questions

- Does SBO still help if the underlying corpus has already been pretrained with whole-word masking?
- Does an analogous loss work at fine-tune time only (no pretraining)?
- Multilingual variant: is there a paper trying SBO on multilingual corpora?

## Related

- [[../sources/2026-05-23-spanbert]] — origin paper
- [[../entities/spanbert-model]] — the resulting model
- [[span-based-ner]] — orthogonal but adjacent
- [[pretraining-objectives-for-ner]] — broader context
