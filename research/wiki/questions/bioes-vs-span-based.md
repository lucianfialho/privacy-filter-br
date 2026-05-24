---
type: question
tags: [architecture, ner-decoding, bioes, span-based]
sources: 0
updated: 2026-05-23
---

# Should we use BIOES or span-based NER?

## Background

Our v3 model uses **BIOES** tagging: each token classified into one of 89 labels (1 O + 22 categories × 4 BIOES tags). Inference outputs a sequence of token labels; the HuggingFace pipeline's `aggregation_strategy="simple"` joins consecutive labels into spans.

**Observed failure:** long PII (CPF "680.075.670-97", card "4111 1111 1111 1111") get fragmented because BERT tokenizers split them into 4-6 sub-tokens. Some sub-tokens get classified as `O` due to ambiguity, breaking the span. Our [[br-pii-guardrail]] patches this with a `merge_consecutive` post-process.

## Alternative: span-based NER

Predict (start, end, label) triples directly instead of per-token labels:
- **Biaffine NER** (Yu et al, 2020): score every possible span via a biaffine head over BERT token pairs.
- **SpanBERT** (Joshi et al, 2020): pretraining objective specifically for span representations.
- **GLiNER** (Zaratiana et al, 2024): zero-shot via prompt-style entity types.

**Advantages:**
- Doesn't suffer from BIOES fragmentation
- Naturally handles nested entities (we don't need this, but doesn't hurt)
- Smoother gradients (loss is per-span, not per-token)

**Disadvantages:**
- O(n²) spans to evaluate at inference (small for short text, expensive for long docs)
- Less mature tooling than BIOES
- Requires re-architecting our finetune pipeline

## Alternative: BIOES + Viterbi decoding

OpenAI Privacy Filter uses Viterbi to decode coherent spans (BIOES has structural constraints like "E must follow B/I, not O"). The HF `aggregation_strategy="simple"` does NOT enforce this.

Could be a quick win: keep BIOES but add Viterbi at inference. Open question: how much does this recover for our model?

## Decision tree (proposed)

```
1. Measure: how often does BIOES fragment vs how often does it miss entirely?
2. If fragmentation is the main issue: try Viterbi (cheap fix).
3. If misclassification (O instead of label) is the main issue: try span-based (architecture change).
4. If both: maybe GLiNER (different paradigm entirely).
```

## Linked papers

- [[../sources/2026-05-23-gliner]] — span-based, zero-shot. Strongly relevant; addresses fragmentation by design.
- [[../sources/2026-05-23-spanbert]] — span-aware **pretraining**, not decoding. Suggests the BIOES-vs-span debate is partly upstream of our v3 architecture: our BERTimbau backbone is not span-aware.
- [[2026-XX-XX-biaffine-ner]] — pending
- [[2026-XX-XX-viterbi-ner]] — pending

## Related concept/entity pages

- [[../concepts/span-based-ner]]
- [[../concepts/zero-shot-ner]]
- [[../entities/gliner-model]]
