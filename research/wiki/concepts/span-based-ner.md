---
type: concept
tags: [ner, decoding, architecture, span-based]
sources: 1
updated: 2026-05-23
---

# Span-based NER

A family of NER approaches that predict `(start, end, label)` triples directly, rather than assigning per-token labels (BIO, BIOES). Instead of a sequence classification problem, NER becomes a **span classification** problem over an enumerable set of candidate spans.

## Why it exists

Per-token tagging schemes (BIO/BIOES) have known weaknesses:

- **Fragmentation.** Tokenizers split long entities (CPF, credit card) into sub-tokens. If any single sub-token is misclassified, the span breaks.
- **No nested entities.** A token can only have one label, so overlapping entities are impossible.
- **Decoding ambiguity.** A token sequence may not form a valid BIOES sequence (E without B, I after O). Naive arg-max ignores this; Viterbi enforces constraints but adds inference cost.

Span-based NER sidesteps all three by scoring spans as atomic units.

## How spans are enumerated

For a sentence of length n, there are O(n²) spans. To stay tractable:

- **Bound max span length K.** GLiNER uses K=12 tokens. Most NER datasets have < 1% of entities exceeding 10 tokens.
- **Filter on simple heuristics.** Exclude spans crossing sentence boundaries, starting/ending on subwords, etc.

Result: linear in n with a constant factor K.

## How spans are scored

Two families:

1. **Span representation + linear classifier.** Build a span embedding (typically concat of start/end token hidden states, sometimes with span-length embedding). Project to label space.
2. **Biaffine scoring** (Yu et al, 2020). Bilinear interaction between start-token and end-token representations, parametrized by a per-label tensor. More expressive but more parameters.

[[gliner-model]] uses approach (1) with a twist: instead of a fixed classifier head, it matches the span embedding against **entity-type embeddings** computed from prompt tokens. Enables zero-shot.

## Inference / decoding

Per-span scores produce per-(span, label) probabilities. Decoding strategies:

- **Greedy.** Threshold each (span, label) at 0.5, keep all above. May produce overlapping spans.
- **Flat NER.** Sort by score, greedily accept non-overlapping spans (priority queue, O(n log n)).
- **Nested NER.** Accept all overlapping spans, let the consumer handle nesting.

## Comparison with BIOES

| | BIOES + token classifier | Span-based |
| --- | --- | --- |
| Decoding | Per-token argmax (or Viterbi) | Per-span sigmoid + non-overlap filter |
| Nested entities | No | Yes |
| Fragmentation | Yes | No |
| Inference cost | O(n × L) where L = labels | O(n × K × L) where K = max span |
| Training signal | Per-token cross-entropy | Per-span binary cross-entropy |

For short text (under ~100 tokens) the span-based cost is comparable; for long documents BIOES wins on raw throughput.

## Relevance to our work

We currently use BIOES in v3 ([[../questions/bioes-vs-span-based]]). Documented downsides:

- CPF/CNPJ/cartão fragmented at sub-token level (we patch this with `merge_consecutive` post-process in `br-pii-guardrail`).
- Person names with title prefixes ("Cliente João Silva") split in unpredictable ways.

A span-based variant would address both, at the cost of re-training and a different inference path.

## Papers / models using this approach

- [[../entities/gliner-model]] — Zaratiana et al, 2023. See [[../sources/2026-05-23-gliner]].
- [[../entities/spanbert-model]] — Joshi et al, 2019 (pretraining for span representations, not a span-decoder per se but the foundational span-aware backbone). See [[../sources/2026-05-23-spanbert]].
- Biaffine NER — Yu et al, 2020. Not yet ingested.
- UniNER — Zhou et al, 2023 (LLM-based, also span-based decoding). Not yet ingested.

## Open questions

- Is K=12 enough for BR addresses? "Rua Senador Vergueiro 1234, Apto 502, Bairro Nossa Senhora de Fátima" easily exceeds 12 tokens after BPE.
- Are there span-based architectures specifically tested on PT-BR?
- How does span-based handle the same sub-token issues during span boundary detection? (Need to verify in code, not just claims.)
