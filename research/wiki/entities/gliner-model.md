---
type: entity
tags: [model, ner, zero-shot, deberta]
sources: 1
updated: 2026-05-23
kind: model
---

# GLiNER

**G**eneralist and **Li**ghtweight **N**amed **E**ntity **R**ecognition.

A family of bidirectional-encoder NER models that handle arbitrary entity types via prompt-style inputs, without per-type classifier heads.

- **Paper:** [arXiv 2311.08526](https://arxiv.org/abs/2311.08526) (Zaratiana et al, 2023). See [[../sources/2026-05-23-gliner]].
- **Code:** [github.com/urchade/GLiNER](https://github.com/urchade/GLiNER)
- **HF org:** [huggingface.co/urchade](https://huggingface.co/urchade)

## Variants

| Variant | Backbone | Params | F1 OOD (avg) | Notes |
| --- | --- | --- | --- | --- |
| GLiNER-S | DeBERTa-v3-small | 50M | 52.7 | Beats ChatGPT (47.5) |
| GLiNER-M | DeBERTa-v3-base | 90M | 57.2 | |
| GLiNER-L | DeBERTa-v3-large | 0.3B | 60.9 | SOTA on CrossNER + MIT |
| GLiNER-Multi | mdeBERTa-v3-base | ~280M | varies | 32.9 avg F1 on 11 languages |

OOD avg = mean F1 across CrossNER (5 domains) + MIT-Movie + MIT-Restaurant.

## Architecture

```
Input:  [ENT] type1 [ENT] type2 ... [SEP] sentence tokens
        └── prompt section ─────────┘   └── text section ──┘

Encoder: DeBERTa-v3 (or mdeBERTa-v3 for multilingual)

Outputs:
  - Type embeddings: hidden state at each [ENT] token
  - Span embeddings: built from start_token + end_token + span_length_emb,
                     for every span up to K=12 tokens in the text section

Scoring: sigmoid(dot(type_emb_i, span_emb_jk)) for each (type i, span j..k)

Decode: greedy, threshold 0.5, optionally non-overlapping (priority queue)
```

See [[../concepts/span-based-ner]] for the span paradigm and [[../concepts/zero-shot-ner]] for the matching mechanic.

## Training

- **Dataset:** [[pile-ner-dataset]] — 44k passages, 240k spans, 13k distinct entity types.
- **Loss:** Binary cross-entropy on (span, type) pairs.
- **Negative sampling:** 50% of (span, type) pairs are negatives (random type names not in the sentence). Critical — paper reports 0% negative sampling causes false-positive bloat.
- **Hyperparams:** AdamW, lr 1e-5 backbone / 5e-5 heads, dropout 0.4, 30k steps, cosine schedule with 10% warmup, max 25 entity types per sentence.
- **Cost:** GLiNER-L = 5h on a single A100.

## Strengths (validated by paper)

- Zero-shot on entity types never seen in training (60.9 F1 avg OOD).
- Beats LLM-based baselines (ChatGPT, UniNER-13B) despite being 30-200x smaller.
- Multilingual transfer without multilingual training data.
- Fast inference (parallel span scoring, not autoregressive).

## Weaknesses / caveats

- Underperforms UniNER by ~3 F1 in **supervised fine-tuning** (when both have access to in-domain labeled data).
- Weaker on social media / tweets / informal text — TweetNER7, BTC suffer.
- Multilingual variant has uneven coverage — strong on Indo-European, weaker on low-resource.
- Max span length K=12 tokens limits applicability for long entities (legal text, BR addresses).
- Not benchmarked on PT-BR directly in the paper.

## Use in our project

Currently **not used.** Candidate for v4 experiments:

1. **As-is multilingual baseline.** Run GLiNER-Multi against our `dataset_br_v3_holdout.jsonl` with our 22 type names translated to natural language ("CPF brasileiro", "endereço residencial", etc). Measure F1.
2. **Fine-tuned on our data.** Take GLiNER-Multi checkpoint, fine-tune on 54k synthetic BR examples, compare against v3.
3. **Inspiration for v3.5.** Adopt span-based decoding in our BERTimbau model without switching backbones. Smaller change, but loses zero-shot.

## Related

- [[../sources/2026-05-23-gliner]] — the source paper
- [[deberta-v3]] — backbone (page TBD)
- [[pile-ner-dataset]] — training data (page TBD)
- [[universalner-uniner]] — closest competitor (page TBD)
- [[../concepts/span-based-ner]]
- [[../concepts/zero-shot-ner]]
- [[../questions/bioes-vs-span-based]]
