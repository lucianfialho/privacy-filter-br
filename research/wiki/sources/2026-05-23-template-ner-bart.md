---
type: source
tags: [ner, few-shot, template-based, bart, seq2seq, domain-adaptation]
sources: 1
updated: 2026-05-23
arxiv: 2106.01760
authors: [Cui, Wu, Liu, Yang, Zhang]
year: 2021
---

# Template-Based Named Entity Recognition Using BART

Cui et al., Zhejiang/Westlake + Microsoft Research Asia — arXiv [2106.01760](https://arxiv.org/abs/2106.01760)

## Abstract (paraphrased)

Few-shot NER tries to transfer from a resource-rich source domain (news, with PER/LOC/ORG) to a target domain with different label sets (movies with CHARACTER/TITLE). Existing methods rely on similarity metrics and can't update model parameters effectively. The authors propose **treating NER as LM ranking via templates**: fine-tune BART with templates like `"⟨span⟩ is a ⟨entity_type⟩ entity"`, and at inference, score each candidate span against each entity-type template; highest-scoring template wins. Hits 92.55 F1 on CoNLL-2003 (rich-resource) and **outperforms fine-tuned BERT by 10-15 F1 points on MIT Movie, MIT Restaurant, ATIS in few-shot setting**.

## Key findings

- **Re-frames NER as seq2seq scoring.** Input = original sentence. Output = filled template. Score = template-conditioned log-likelihood. No softmax/CRF output layer.
- **No output layer means no retraining for new categories.** Adding "PIX_KEY" category just requires defining a template — no architecture change. Closest pre-2023 paper to GLiNER's zero-shot promise.
- **Negative templates required.** `"⟨span⟩ is not a named entity"` lets the model score non-entities consistently. Without negatives, everything matches some template.
- **Massive gains in low-resource regime.** MIT Movie: BERT-base 49.6 → Template 65.0 F1 (+15.4). Smaller gap as data grows.
- **Span enumeration over all candidates.** Like GLiNER and LUKE — enumerate all spans up to max length, score each with template-scoring.
- **Slow at inference.** Per-span template scoring is O(n × |types|) BART forward passes per sentence. Much slower than a single CRF decode.

## How this relates to our work

This is the **direct ancestor of zero-shot NER paradigms like GLiNER**, but using a different mechanism (template scoring with a seq2seq LM) instead of type-name embeddings + matching.

Implications for us:

1. **Add new PII categories without retraining = templates.** If our v4 had a template like `"⟨span⟩ é uma chave PIX"`, we could add PIX-key detection by writing one template. Same promise as [[../concepts/zero-shot-ner]].
2. **BART seq2seq is overkill for our latency budget.** Their inference is many BART forward passes per sentence. Our <100ms target rules this out.
3. **The pattern of "negative class via template" is portable.** We could use a similar trick during NER fine-tuning — explicitly train a "this is not PII" template to balance precision.

## What we can actually use

- **Validation that template-based / type-aware NER is competitive in low-resource regimes.** We're not in a low-resource regime (~2500/category), so this matters less. But for new categories where we have <100 examples, the template approach beats fine-tuning.
- **Negative-class explicit training.** Worth experimenting with at fine-tune time. We have an implicit "O" class in BIOES, but we don't explicitly train "this token is definitely not PII" with examples.

## Concerns / critique

- **Latency is killer.** BART per-span scoring at inference is ~100x slower than CRF or per-span classifier. Hard sell for a local guardrail.
- **Template wording matters.** "Bangkok is a location" vs "Bangkok is a place" — different templates, different scores. Engineering effort moves from architecture to prompt design.
- **No PT-BR variant tested.** English only. Would need PT BART (mBART or BARTimbau if exists) for our use case.
- **Single-token span bias.** Their templates implicitly assume single-token spans for the candidate. Multi-token entities work but performance is uneven.

## Open questions

- Does mBART work as a multilingual template-NER backbone? Untested in paper.
- Can we distill template-NER outputs into a fast student model? That would resolve the latency problem.
- What's the inference cost for K candidate spans × T templates × L BART decode length? Need actual measurements.

## Comparison with GLiNER (already ingested)

| | Template-NER (Cui 2021) | GLiNER (Zaratiana 2023) |
| --- | --- | --- |
| Backbone | BART (seq2seq) | DeBERTa-v3 (encoder) |
| New-category support | Add template | Add type-name to prompt |
| Inference speed | Slow (per-span LM scoring) | Fast (parallel matching) |
| Decoding | Argmax over template scores | Threshold + greedy non-overlap |
| BR support | Not tested | Not tested but multilingual variant exists |

**GLiNER strictly dominates Template-NER for our use case** — same flexibility, much faster. But Template-NER is the conceptual ancestor and useful to understand the lineage.

## Related

- [[../concepts/zero-shot-ner]] — same paradigm, different mechanism
- [[../concepts/lm-based-data-augmentation]] — uses LM at training time, this paper uses LM at inference time
- [[../entities/gliner-model]] — successor approach, much better engineered for inference
- [[../questions/bioes-vs-span-based]] — Template-NER also abandons BIOES
