---
type: concept
tags: [pii, ner, llm, zero-shot, gpt-4, paradigm-comparison]
sources: 1
updated: 2026-05-23
---

# LLM-based PII detection

Using large generative language models (GPT-4, Claude, Llama, etc) to identify and redact PII via prompts, rather than training a dedicated NER model.

## The paradigm

```
Input:  raw text + prompt that lists target PII categories with examples
Model:  generative LLM (closed API or local OSS)
Output: same text with PII redacted (or list of identified entities)
```

The model treats PII detection as instruction-following + NER, leveraging in-context learning rather than supervised fine-tuning.

## Variants

1. **Cloud frontier LLM** (GPT-4, Claude). High accuracy, high cost, data-residency issues. See [[../sources/2026-05-23-deid-gpt]].
2. **Local instruction-tuned LLM** (Llama-2, Falcon, Mistral). Lower cost, lower accuracy. Paper reports they fail catastrophically (~0.60 accuracy) on i2b2 de-id.
3. **Local NER-fine-tuned encoder** (BERTimbau, ClinicalBERT). Our v3. Lower cost, *higher* accuracy than small instruction-tuned LLMs on the same task.
4. **Hybrid: small NER + LLM fallback for uncertainty.** Use the small model for confident cases, escalate to LLM only for ambiguous spans. Not in literature widely — could be a contribution.

## When LLM-based PII detection wins

- **New categories at inference time.** Rule-based and fine-tuned NER need a category list at training time. LLMs accept new categories in the prompt.
- **Multilingual / low-resource languages.** A single GPT-4 prompt works for PT, EN, ES, etc. A fine-tuned NER needs per-language training data.
- **Reasoning across ambiguity.** "Mr. Silva" — is "Silva" the person, or part of company name "Silva Engenharia"? An LLM can use surrounding context. A token-classifier may guess wrong.
- **Low-volume use.** API cost dominates at scale; for hundreds of docs per day, GPT-4 is cheaper than maintaining a training pipeline.

## When LLM-based PII detection loses

- **Data residency / compliance.** Cannot send patient/citizen data to OpenAI under HIPAA / LGPD / similar. Cloud LLM is a non-starter for many use cases — including ours.
- **Cost at scale.** $0.01-$0.10 per call adds up. At 100k docs/day, $1k-10k/day. A local NER costs ~0 to run.
- **Latency.** GPT-4 API: ~2-5 seconds. Our v3 target: <100ms. 50x difference.
- **Auditability.** Open-weight encoder we can inspect, replay, freeze. GPT-4 is a moving target.
- **Determinism.** Same prompt, same input, different output sometimes. Not acceptable for compliance.
- **Small local LLMs are surprisingly bad.** Llama-2-7b on i2b2 de-id = 0.612. Not competitive with fine-tuned 110M encoder.

## Decision matrix

| Use case | Best approach |
| --- | --- |
| High-volume, on-premise, regulated data | Small fine-tuned encoder ← **us** |
| Low-volume, cloud-tolerant, varied categories | Cloud LLM |
| Mixed-criticality, with budget | Encoder + LLM fallback for low-confidence cases |
| Multilingual, no per-language training data | Multilingual encoder ([[zero-shot-ner]]) or cloud LLM |

## Implications for `br-pii-guardrail`

Our library has 3 layers today:
1. Regex + checksum (CPF/CNPJ/cartão/CEP)
2. Schema-aware scanners (JSON/CSV/PDF)
3. NER fallback (BERTimbau v3)

Possible 4th layer (optional, opt-in):
4. **LLM fallback for ambiguous text.** When NER confidence is low AND regex layer didn't fire, optionally escalate to a configured LLM (GPT-4, Claude, local Llama, etc) with a 3-part PII prompt. Useful for free-text fields where regex/schemas don't apply and NER is uncertain.

Design constraints:
- Off by default. User must explicitly enable and configure provider/endpoint.
- Per-call data residency check (warn if config sends to non-GDPR-equivalent region).
- Latency budget knob (skip LLM call if call would exceed user's max latency).
- Cache LLM results when input hash matches (cost control).

## Open questions

- Does GPT-4 with our 22-category LGPD prompt beat v3 on BR PII? (Untested.)
- Can we distill GPT-4 PII labels into a smaller student model? Already a known technique for general NER — does it work for our specific category set?
- Where exactly does v3 confidence cross the "should escalate" threshold? Need to define before building the fallback.

## Related

- [[../sources/2026-05-23-deid-gpt]] — primary source
- [[prompt-engineering-for-pii]] — the prompt template
- [[zero-shot-ner]] — encoder-based zero-shot (e.g. GLiNER) — a different way to get "new categories without retraining"
- [[../questions/synthetic-data-quality]] — LLM as labeler (related but distinct)
