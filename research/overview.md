---
type: overview
updated: 2026-05-23
---

# Overview — Improving privacy-filter-br

## Running thesis (as of 2026-05-23)

We trained `privacy-filter-br`: BERTimbau (110M) fine-tuned on 54k synthetic BR examples covering 22 PII categories. F1 macro 0.9934 on a synthetic holdout.

**It works for structured data** (JSON/CSV) when combined with regex+checksum recognizers via `br-pii-guardrail`. Schema-aware detection covers ~90% of Analytics Copilot input where PII comes in fields with known names (`cpf`, `email`, `order_id`).

**It struggles in free text.** Concrete failure modes observed:
- `private_person` misses first half of compound names ("João Silva" → only "Silva" detected)
- BIOES sub-token fragmentation: long PII split into 2+ spans the aggregator doesn't merge
- `private_phone` (F1 0.96) — most variability of any category
- Card numbers misclassified as `private_transaction_id` (both are long digit strings)
- Without verbal prefix ("email: X"), recall on `private_email` drops sharply

The first two are also patched in `br-pii-guardrail` (spaCy fallback for person, post-merge of consecutive spans).

## Hypotheses to test (link to questions/)

1. **Synthetic-to-real gap is the dominant error source.** Hardly an insight — but the magnitude matters. Need to quantify recall drop on real BR documents before deciding what to invest in.
2. **More synthetic data won't help much.** v1 (11k) → v3 (54k) gained ~2% F1. v3 → v4 (200k) might gain <0.5%. Diminishing returns.
3. **Better synthetic data structure helps more than more of it.** Specifically: more contextual diversity (PII without verbal prefix, names in unusual positions, OCR-style noise).
4. **Span-based NER may outperform BIOES** for long PII like card numbers and order IDs, eliminating the sub-token fragmentation issue.
5. **Multi-task pretraining (NER + char-level)** might help with formatting variations (CPF formatted vs raw, phone with/without 9).

## Constraints (anchor every decision)

- **Latency:** model is in the critical path of every Analytics Copilot request. Sub-100ms p99 ideal.
- **Memory:** target ≤ 1GB RAM, runs on CPU.
- **No real PII for training.** Cannot use customer data even with consent — LGPD risk.
- **Recall ≥ precision.** A false positive is annoying; a false negative is a vazamento.

## Next concrete steps

1. **Ingest 5-8 papers** from `wiki/questions/initial-reading-list.md` via paper7.
2. **Synthesize across them**: are our hypotheses supported by the literature?
3. **Design v4 dataset improvements** based on synthesis.
4. **Eventually**: validate on small set of real BR docs (your own NFs, extratos, etc — not customer data).

## What we will NOT do (yet)

- Retrain v3 today. Need more evidence first.
- Switch base model. BERTimbau 110M is the right size for our constraints.
- Build a "better" version of `br-pii-guardrail`. The 3-layer architecture is right; we should harden the existing layers.
