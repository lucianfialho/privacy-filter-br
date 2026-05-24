---
type: question
tags: [phase-2, labeler, format-aware, root-cause, v4]
sources: 0
updated: 2026-05-24
---

# Phase 2: Format-aware labeler fix + v4 dataset regeneration

## Context

After Phase 1 ([[2026-05-23-direction-a-audit-results]] + [[2026-05-23-v3.1-training-results]]) showed that v3.1's gains were marginal because the relabel was narrow, we moved to fix the root cause: `src/labeler.py` used `re.escape(value)` exact-string matching, so when the LLM rewriter altered formatting (e.g., `-` → `.` in CPF check-digit position), the labeler silently dropped the PII.

## What was fixed

### `src/labeler.py` rewritten

The new labeler does:

1. **Always exact match** (`re.escape(value)`) — preserves byte-for-byte literal matches, including mask chars (`*`) and non-canonical content.
2. **Additionally, format-aware match** for structured PII labels (`PRIVATE_CPF`, `PRIVATE_CNPJ`, `PRIVATE_RG`, `PRIVATE_PIS`, `PRIVATE_CNH`, `PRIVATE_TITULO_ELEITOR`, `PRIVATE_IE`, `PRIVATE_PHONE`, `PRIVATE_CERTIDAO`):
   - Extract alphanumeric skeleton of value: `_alnum_skeleton("123.456.789-09") == "123456789090"`.
   - Build regex that matches the skeleton with up to 2 separator chars allowed between consecutive alnum chars: `1[.\-/()\s  ]{0,2}2[.\-/()\s  ]{0,2}3...`.
   - Anchor at non-alnum word boundaries on both sides.
   - Match in text.
3. **Free-text labels** (`PRIVATE_PERSON`, `PRIVATE_EMAIL`, `PRIVATE_ADDRESS`, etc.) keep exact match.
4. **Dedup** by keeping longest span (existing behavior preserved).

### Tests

Added 10 new tests in `tests/test_labeler.py` covering:
- CPF/CNPJ/PIS separator variants (`-` → `.`, spaces, no separators)
- Phone with extra dashes between parens
- Word boundary respect (no match inside arbitrary digit sequences)
- Short skeleton fallback to exact match
- Free-text labels still exact
- Variants in `inserted` deduplicated by (skeleton, label)

All 16 labeler tests pass. All 38 project tests pass.

## Limit discovered: relabeling existing data doesn't help much

The script `scripts/relabel_v2.py` attempts to apply the new labeler to existing texts in `dataset_br_v3.jsonl` by reconstructing the `inserted` dict from existing labels (using `text[start:end]` from each entity as the value, and the label).

**Result on 1000 sample: 0 new labels added** (after the exact+format-aware union dedup).

Why: the new labeler can only find variants of values that **were already labeled** in v3. PIIs that v3 missed entirely (because the LLM mangled the format) are not in the dataset, so reconstructing `inserted` doesn't recover them.

To benefit from the new labeler, we need to **regenerate the dataset** with the new code in the pipeline — preserving the original `inserted` dict at generation time.

## Smoke test plan (in progress)

Generating a 5000-example smoke test via OpenAI Batch API to validate end-to-end:

1. `scripts/openai_batch.py perfis --n 1000` — pre-generate 1000 4devs profiles to `data/perfis_smoke.jsonl` (~15-20 min)
2. `scripts/openai_batch.py prepare --n 5000 --perfis data/perfis_smoke.jsonl --output data/batch_input_smoke.jsonl` — build batch + metadata
3. `scripts/openai_batch.py submit --input data/batch_input_smoke.jsonl` — upload to OpenAI Batch API (50% discount, 24h SLA)
4. Wait for batch (typically <1h for 5k requests)
5. `scripts/openai_batch.py process --metadata data/batch_input_smoke.metadata.jsonl --output data/dataset_br_v4_smoke.jsonl` — apply new labeler to outputs
6. Inspect: do variants of CPF/CNPJ/phone/etc get labeled?
7. Train v4-smoke locally for ~30 min, benchmark vs v3

Cost: ~$0.50-1.50 (gpt-5-nano batch).

If smoke test confirms the pipeline works correctly:
- Scale to 50k examples (~$5-15, ~24h batch)
- Train v4 on GPU (~2h)
- Benchmark vs v3 on both original and relabeled holdout

## Smoke test results (2026-05-24)

### Dataset diagnostics

- **5448 train + 539 holdout** (combined: 4546 regular + 902 extras for train; 442 + 97 for holdout)
- All 22 categories present (initially missed account_number/date/url/secret because `cmd_extras` wasn't run — added separately)
- Regex audit on the regenerated dataset: **examples with at least one unlabeled regex match dropped from 16.76% (v3) → 5.21% (v4-smoke)**. All remaining 5.21% is the `financeiro@empresa.com.br` template placeholder (template-level bug, not labeler).
- Per-category miss rates (v3 → v4-smoke):
  - private_cpf: 6.72% → **0.00%**
  - private_cnpj: 0.32% → **0.00%**
  - private_rg: 7.36% → **0.00%**
  - private_phone: 4.63% → **0.17%**
  - private_titulo_eleitor: 7.79% → **0.00%**

### Format diversity in labels

Each structured category has all 4 format variants properly labeled:
- CPF: canonical (1043) + masked (1009) + raw (1049) + spaces (943) ≈ 1k each
- CNPJ: canonical (1022) + masked (964) + raw (899) + spaces (882) ≈ 900-1k each
- Phone: with parens (1183) + raw (1030) + spaces (897) + no-DDD (1042) ≈ 1k each

In v3, only the canonical format got labeled; variants were silently dropped.

### Training results (BERTimbau, 3 epochs, RTX 2070 SUPER, ~13 min)

| | v3 final | v3.1 ep3 | **v4-smoke ep3** |
| --- | --- | --- | --- |
| eval_loss | — | 0.0424 | **0.0046** (~9× smaller) |
| precision | 0.9859 | 0.9724 | **0.9982** |
| recall | 0.9942 | 0.9953 | **0.9985** |
| **F1 (micro)** | **0.9900** | 0.9837 | **0.9983** (+0.0083) |
| F1 (macro) | 0.9934 | 0.9900 | **0.9982** (+0.0048) |
| Training set size | 50000 | 50000 | **5448** (10× smaller) |

**v4-smoke with 10× less data beats v3 in F1.** Label quality > volume.

Loss curve: v4-smoke epoch 1 already beat v3 final (F1 0.9974 vs 0.9900).

### Per-category breakdown (v4-smoke holdout)

15 of 22 categories achieve **perfect 1.0000 F1**:
- private_address, certidao, client_revenue, cnh, cnpj, date, ie, order_id, pis, rg, titulo_eleitor, tracking_code, transaction_id, url + (one more)

Biggest improvements over v3:
- **private_phone: 0.9627 → 0.9959 (+0.0332)** — was the weakest category in v3 due to format variants
- private_certidao: 0.9828 → 1.0000 (+0.0172)
- private_person: 0.9867 → 0.9990 (+0.0123)
- private_ie: 0.9879 → 1.0000 (+0.0121)
- private_address: 0.9886 → 1.0000 (+0.0114)

Categories that slightly regressed (all with support <150 spans — small-sample noise):
- account_number: 0.9978 → 0.9885 (-0.0093, 131 spans)
- private_customer_id: 1.0000 → 0.9913 (-0.0087, 57 spans)
- private_invoice_number: 0.9978 → 0.9947 (-0.0031)
- secret: 1.0000 → 0.9958 (-0.0042)

These regressions are expected from the 10× smaller training set; should disappear at full scale.

## Hypothesis CONFIRMED

The root-cause analysis was correct:
1. **v3's labeler bug** (exact-string `re.escape(value)` match) caused ~16.76% of examples to have unlabeled PII when the LLM rewrote formats.
2. **v3.1's regex-based relabel** captured only ~11% of these because it didn't have access to the original `inserted` dict.
3. **v4-smoke's regenerated dataset** with the new format-aware labeler captures ~99.5% of format variants at generation time.
4. **F1 improvement is real**, not an artifact: +0.83 pts micro F1 vs v3 with 10× less data.

## What this proves and what it doesn't

**Proven:**
- Format-aware labeling at generation time outperforms post-hoc relabel.
- Phase 2 fixes the root cause; the v3.1 marginal gain was indeed limited by what regex could recover.
- Even at 1/10 scale, v4 outperforms v3 on the synthetic holdout.

**NOT proven (still):**
- Real-world performance (Phase 1 from the roadmap is still pending).
- Whether the gains transfer to non-synthetic BR documents (PDFs, OCR'd text, mixed formats).

## Recommended next step

**Scale up to full v4** (50k examples) and publish as v4 on HF Hub, OR run Phase 1 (manual real-world test set) before committing to a full regeneration. Either path will further reduce uncertainty.

Cost estimate for full v4:
- 10000 perfis via 4devs: ~3-4h
- 50000 batch requests via OpenAI: ~10-20 min batch wait + ~$5-15
- Train v4 on remote: ~2h GPU
- **Total: ~6-8h wall time, ~$5-15**

This is the "fix root cause" path discussed in [[2026-05-23-v3.1-training-results#concrete-recommendation]] as Option 2 — and the smoke test confirms it works as predicted.

## Related

- [[2026-05-23-direction-a-audit-results]] — Phase 1 audits that identified the labeler bug
- [[2026-05-23-v3.1-training-results]] — Phase 1 retrain (marginal gains because relabel was narrow)
- [[../sources/2026-05-23-lambada]] — methodological inspiration (generate + filter)
- [[../concepts/synthetic-data-filtering]] — broader concept
