---
type: question
tags: [v4, postmortem, distribution-shift, boundary-bug, cross-holdout]
sources: 0
updated: 2026-05-24
---

# v4 post-mortem: cross-holdout reveals over-fitting (2026-05-24)

## TL;DR

v4 trained beautifully on its own holdout (F1 0.9992) but **catastrophically fails on v3's holdout (F1 0.5534)**. v3, in contrast, achieves F1 0.9947 on v4's holdout — better than its own (0.9901). v4 over-fit to the gpt-5-nano rewriter's output style. v3 remains the production default. v4 stays public on HF Hub as a research artifact.

## What we ran

After completing v4 training and seeing F1 0.9992 on its own holdout, the user paused before publishing to PyPI and asked: "is this really better than v3?"

Designed a 2×2 cross-holdout benchmark in `scripts/benchmark_v3_v4_cross.py`:

| | holdout_v3 (Haiku-generated) | holdout_v4 (gpt-5-nano-generated) |
| --- | --- | --- |
| v3 | 0.9901 ✓ | 0.9947 |
| v4 | **0.5534** ⚠️ | 0.9992 ✓ |

Each cell uses `seqeval` BIOES F1 (same metric as `notebooks/finetune_v3_local.py`).

**Sanity checks passing:**
- v3 on holdout_v3 = 0.9901 (matches published 0.9900)
- v4 on holdout_v4 = 0.9992 (matches published 0.99923)

**Cross-cells, the unexpected:**
- v4 vs v3 on holdout_v3: **−0.4367**
- v4 vs v3 on holdout_v4: +0.0046

## What we found (per-category diagnosis)

`scripts/diagnose_v4_on_v3_holdout.py` revealed that v4 fails on **all 22 categories** with similar TP/FP/FN ratios:

| category | v4 F1 | TP | FP | FN |
| --- | --- | --- | --- | --- |
| account_number | 0.00 | very low | 591 | 269 |
| private_cnpj | 0.00 | very low | 7493 | 3725 |
| private_date | 0.00 | very low | 1062 | 529 |
| private_cnh | 0.00 | very low | 1117 | 501 |
| private_rg | 0.0017 | 2 | 1587 | 765 |
| private_phone | 0.0255 | 155 | 8065 | 3803 |
| private_person | 0.0703 | 579 | 10413 | 4903 |

Only 3 examples in 4929 had a **same-position re-label** (gold=X, v4=Y). So v4 isn't semantically confused — it gets the type right, but **boundaries wrong**.

## Root cause: boundary-bleed into markdown

`scripts/diff_v4_vs_v3holdout_spans.py` showed v4's predictions consistently extend or contract beyond the gold span:

```
GOLD: 'Manuel Yuri da Rocha' (20 chars)
V4:   'Manuel Yuri da'       (14 chars)  ← trunca o sobrenome

GOLD: '61617-132'             (9 chars)
V4:   '61617-132\n- *'       (13 chars)  ← estende em markdown

GOLD: '47502665587'           (11 chars)
V4:   '** 47502665587\n- *'  (18 chars)  ← prefix e suffix em markdown

GOLD: '(85) 99272-8228'      (15 chars)
V4:   '(85) 99272-8228\n\n---' (20 chars) ← estende em separadores
```

**Distribuição de delta total (|start_diff| + |end_diff|) nas amostras:**

| Δ | % |
| --- | --- |
| 1 | 17% |
| 4 | 17% |
| 5 | 17% |
| 6 | 22% |
| 7 | 22% |
| 8 | 6% |

**0% das amostras observadas tinha Δ=0.** v4 sempre erra boundary, mesmo quando categoria está correta.

## Why this happened

v4 dataset was generated via `openai_batch.py` using **gpt-5-nano** with `reasoning_effort=minimal`. The model produced text with extremely consistent markdown structure:

```
- **CPF:** 47502665587
- **RG:** 48.699.916-6
- **PIS:** 54649663372
```

Every PII surrounded by `**marker**: value\n`. The format-aware labeler placed labels correctly at the value characters, but v4 saw thousands of examples of this exact pattern and **learned to extend predictions into the surrounding markdown**.

v3 dataset was generated with **Claude Haiku** via CLI subprocess. Haiku produces more variable text — sometimes with markdown, sometimes inline ("o CPF do cliente é 47502665587"), sometimes in tables. v3 learned more diverse boundaries.

This is a textbook **distribution shift via training-time monoculture** problem.

## Why we didn't catch this earlier

1. **Smoke test (5448 examples) had the same gpt-5-nano output style** → F1 0.9983 didn't reveal the bug, because boundaries were consistent with the training distribution.
2. **Full v4 training measured against v4 holdout only** → 0.9992 confirmed in-distribution accuracy, not generalization.
3. **The audit scripts checked LABEL quality, not BOUNDARY consistency** → labels were correct (skeleton-matched the values), but the trained model still bled into markdown.
4. **No cross-holdout baseline ran before publishing** → publishing v4 to HF Hub happened before the cross-holdout was even designed.

The user's instinct to pause before PyPI and ask "is this really better?" was the saving grace.

## What we did about it

- **Reverted Gradio Space to v3** — production demo is back to the model that generalizes.
- **Reverted `br-pii-guardrail` lib docs to reference v3** as the recommended model.
- **Updated repo README** to:
  - Mark v3 as production default
  - Mark v4 as experimental research artifact
  - Document the cross-holdout finding prominently
- **Kept v4 model on HF Hub** as a public artifact (don't break links for anyone who already downloaded).
- **Did NOT publish `br-pii-guardrail` 0.1.2 to PyPI.**

## What was actually valuable from this whole exercise

Even though v4 didn't ship, this work produced:

1. **Format-aware labeler (`src/labeler.py`)** — real fix for a real bug (16.76% of v3 labels had unlabeled regex-matchable PII). The labeler code itself is correct; the dataset built with it just has homogeneous text style.
2. **OpenAI Batch API pipeline (`scripts/openai_batch.py`)** — 50% cost discount, ~$3 for 50k examples. Reusable for v5 with different rewriter mix.
3. **Cross-holdout methodology** — caught a distribution-shift problem that the standard in-distribution holdout would have hidden.
4. **8 audit/benchmark/diagnostic scripts** — full toolkit for future experiments.
5. **Research wiki with 8 papers ingested** — including LAMBADA (Anaby-Tavor 2019) which directly warned about this kind of LM-rewriter monoculture.

## v5 plan (informed by this post-mortem)

To get a model that actually beats v3, we need to:

1. **Diverse rewriter mix:** combine Haiku output (v3 dataset) + gpt-5-nano output (v4 dataset) + ideally a third source (Llama, Mistral, etc). Train on union → model can't overfit to one style.

2. **Boundary-tightening post-process in `br-pii-guardrail`:** strip leading/trailing `**`, `\n`, whitespace, `- ` from NER predictions before returning. Cheap, helps even with v3 boundaries on noisy real-world inputs.

3. **Pre-augmentation: strip markdown from 30% of training examples** to teach the model that markdown is context, not entity.

4. **Cross-holdout as standard CI check:** before publishing any new model, benchmark against every previous version's holdout. Distribution shift becomes detectable automatically.

5. **Real-world test set (Phase 1):** still pending. 50-100 manually-labeled real BR docs. Without this, no synthetic F1 is meaningful for production.

## Files touched in this post-mortem

- `spaces/privacy-filter-br-demo/{app.py, README.md}` — reverted to v3
- `br-pii-guardrail/{README.md, src/br_pii_guardrail/spacy_ner.py}` — reverted to v3
- `README.md` — comprehensive rewrite with cross-holdout findings prominently displayed
- `research/wiki/questions/2026-05-24-v4-postmortem.md` — this page
- `scripts/benchmark_v3_v4_cross.py` — the benchmark that revealed the issue
- `scripts/diagnose_v4_on_v3_holdout.py` — per-category diagnosis
- `scripts/diff_v4_vs_v3holdout_spans.py` — side-by-side boundary comparison

## Lessons (for the next time)

1. **A 99% F1 on your own holdout means "consistent with my training" not "good model".** Cross-holdout is the cheap honesty check.

2. **Single-source LM rewriters create stylistic monoculture in training data.** Mix at least 2-3 different rewriters or risk distribution overfitting.

3. **Markdown-formatted training text is a trap.** The model learns the markdown as part of entity context. Either strip during training or augment with markdown-stripped variants.

4. **Pause before publishing.** The user's instinct to ask "is this really better?" before PyPI saved deploying a regression to production users.

5. **Boundary metrics are not the same as classification metrics.** seqeval BIOES F1 conflates "wrong category" with "right category, wrong boundary". They have very different fix paths.

## Related

- [[model-evolution]] — living doc covering v3 → v3.1 → v4 → v5 → v6 → v7 (v4 section + Phase 2 labeler fix live there now)
- [[../sources/2026-05-23-lambada]] — LAMBADA explicitly warned about data drifting from single-source generation
