# Research Log

Chronological record of ingests, queries, and lint passes.

Format: `## [YYYY-MM-DD] <type> | <subject>` where `<type>` is one of: `ingest | query | lint | scaffold`.

To get the last N entries: `grep "^## \[" log.md | tail -N`.

---

## [2026-05-23] scaffold | Initial wiki structure

- Created `research/` following the Karpathy LLM-wiki pattern.
- Schema: `CLAUDE.md` with ingest/query/lint operations.
- Layout: `raw/`, `wiki/{sources,concepts,entities,questions}`, `index.md`, `log.md`, `overview.md`.
- Bootstrapped initial research questions in `wiki/questions/`:
  - `bioes-vs-span-based.md`
  - `person-detection-failure-modes.md`
  - `synthetic-data-quality.md`
  - `initial-reading-list.md` — ~15 papers to ingest first via paper7.
- No sources ingested yet.

## [2026-05-23] ingest | GLiNER: Generalist Model for NER using Bidirectional Transformer

- paper7 id: 2311.08526 (Zaratiana et al, 2023)
- raw: `raw/2311.08526-gliner.md` (177 lines)
- touched:
  - `wiki/sources/2026-05-23-gliner.md` (new)
  - `wiki/concepts/span-based-ner.md` (new)
  - `wiki/concepts/zero-shot-ner.md` (new)
  - `wiki/entities/gliner-model.md` (new)
  - `wiki/questions/bioes-vs-span-based.md` (linked source + new concept/entity pages)
  - `index.md` (added source + concepts + entity, marked GLiNER as ingested)
- key takeaways:
  - Span-based architecture eliminates BIOES fragmentation by design — directly relevant to our CPF/CNPJ/cartão sub-token issue.
  - Zero-shot type matching lets us add new PII categories (PIX key, voter ID, CTPS) without retraining.
  - GLiNER-Multi (mdeBERTa-v3-base, ~280M) supports PT but is NOT benchmarked on PT-BR in the paper. Needs our own eval.
  - K=12 max span length flagged as risk for BR addresses.
  - 50% negative entity sampling at training time is critical for precision/recall balance.
- proposed v4 experiment: fine-tune GLiNER-Multi on our 54k synthetic dataset, compare F1+latency vs v3 BERTimbau+BIOES. Cost ~5h A100.
- next paper: SpanBERT (1907.10529) — Tier 1, fills the supervised-span-pretraining gap GLiNER references.

## [2026-05-23] ingest | SpanBERT: Improving Pre-training by Representing and Predicting Spans

- paper7 id: 1907.10529 (Joshi et al, 2019)
- raw: `raw/1907.10529-spanbert.md` (244 lines)
- touched:
  - `wiki/sources/2026-05-23-spanbert.md` (new)
  - `wiki/concepts/span-boundary-objective.md` (new)
  - `wiki/concepts/pretraining-objectives-for-ner.md` (new)
  - `wiki/entities/spanbert-model.md` (new)
  - `wiki/concepts/span-based-ner.md` (linked SpanBERT entity)
  - `wiki/questions/bioes-vs-span-based.md` (linked source, reframed question)
  - `index.md` (added source + 2 concepts + 1 entity, marked SpanBERT as ingested)
- key takeaways:
  - Two innovations: (a) mask **contiguous spans** instead of single tokens, (b) **Span Boundary Objective (SBO)** that predicts each masked token from boundary representations only.
  - **Single-sequence training beats NSP.** NSP appears net-harmful for span tasks.
  - **WWM is partial span-awareness.** BERTimbau (our backbone) uses WWM but not span masking or SBO. There's a real gap.
  - From ablations: WWM gains +0.5-1.0 over plain MLM; span masking adds another +1.0; SBO adds another +0.7-2.7 (biggest gain on coref).
  - **No public PT or multilingual SpanBERT.** Would have to train it.
- reframed question: our [[bioes-vs-span-based]] debate is partly upstream of our v3 architecture. Even with a span-based decoder (like GLiNER), the backbone needs to be span-aware to maximize gains.
- proposed v4 experiments (in order of cost):
  1. Add auxiliary SBO-style head during NER fine-tuning. Cheap, novel.
  2. Test SpanBERT-base-cased (English) as cross-lingual baseline. ~1h.
  3. Continual pretrain BERTimbau with span masking + SBO on PT-BR corpus. ~50-200h GPU.
- next paper: Microsoft Presidio architecture (Tier 1) — closest production reference for our use case; or DEID-GPT (2303.11032) for clinical PII with synthetic data parallels.

## [2026-05-23] ingest | DeID-GPT: Zero-shot Medical Text De-Identification by GPT-4

- paper7 id: 2303.11032 (Liu et al, 2023)
- raw: `raw/2303.11032-deid-gpt.md` (280 lines)
- skipped Tier-1 item #3 (Microsoft Presidio) — no arXiv, needs separate docs-based ingest workflow. Marked "deferred" in reading list tracker.
- touched:
  - `wiki/sources/2026-05-23-deid-gpt.md` (new)
  - `wiki/concepts/llm-based-pii-detection.md` (new)
  - `wiki/concepts/prompt-engineering-for-pii.md` (new)
  - `wiki/entities/i2b2-uthealth-dataset.md` (new)
  - `wiki/questions/synthetic-data-quality.md` (linked DeID-GPT as external validation)
  - `wiki/questions/initial-reading-list.md` (status tracker updated)
  - `index.md` (added source + 2 concepts + 1 entity)
- key takeaways:
  - GPT-4 + **explicit 3-part HIPAA prompt** = 0.99 entity accuracy on i2b2, beats fine-tuned ClinicalBERT (0.974), RoBERTa (0.947), BERT (0.798).
  - **Prompt engineering is the dominant lever** — ChatGPT goes from 0.686 (implicit) to 0.929 (explicit), +24 points.
  - **Small open-source LLMs fail catastrophically.** Llama-2-7b = 0.612, Falcon-7b = 0.605. Failure mode: "fundamental misunderstanding of the task". Validates our choice to fine-tune a small encoder rather than instruction-tune a small LLM.
  - **i2b2/UTHealth is fully PHI-synthetic** (real notes, surrogate identifiers) — direct external validation for our synthetic-data pipeline. Counters "your benchmark is just synthetic" critique.
  - **Data residency is the killer.** Paper itself flags that GPT-4 API cannot be used in hospital settings. Same applies to LGPD/BR — validates our on-device architecture.
  - **Metric caveat:** paper uses accuracy (TP+TN)/(TP+TN+FP+FN), which inflates with abundant true negatives. Not directly comparable to NER F1.
- 3-part prompt template captured in [[../concepts/prompt-engineering-for-pii]] with a draft PT-BR adaptation for LGPD categories.
- proposed experiments (low cost):
  1. Run GPT-4 with PT-BR LGPD prompt on `dataset_br_v3_holdout.jsonl`. Compare against v3. Estimated cost: ~$30-50 in API.
  2. Use GPT-4 as a labeler on real (public) BR documents to bootstrap real-data training set.
  3. Add optional opt-in LLM fallback layer to `br-pii-guardrail` (off by default, configurable provider).
- next paper: Anaby-Tavor "Do Not Have Enough Data?" (1911.03118) — synthetic data quality controls. Directly addresses [[../questions/synthetic-data-quality]] hypotheses.

## [2026-05-23] ingest | LAMBADA: Do Not Have Enough Data? Deep Learning to the Rescue (Anaby-Tavor et al, 2019)

- paper7 id: 1911.03118
- raw: `raw/1911.03118-anaby-tavor.md` (210 lines)
- touched:
  - `wiki/sources/2026-05-23-lambada.md` (new)
  - `wiki/concepts/lm-based-data-augmentation.md` (new)
  - `wiki/concepts/synthetic-data-filtering.md` (new)
  - `wiki/questions/synthetic-data-quality.md` (linked LAMBADA + 4 concrete next experiments)
  - `wiki/questions/initial-reading-list.md` (status tracker updated)
  - `index.md` (added source + 2 concepts)
- key takeaways:
  - **LAMBADA pipeline = our pipeline + filter step.** Fine-tune LM → generate 10× → **filter via baseline classifier** → retrain. We do generation but skip filtering. **This is the architectural gap.**
  - **The filter is essential, not optional.** Generate 10× more than needed and throw away 90% if the baseline disagrees.
  - **Filtering works across classifier types** (SVM, LSTM, BERT). Method is architecture-agnostic.
  - **Biggest gains in small-data regime** (5-100 examples/class). We're at ~2500/class — **doesn't help our expansion**, but **filter insight transfers**.
  - **LAMBADA beats real unlabeled data with weak labeling.** Surprising — synthetic+filtered > unlabeled+weak-label. Suggests LM injects useful prior.
  - **Per-category re-balancing is a free win.** They generate exactly N_y per class, handling imbalance directly. We let templates determine ratios.
- concrete next experiments captured in [[synthetic-data-quality]]:
  1. Round-trip filter audit (does gpt-5-nano drop inserted PII?)
  2. v3 disagreement audit (where does v3 disagree with string-match labels?)
  3. Per-category count audit (which categories are under-represented?)
  4. Conditional filter retrain if audits show >5% problematic examples
- comparison with DeID-GPT: LAMBADA puts the LLM at TRAINING time; DeID-GPT puts it at INFERENCE time. Opposite roles. LAMBADA's pattern is more applicable to our local-small-model architecture.
- next paper: pivot from synthetic-data thread back to architecture — **BERTimbau original paper (2002.06424)** to understand our backbone's pretraining specifics; or **LUKE (2010.01057)** for entity-aware transformer as v4 backbone candidate.

## [2026-05-23] ingest | Mass-ingest: LUKE + Template-NER + PTT5 + Dai-Adel NER augmentation

**User request:** "ingesta logo a porra toda". Batch-ingest of Tier 2/3 reading list.

### Reading list correction

5 of 6 attempted arXiv IDs in the original reading list were **hallucinated** (LLM-invented IDs pointing to unrelated papers). Confirmed wrong:
- 2002.06424 (BERTimbau) → Crone, joint NER/RE
- 2009.05028 (BERTimbau alt) → quantum annealing
- 2007.10760 (Karimi) → backdoor attacks survey
- 2004.10193 (NER calibration) → astrophysics
- 1607.00501 (distant supervision) → image classification

Only 2010.01057 (LUKE) was correct. Found two relevant replacements during search: PTT5 (2008.09144) and Dai-Adel (2010.11683). Tier-3 ID 2106.01760 (Cui Template-NER) was also correct.

### Papers ingested in this batch

1. **[[../sources/2026-05-23-luke]]** — LUKE: entity-aware self-attention + entities as tokens. SOTA on CoNLL-03 (94.3 F1) and 4 other entity tasks. English-only; no PT variant exists. **Direction C candidate** (backbone replacement) if we had a PT-LUKE.

2. **[[../sources/2026-05-23-template-ner-bart]]** — Cui et al Template-NER: re-frame NER as seq2seq LM template scoring. Conceptual ancestor of GLiNER, much slower at inference. Useful as historical context; GLiNER strictly dominates for our use case.

3. **[[../sources/2026-05-23-ptt5]]** — PTT5 (UNICAMP/NeuralMind): T5 pretrained on BrWac. Same group as BERTimbau. Confirms PT-monolingual > multilingual for PT-BR tasks. Anchor reference for any future PT-BR continual pretraining.

4. **[[../sources/2026-05-23-dai-ner-augmentation]]** — Dai & Adel: 4 simple NER augmentations (LwTR, SR, MR, SiS). **Mention Replacement is directly applicable to our pipeline as online augmentation** during fine-tuning. Validates [[../questions/synthetic-data-quality]] H3 (controlled noise > more clean data).

### Touched files

- `wiki/sources/2026-05-23-luke.md` (new)
- `wiki/sources/2026-05-23-template-ner-bart.md` (new)
- `wiki/sources/2026-05-23-ptt5.md` (new)
- `wiki/sources/2026-05-23-dai-ner-augmentation.md` (new)
- `wiki/questions/initial-reading-list.md` (full rewrite with status corrections + "5 IDs hallucinated" note)
- `index.md` (8 sources now indexed; added 3-direction strategic synthesis: filter, switch decoder, switch backbone)

### Strategic synthesis after 8 papers

Three v4 directions, ranked by cost:

- **A — Filter existing dataset.** Cheap. Sources: LAMBADA + Dai-Adel. Action: 4 audits on dataset_br_v3.jsonl (round-trip check, v3-disagreement, per-category count, filter+retrain). See [[../questions/synthetic-data-quality]].
- **B — Switch decoder to span-based + zero-shot.** Medium. Sources: GLiNER + LUKE + Template-NER. Action: fine-tune GLiNER-Multi on our 54k dataset, ~5h A100.
- **C — Switch backbone to span-aware or entity-aware PT.** High. Sources: SpanBERT + LUKE + PTT5. Action: continual pretrain BERTimbau on BrWac with SBO objective, ~50-200h A100.

Recommended order: A → B → C.

### Still deferred (need correct IDs or alternative source)

- BERTimbau itself (Souza et al, BRACIS 2020 — likely no arXiv preprint)
- Microsoft Presidio (no arXiv — GitHub docs)
- PII Masker, AssIN benchmarks, OpenAI PF (no IDs)
- NER calibration / distant supervision / CRF-Viterbi (wrong IDs, need lookup)

### Lessons learned

- Always verify arXiv IDs by reading first ~10 lines of paper7 output. Title + authors + year on lines 2-5 confirm match.
- LLM-drafted reading lists need cross-verification (paper title + first author) before fetch.
