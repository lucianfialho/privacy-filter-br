# Research Wiki Index

Knowledge base for improving `privacy-filter-br-v3` (BR PII detection NER) and the `br-pii-guardrail` library.

See [[CLAUDE]] for schema and conventions, [[overview]] for the running thesis.

## Sources (ingested papers)

### Architecture / NER paradigms
- [[wiki/sources/2026-05-23-gliner]] — GLiNER: Generalist Model for NER using Bidirectional Transformer (Zaratiana et al, 2023)
- [[wiki/sources/2026-05-23-spanbert]] — SpanBERT: Improving Pre-training by Representing and Predicting Spans (Joshi et al, 2019)
- [[wiki/sources/2026-05-23-luke]] — LUKE: Deep Contextualized Entity Representations (Yamada et al, 2020)
- [[wiki/sources/2026-05-23-template-ner-bart]] — Template-Based NER Using BART (Cui et al, 2021)

### PII detection at inference time (LLM-based)
- [[wiki/sources/2026-05-23-deid-gpt]] — DeID-GPT: Zero-shot Medical Text De-Identification by GPT-4 (Liu et al, 2023)

### Synthetic data + augmentation
- [[wiki/sources/2026-05-23-lambada]] — Do Not Have Enough Data? LAMBADA (Anaby-Tavor et al, 2019)
- [[wiki/sources/2026-05-23-dai-ner-augmentation]] — Simple Data Augmentation for NER (Dai & Adel, 2020)

### PT-BR pretraining context
- [[wiki/sources/2026-05-23-ptt5]] — PTT5: T5 on BR Portuguese (Carmo et al, 2020)

## Concepts

- [[wiki/concepts/span-based-ner]] — predicting (start, end, label) triples
- [[wiki/concepts/zero-shot-ner]] — supplying type names at inference time
- [[wiki/concepts/span-boundary-objective]] — SpanBERT's SBO loss
- [[wiki/concepts/pretraining-objectives-for-ner]] — MLM vs WWM vs span vs RTD
- [[wiki/concepts/llm-based-pii-detection]] — LLM-as-detector paradigm
- [[wiki/concepts/prompt-engineering-for-pii]] — DeID-GPT's 3-part template
- [[wiki/concepts/lm-based-data-augmentation]] — LAMBADA-style generate+filter
- [[wiki/concepts/synthetic-data-filtering]] — classifier-confidence filtering

## Entities

### Models
- [[wiki/entities/gliner-model]] — GLiNER (DeBERTa-v3, span-based, zero-shot)
- [[wiki/entities/spanbert-model]] — SpanBERT (English, span-aware pretraining)

### Datasets
- [[wiki/entities/i2b2-uthealth-dataset]] — clinical de-id benchmark

## Open Questions

- [[wiki/questions/bioes-vs-span-based]] — informed by GLiNER + SpanBERT + LUKE + Template-NER
- [[wiki/questions/person-detection-failure-modes]] — Dai-Adel mention-replacement is directly applicable
- [[wiki/questions/synthetic-data-quality]] — DeID-GPT validation + LAMBADA filter + Dai-Adel mention-replacement
- [[wiki/questions/2026-05-23-direction-a-audit-results]] — 4-audit findings: labeler bug confirmed, 99.30% v3 disagreement
- [[wiki/questions/2026-05-23-v3.1-training-results]] — v3.1 trained on relabeled dataset; 2×2 benchmark confirms hypothesis
- [[wiki/questions/2026-05-24-v4-full-training-results]] — v4 training results (⚠️ superseded by post-mortem)
- [[wiki/questions/2026-05-24-v4-postmortem]] — cross-holdout reveals v4 overfit; v3 became default temporarily
- [[wiki/questions/2026-05-24-v5-results]] — v5 (mixed Haiku + gpt5nano) generalizes both styles
- [[wiki/questions/2026-05-25-v6-results]] — v6 (+ 10 narrative templates) fixes template-coverage failures; new production default
- [[wiki/questions/initial-reading-list]] — tracker (note: 5 original IDs were hallucinated, see notes)

## Strategic synthesis (where the papers point us)

After ingesting 8 papers, three concrete directions for v4 emerged:

### Direction A — Filter the existing dataset
**Cost:** low. **Source:** [[wiki/sources/2026-05-23-lambada]] + [[wiki/sources/2026-05-23-dai-ner-augmentation]].

Run audits on `dataset_br_v3.jsonl`: (1) round-trip check (does gpt-5-nano drop PII in rewriting?), (2) v3-disagreement audit, (3) per-category count audit. Filter or augment based on findings. See [[wiki/questions/synthetic-data-quality]] for the 4 concrete experiments.

### Direction B — Switch decoder to span-based + zero-shot capable
**Cost:** medium. **Source:** [[wiki/sources/2026-05-23-gliner]] + [[wiki/sources/2026-05-23-luke]] + [[wiki/sources/2026-05-23-template-ner-bart]].

Replace BIOES with span enumeration + matching. Best candidate: fine-tune GLiNER-Multi on our 54k dataset, compare with v3. Resolves fragmentation and adds zero-shot extensibility for new PII categories. Estimated cost: 5h A100 + data loader wiring.

### Direction C — Switch backbone to span-aware or entity-aware pretraining
**Cost:** high. **Source:** [[wiki/sources/2026-05-23-spanbert]] + [[wiki/sources/2026-05-23-luke]] + [[wiki/sources/2026-05-23-ptt5]].

Continual pretraining of BERTimbau (or PT-base alternatives) with SpanBERT-style SBO objective or LUKE-style entity-aware attention. No public PT variant exists for either — would need to train from scratch on BrWac. Estimated cost: 50-200h A100 GPU.

**Recommended order:** A → B → C. Cheap wins first.

## Recommended starting sources (status)

- ~~GLiNER~~ ✅
- ~~SpanBERT~~ ✅
- Microsoft Presidio (deferred — needs GitHub docs)
- ~~DEID-GPT~~ ✅
- ~~LAMBADA~~ ✅
- ~~BERTimbau~~ → substituted by PTT5
- ~~LUKE~~ ✅
- ~~Karimi augmentation~~ → substituted by Dai & Adel
- ~~Few-shot NER (Template-NER w/ BART)~~ ✅
- Tier 3 papers: mostly deferred due to wrong/missing arXiv IDs

See [[wiki/questions/initial-reading-list]] for full status.
