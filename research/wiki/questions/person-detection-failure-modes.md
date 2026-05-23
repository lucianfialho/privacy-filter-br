---
type: question
tags: [person-ner, failure-modes, dataset-quality]
sources: 0
updated: 2026-05-23
---

# Why does `private_person` fail on free text?

## Observation

In v3 sanity tests, "Cliente João Silva" yielded only "Silva" detected (score 0.531), missing "João". On benchmark (synthetic), F1 was 0.9867 — best in class for our model, yet visibly weakest on real-feeling inputs.

Related: spaCy `pt_core_news_lg` correctly tagged "Cliente João Silva" as MISC (full span including prefix), and after stripping the prefix the lib gets the right answer. So **spaCy is doing what our model should be doing.**

## Failure modes catalog

Each will be tested + documented as we ingest papers and run more experiments.

1. **First-name dropout.** "João Silva" → only "Silva". Hypothesis: in 4devs synthetic data, first names appear ~always after a vocative ("Cliente X", "Sr. X"), so the model learns to mark only the trailing surname.

2. **Compound surnames.** "Maria de Souza Pereira" (4 tokens). BIOES requires B-I-I-E sequence. If any token classifies as O, the span breaks. Test: how often does the middle "de" get classified as O?

3. **Names that look like common words.** Brazilian first names include "Pedra", "Sol", "Lua", "Sereia". Model may classify them as O because pretraining never saw them as PER.

4. **Single-token names.** "Madonna", "Pelé", "Neymar" — no BIOES `S-PERSON` was emitted in our test set. Check: does our training data include single-token person names?

5. **Names + title.** "Dr. Pedro", "Sr. Silva" — spaCy includes the title in the span. Our model? Unclear, needs eval.

## Why does spaCy do better?

Likely: spaCy `pt_core_news_lg` pretrained on a much larger and more naturalistic PT corpus (Wikipedia, CONLL). Our synthetic data has a narrow distribution of how names appear in text — always as a leaf in a Jinja template.

## Experiments to run

1. **Generate 1k examples with names in unusual positions** (start of sentence, mid-clause, possessive, etc) and retrain. Measure delta on holdout that has the same diversity.
2. **Mix in spaCy-labeled CONLL-PT data** as additional silver labels. Distant supervision approach.
3. **Replace BIOES with span-based prediction** (biaffine) — would solve fragmentation but adds complexity.

## Linked papers (when ingested)

- [[2026-XX-XX-gliner]] — does zero-shot improve recall on unseen person variations?
- [[2026-XX-XX-karimi-augmentation]] — entity replacement augmentation specifically for names
- [[2026-XX-XX-anaby-tavor]] — quality control in LLM-generated synthetic data
