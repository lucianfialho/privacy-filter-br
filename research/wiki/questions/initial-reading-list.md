---
type: question
tags: [reading-list, planning, bootstrap]
updated: 2026-05-23
---

# Initial reading list

~15 papers to ingest first, each tied to a hypothesis from [[overview]]. Order is rough priority. Use paper7 to fetch.

## Tier 1 — closest to our problems

1. **GLiNER (Zaratiana et al, 2024)** — `2311.08526`
   Zero-shot NER via prompt-style entity types. If results transfer to PT-BR, it could replace our BIOES head for new categories without retraining.

2. **SpanBERT (Joshi et al, 2020)** — `1907.10529`
   Pretraining objective designed for span representations. Relevant to our [[span-based-vs-bioes]] question.

3. **Microsoft Presidio architecture** — github docs, no arXiv
   How they compose recognizers + NER + anonymizers. Our [[br-pii-guardrail]] is a BR version of this. What are they doing differently?

4. **DEID-GPT (Liu et al, 2023)** — `2303.11032`
   GPT-4 for clinical de-identification. Validates LLM-based PII detection. Compares with rule-based and BERT.

5. **Anaby-Tavor "Do Not Have Enough Data?"** — `1911.03118`
   Synthetic data via LLM rewriting (exactly what we did). What are the established quality controls?

## Tier 2 — adjacent / supporting

6. **BERTimbau (Souza et al, 2020)** — `2002.06424`
   Our base model. Need to understand its pretraining corpus and known biases for NER.

7. **LUKE (Yamada et al, 2020)** — `2010.01057`
   Entity-aware transformer. Better at entity-level tasks than BERT. Could it replace BERTimbau for v4?

8. **AssIN/PT-NER benchmarks**
   Existing PT-BR NER datasets/benchmarks. We've trained on synthetic only — what would a fair benchmark look like?

9. **PII Masker (Chen et al, 2022)** — find arXiv id
   Adversarial robustness in PII detection. How easy is it to fool a NER guardrail?

10. **Karimi "Augmentation for NER"** — `2007.10760`
    Data augmentation techniques specific to NER (entity replacement, mention swap, etc).

## Tier 3 — for later

11. **Calibration of NER models** — `2004.10193`
    Token-level confidence calibration. Relates to our threshold tuning for `[[private_phone]]` (lowest F1).

12. **OpenAI Privacy Filter model card**
    Their architecture (1.5B MoE, Viterbi decoding). We deliberately picked a smaller model — what trade-offs did we make?

13. **Distant supervision NER** — `1607.00501` (Shang et al)
    Generating NER labels without manual annotation. Our pipeline is exactly this. State of art?

14. **CRF / Viterbi decoding for NER** — Lafferty et al 2001 (CRF), Bidirectional inference papers
    OpenAI uses Viterbi for span coherence. Could fix our BIOES fragmentation issue.

15. **Few-shot NER (Cui et al, 2021)** — `2106.01760`
    Template-based few-shot. Useful for adding new categories (e.g. PIX key, MAC address) without retraining.

## How to ingest

```bash
# Example: ingest GLiNER paper
cd research/
paper7 fetch 2311.08526 --output raw/

# Then in Claude:
# "Read raw/2311.08526.md and ingest into wiki per CLAUDE.md."
```

## Tracking

| # | Paper | Status | Date | Notes |
|---|---|---|---|---|
| 1 | GLiNER | pending | | |
| 2 | SpanBERT | pending | | |
| 3 | Presidio | pending | (no arXiv, find docs) | |
| 4 | DEID-GPT | pending | | |
| 5 | Anaby-Tavor | pending | | |
| 6 | BERTimbau | pending | | |
| 7 | LUKE | pending | | |
| 8 | PT-NER benchmarks | pending | (lookup) | |
| 9 | PII Masker | pending | (lookup) | |
| 10 | Karimi augmentation | pending | | |
| 11 | NER calibration | pending | | |
| 12 | OpenAI PF model card | pending | (no arXiv) | |
| 13 | Distant supervision | pending | | |
| 14 | CRF/Viterbi | pending | | |
| 15 | Few-shot NER | pending | | |
