---
type: question
tags: [reading-list, planning, bootstrap]
updated: 2026-05-23
---

# Initial reading list

~15 papers to ingest first, each tied to a hypothesis from [[overview]]. Order is rough priority. Use paper7 to fetch.

## Status note (2026-05-23)

**5 of the original arXiv IDs in this list were hallucinated** (the LLM that drafted the list invented plausible-looking IDs that pointed to unrelated papers). Confirmed wrong:

- BERTimbau "2002.06424" → actually Crone, joint NER/RE
- BERTimbau alternate "2009.05028" → quantum annealing (Pelofske et al)
- Karimi "2007.10760" → backdoor attacks survey (Gao et al)
- NER calibration "2004.10193" → astrophysics (Wittor et al)
- Distant supervision "1607.00501" → image classification (Dong et al)

**Replacements found and ingested:**

- Instead of BERTimbau → [[../sources/2026-05-23-ptt5]] (same UNICAMP/NeuralMind group, PT-BR T5)
- Instead of Karimi → [[../sources/2026-05-23-dai-ner-augmentation]] (Dai & Adel 2020, simple NER augmentation — likely what list intended)

**Still missing (need correct IDs or alternative search):**

- BERTimbau original paper itself (Souza et al, BRACIS 2020 — may not have arXiv)
- NER calibration / token confidence (search for "Jiang et al confidence NER" or similar)
- Distant supervision NER (Shang et al original)
- CRF/Viterbi (Lafferty 2001 — pre-arXiv era)
- PII Masker, AssIN benchmarks, OpenAI PF — no arXiv IDs in list

## Tier 1 — closest to our problems

1. **GLiNER** — `2311.08526` ✅ ingested
2. **SpanBERT** — `1907.10529` ✅ ingested
3. **Microsoft Presidio** — no arXiv, deferred (needs GitHub docs ingest)
4. **DEID-GPT** — `2303.11032` ✅ ingested
5. **Anaby-Tavor LAMBADA** — `1911.03118` ✅ ingested

## Tier 2 — adjacent / supporting

6. **BERTimbau** — original ID was wrong; replaced with PTT5 ([[../sources/2026-05-23-ptt5]]) as proxy for UNICAMP/NeuralMind PT-BR pretraining context. True BERTimbau paper search still pending.
7. **LUKE** — `2010.01057` ✅ ingested
8. **AssIN/PT-NER benchmarks** — no specific ID, deferred
9. **PII Masker** — no arXiv ID provided, deferred
10. **Dai & Adel NER augmentation** — `2010.11683` ✅ ingested (replaces "Karimi")

## Tier 3 — for later

11. **NER calibration** — original ID was wrong, deferred
12. **OpenAI Privacy Filter** — no arXiv, deferred
13. **Distant supervision NER** — original ID was wrong, deferred
14. **CRF / Viterbi** — pre-arXiv, deferred (textbook material)
15. **Few-shot NER (Cui et al Template-NER)** — `2106.01760` ✅ ingested (correct ID, was Tier 3)

## Tracking

| # | Paper | Status | Date | Notes |
|---|---|---|---|---|
| 1 | GLiNER | ✅ ingested | 2026-05-23 | [[../sources/2026-05-23-gliner]] |
| 2 | SpanBERT | ✅ ingested | 2026-05-23 | [[../sources/2026-05-23-spanbert]] |
| 3 | Presidio | deferred | | no arXiv; needs GitHub docs ingest |
| 4 | DEID-GPT | ✅ ingested | 2026-05-23 | [[../sources/2026-05-23-deid-gpt]] |
| 5 | Anaby-Tavor (LAMBADA) | ✅ ingested | 2026-05-23 | [[../sources/2026-05-23-lambada]] |
| 6 | BERTimbau | substituted | 2026-05-23 | PTT5 ([[../sources/2026-05-23-ptt5]]) used as proxy |
| 7 | LUKE | ✅ ingested | 2026-05-23 | [[../sources/2026-05-23-luke]] |
| 8 | PT-NER benchmarks | deferred | | needs lookup |
| 9 | PII Masker | deferred | | needs lookup |
| 10 | Karimi augmentation | substituted | 2026-05-23 | Dai & Adel ([[../sources/2026-05-23-dai-ner-augmentation]]) used as proxy |
| 11 | NER calibration | deferred | | wrong ID, needs lookup |
| 12 | OpenAI PF model card | deferred | | no arXiv |
| 13 | Distant supervision | deferred | | wrong ID, needs lookup |
| 14 | CRF/Viterbi | deferred | | pre-arXiv |
| 15 | Few-shot NER (Template-NER w/ BART) | ✅ ingested | 2026-05-23 | [[../sources/2026-05-23-template-ner-bart]] |

## How to ingest

```bash
# Example: ingest a paper
cd research/
paper7 get <arxiv-id> --detailed > raw/<id>-<slug>.md

# Then in Claude:
# "ingest o próximo paper da reading list"
```

## Lessons learned

- **Always verify arXiv IDs before fetching.** First 4 lines of paper7 output show the actual title and authors — check these match expectations before writing a wiki page.
- **The original reading list was LLM-generated.** Several IDs were plausible-looking but invented. Future reading lists should include the paper title + first-author last name to enable cross-check.
