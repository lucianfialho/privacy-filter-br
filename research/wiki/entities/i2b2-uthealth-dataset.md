---
type: entity
tags: [dataset, benchmark, de-identification, clinical, synthetic-surrogates]
sources: 1
updated: 2026-05-23
kind: dataset
---

# i2b2/UTHealth 2014 De-Identification Challenge

The canonical benchmark for clinical-text de-identification. Used by DeID-GPT and most published de-id baselines (ClinicalBERT, RoBERTa fine-tunes, etc).

- **Origin:** i2b2 = "Informatics for Integrating Biology and the Bedside", a network funded by NIH. UTHealth = University of Texas Health Science Center at Houston.
- **Year:** 2014 shared task.
- **Access:** Restricted — Blavatnik Institute of Biomedical Informatics at Harvard grants access on request.
- **Standard:** HIPAA 18-category PHI.

## Composition

- **1,304 free-form clinical notes** from 296 diabetic patients.
- Notes include physician consultation, nursing assessments, discharge reports, procedure/operative reports, radiology/pathology reports.
- All PHI manually annotated by humans, then **replaced with realistic surrogates** before release.

**Example:**
- Original: "Mr. James McCarthy visited on 12/01/2013"
- Released: "Mr. Joshua Howard visited on 04/01/2060"

Names, professions, locations, ages, dates, contacts, IDs — all swapped to plausible-looking fake values.

## Why surrogate data matters for us

The i2b2 dataset is **fully synthetic at the PHI level** (real notes with fake PII inserted). Yet it's the accepted benchmark for de-id research. This is **direct external validation** of the synthetic-data approach we use in `dataset_br_v3.jsonl`:

- 4devs API generates valid-format BR PII (CPF/CNPJ/RG with checksum) — same as i2b2's surrogate names.
- Jinja templates generate document scaffolding (NF, contrato, holerite, etc) — analogous to clinical-note structure.
- GPT-5-nano rewrites for naturalness — i2b2 keeps real linguistic structure.

The community accepts i2b2 as a credible benchmark, which weakens the critique that our v3 holdout F1 of 0.9934 is "just synthetic". It's the same level of syntheticness as the gold-standard de-id benchmark.

## Reported baseline performance on i2b2

From DeID-GPT (Table 2):

| Method | Accuracy |
| --- | --- |
| BERT (fine-tuned) | 0.798 |
| RoBERTa (fine-tuned) | 0.947 |
| ClinicalBERT (fine-tuned) | 0.974 |
| ChatGPT (implicit prompt) | 0.686 |
| ChatGPT (explicit prompt) | 0.929 |
| GPT-4 (implicit prompt) | 0.908 |
| GPT-4 (explicit prompt) | 0.99 |
| mT0 | 0.824 |
| Falcon-7b | 0.605 |
| Llama-1-7b | 0.609 |
| Llama-2-7b | 0.612 |
| Flan-T5-base | 0.737 |

**Caveat:** "accuracy" here = (TP+TN)/(TP+TN+FP+FN). This is not standard NER F1. The numbers are not directly comparable to our v3's 0.9934 F1 macro on `dataset_br_v3_holdout.jsonl`. But the ordering across methods is informative.

## Limitations as a benchmark

- **English-only.** No BR/PT equivalent of comparable scope exists. (Worth checking: is there a public PT de-id dataset?)
- **Clinical domain only.** Doesn't cover financial, legal, e-commerce text — which is our actual use case.
- **HIPAA categories only.** Missing CPF, CNPJ, CEP, voter ID, etc. Not LGPD-aligned.
- **Surrogate replacement loses some signal.** Statistical patterns in surrogate values may differ from real ones (e.g., dates clustered around release year).
- **Access is gated.** Cannot just download and run experiments. Useful for reading published numbers, not for replication.

## How we should use this

1. **As a reference for published baselines.** "ClinicalBERT 0.974, GPT-4 0.99" gives us a sanity check for what's possible.
2. **As a justification for our synthetic-data approach.** When stakeholders question whether synthetic surrogates are credible, point to i2b2.
3. **As a template for our own evaluation protocol.** Per-HIPAA-category accuracy is a thin metric; we should do per-LGPD-category F1, precision, recall, and a confusion matrix.

## Open questions

- Is there a published PT-BR equivalent of i2b2? Worth searching.
- Does Harvard grant access to non-clinical researchers? We could potentially do cross-lingual transfer experiments.
- What's the standard evaluation script everyone uses for i2b2? Is it released?

## Related

- [[../sources/2026-05-23-deid-gpt]] — uses this as the only benchmark
- [[../concepts/llm-based-pii-detection]]
- [[../questions/synthetic-data-quality]] — i2b2 is external validation for synthetic-PII data
- [[hipaa-vs-lgpd]] — category mapping (TBD)
- [[clinicalbert]] — TBD
