---
type: source
tags: [data-augmentation, ner, low-resource, biomedical, mention-replacement, token-replacement]
sources: 1
updated: 2026-05-23
arxiv: 2010.11683
authors: [Dai, Adel]
year: 2020
---

# An Analysis of Simple Data Augmentation for Named Entity Recognition

Dai & Adel, Bosch Center for AI + Sydney + CSIRO — arXiv [2010.11683](https://arxiv.org/abs/2010.11683)

> **Note:** Fetched while looking for "Karimi NER augmentation" (the reading list had a hallucinated arXiv ID). This Dai & Adel paper covers the same topic — simple data augmentation techniques for NER — and is the most-cited modern reference for the area. Likely what the reading list meant.

## Abstract (paraphrased)

Sentence-level NLP has well-studied augmentation (synonym replacement, EDA, back-translation). NER is token-level — does the same work? The authors design and compare four simple NER augmentation techniques on i2b2-2010 (biomedical) and MaSciP (materials science) datasets, across BiLSTM-CRF and BERT models. **All four boost performance in small-data regimes**; gains shrink as training data grows.

## The four augmentation techniques

1. **Label-wise Token Replacement (LwTR).** With probability p, replace each token with another token of the same NER label drawn from a per-label vocabulary. Preserves label sequence.
2. **Synonym Replacement (SR).** With probability p, replace each token with a WordNet synonym. Limited for low-resource languages without good WordNets.
3. **Mention Replacement (MR).** With probability p, replace each entity mention with another mention of the same entity type from the training set. Preserves sentence structure, swaps entity content.
4. **Shuffle within Segments (SiS).** Within each segment (consecutive tokens with same label), shuffle the order.

## Key findings

- **All four help on small data.** 50 training sentences, i2b2-2010: BERT baseline ~70 F1 → augmented ~75-77 F1. Larger absolute gains for BiLSTM than BERT.
- **Mention Replacement is the strongest single technique.** Especially on biomedical data where entity types are well-defined.
- **Combining all four > any single one.** Diminishing returns but still positive.
- **Past ~1000 sentences, gains shrink to <0.5 F1.** Augmentation is a low-resource trick, not a magic boost.
- **BERT benefits less than BiLSTM.** Pretrained models have already absorbed augmentation-like variation during pretraining. Still gain a little.

## How this relates to our work

This is the **augmentation paper closest to our concrete problem.** Several techniques are directly applicable to our v3 dataset construction or to v3.1 retraining:

1. **Mention Replacement is essentially what 4devs already does for us.** We swap valid CPFs/CNPJs/RGs in templates. We could extend this by swapping entity *values* within already-generated examples at training time (online augmentation) rather than only at generation time.
2. **Label-wise Token Replacement could help our "common-word names" failure mode.** If `private_person` includes names that look like common words ("Silva" = "silvae" = forest tree), LwTR over the person-label vocabulary would expose v3 to more name variants per sentence.
3. **Shuffle within Segments** doesn't apply directly to our case — most BR PII spans are short and order-preserving (CPF, CNPJ, address have specific format).
4. **Synonym Replacement requires a PT WordNet.** OpenWordnet-PT exists but is smaller than English WN. Could work for non-PII tokens to add general PT diversity.

## What we can actually use

- **Online MR during training.** At each training batch, randomly swap each entity span with another of the same type drawn from a per-category pool (we have one — 4devs profiles). Adds variability without regenerating the dataset.
- **Online LwTR for non-PII tokens.** Token-replacement within the "O" class to add lexical diversity, leaving entity spans untouched.
- **Skip SiS.** Doesn't help for our entity types.
- **Combination experiment.** Try MR + LwTR jointly. Paper shows combinations help.

## Direct mapping to our hypotheses

[[../questions/synthetic-data-quality]] H3 says: *"Adding 5-10k examples with controlled noise (OCR-like) would generalize better than adding 50k more clean examples."* Dai-Adel's MR is a form of controlled noise — replacing entity values while preserving structure. **Their empirical results back this hypothesis up.**

[[../questions/person-detection-failure-modes]]: their MR specifically addresses entity coverage by sampling more diverse mentions per type. Could improve recall on long-tail names.

## Concerns / caveats

- **Tested on biomedical only.** Generalization to BR PII is plausible but not empirical.
- **2020 paper, BERT-base baseline.** Modern backbones (DeBERTa, BERTimbau) already absorb more. Gains may be smaller for us.
- **"Small data" is <1000 sentences. We have 54k.** We're outside the regime where MR gives large gains. But for under-represented categories (private_phone, private_voucher), we may have effectively <500 examples — augmentation could help there.

## Open questions

- Per-category augmentation policy: should we MR-augment only under-represented categories?
- Online (per-batch) vs offline (dataset expansion) augmentation: which is more practical for our pipeline?
- Does MR interact with our existing LM-based generation? Doing both might be redundant or compounding — needs experimentation.

## Direct contrast with LAMBADA

| | Dai & Adel (this paper) | LAMBADA ([[../sources/2026-05-23-lambada]]) |
| --- | --- | --- |
| Method | Surface-level token/mention swaps | Generate full sentences with fine-tuned LM |
| Task | NER (token-level) | Text classification (sentence-level) |
| Cost | ~free, online | Fine-tune LM, generate, filter |
| Best regime | <1000 examples | <100 examples |
| Quality control | None — assumes label preservation | Classifier-confidence filter |

For NER specifically, Dai & Adel is more applicable than LAMBADA because mention replacement preserves token labels deterministically. LAMBADA generates whole sentences which need re-labeling.

## Related

- [[../concepts/lm-based-data-augmentation]] — orthogonal augmentation paradigm
- [[../concepts/synthetic-data-filtering]] — augmentation needs no filter when labels are preserved by construction
- [[../questions/synthetic-data-quality]] — directly addresses H3
- [[../questions/person-detection-failure-modes]]
- [[../sources/2026-05-23-lambada]]
