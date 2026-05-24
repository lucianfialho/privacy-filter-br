---
type: concept
tags: [synthetic-data, filtering, self-distillation, dataset-quality]
sources: 1
updated: 2026-05-23
---

# Synthetic data filtering (classifier-confidence)

A quality-control step where a baseline model trained on a subset of synthetic data is used to *score* the rest of the synthetic pool and discard low-quality examples. Closely related to self-distillation, weak supervision, and active learning.

## The mechanic

```
1. Have a pool D_synth of N generated examples.
2. Split: train interim model h on a subset (or on real labeled data).
3. For each remaining example x in D_synth:
     - Predict h(x) → ŷ
     - Compute confidence c(x) = h's softmax max (or calibrated equivalent)
4. Keep examples where:
     - ŷ matches the synthetic label, AND
     - c(x) > threshold τ
5. Train final model on the kept examples.
```

[[../sources/2026-05-23-lambada]] (LAMBADA) introduced this for text classification. Same pattern shows up in:

- **Pseudo-labeling** (semi-supervised learning, Lee 2013)
- **Self-training** (Yarowsky 1995)
- **Noisy Student** (Xie et al, 2019)
- **Knowledge distillation** (Hinton et al, 2015) — with the teacher being a different model

## Why this works

Synthetic data has noise. Confidence filtering exploits two properties:

1. **The interim model has learned the easy patterns first.** Easy examples have high confidence; hard or noisy ones have low confidence. Cutting low-confidence drops noise more than signal.
2. **Disagreement = generation error.** If the model trained on the labels disagrees with the synthetic label on a synthetic example, one of them is wrong. Empirically, more often the synthetic label is wrong (generator screwed up, post-processing screwed up, or the example is genuinely ambiguous).

## Risks

- **Confirmation bias.** The filter keeps examples the model already agrees with, discarding ones it might learn from. If the model is wrong about something systematic, filtering reinforces the error.
- **Mode collapse.** Over-aggressive filtering shrinks the dataset diversity. The retrained model performs *better on the filtered distribution* but may generalize *worse* to the original target distribution.
- **Calibration matters.** "Confidence above τ" assumes the model is well-calibrated. BERT-style models are notoriously overconfident; raw softmax values shouldn't be trusted as probabilities. Temperature scaling or other calibration ([[ner-calibration]] TBD) helps.

## Variants

### Cross-model agreement filter

Train two different models (different architectures or different seeds) on the synthetic data. Keep only examples both agree on. Catches model-specific biases that single-model filtering misses.

### External-validator filter

Use a model trained on **real** (different-distribution) labeled data as the filter. The filter doesn't need to be best in class — it just needs different inductive biases than the generator.

### Round-trip filter

For NER specifically: generate sentence → string-match labels → run NER → check NER spans match string-match spans. Discard mismatches. Cheap and catches generator-induced span shifts.

## How to apply in our pipeline

Our v3 dataset has 54k examples with no filtering. Proposed steps:

1. **Audit step (no retraining yet).** For each of 54k examples, run v3 inference. Compute:
   - `disagreement_rate` = examples where v3 misses or mislabels an inserted PII
   - `low_conf_rate` = examples where v3 predicts the right span but with confidence < 0.7
   - `pii_drop_rate` = examples where gpt-5-nano dropped at least one inserted PII span in rewriting
2. **Report and decide.** If disagreement is <2%, filtering isn't worth the retraining cost. If >5%, filtering likely helps.
3. **Conservative filter.** Drop only examples with explicit PII drop (round-trip filter) + disagreement on all spans. Don't drop low-confidence examples in first iteration (avoid confirmation bias on hard cases).
4. **Retrain v3.1 on filtered set.** Compare F1 on the original holdout (not the filtered training subset).
5. **If positive, iterate once.** Don't iterate more than 2-3 times to limit drift.

## Open questions

- What's the actual disagreement rate in our 54k examples? Need to measure.
- Is "v3 confidence" calibrated enough to use as a threshold, or do we need temperature scaling first?
- Should we use an external NER (e.g., spaCy `pt_core_news_lg` or GLiNER-Multi) as a cross-architecture validator?
- How do we handle examples where v3 disagrees with the label but is *right* (i.e., the label is wrong but v3 catches it)? Currently impossible to detect without manual review.

## Related

- [[../sources/2026-05-23-lambada]] — origin
- [[lm-based-data-augmentation]] — the broader pipeline
- [[../questions/synthetic-data-quality]] — directly relevant
- [[ner-calibration]] — TBD; calibration is a prerequisite for confidence filtering
