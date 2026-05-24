---
type: concept
tags: [data-augmentation, lm-generation, synthetic-data, training-pipeline]
sources: 1
updated: 2026-05-23
---

# LM-based data augmentation

Using a fine-tuned (or prompted) language model to generate additional labeled training examples for a downstream supervised task. Originated as a small-data text classification technique; now broadly used wherever labeled data is expensive.

## The canonical recipe ([[../sources/2026-05-23-lambada]] / LAMBADA)

1. **Fine-tune a generator** (GPT-2 originally, modern variants use GPT-4, Claude, Llama-fine-tuned) on the small labeled set, conditioning on class labels: `label [SEP] sentence [EOS]`.
2. **Generate excess samples** — typically 5-10× the target size per class.
3. **Filter aggressively** using a baseline classifier trained on the original data. Keep only samples the classifier confidently agrees with.
4. **Retrain final model** on the union of original + filtered synthetic.

Key insight: **generation is cheap, filtering is the quality lever**. Generate 10× and throw away 90% if needed.

## When it works

- **Small labeled data** (5-100 examples per class). Bigger gains for smaller datasets.
- **Imbalanced classes.** Generation can target under-represented classes specifically. LAMBADA's biggest gains came on ATIS, the most imbalanced dataset tested.
- **Class names with semantic meaning.** "Flight time", "Aircraft", "City" carry semantic content the LM can use to condition generation. Opaque class IDs ("class_42") work less well.

## When it doesn't help (or hurts)

- **Plenty of real data already.** Past ~100 examples per class, gains shrink. Past ~1000, often no measurable improvement.
- **Domain shift between LM pretraining and target domain.** Fine-tuning GPT-2 on clinical notes works less well than fine-tuning on general English — the LM's prior is wrong.
- **Iterative regeneration without diversity control.** "Generated → trained → generated again" can cause mode collapse: the synthesized data becomes a narrower subset of the real distribution. Original LAMBADA flags this as "data drifting" — it's the same phenomenon.

## Why filtering matters

Generated examples have three failure modes:

1. **Label drift** — the generator emits a sentence that doesn't actually belong to the requested class. Filter catches this (h(x) != y).
2. **Style monotony** — all generated sentences sound the same. Filter doesn't directly catch this, but generation pool sampling (top-k, top-p, temperature) can mitigate.
3. **Hallucinated content** — names of products, places, people that don't exist; impossible facts. Filter only catches this if the classifier is trained well enough to recognize implausibility, which is rare.

The classifier-confidence filter only addresses (1). Modern pipelines add additional checks: deduplication, length filters, similarity-to-training cap, factuality checks.

## How our pipeline maps to this

Our v3 dataset construction:

```
1. 4devs API → valid BR PII (CPF, CNPJ, RG with checksums)
2. Jinja templates → ~18 BR document types with PII slots filled
3. GPT-5-nano → rewrite for naturalness
4. String-match → label spans by finding inserted PII in rewrite
5. NO FILTER → keep all 54k examples
```

In LAMBADA terms:
- Steps 1-3 = generation
- Step 4 = labeling (we use string-match, LAMBADA uses generation-time label)
- Step 5 = filtering, which we **skip**

This is the architectural gap. We could add:

```
6. Train interim NER on 80% of v3 data
7. Score remaining 20% with interim NER
8. For each example, compute (a) string-match labels, (b) NER predictions
9. Drop examples where (a) and (b) disagree, or where NER confidence < τ
10. Drop examples where gpt-5-nano dropped any inserted PII
11. Retrain v3.1 on filtered set
```

## How to handle NER-specific constraints

LAMBADA generates whole sentences with a single class label. For NER we have multiple spans per sentence with per-span labels. Adapting the filter:

- **Span-level filter.** For each (start, end, label) triple, score with NER. If NER doesn't predict ANY span at (start, end) → drop the example.
- **Sentence-level filter.** Compute per-sentence accuracy of NER predictions against string-match labels. If accuracy < τ → drop the example.
- **Confidence filter.** Average per-span confidence from NER. Below threshold → drop.

We can combine all three.

## Open questions

- What's the optimal filter threshold τ? LAMBADA uses "top-K by confidence" — implicitly leaves this to the classifier's calibration.
- Does iterative filtering converge to a good dataset, or does it shrink-and-bias?
- Can we use an *external* filter (e.g., a different NER architecture) to catch what v3 misses? Cross-architecture agreement might be a better signal.

## Related

- [[../sources/2026-05-23-lambada]] — origin paper
- [[synthetic-data-filtering]] — the filter mechanic specifically
- [[llm-based-pii-detection]] — opposite use of LLM (inference, not training)
- [[../questions/synthetic-data-quality]] — directly relevant
