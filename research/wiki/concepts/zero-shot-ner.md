---
type: concept
tags: [ner, zero-shot, generalization, prompting]
sources: 1
updated: 2026-05-23
---

# Zero-shot NER

NER models that can recognize entity types **not seen during training**, by supplying the type name (or description) at inference. Contrast with the traditional setup, where the label space is fixed at training time and adding a new category requires re-training.

## Two flavors

1. **Generative / LLM-based.** ChatGPT, UniNER, Vicuna. The user prompts the model with "Find all PERSON entities in: ..." and the model emits text. Strong generality, slow inference, hard to bound outputs.

2. **Encoder + matching.** A bidirectional encoder produces span representations AND type-name representations in the same latent space. NER becomes a similarity-matching problem. [[gliner-model]] is the canonical example. Fast, bounded outputs, but no chain-of-thought.

## How matching-based zero-shot works ([[gliner-model]] specifically)

Input format: `[ENT] type1 [ENT] type2 [SEP] sentence text`

- Each `[ENT]` token is a learned special token whose hidden state at the output serves as the **type embedding** for the type-name that follows.
- Each candidate span in `sentence text` produces a **span embedding**.
- Score = `sigmoid(dot(type_embedding, span_embedding))`.
- Threshold at 0.5 to decide if the span is an entity of that type.

Zero-shot works because the encoder has been trained on a **wide variety** of (type, span) pairs (13k distinct types in Pile-NER), so the matching function generalizes to novel type names via their token-level semantics.

## What "zero-shot" actually means in NER

There are subtle gradations:

- **Strict zero-shot.** Type name never seen during training. E.g. ask for "BR PIX key" when training only saw English Pile-NER. Performance varies wildly.
- **Cross-domain zero-shot.** Type name is semantically similar to something seen in training, but in a new domain. E.g. "person" seen in news → "patient" in clinical notes. Usually works well.
- **Few-shot via prompt extension.** Add 1-3 examples of the type to the prompt. Different from zero-shot, but commonly conflated.

[[gliner-model]] reports strong **strict zero-shot** numbers (60.9 F1 avg on CrossNER + MIT benchmarks).

## Costs vs benefits

**Benefits:**
- Add new PII categories without re-training (e.g. `private_pix_key`).
- Reduces synthetic-data generation burden ([[../questions/synthetic-data-quality]]).
- Composes with downstream rule-based validators (regex for format, checksum for validity).

**Costs:**
- Worse than supervised fine-tuning when fine-tuning data is plentiful (GLiNER reports lagging UniNER by ~3 F1 points when fine-tuned).
- Type-name wording matters. "credit_card" vs "cartão de crédito" vs "card number" produce different results.
- Hard to debug — no clear correspondence between training data and per-type performance.

## Relevance to our work

We want to support 22 PII categories today, but the list could grow:

- `private_pix_key` (UUID, phone, email, CPF, EVP)
- `private_bank_account` (agência + conta)
- `private_voter_id` (título de eleitor)
- `private_work_card` (CTPS)
- `private_voucher_id` (vale-transporte, vale-refeição numbers)

Each new category currently costs us ~5-10k synthetic examples + 1 retrain. Zero-shot would shortcut this.

**Open question:** does zero-shot work well enough for BR PII to be a primary mechanism, or only good for "long-tail" categories we don't bother training on?

## Failure modes to watch for

- **Type-name collision.** "Person" and "patient" both match human-name spans. The model can't tell which type is intended without context.
- **Brittle to phrasing.** "CPF" vs "Brazilian individual taxpayer ID" — same concept, different prompt embeddings.
- **No checksum/format awareness.** Zero-shot model sees "123.456.789-00" and says "looks like a CPF", but doesn't verify the check digit. Need a downstream validator. (Our [[../entities/br-pii-guardrail]] regex layer does this.)

## Related

- [[span-based-ner]] — most zero-shot encoder approaches are span-based
- [[gliner-model]]
- [[../sources/2026-05-23-gliner]]
