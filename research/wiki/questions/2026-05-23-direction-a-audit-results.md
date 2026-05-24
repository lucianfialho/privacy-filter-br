---
type: question
tags: [audit-results, dataset-quality, v3, filter-and-retrain, lambada-applied]
sources: 0
updated: 2026-05-23
---

# Direction A: dataset audit results (2026-05-23)

Findings from running the 4 audits proposed in [[synthetic-data-quality]] after ingesting LAMBADA + Dai-Adel.

## Audit 1 — Round-trip filter (regex-based unlabeled PII detection)

**Script:** `scripts/audit_round_trip.py`
**Method:** Run conservative regexes for canonical BR PII patterns (CPF, CNPJ, email, phone, RG, PIS, URL, titulo eleitor) on every text. Compare matches against existing labeled spans. Report regex matches NOT covered by any label.

**Result: 16.76% of examples (8355 / 49842) have at least one regex-detectable PII pattern that is unlabeled.**

Per-pattern miss rates:

| pattern | unlabeled | total matches | miss-rate |
| --- | --- | --- | --- |
| private_cep | 3401 | 35137 | 9.68% |
| private_titulo_eleitor | 393 | 5043 | 7.79% |
| private_rg | 276 | 3752 | 7.36% |
| private_cpf | 1531 | 22785 | 6.72% |
| private_email | 2944 | 45803 | 6.43% |
| private_phone | 904 | 19532 | 4.63% |
| private_cnh | 80 | 29500 | 0.27% |
| private_cnpj | 47 | 14734 | 0.32% |
| private_pis | 13 | 13 | 100% |
| private_url | 8 | 2735 | 0.29% |

**Caveats:**
- `private_cep` 9.68% inflated by regex false positives — 8-digit chunks in Mercado Livre order IDs ("ML-2026-76525706") match the CEP pattern.
- `private_cnh` (11 digits) collides with IE / generic IDs.
- `private_pis` 100% is suspicious — only 13 total matches; likely the regex itself is off.

**Concrete example of a real labeler bug:**
Line 0 contains `**CPF:** 320.575.016.04`. The label is missing because the inserted CPF was probably `320.575.016-04` and gpt-5-nano rewriter changed `-` to `.`. The labeler does exact-string match (see `src/labeler.py`), so the variant goes unlabeled.

## Audit 2 — Per-category count

**Script:** `scripts/audit_category_counts.py`
**Method:** Count examples per category and entities per category in dataset_br_v3.jsonl.

**Result: 22 categories, 49842 examples, 374877 entity spans.**

Top categories (>60% of examples):
- private_person: 93.98%
- private_email: 79.57%
- private_cpf: 79.96%
- private_phone: 67.66%
- private_cnpj: 65.22%
- private_address: 61.49%

Under-represented (<5% of examples):
- account_number: 3.50%
- private_url: 3.64%
- secret: 3.64%
- **private_date: 3.64%** — surprising; nearly every BR document has a date

**Finding 1:** Pre-audit hypothesis — that private_phone (lowest F1) was under-represented — is **wrong**. Phone is in 67.66% of examples. The low F1 has a different cause (probably format variability or boundary issues).

**Finding 2:** `private_date` at 3.64% is almost certainly an instrumentation bug. The labeler probably never includes date as an inserted PII; instead, dates flow through naturally in text without being captured. Fix at labeler/template level.

## Audit 3 — v3 disagreement on training set

**Script:** `scripts/audit_v3_disagreement.py` (sample = 1000)
**Method:** Run v3 (`checkpoints/v3-local`) on a random sample of training examples. Compare predicted spans with gold labels.

**Result: only 0.70% of examples have v3 fully agreeing with gold. v3 finds MORE entities than labeled in 99.30% of examples.**

- v3 predictions are highly confident (mean 0.995, median 1.000)
- only 0.41% of predictions have confidence < 0.70
- only 1.54% of predictions have confidence < 0.90

Categories where v3 finds extras vs gold:
- private_person: +2031 (most common)
- private_cnpj: +1305
- private_cpf: +1297
- private_client_revenue: +712
- private_email: +429
- private_ie, rg, date, certidao, address: 100-300 each

**Two interpretations of this disagreement:**

1. **Labeler miss (dominant).** v3 has learned more PII patterns than the exact-string-match labeler captures. This is the same root cause as Audit 1, but for free-text categories (names, addresses) that regex can't catch.
2. **v3 fragmentation artifacts.** Some "extra" predictions are boundary noise — e.g., predicting `**` (markdown bold) as CPF, or stray ending punctuation. Inflates the disagreement count but isn't real PII.

Confidence is very tight (0.995 mean) even on artifacts, so confidence threshold alone won't filter artifacts. Need explicit boundary cleanup.

**Caveat:** sample was 1000 of 49842; truncated 3 examples to 1500 chars to fit within BERT's 512-token limit.

## Audit 4 — Filter+retrain (preparation done; retrain pending)

**Script:** `scripts/relabel_dataset.py`
**Method:** Augment original labels with regex-detected PII patterns that have no false-positive collision risk. Skip categories where regex collides (CEP, CNH, IE, etc).

**Result: `data/dataset_br_v3_relabeled.jsonl` with 6209 new labels added across 5503 examples (11.04% of dataset).**

New labels added per category:
- private_email: 2944 (biggest fix)
- private_cpf: 1546
- private_phone: 1003
- private_titulo_eleitor: 372
- private_rg: 276
- private_cnpj: 47
- private_pis: 13
- private_url: 8

**Strategy:** Option B from the audit plan — fix the labels, not the labeler. We kept all 374877 original labels (they're correct for the spans they cover) and only ADDED regex matches that fall outside existing labeled spans.

**Skipped categories** (false-positive risk too high): cep, cnh, account_number, customer_id, ie, invoice_number, order_id, tracking_code, transaction_id, secret, date, certidao, client_revenue.

For these, fixing requires either (a) a smarter labeler that knows BR PII format conventions per category, or (b) a labeler that has access to the original `inserted` dict from the generator (preserved before string-match).

## Synthesis

**The hypothesis was right.** Our pipeline missed labels at scale. LAMBADA's prescription (filter+retrain) applies, but the cause is upstream of generation — it's the exact-string labeler in `src/labeler.py`.

**Next concrete step (user decision):** retrain v3 on `data/dataset_br_v3_relabeled.jsonl` using the same finetune script as v3 (`notebooks/Privacy_Filter_BR_v3_Finetune.ipynb` or `scripts/finetune_v3.py`). Compare F1 on `data/dataset_br_v3_holdout.jsonl` (unchanged). Expected:

- Recall up on private_email, private_cpf, private_phone (the categories with most new labels).
- Precision either steady or slightly up (we didn't change existing labels).
- F1 macro should improve, especially on the under-represented categories.

**Cost:** comparable to original v3 training (~few hours on a single GPU; check `scripts/run_overnight.sh` for the script setup).

**Deferred for future audits / fixes:**

1. **Fix the labeler at source.** Replace `re.escape(value)` exact match with a format-aware matcher that handles separator variants (`-` ↔ `.` ↔ space ↔ `/`). Apply during next dataset regeneration.
2. **Fix the date instrumentation.** Templates emit dates but `inserted` dict probably doesn't include them. Add date as a first-class PII slot.
3. **Boundary cleanup post-process.** v3 produces artifacts like `**` labeled as CPF. Add a post-process step (in `br-pii-guardrail`?) that drops spans matching markdown/punctuation patterns.

## Related

- [[synthetic-data-quality]] — origin question, hypotheses H1/H2/H3
- [[../sources/2026-05-23-lambada]] — methodological inspiration
- [[../sources/2026-05-23-dai-ner-augmentation]] — alternative path (online MR augmentation)
- [[../concepts/synthetic-data-filtering]] — what we just did, but in a slightly different form (relabel, don't filter out)
