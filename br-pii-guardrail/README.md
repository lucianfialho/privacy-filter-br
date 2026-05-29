# br-pii-guardrail

**Brazilian PII detection + tokenization for LLM workflows.** Combines regex+checksum recognizers, schema-aware scanners (JSON/CSV), and an optional NER fallback into a single lib. Built for Analytics Copilot-style products that need to send data through external LLMs without leaking PII.

## What it does

```
INPUT text/JSON/CSV  →  Guardrail.scan_*()  →  [Match(start, end, label, source), ...]
                                                       ↓
                                            Tokenizer.tokenize()
                                                       ↓
                                  masked text  +  vault (AES-GCM encrypted)
                                                       ↓
                                  ─── send to GPT-4o / Claude / etc ───
                                                       ↓
                                            Tokenizer.detokenize()
                                                       ↓
                                              original PII restored
```

## Why a guardrail (not just a detector)

The pattern lets you:
- Persist **tokenized** data in DB → backup leak doesn't expose PII
- Run analytics on tokens (`COUNT(DISTINCT cpf_token)` works)
- Only decrypt when the data subject themselves needs to read

## Install

```bash
pip install br-pii-guardrail               # regex + tokenizer only
pip install br-pii-guardrail[ner]          # + NER fallback for free text
pip install br-pii-guardrail[all]          # + PDF support
```

## Quick start

```python
from br_pii_guardrail import Guardrail, Tokenizer, derive_tenant_key
import os

# Setup
master_key = bytes.fromhex(os.environ["PII_MASTER_KEY"])  # 32 bytes
tok = Tokenizer(derive_tenant_key(master_key, tenant_id="acme"))
guard = Guardrail.default()  # all BR recognizers enabled

# Free text
text = "Cliente João, CPF 680.075.670-97, telefone (11) 98765-4321"
matches = guard.scan_text(text)
masked, vault = tok.tokenize(text, matches)
# masked  = 'Cliente João, CPF [CPF_a3f8c2], telefone [PHONE_7b1d4e]'
# vault   = {'[CPF_a3f8c2]': '<aes-gcm blob>', ...}

# Send `masked` to GPT-4o, save `masked + vault` to DB
response = llm.complete(masked)

# Restore for end-user display
original = tok.detokenize(response, vault)
```

## JSON scan (structured input)

```python
data = {"customer": {"cpf": "680.075.670-97", "name": "João"},
        "order_id": "ML-2024-789456", "amount": 1250.0}
matches = guard.scan_json(data)
# Schema-aware: detects "cpf" key automatically, validates with checksum
```

## CSV scan

```python
from br_pii_guardrail.scanners import scan_csv_text
csv_text = open("clientes.csv").read()
results = scan_csv_text(csv_text, guard.recognizers)
# Returns {column_name: [Match, ...]} grouped by column
```

## Optional NER fallback

For prose / free text where regex misses (descriptions, comments, etc):

```python
from br_pii_guardrail.ner import NER

ner = NER(model_path="lucianfialho/privacy-filter-br")  # HF Hub canonical (latest = v7)
# Pin a version: NER(model_path="lucianfialho/privacy-filter-br", revision="v8.1")
guard = Guardrail.default(ner=ner)
guard.use_ner_fallback = True  # only run NER when regex finds nothing

matches = guard.scan_text("comentário do cliente: bla bla...")
```

## Categories detected

**BR-specific (with checksum where applicable):**
`private_cpf`, `private_cnpj`, `private_rg`, `private_cnh`, `private_pis`, `private_titulo_eleitor`, `private_certidao`, `private_ie`, `private_order_id`, `private_tracking_code`, `private_invoice_number`, `private_client_revenue`, `private_transaction_id`, `private_customer_id`

**OAI-compatible:**
`private_person`, `private_email`, `private_phone`, `private_address`, `private_date`, `private_url`, `secret`, `account_number`

## License

MIT
