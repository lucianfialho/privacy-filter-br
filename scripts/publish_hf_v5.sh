#!/bin/bash
# Publish v5 NER model to HuggingFace Hub.

set -euo pipefail

REPO_NAME="${HF_REPO:-lucianfialho/privacy-filter-br-v5}"
MODEL_DIR="${MODEL_DIR:-checkpoints/v5-local}"

echo "=== Publishing to HuggingFace Hub ==="
echo "  repo: $REPO_NAME"
echo "  source: $MODEL_DIR"
echo ""

if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
    echo "ERROR: $MODEL_DIR/model.safetensors not found" >&2
    exit 1
fi

cat > "$MODEL_DIR/README.md" <<EOF
---
language: pt
license: mit
library_name: transformers
tags:
  - token-classification
  - ner
  - pii
  - portuguese
  - brazilian-portuguese
  - lgpd
base_model: neuralmind/bert-base-portuguese-cased
metrics:
  - f1
  - precision
  - recall
---

# Privacy Filter BR v5

Token classification model for Brazilian Portuguese PII detection. Fine-tuned from BERTimbau (\`neuralmind/bert-base-portuguese-cased\`, 110M params) on a **100k synthetic dataset combining two LLM rewriters** (Claude Haiku + gpt-5-nano) covering 22 categories.

**v5 key improvement over v3/v4:** mixed-rewriter training to avoid the distribution-shift trap that affected v4. v4 was trained on gpt-5-nano-only text and over-fit to its markdown style — F1 dropped to 0.55 on Haiku-style holdout. v5 sees both distributions and generalizes.

## Performance

### Cross-style benchmark (seqeval BIOES F1)

| Model | Haiku-style holdout | gpt5nano-style holdout | Spread |
| --- | --- | --- | --- |
| v3 | 0.9901 | 0.9947 | 0.0046 |
| v4 | **0.5534** ⚠️ | 0.9992 | **0.4458** |
| **v5** | **0.9912** ✓ | **0.9992** ✓ | **0.0080** ✓ |

v5 strictly dominates v3 on both styles. v5 strictly dominates v4 on Haiku-style.

### Per-category v5 (combined holdout, 9893 examples / 67601 spans)

| Metric | Value |
| --- | --- |
| Micro F1 | 0.9955 |
| Macro F1 | **0.9989** |
| Precision | 0.9938 |
| Recall | 0.9972 |

**Biggest gains vs v3:**

| Category | v3 F1 | v5 F1 | Δ |
| --- | --- | --- | --- |
| private_phone | 0.9627 | 0.9863 | **+0.0236** |
| private_certidao | 0.9828 | 0.9916 | +0.0088 |
| private_ie | 0.9879 | 0.9959 | +0.0080 |
| private_person | 0.9867 | 0.9938 | +0.0071 |
| private_address | 0.9886 | 0.9952 | +0.0066 |
| private_transaction_id | 0.9946 | 0.9996 | +0.0050 |
| private_cpf | 0.9911 | 0.9957 | +0.0046 |

**Known regressions (small):**

| Category | v3 F1 | v5 F1 | Δ |
| --- | --- | --- | --- |
| private_rg | 0.9948 | 0.9912 | -0.0036 |
| private_customer_id | 1.0000 | 0.9995 | -0.0005 |

## Known limitations (failure cases v5 inherits)

These are **template coverage** gaps, not model architecture issues:

1. **Customer IDs with similar prefix to order IDs:** strings like \`CUST-998877\` are usually labeled as \`private_order_id\` instead of \`private_customer_id\`. The templates produce both with similar formatting and don't teach the model to use context (the word "cliente" preceding) for disambiguation.

2. **Revenue in free-form sentences:** patterns like "faturou R\$ 50.000,00 em outubro" are often missed. Templates put revenue values in dashboards/tables, rarely in narrative text.

Both are addressable with template improvements + a real-world test set (planned as Phase 1).

## Categories detected (22)

**OAI-compatible:** account_number, private_address, private_date, private_email, private_person, private_phone, private_url, secret

**BR-specific:** private_cpf, private_cnpj, private_rg, private_cnh, private_pis, private_titulo_eleitor, private_certidao, private_ie, private_order_id, private_tracking_code, private_invoice_number, private_client_revenue, private_transaction_id, private_customer_id

## Usage

\`\`\`python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

tok = AutoTokenizer.from_pretrained("$REPO_NAME")
model = AutoModelForTokenClassification.from_pretrained("$REPO_NAME")
ner = pipeline("token-classification", model=model, tokenizer=tok, aggregation_strategy="simple")

text = "Cliente João Silva, CPF 680.075.670-97, pedido ML-2024-789456"
for ent in ner(text):
    print(f"{ent['entity_group']}: {text[ent['start']:ent['end']]}")
\`\`\`

## Recommended: combine with \`br-pii-guardrail\`

For LGPD compliance with regex+checksum (deterministic recall for canonical formats) + reversible AES-GCM tokenization per tenant:

\`\`\`python
from br_pii_guardrail import Guardrail
from br_pii_guardrail.ner import NER

ner = NER("$REPO_NAME")
guard = Guardrail.default(ner=ner)
tokenized, vault = guard.tokenize(text, tenant_id="cliente-x")
\`\`\`

## Training details

- Base: \`neuralmind/bert-base-portuguese-cased\` (BERTimbau, 110M)
- Dataset: 99900 examples (dataset_br_v3.jsonl 49842 Haiku + dataset_br_v4.jsonl 49900 gpt-5-nano, all 22 categories)
- Epochs: 3 (best checkpoint = epoch 3)
- Effective batch: 16 (batch 8 × grad_accum 2)
- LR: 3e-5
- Hardware: RTX 2070 SUPER 8GB
- Training time: ~3h45min

## Pipeline summary

\`\`\`
4devs API → BR PII profiles (CPF/CNPJ/RG with checksums)
    ↓ variants.py: 4 format variants per field (canonical, raw, masked, spaces)
    ↓ Jinja templates: 18 regular + 4 extras
prompts split between TWO LLM rewriters:
    - Claude Haiku via CLI (varied prose style)
    - gpt-5-nano via OpenAI Batch API (consistent markdown style)
    ↓ src/labeler.py: format-aware skeleton matching
    ↓ src/validator.py
100k train / 10k holdout (concat of v3 + v4 datasets)
    ↓ BERTimbau fine-tune (3 epochs, ~3h45min)
checkpoints/v5-local/
\`\`\`

## License

MIT
EOF

echo "Created model card at $MODEL_DIR/README.md"
echo ""
echo "=== Uploading to $REPO_NAME ==="

python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('$REPO_NAME', exist_ok=True, repo_type='model')
api.upload_folder(folder_path='$MODEL_DIR', repo_id='$REPO_NAME', repo_type='model')
print('Done. Visit: https://huggingface.co/$REPO_NAME')
"
