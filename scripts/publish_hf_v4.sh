#!/bin/bash
# Publish v4 NER model to HuggingFace Hub.
#
# Prereqs:
#   1. HF account: https://huggingface.co/join
#   2. Create a "write" token: https://huggingface.co/settings/tokens (type=write)
#   3. Install: pip install huggingface_hub
#
# Usage:
#   export HF_TOKEN=hf_xxxxxxxxxxxxxxxx  (or run `huggingface-cli login`)
#   bash scripts/publish_hf_v4.sh

set -euo pipefail

REPO_NAME="${HF_REPO:-lucianfialho/privacy-filter-br-v4}"
MODEL_DIR="${MODEL_DIR:-checkpoints/v4-local}"

echo "=== Publishing to HuggingFace Hub ==="
echo "  repo: $REPO_NAME"
echo "  source: $MODEL_DIR"
echo ""

if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
    echo "ERROR: $MODEL_DIR/model.safetensors not found" >&2
    exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
    if ! huggingface-cli whoami >/dev/null 2>&1; then
        echo "ERROR: Not logged in to HF. Run: huggingface-cli login" >&2
        exit 1
    fi
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

# Privacy Filter BR v4

Token classification model for Brazilian Portuguese PII detection. Fine-tuned from BERTimbau (\`neuralmind/bert-base-portuguese-cased\`, 110M params) on a 54k synthetic dataset covering 22 categories.

**v4 key improvement over v3:** the synthetic dataset is regenerated with a new **format-aware labeler** (\`src/labeler.py\`) that captures separator variants the original \`re.escape()\` exact match missed (e.g., \`123.456.789-00\` vs \`123.456.789.00\` vs \`123 456 789 00\`). This fixes a labeler bug that caused ~16.76% of v3 training examples to have at least one unlabeled regex-detectable PII.

## Performance (synthetic holdout, n=4964)

| Metric | v3 | **v4** | Δ |
| --- | --- | --- | --- |
| Macro F1 | 0.9934 | **0.9989** | **+0.0055** |
| Micro F1 | 0.9900 | **0.99923** | **+0.0092** |
| Macro precision | 0.9905 | **0.9992** | +0.0087 |
| Macro recall | 0.9962 | **0.9985** | +0.0023 |

**13 of 22 categories now have perfect F1 = 1.0000.**

### Biggest per-category improvements

| Category | v3 F1 | v4 F1 | Δ |
| --- | --- | --- | --- |
| private_phone | 0.9627 | 0.9989 | **+0.0362** |
| private_certidao | 0.9828 | 1.0000 | +0.0172 |
| private_person | 0.9867 | 0.9997 | +0.0130 |
| private_ie | 0.9879 | 1.0000 | +0.0121 |
| private_address | 0.9886 | 1.0000 | +0.0114 |
| private_cpf | 0.9911 | 0.9993 | +0.0082 |
| private_cnpj | 0.9931 | 0.9997 | +0.0066 |

### Known regression (vs v3)

| Category | v3 F1 | v4 F1 | Δ |
| --- | --- | --- | --- |
| private_rg | 0.9948 | 0.9868 | -0.0080 |

Cause under investigation — likely RG/IE format confusion. Not a blocker; will be addressed in v4.1.

## Categories detected

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
    print(ent)
\`\`\`

## Recommended companion lib

For production use as an LGPD guardrail (regex+checksum+tokenization), use the **\`br-pii-guardrail\`** library which uses this model as an NER fallback layer:

\`\`\`python
from br_pii_guardrail import Guardrail
from br_pii_guardrail.ner import NER
guard = Guardrail.default(ner=NER("$REPO_NAME"))
\`\`\`

## Training details

- Base: \`neuralmind/bert-base-portuguese-cased\`
- Dataset: 54864 synthetic examples (1507 4devs profiles × 33 reuses × variant formats, generated via OpenAI Batch API with gpt-5-nano)
- Epochs: 3 (best checkpoint = epoch 2)
- Effective batch: 16
- LR: 3e-5
- Hardware: RTX 2070 SUPER 8GB
- Training time: 1h55min

## Pipeline upgrades from v3

1. **Format-aware labeler** (\`src/labeler.py\`) — uses alphanumeric skeleton matching with optional separators between alnum chars, anchored at word boundaries. Tolerates \`-\` ↔ \`.\` ↔ space ↔ \`/\` ↔ parentheses.
2. **Generation via OpenAI Batch API** — 50% cost discount, 24h SLA, cleaner pipeline.
3. **Comprehensive smoke validation** — regex audit confirmed labeler misses dropped from 16.76% (v3) to 0% on canonical structured PII formats.

## Limitations

- Trained on **synthetic** data. Real-world F1 not yet measured.
- Brazilian Portuguese only.
- private_rg slightly regressed vs v3 (see above).
- Recommend combining with regex+checksum guardrails (see \`br-pii-guardrail\`).

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
