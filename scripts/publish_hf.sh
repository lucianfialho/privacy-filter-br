#!/bin/bash
# Publish v3 NER model to HuggingFace Hub.
#
# Prereqs:
#   1. HF account: https://huggingface.co/join
#   2. Create a "write" token: https://huggingface.co/settings/tokens (type=write)
#   3. Install: pip install huggingface_hub
#
# Usage:
#   export HF_TOKEN=hf_xxxxxxxxxxxxxxxx  (or run `huggingface-cli login`)
#   bash scripts/publish_hf.sh

set -euo pipefail

REPO_NAME="${HF_REPO:-lucianfialho/privacy-filter-br-v3}"
MODEL_DIR="${MODEL_DIR:-checkpoints/v3-local}"

echo "=== Publishing to HuggingFace Hub ==="
echo "  repo: $REPO_NAME"
echo "  source: $MODEL_DIR"
echo ""

# Sanity check the model dir
if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
    echo "ERROR: $MODEL_DIR/model.safetensors not found" >&2
    echo "Run training first or update MODEL_DIR." >&2
    exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN env var not set. Falling back to cached login..."
    if ! huggingface-cli whoami >/dev/null 2>&1; then
        echo "ERROR: Not logged in to HF. Run: huggingface-cli login" >&2
        exit 1
    fi
fi

# Copy README + license INTO the model dir (HF Hub shows it as model card)
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

# Privacy Filter BR v3

Token classification model for Brazilian Portuguese PII detection. Fine-tuned from BERTimbau (\`neuralmind/bert-base-portuguese-cased\`, 110M params) on a 54k synthetic dataset covering 22 categories.

## Performance (synthetic holdout, n=4929)

| Metric | Value |
| --- | --- |
| Macro F1 | **0.9934** |
| Macro Precision | 0.9905 |
| Macro Recall | 0.9962 |
| Micro F1 | 0.9900 |

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
- Dataset: 54k synthetic examples (4devs API + Jinja templates + LLM rewriting)
- Epochs: 3
- Effective batch: 16
- LR: 3e-5
- Hardware: RTX 2070 SUPER 8GB

## Limitations

- Trained on **synthetic** data. Expect F1 drop of 5-10% on real-world text.
- Brazilian Portuguese only.
- Weakest categories: \`private_phone\` (F1 0.96) due to format variability, \`private_person\` (F1 0.99) due to name ambiguity.
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
