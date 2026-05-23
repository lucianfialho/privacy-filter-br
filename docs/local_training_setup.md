# Local Training Setup — RTX 2070 8GB (or similar)

Training Privacy Filter BR v3 on a local Turing+ GPU using BERTimbau as base.

## What you get
- **Base model:** `neuralmind/bert-base-portuguese-cased` (110M params)
- **Target taxonomy:** same v3 hybrid (22 categories, 89 BIOES labels)
- **Expected training time:** ~1-2h for 3 epochs on RTX 2070
- **Memory usage:** ~4-5GB (fits comfortably in 8GB)
- **Expected F1:** 0.93-0.96 (no OAI transfer, but 54k examples)

---

## 1. Clone repo on the office PC

```bash
git clone <repo-url> privacy-filter-br
cd privacy-filter-br
```

## 2. Check NVIDIA driver

```bash
nvidia-smi
# Expected: shows RTX 2070, driver version >= 525, CUDA version >= 12.0
```

If `nvidia-smi: command not found`, install drivers first:
```bash
# Ubuntu/Debian:
sudo apt update && sudo apt install nvidia-driver-535
sudo reboot
```

## 3. Install Python deps

```bash
# Python 3.10+ (most Ubuntu LTS have this by default)
python3 --version

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets seqeval accelerate
```

**Verify CUDA works:**
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True NVIDIA GeForce RTX 2070
```

## 3. Get the dataset

Copy these two files from the Mac (they're not in git — `.gitignore` excludes `*.jsonl`):

```
data/dataset_br_v3.jsonl           # 49.842 train, ~65MB
data/dataset_br_v3_holdout.jsonl   # 4.929 holdout, ~6.4MB
```

Options: scp, rsync, USB stick, Google Drive — whatever's faster.

## 4. Run training

```bash
python notebooks/finetune_v3_local.py \
    --train-file data/dataset_br_v3.jsonl \
    --eval-file data/dataset_br_v3_holdout.jsonl \
    --output-dir checkpoints/v3-local \
    --epochs 3 \
    --batch-size 8 \
    --grad-accum 2 \
    --lr 3e-5 \
    --max-length 256
```

**If OOM happens:**
- Drop `--batch-size 4` and `--grad-accum 4` (same effective batch, less memory)
- Drop `--max-length 128` (only if your texts are short — most BR examples fit in 256 tokens)

**Watch GPU memory live in another terminal:**
```bash
watch -n 1 nvidia-smi
```
Expected: ~4-5GB usage with default settings.

## 5. Output

After training (~1-2h), you'll have:
```
checkpoints/v3-local/
├── config.json
├── model.safetensors          # ~440MB
├── tokenizer.json
├── special_tokens_map.json
├── benchmark.txt              # per-label F1 report
└── checkpoint-XXXX/           # intermediate checkpoints
```

Read `benchmark.txt` to see per-class F1. If macro F1 ≥ 0.93, you're good for production use.

## 6. Test inference

```bash
python -c "
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

m = AutoModelForTokenClassification.from_pretrained('checkpoints/v3-local').cuda().eval()
t = AutoTokenizer.from_pretrained('checkpoints/v3-local')
ner = pipeline('token-classification', model=m, tokenizer=t,
               aggregation_strategy='simple', device=0)

samples = [
    'João Silva, CPF 123.456.789-00, pedido ML-2024-789456.',
    'Empresa Acme, CNPJ 12.345.678/0001-90, fatura FAT-2025-001.',
    'API Key: sk-proj-abc123def. Cartão 4111 1111 1111 1111.',
]
for s in samples:
    print('TEXT:', s)
    for e in ner(s):
        print(f'  {e[\"entity_group\"]:25} | {e[\"word\"]!r} | {e[\"score\"]:.3f}')
    print()
"
```

## 7. Plug into Analytics Copilot

The saved model loads with standard HF API (no PEFT/LoRA required):
```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL_PATH = "/path/to/checkpoints/v3-local"
ner_model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
```

Then combine with `src/tokenizer.py` for the encrypt/decrypt flow:
```python
from src.tokenizer import Tokenizer, derive_tenant_key
import os

key = derive_tenant_key(bytes.fromhex(os.environ["PII_MASTER_KEY"]), tenant_id)
tok = Tokenizer(key)

# 1. Detect PII spans
preds = ner_pipeline(user_message)
entities = [{"start": p["start"], "end": p["end"], "label": p["entity_group"].lower()}
            for p in preds]

# 2. Tokenize
masked, vault = tok.tokenize(user_message, entities)

# 3. Send `masked` to GPT-4o, save `masked + vault` to DB
# 4. Detokenize on demand: tok.detokenize(response, vault)
```

---

## Troubleshooting

**`CUDA out of memory`:** Decrease `--batch-size` (4 or 2), increase `--grad-accum` proportionally. Or `--max-length 128`.

**Very slow training (>5h):** Check `nvidia-smi` — if GPU util is <80%, CPU/disk is bottleneck. Increase `--dataloader_num_workers` (already 2). Or quit other apps.

**`bf16 not supported`:** Expected on Turing. Script auto-falls back to fp16. Just ignore the warning.

**F1 stuck at ~0.5 on some labels:** Probably the OAI categories (date/url/secret/account) where we only have 2-6k examples. Increase epochs to 5 or generate more extras via `scripts/openai_batch.py extras --n 5000`.
