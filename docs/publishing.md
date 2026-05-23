# Publishing Guide

Two deliverables to publish:

1. **`lucianfialho/privacy-filter-br-v3`** — NER model on HuggingFace Hub
2. **`br-pii-guardrail`** — Python lib on PyPI

---

## 1. HuggingFace Hub (NER model)

### Prereqs

```bash
pip install huggingface_hub
huggingface-cli login   # paste write token from https://huggingface.co/settings/tokens
```

### Publish

```bash
# Defaults: REPO=lucianfialho/privacy-filter-br-v3, MODEL_DIR=checkpoints/v3-local
bash scripts/publish_hf.sh

# Override:
HF_REPO=youruser/whatever MODEL_DIR=/path/to/model bash scripts/publish_hf.sh
```

The script creates a model card (`README.md`) automatically and uploads everything in `checkpoints/v3-local/` (model + tokenizer + config).

### Test public model

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

REPO = "lucianfialho/privacy-filter-br-v3"
tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForTokenClassification.from_pretrained(REPO)
ner = pipeline("token-classification", model=model, tokenizer=tok, aggregation_strategy="simple")
ner("CPF 680.075.670-97")
```

---

## 2. PyPI (br-pii-guardrail lib)

### Prereqs

```bash
pip install --upgrade build twine
```

Get tokens at https://pypi.org/manage/account/token/ (and https://test.pypi.org/manage/account/token/ for testing).

Config in `~/.pypirc`:
```ini
[pypi]
username = __token__
password = pypi-xxxxxxxxxxxxxxxxxxxxxxxxxxxx

[testpypi]
username = __token__
password = pypi-yyyyyyyyyyyyyyyyyyyyyyyyyyy
```

### Publish to TestPyPI first (RECOMMENDED)

```bash
PYPI_TEST=1 bash scripts/publish_pypi.sh
```

Verify install:
```bash
pip install --index-url https://test.pypi.org/simple/ br-pii-guardrail
python -c "from br_pii_guardrail import Guardrail; print(Guardrail.default().recognizers)"
```

### Publish to production PyPI

```bash
bash scripts/publish_pypi.sh
# confirms with [y/N] before uploading
```

After publish:
```bash
pip install br-pii-guardrail
pip install br-pii-guardrail[ner]    # with NER fallback
pip install br-pii-guardrail[pdf]    # with PDF support
pip install br-pii-guardrail[all]    # everything
```

### Bump version

Edit two files together to keep them in sync:
- `br-pii-guardrail/pyproject.toml` → `version = "0.2.0"`
- `br-pii-guardrail/src/br_pii_guardrail/__init__.py` → `__version__ = "0.2.0"`

Tag the commit so PyPI version matches git:
```bash
git tag v0.2.0
git push origin v0.2.0
```

---

## After publishing both

Update the Analytics Copilot issue to reference the public model + PyPI lib:
```bash
gh issue comment 274 --repo metricasboss/analytics-copilot \
  --body "Published: br-pii-guardrail on PyPI, privacy-filter-br-v3 on HF Hub. \`pip install br-pii-guardrail[ner]\` ready to integrate."
```
