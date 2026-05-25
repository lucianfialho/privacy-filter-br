---
title: Privacy Filter BR v5 Demo
emoji: 🔒
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.32.1
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Detector de PII em português brasileiro (CPF, CNPJ, B2B IDs)
---

# Privacy Filter BR v5 — Interactive Demo

Demo Gradio do modelo [`lucianfialho/privacy-filter-br-v5`](https://huggingface.co/lucianfialho/privacy-filter-br-v5).

Cole qualquer texto BR pra ver os PIIs detectados com highlights coloridos por categoria.

**22 categorias detectadas:**
- Pessoais: CPF, CNPJ, RG, CNH, PIS, título eleitor, certidão, IE, nome, email, telefone, endereço
- B2B: order_id, customer_id, tracking, invoice, transaction_id, client_revenue
- Técnicas: URL, secret/API key, account_number (cartão Luhn), date

Para uso em produção LGPD, veja a lib [`br-pii-guardrail`](https://github.com/lucianfialho/privacy-filter-br/tree/main/br-pii-guardrail) que combina regex+checksum+tokenization.
