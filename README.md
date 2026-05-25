# Privacy Filter BR

Detecção de PII em documentos brasileiros (NF-e, holerite, contrato, certidão, e-mail corporativo, dashboards, etc) — fine-tune do **BERTimbau** (110M params) cobrindo 22 categorias de PII com formato canônico brasileiro.

Repositório completo: modelo + lib de produção + research wiki + scripts de auditoria + Gradio demo.

## Componentes

| Componente | Onde | Estado |
| --- | --- | --- |
| **Modelo v5 (production default)** | [`lucianfialho/privacy-filter-br-v5`](https://huggingface.co/lucianfialho/privacy-filter-br-v5) | publicado |
| Modelo v3 (anterior) | [`lucianfialho/privacy-filter-br-v5`](https://huggingface.co/lucianfialho/privacy-filter-br-v5) | publicado |
| Modelo v4 (experimental, **não use**) | [`lucianfialho/privacy-filter-br-v4`](https://huggingface.co/lucianfialho/privacy-filter-br-v4) | ⚠️ vê post-mortem |
| **Demo Gradio** | [`spaces/privacy-filter-br-demo`](https://huggingface.co/spaces/lucianfialho/privacy-filter-br-demo) | live (v5) |
| **Lib `br-pii-guardrail`** | [`br-pii-guardrail/`](./br-pii-guardrail) | 0.1.3 (regex+checksum+NER+AES vault) |
| **Research wiki** | [`research/`](./research) | 9 papers + análises Phase 1/2 |

## Performance

**Métrica:** seqeval BIOES F1.

| Modelo | Haiku-style holdout | gpt5nano-style holdout | Spread |
| --- | --- | --- | --- |
| v3 (Haiku-trained) | 0.9901 | 0.9947 | 0.0046 |
| v4 (gpt5nano-only) | **0.5534** ⚠️ | 0.9992 | **0.4458** 💥 |
| **v5 (mixed)** | **0.9912** ✓ | **0.9992** ✓ | **0.0080** ✓ |

**v5 estritamente domina v3 e v4.** Sem regressão em nenhum estilo de texto.

### Resumo da jornada

1. **v3** treinado com Claude Haiku como rewriter — F1 0.9900 no próprio holdout, generaliza ok.
2. **v3.1** com regex relabel parcial — F1 caiu (-0.0063) por buggy-holdout penalizing.
3. **v4** com gpt-5-nano + format-aware labeler — F1 0.9992 no próprio holdout MAS apenas 0.5534 no Haiku-style. Aprendeu a estender boundaries em markdown decorators (`**`, `\n`). Post-mortem em [`2026-05-24-v4-postmortem`](./research/wiki/questions/2026-05-24-v4-postmortem.md).
4. **v5** combina datasets v3+v4 (~100k) — modelo vê AS DUAS distribuições e generaliza ambas. F1 0.9912/0.9992 nos 2 estilos. Detalhes em [`2026-05-24-v5-results`](./research/wiki/questions/2026-05-24-v5-results.md).

## ⚠️ Caveat de honestidade

**F1 0.99 é em holdout sintético da mesma distribuição do treino.** Em inputs reais o desempenho cai. Exemplo concreto:

```
Input: "Pedido ML-2024-789456 do cliente CUST-998877. CNPJ 11.222.333/0001-81,
        faturou R$ 50.000,00 em outubro."

v5 prevê (esperado para produção):
  ✅ ML-2024-789456 → private_order_id
  ❌ CUST-998877 → mislabeled ou not detected (depending)
  ✅ 11.222.333/0001-81 → private_cnpj
  ❌ R$ 50.000,00 → não detectado em frases informais
```

Use [`br-pii-guardrail`](./br-pii-guardrail) para combinar regex+checksum (deterministic, ~100% recall em formatos canônicos) com a NER (cobertura para texto livre).

## Quick start

### Como modelo standalone

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

tok = AutoTokenizer.from_pretrained("lucianfialho/privacy-filter-br-v5")
model = AutoModelForTokenClassification.from_pretrained("lucianfialho/privacy-filter-br-v5")
ner = pipeline("token-classification", model=model, tokenizer=tok,
               aggregation_strategy="simple")

text = "Cliente João Silva, CPF 680.075.670-97, pedido ML-2024-789456"
for ent in ner(text):
    print(f"{ent['entity_group']:<25} {text[ent['start']:ent['end']]!r}")
```

### Como guardrail LGPD (recomendado pra produção)

```python
from br_pii_guardrail import Guardrail
from br_pii_guardrail.ner import NER

# Combina regex+checksum (CPF/CNPJ/cartão) + scanners (JSON/CSV/PDF) + NER
ner = NER("lucianfialho/privacy-filter-br-v5")
guard = Guardrail.default(ner=ner)

# Tokeniza com AES-GCM (vault reversível por tenant) antes de mandar pra LLM
tokenized, vault = guard.tokenize(text, tenant_id="cliente-x")
# ... chama LLM com `tokenized`...
plain = guard.detokenize(llm_response, vault)
```

## Categorias (22)

**OAI-compatible (8):** `account_number`, `private_address`, `private_date`, `private_email`, `private_person`, `private_phone`, `private_url`, `secret`

**BR-específicas (14):** `private_cpf`, `private_cnpj`, `private_rg`, `private_cnh`, `private_pis`, `private_titulo_eleitor`, `private_certidao`, `private_ie`, `private_order_id`, `private_tracking_code`, `private_invoice_number`, `private_client_revenue`, `private_transaction_id`, `private_customer_id`

Cada categoria estruturada (CPF/CNPJ/RG/PIS/CNH/IE/phone/etc) reconhece **múltiplas variantes de formato**:
- canonical: `680.075.670-97`
- raw: `68007567097`
- com espaços: `680 075 670 97`
- mascarado: `680.075.***-**`
- com separador alternativo: `680.075.670.97` (dot em vez de dash)

## Pipeline de geração (v3 e v4)

```
4devs API (profile gen)
   ↓ rate-limited, retries em 429
N perfis (CPF/CNPJ/RG válidos com checksum + nome + endereço + ...)
   ↓ variants.py (4 variantes por campo: canonical, raw, masked, espaços)
   ↓ Jinja templates (18 templates regulares + 4 extras)
N prompts pra LLM rewriter
   ↓ v3: Claude Haiku via CLI (mais diversidade textual)
   ↓ v4: gpt-5-nano via Batch API (50% discount, mais uniforme)
text + inserted dict
   ↓ src/labeler.py — format-aware skeleton matching (v4)
   ↓ src/validator.py
~50k train + ~5k holdout
   ↓ BERTimbau fine-tune (notebooks/finetune_v3_local.py)
checkpoints/v[3|4]-local/
```

## Arquitetura

- **Base:** `neuralmind/bert-base-portuguese-cased` (BERTimbau, 110M params, NeuralMind/UNICAMP)
- **Output head:** 89 labels (1 O + 22 categorias × 4 BIOES)
- **Decoding:** HF pipeline `aggregation_strategy="simple"` ou `"first"`

## Estrutura do repo

```
privacy-filter-br/
├── src/                          # generator + labeler + 4devs client + variants
│   ├── labeler.py                # format-aware skeleton matching
│   ├── variants.py               # 4 variantes por campo PII
│   ├── pessoa.py / fodevs.py     # 4devs API client
│   └── validator.py
├── templates/                    # 18 Jinja2 templates BR (NF, holerite, etc)
├── notebooks/finetune_v3_local.py # treino (8GB GPU friendly, MPS support)
├── scripts/                      # CLI tools
│   ├── openai_batch.py           # pipeline batch v4 (perfis → prepare → submit → process)
│   ├── audit_*.py                # auditorias dataset
│   ├── diagnose_v4_on_v3_holdout.py
│   ├── diff_v4_vs_v3holdout_spans.py
│   ├── benchmark_*.py            # cross-holdout 2x2
│   ├── publish_hf_v4.sh
│   └── publish_pypi.sh
├── br-pii-guardrail/             # lib de produção (regex+NER+AES vault)
├── spaces/privacy-filter-br-demo/ # Gradio Space
├── tests/                        # pytest suite (38 tests)
└── research/                     # LLM-maintained wiki (8 papers + análises)
    ├── CLAUDE.md
    ├── wiki/{sources,concepts,entities,questions}/
    └── log.md
```

## Casos de uso

**LGPD compliance:**
- Mascarar CPF/CNPJ em NF-e antes de auditoria externa
- Limpar contratos antes de enviar a sub-fornecedores
- Compliance em sistemas RH/Folha (CPF + dados bancários + PIS)

**Sigilo contratual / NDA:**
- BPO fiscal removendo PII antes de relatórios externos
- Agências mascarando dados de cliente em apresentações
- Logs internos sem vazamento de dados sensíveis

**LLM safe guardrail:**
- Tokenizar PII antes de mandar pra OpenAI/Claude (vault reversível por tenant)
- Validar saída de LLM antes de mostrar pra usuário final

**Treinamento de IA:**
- Limpar datasets corporativos antes de fine-tuning
- Anonimizar dados de prod pra ambiente dev

## Roadmap

- [x] v1: 13 categorias, fine-tune leve
- [x] v2: BERTimbau full fine-tune (22 categorias)
- [x] v3: dataset 50k via Haiku rewriter
- [x] v3.1: regex relabel + retrain (testado, abandonado por buggy-holdout)
- [x] v4: dataset 50k via gpt-5-nano + format-aware labeler (experimental, ver post-mortem)
- [x] **v5: dataset misto (Haiku + gpt-5-nano = ~100k) — production default**
- [ ] Phase 1: 50-100 docs reais labelizados manualmente → medir F1 real
- [ ] Avaliar GLiNER-Multi fine-tuned como v5 alternative

Decisões e benchmarks documentados em [`research/wiki/`](./research/wiki). Veja [`research/index.md`](./research/index.md) pra navegar.

## Licença

MIT (modelo + lib + scripts). Base BERTimbau é MIT-compatible.

## Créditos

- **Base model:** [BERTimbau](https://github.com/neuralmind-ai/portuguese-bert) (NeuralMind / UNICAMP)
- **PII generator:** [4devs.com.br](https://www.4devs.com.br) (gerador de PII brasileiro válido)
- **LLM rewriter:** Claude Haiku (v3 dataset), OpenAI gpt-5-nano (v4 dataset)

## Citação

```bibtex
@misc{privacy-filter-br-2026,
  author = {Lucian Fialho},
  title = {Privacy Filter BR — NER for Brazilian PII detection},
  year = {2026},
  publisher = {HuggingFace},
  url = {https://huggingface.co/lucianfialho/privacy-filter-br-v5}
}
```
