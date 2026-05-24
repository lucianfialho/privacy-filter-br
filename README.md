# Privacy Filter BR

Detecção de PII em documentos brasileiros (NF-e, holerite, contrato, certidão, e-mail corporativo, dashboards, etc) — fine-tune do **BERTimbau** (110M params) cobrindo 22 categorias de PII com formato canônico brasileiro.

Repositório completo: modelo + lib de produção + research wiki + scripts de auditoria + Gradio demo.

## Componentes

| Componente | Onde | Estado |
| --- | --- | --- |
| **Modelo v3 (production default)** | [`lucianfialho/privacy-filter-br-v3`](https://huggingface.co/lucianfialho/privacy-filter-br-v3) | publicado |
| **Modelo v4 (experimental)** | [`lucianfialho/privacy-filter-br-v4`](https://huggingface.co/lucianfialho/privacy-filter-br-v4) | ⚠️ vê caveat abaixo |
| **Demo Gradio** | [`spaces/privacy-filter-br-demo`](https://huggingface.co/spaces/lucianfialho/privacy-filter-br-demo) | live (v3) |
| **Lib `br-pii-guardrail`** | [`br-pii-guardrail/`](./br-pii-guardrail) | 0.1.2 (regex+checksum+NER+AES vault) |
| **Research wiki** | [`research/`](./research) | 8 papers + análises Phase 1/2 |

## Performance

**Métrica:** seqeval BIOES F1 em holdout sintético.

| Modelo | F1 (próprio holdout) | F1 (cross-holdout v3) | Notas |
| --- | --- | --- | --- |
| v3 | 0.9900 | 0.9901 | **production default** — generaliza bem |
| v3.1 | 0.9837 | — | regex relabel intermediário |
| v4 | 0.9992 | **0.5534** ⚠️ | overfit ao estilo de gpt-5-nano (rewriter usado no v4 dataset) |

**Achado crítico cross-holdout:** v4 atinge F1 0.9992 no próprio holdout (gerado com gpt-5-nano), mas **despenca para 0.5534 no holdout do v3** (gerado com Claude Haiku). v3 ao contrário **generaliza** — atinge F1 0.9947 no holdout do v4.

Diagnóstico: v4 aprendeu a estender boundaries em decoradores markdown (`**`, `\n`, `- `) que gpt-5-nano sempre produz. Em texto Haiku-style ou texto livre, esses padrões somem e v4 falha.

**Por isso v3 continua sendo o default para produção.** v4 fica disponível como artefato de pesquisa.

Detalhes em [`research/wiki/questions/2026-05-24-v4-postmortem.md`](./research/wiki/questions/2026-05-24-v4-postmortem.md).

## ⚠️ Caveat de honestidade

**F1 0.99 é em holdout sintético da mesma distribuição do treino.** Em inputs reais o desempenho cai. Exemplo concreto:

```
Input: "Pedido ML-2024-789456 do cliente CUST-998877. CNPJ 11.222.333/0001-81,
        faturou R$ 50.000,00 em outubro."

v3 prevê (esperado para produção):
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

tok = AutoTokenizer.from_pretrained("lucianfialho/privacy-filter-br-v3")
model = AutoModelForTokenClassification.from_pretrained("lucianfialho/privacy-filter-br-v3")
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
ner = NER("lucianfialho/privacy-filter-br-v3")
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
- [x] **v3: dataset 50k via Haiku rewriter — production default**
- [x] v3.1: regex relabel + retrain (testado, abandonado por buggy-holdout)
- [x] v4: dataset 50k via gpt-5-nano + format-aware labeler (experimental, ver post-mortem)
- [ ] v5: dataset misto (Haiku + gpt-5-nano + outros) + boundary post-processing
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
  url = {https://huggingface.co/lucianfialho/privacy-filter-br-v3}
}
```
