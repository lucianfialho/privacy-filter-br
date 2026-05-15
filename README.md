# Privacy Filter BR

Fine-tune do [OpenAI Privacy Filter](https://huggingface.co/openai/privacy-filter) especializado em **detecção de PII em documentos comerciais brasileiros**.

**Foco:** documentos B2B (NF-e, holerite, contrato, certidão, cadastro, comunicado, relatório, e-mail corporativo) com categorias específicas que outros modelos PT-BR não cobrem.

**Não substitui** [arthrod/gliner-opf-ptbr-pii-v1](https://huggingface.co/arthrod/gliner-opf-ptbr-pii-v1) — **complementa**. Use o arthrod para PII pessoal genérico, este para PII em documentos comerciais brasileiros.

## Performance

F1 macro: **0.9755** em holdout interno (1.047 exemplos sintéticos do mesmo pipeline de treino).

| Categoria | F1 |
|---|---|
| PRIVATE_CERTIDAO | 0.9963 |
| PRIVATE_TITULO_ELEITOR | 0.9976 |
| PRIVATE_ADDRESS | 0.9974 |
| PRIVATE_PIS | 0.9970 |
| PRIVATE_RG | 0.9898 |
| PRIVATE_CPF | 0.9906 |
| PRIVATE_PERSON | 0.9803 |
| PRIVATE_EMAIL | 0.9701 |
| PRIVATE_CNH | 0.9639 |
| PRIVATE_IE | 0.9452 |
| PRIVATE_CNPJ | 0.9423 |
| PRIVATE_PHONE | 0.9354 |

⚠️ **Importante:** F1 medido em holdout sintético da mesma distribuição do treino. Performance em documentos reais pode ser menor — validação em produção recomendada antes de uso crítico.

## Categorias

13 categorias específicas de PII brasileiro, em formato BIOES (53 classes total = 1 background + 13×4):

```
PRIVATE_CPF              CPF
PRIVATE_CNPJ             CNPJ
PRIVATE_RG               RG
PRIVATE_CNH              CNH
PRIVATE_PIS              PIS/PASEP
PRIVATE_TITULO_ELEITOR   Título de Eleitor
PRIVATE_CERTIDAO         Número de certidão
PRIVATE_IE               Inscrição Estadual (27 formatos)
PRIVATE_PERSON           Nome
PRIVATE_EMAIL            E-mail
PRIVATE_PHONE            Telefone/celular
PRIVATE_ADDRESS          CEP + endereço
PRIVATE_DATE             Data de nascimento
```

**Diferenciais frente a outros modelos PT-BR:**
- ✅ CNPJ como categoria distinta (não agrupado em `account_number`)
- ✅ Inscrição Estadual com 27 formatos diferentes por estado
- ✅ CNH, Título de Eleitor e Certidão como categorias separadas
- ✅ Variantes de formato: formatado (`680.075.670-97`), raw (`68007567097`), parcial (`680.075.***-**`), com espaços (`680 075 670 97`)

## Casos de Uso

**LGPD compliance em B2B:**
- Mascarar CPF de tomador em NF-e antes de auditoria
- Detectar dados sensíveis em holerite (Art. 5º II LGPD)
- Limpar contratos antes de enviar a sub-fornecedores
- Compliance em sistemas RH/Folha

**Sigilo contratual / NDA:**
- Agências e consultorias mascarando dados de cliente
- BPO fiscal removendo PII antes de relatórios externos
- Logs internos sem vazamento de dados sensíveis

**Treinamento de IA:**
- Limpar datasets corporativos antes de fine-tuning
- Anonimizar dados de produção para ambiente de dev

## Uso

```python
from transformers import pipeline

pipe = pipeline(
    'token-classification',
    model='metricasboss/privacy-filter-br',
    aggregation_strategy='first',
    trust_remote_code=True,
)

text = "Cliente João Silva (CPF 680.075.670-97, RG 27.141.489-3) realizou compra. CNPJ emitente: 72.682.864/0001-41."
for r in pipe(text):
    print(f"{r['entity_group']}: {r['word']}")
```

Output esperado:
```
PRIVATE_PERSON: João Silva
PRIVATE_CPF: 680.075.670-97
PRIVATE_RG: 27.141.489-3
PRIVATE_CNPJ: 72.682.864/0001-41
```

## Detalhes Técnicos

- **Base:** openai/privacy-filter (1.5B total / 50M active params, MoE)
- **Arquitetura:** bidirectional token classifier
- **Output head:** 53 classes (1 + 13 categorias × 4 BIOES)
- **Fine-tuning:** LoRA (r=16, alpha=32) em q_proj, v_proj + classification head full
- **Trainable params:** 328k (0.02% do total)
- **Training data:** 11.000 exemplos sintéticos (4devs API + Mixtral 8x7B reescreve em PT-BR)
- **Training:** 5 épocas no Colab A100, ~1.5h, F1 saturou em ~3 épocas
- **License:** Apache 2.0 (mesma do base model)

## Limitações

1. **Dataset 100% sintético** — não foi treinado em documentos reais brasileiros
2. **Holdout same-distribution** — F1 reportado é otimista comparado a uso real
3. **Domínio comercial estreito** — 8 tipos de documento. Pode falhar em outros tipos
4. **CPF raw (sem formatação)** — taxa de erro maior (confunde com outros números)
5. **Inscrição Estadual** — categoria mais fraca (F1 0.945) por causa dos 27 formatos diferentes

## Comparação

| | privacy-filter-br | arthrod/gliner-opf-ptbr-pii-v1 | OpenMed multilingual |
|---|---|---|---|
| Foco | Documentos B2B brasileiros | PII pessoal PT-BR genérico | Multilingual (16 línguas) |
| Training data | 11k sintético | 914k natural-text | AI4Privacy mix |
| CNPJ específico | ✅ | ❌ (em `account_number`) | ❌ |
| IE 27 estados | ✅ | ❌ | ❌ |
| Person granular (first/middle/last) | ❌ | ✅ | ✅ |
| LGPD sensitive (Art. 5º II) | ❌ | ✅ | parcial |

**Recomendação:** combine os dois. arthrod para PII pessoal, este para documentos comerciais.

## Roadmap

- [ ] v2: combinar com dataset arthrod (914k natural) + 100k novos B2B-específicos
- [ ] v2: adicionar categorias DARF, PIX key, conta bancária, NF-e key
- [ ] Validação em corpus real B2B (ainda pendente)
- [ ] Avaliar quantização para CPU (50M params ativos = inferência leve)

## Citação

Se usar em pesquisa:

```bibtex
@misc{privacy-filter-br-2026,
  author = {Métricas Boss},
  title = {Privacy Filter BR — Brazilian B2B PII Detection},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/metricasboss/privacy-filter-br}
}
```

## Créditos

- **Base model:** OpenAI Privacy Filter (Apache 2.0)
- **Inspiração para PT-BR:** arthrod/gliner-opf-ptbr-pii-v1
- **Dados sintéticos:** 4devs.com.br (gerador de PII brasileiro)

## Licença

Apache 2.0
