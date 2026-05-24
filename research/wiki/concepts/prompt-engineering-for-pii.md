---
type: concept
tags: [prompt-engineering, llm, pii, de-identification]
sources: 1
updated: 2026-05-23
---

# Prompt engineering for PII detection

How to structure a prompt so an LLM reliably finds and redacts PII. Primary source: [[../sources/2026-05-23-deid-gpt]] (DeID-GPT, Liu et al, 2023).

## The DeID-GPT 3-part template

```
TASK STATEMENT
   You are an expert in de-identifying clinical notes.
   Your task is to anonymize the following clinical note in compliance with HIPAA.

COMMAND
   Replace all the following information with the term "[redacted]":

SPECIFIC RULES
   - Redact any patient names, e.g., "Mr. James McCarthy"
   - Redact any geographic information smaller than state, e.g., "3970 Longview Drive, Boston"
   - Redact any date elements except year, e.g., "12/01/2013"
   - Redact phone numbers, e.g., "555-1234"
   - Redact email addresses
   - Redact social security numbers, e.g., "123-45-6789"
   - Redact medical record numbers
   - Redact account numbers
   - ... (one rule per HIPAA category, each with a concrete example)

INPUT
   [clinical note text here]
```

## What each part does

1. **Task statement** — sets role and goal. The autoregressive nature of GPT models means information stated first conditions everything that follows. Tell the model what it's doing before how to do it.

2. **Command** — defines the action verb and output token. "Replace X with `[redacted]`" gives the model a deterministic output format that's also easy to post-process (regex on `[redacted]`).

3. **Specific rules with examples** — one rule per category. Each rule has a concrete example. Examples are not optional — they anchor the model's category understanding to a specific lexical pattern.

## Empirical impact (DeID-GPT)

|Prompt type | ChatGPT (GPT-3.5) | GPT-4 |
| --- | --- | --- |
| Implicit ("Please anonymize this clinical note") | 0.686 | 0.908 |
| Explicit (3-part template) | **0.929** | **0.99** |

+24 points on ChatGPT from prompt structure alone. GPT-4 is more robust to bad prompts (+8 from explicit), but still benefits significantly.

## Anti-patterns flagged by the paper

1. **Stating only the task.** "Please anonymize this" → model guesses categories, misses many.
2. **Punctuation errors.** Extra period after the command made ChatGPT fail to parse. LLMs are surprisingly brittle to syntax in instructions.
3. **Multiple task statements.** Confuses the model about priority. One task per prompt.
4. **Not specifying output format.** Without "replace with `[redacted]`" the model invents formats (`<PII>`, `***`, removes entirely, etc) and post-processing breaks.

## Adapting for our LGPD-aligned PII

For `br-pii-guardrail` LLM fallback, the prompt should:

1. State LGPD context (not HIPAA) — "remova informações pessoais conforme LGPD".
2. Use Portuguese category names (`CPF`, `CNPJ`, `CEP`, `endereço`, etc) since the LLM has seen these in its training data — don't translate to "Brazilian individual taxpayer ID".
3. Include 2-3 concrete BR examples per category (real format, fake values):
   - CPF: "123.456.789-00"
   - CNPJ: "12.345.678/0001-90"
   - CEP: "01310-100"
   - RG: "12.345.678-9"
   - Telefone: "(11) 98765-4321"
4. Specify deterministic output token: replace with `[REDACTED:CATEGORY]` so we can both (a) confirm a redaction happened and (b) know which category for downstream tokenization/restoration.

## Draft PT-BR prompt template

```
Você é um especialista em remover informações pessoais identificáveis (PII) 
de textos em português brasileiro, conforme a LGPD.

Sua tarefa: substituir todas as informações pessoais no texto abaixo pelo 
formato `[REDACTED:CATEGORIA]`, onde CATEGORIA é uma das listadas abaixo.

Categorias e exemplos:
- private_person: nome de pessoa, ex: "João da Silva"
- private_cpf: CPF, ex: "123.456.789-00"
- private_cnpj: CNPJ, ex: "12.345.678/0001-90"
- private_rg: RG, ex: "12.345.678-9"
- private_email: email, ex: "joao@exemplo.com"
- private_phone: telefone, ex: "(11) 98765-4321"
- private_address: endereço, ex: "Rua das Flores, 123, Centro"
- private_cep: CEP, ex: "01310-100"
- ... (one rule per category, with example)

Preserve toda a estrutura e o significado do texto. Substitua APENAS as 
informações pessoais listadas.

Texto:
[INPUT]
```

## Open questions

- Does few-shot (2-3 fully-redacted examples in the prompt) beat zero-shot with category descriptions? DeID-GPT only tests zero-shot.
- Does Chain-of-Thought ("First identify all entities, then output redacted text") help or hurt? Untested in paper.
- How does prompt length affect cost vs accuracy? 22 categories × 3 examples = ~600 tokens of prompt overhead per call. Worth it?
- Are model-specific prompts (one for GPT-4, one for Claude, one for Llama) substantially different?

## Related

- [[../sources/2026-05-23-deid-gpt]] — empirical source
- [[llm-based-pii-detection]] — the broader paradigm
- [[zero-shot-ner]] — encoder-based equivalent (uses type embeddings instead of prompt text)
