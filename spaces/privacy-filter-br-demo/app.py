"""Privacy Filter BR — Gradio Demo for HF Spaces."""
import os
import json
from collections import defaultdict

import gradio as gr
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

MODEL = os.environ.get("MODEL_ID", "lucianfialho/privacy-filter-br")

print(f"Loading {MODEL}...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForTokenClassification.from_pretrained(MODEL).eval()
ner = pipeline(
    "token-classification", model=model, tokenizer=tok,
    aggregation_strategy="simple", device=-1,
)

COLORS = {
    "private_cpf":             "#ef4444",
    "private_cnpj":            "#f97316",
    "private_rg":              "#f59e0b",
    "private_cnh":             "#fbbf24",
    "private_pis":             "#eab308",
    "private_titulo_eleitor":  "#a3e635",
    "private_certidao":        "#84cc16",
    "private_ie":              "#22c55e",
    "private_person":          "#10b981",
    "private_email":           "#14b8a6",
    "private_phone":           "#06b6d4",
    "private_address":         "#0ea5e9",
    "private_date":            "#3b82f6",
    "private_url":             "#6366f1",
    "private_order_id":        "#8b5cf6",
    "private_tracking_code":   "#a855f7",
    "private_invoice_number":  "#c026d3",
    "private_customer_id":     "#d946ef",
    "private_transaction_id":  "#ec4899",
    "private_client_revenue":  "#f43f5e",
    "account_number":          "#94a3b8",
    "secret":                  "#dc2626",
}


def highlight_html(text, entities):
    if not entities:
        return f"<div style='padding:1rem;color:#888;'>Nenhum PII detectado.</div>"
    out = []
    last = 0
    for ent in sorted(entities, key=lambda e: e["start"]):
        s, e = ent["start"], ent["end"]
        if s < last:
            continue
        if s > last:
            seg = text[last:s].replace("<", "&lt;").replace("\n", "<br>")
            out.append(f"<span style='color:#9ca3af;'>{seg}</span>")
        color = COLORS.get(ent["entity_group"], "#64748b")
        label = ent["entity_group"].replace("private_", "")
        out.append(
            f"<span style='background:{color}33;border:1.5px solid {color};"
            f"border-radius:6px;padding:3px 8px;margin:2px 3px;display:inline-block;'>"
            f"<b style='color:{color};'>{text[s:e]}</b>"
            f"<sub style='color:{color};font-weight:700;font-size:0.65em;margin-left:5px;"
            f"text-transform:uppercase;'>{label}</sub></span>"
        )
        last = e
    if last < len(text):
        seg = text[last:].replace("<", "&lt;").replace("\n", "<br>")
        out.append(f"<span style='color:#9ca3af;'>{seg}</span>")
    return (
        f"<div style='line-height:2.4;padding:1.2rem;font-family:system-ui,sans-serif;"
        f"font-size:0.95em;background:rgba(255,255,255,0.02);border-radius:8px;'>"
        f"{''.join(out)}</div>"
    )


def stats_md(entities):
    if not entities:
        return "*Cole um texto e clique em **Detectar PII** pra ver as estatísticas.*"
    counts = defaultdict(list)
    for e in entities:
        counts[e["entity_group"]].append(e["score"])
    rows = ["| Categoria | Qtd | Confiança média |", "|:---|:---:|:---:|"]
    for label, scores in sorted(counts.items(), key=lambda x: -len(x[1])):
        avg = sum(scores) / len(scores)
        rows.append(f"| `{label}` | {len(scores)} | {avg:.3f} |")
    return "\n".join(rows)



def merge_consecutive(entities, max_gap=3):
    """Junta spans consecutivos com mesmo label (gap <= max_gap chars)."""
    if not entities:
        return entities
    sorted_ents = sorted(entities, key=lambda e: e["start"])
    merged = [dict(sorted_ents[0])]
    for ent in sorted_ents[1:]:
        prev = merged[-1]
        same_label = ent["entity_group"] == prev["entity_group"]
        small_gap = ent["start"] - prev["end"] <= max_gap
        if same_label and small_gap:
            prev["end"] = ent["end"]
            prev["word"] = (prev["word"] + " " + ent["word"]).replace("  ", " ").strip()
            prev["score"] = max(prev["score"], ent["score"])
        else:
            merged.append(dict(ent))
    return merged


def analyze(text):
    if not text or not text.strip():
        return (
            "<div style='padding:1rem;color:#888;'>Cole um texto pra começar.</div>",
            "*nenhum input*",
            "[]",
        )
    entities = ner(text)
    entities = merge_consecutive(entities)
    clean = [
        {k: (float(v) if hasattr(v, "item") else v) for k, v in e.items()}
        for e in entities
    ]
    return (
        highlight_html(text, clean),
        stats_md(clean),
        json.dumps(clean, ensure_ascii=False, indent=2),
    )


EXAMPLES = [
    ["Cliente João Silva, CPF 680.075.670-97, telefone (11) 98765-4321, email joao@empresa.com.br, mora na Rua das Flores 123, São Paulo/SP."],
    ["Pedido ML-2024-789456 do cliente CUST-998877. Empresa Acme Comércio Ltda, CNPJ 11.222.333/0001-81, faturou R$ 50.000,00 em outubro. Rastreio Correios BR123456789BR. Fatura FAT-2025-001."],
    ["Em 15/01/2026, acesse https://portal.empresa.com/reset?token=abc123def456 antes do prazo. API Key: sk-proj-1234567890abcdef. Cartão de teste: 4111 1111 1111 1111. Conta: Ag. 1234 / Conta 56789-0."],
    ["Funcionário Maria Souza, PIS 123.45678.90-1, RG 12.345.678-9, endereço Rua das Flores 123, CEP 01234-567, São Paulo/SP. CNH 12345678901 vencimento 31/12/2027."],
]

with gr.Blocks(title="Privacy Filter BR v6", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🇧🇷 Privacy Filter BR v6 — Demo
        Detector de PII em português brasileiro. **22 categorias**: CPF, CNPJ, RG, telefone, email, endereço, e categorias B2B (order_id, customer_id, tracking, invoice, revenue).

        - **F1 macro 0.9934** no holdout sintético
        - Modelo: [`lucianfialho/privacy-filter-br`](https://huggingface.co/lucianfialho/privacy-filter-br) — BERTimbau 110M (latest = v7, boundary-merged, Phase 1 CVM F1 0.90 overlap)
        - Lib de produção (regex+checksum+tokenization): [`br-pii-guardrail`](https://github.com/lucianfialho/privacy-filter-br/tree/main/br-pii-guardrail)
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="Cole o texto pra escanear",
                placeholder="Ex: João Silva, CPF 123.456.789-00, pedido ML-2024-789456...",
                lines=10,
            )
            with gr.Row():
                btn = gr.Button("🔍 Detectar PII", variant="primary", scale=2)
                clear_btn = gr.Button("Limpar", scale=1)
        with gr.Column(scale=3):
            highlights = gr.HTML(
                label="Detecções com highlights",
                value="<div style='padding:1rem;color:#888;'>Cole um texto e clique em <b>Detectar PII</b>.</div>",
            )
            stats = gr.Markdown(value="*Resultados aparecem aqui após a detecção.*")
            with gr.Accordion("JSON output (para debug)", open=False):
                json_out = gr.Code(language="json", lines=12, value="[]")

    gr.Examples(
        examples=EXAMPLES, inputs=input_text,
        label="Exemplos clicáveis",
        examples_per_page=4,
    )

    btn.click(analyze, inputs=input_text, outputs=[highlights, stats, json_out])
    input_text.submit(analyze, inputs=input_text, outputs=[highlights, stats, json_out])
    clear_btn.click(
        lambda: (
            "<div style='padding:1rem;color:#888;'>Cole um texto e clique em <b>Detectar PII</b>.</div>",
            "*Resultados aparecem aqui após a detecção.*",
            "[]",
        ),
        outputs=[highlights, stats, json_out],
    )

if __name__ == "__main__":
    demo.launch()
