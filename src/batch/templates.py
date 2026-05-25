"""Template lists for OpenAI Batch pipeline."""
from __future__ import annotations

TEMPLATES = [
    # Structured documents (v1-v5 baseline)
    "email", "nfe", "contrato", "holerite",
    "certidao", "cadastro", "comunicado", "relatorio",
    "nfe_completa", "darf", "boleto",
    "comprovante_pix", "extrato_bancario", "fatura_servico",
    "pedido_marketplace", "dashboard_vendas",
    "comprovante_delivery", "relatorio_faturamento",
    # Narrative / conversational (v6 — CUST/order disambig + revenue in prose)
    "artigo_noticia", "email_conversacional", "doc_tecnico",
    "nota_livre", "dialogo_chat", "email_thread",
    "comentario_sistema", "artigo_blog",
    "rh_perfil_narrativo", "incident_report",
]

EXTRAS_TEMPLATES = [
    "extras_devops_log", "extras_notification_email",
    "extras_bank_statement", "extras_api_docs",
]
