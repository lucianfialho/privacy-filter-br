import random
from dataclasses import dataclass, field
from src.pessoa import gerar_perfil_completo
from src.variants import get_variants_for_perfil, pick_variant
from src.labeler import label_text
from src.validator import validate_example, ValidationResult
from src.haiku import HaikuGenerator

TEMPLATES = [
    # Originais
    "email", "nfe", "contrato", "holerite",
    "certidao", "cadastro", "comunicado", "relatorio",
    # B2B novos
    "nfe_completa", "darf", "boleto",
    "comprovante_pix", "extrato_bancario", "fatura_servico",
]


@dataclass
class GeneratorStats:
    total: int = 0
    valid: int = 0
    discarded: int = 0
    discard_reasons: dict = field(default_factory=dict)

    def record(self, example, reason: str | None):
        self.total += 1
        if example is not None:
            self.valid += 1
        else:
            self.discarded += 1
            if reason:
                self.discard_reasons[reason] = self.discard_reasons.get(reason, 0) + 1


def generate_example(haiku: HaikuGenerator | None = None) -> dict | None:
    """
    Generates one labeled example. Returns None if any step fails.
    Orchestrates: 4devs → variants → Haiku → validate → label.
    """
    if haiku is None:
        haiku = HaikuGenerator()

    try:
        perfil = gerar_perfil_completo()
    except Exception:
        return None
    variants = get_variants_for_perfil(perfil)

    chosen = {}
    inserted = {}

    for field_name, vs in variants.items():
        if not vs:
            continue
        value, label = pick_variant(vs)
        chosen[f"{field_name}_valor"] = value
        chosen[f"{field_name}_label"] = label.replace("PRIVATE_", "").title()
        # Register ALL variants so labeler finds the value even if Claude reformats it
        for v, lbl in vs:
            if v:
                inserted[v] = lbl

    inserted[perfil["nome"]] = "PRIVATE_PERSON"
    inserted[perfil["email"]] = "PRIVATE_EMAIL"
    chosen["nome"] = perfil["nome"]
    chosen["email"] = perfil["email"]
    chosen["cidade"] = perfil["cidade"]
    chosen["estado"] = perfil["estado"]
    chosen["data_nasc"] = perfil["data_nasc"]
    chosen["endereco"] = perfil["endereco"]

    template_name = random.choice(TEMPLATES)
    try:
        text = haiku.generate(template_name, chosen)
    except Exception:
        return None

    example = label_text(text, inserted)
    result = validate_example(example)
    if result != ValidationResult.VALID:
        return None

    example["template"] = template_name
    return example
