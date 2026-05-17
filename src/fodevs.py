import random
import requests
import json
import time

FODEVS_URL = "https://www.4devs.com.br/ferramentas_online.php"
HEADERS = {
    "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
    "x-requested-with": "XMLHttpRequest",
}


class Fodevs4:
    def __init__(self, base_url: str = FODEVS_URL, delay: float = 1.0):
        self.base_url = base_url
        self.delay = delay

    def _post(self, data: str, retries: int = 5) -> str:
        for attempt in range(retries):
            try:
                time.sleep(self.delay)
                resp = requests.post(self.base_url, headers=HEADERS, data=data, timeout=15)
                if resp.status_code == 429:
                    wait = 15 + 5 * attempt + random.uniform(0, 5)
                    if attempt == retries - 1:
                        resp.raise_for_status()
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.text.strip()
            except requests.HTTPError:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt + random.uniform(0, 1))
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt + random.uniform(0, 1))

    def gerar_cpf(self) -> str:
        return self._post("acao=gerar_cpf&pontuacao=S")

    def gerar_cnpj(self) -> str:
        return self._post("acao=gerar_cnpj&pontuacao=S")

    def gerar_rg(self) -> str:
        return self._post("acao=gerar_rg&pontuacao=S")

    def gerar_cnh(self) -> str:
        return self._post("acao=gerar_cnh")

    def gerar_titulo_eleitor(self, estado: str = "SP") -> str:
        return self._post(f"acao=gerar_titulo_eleitor&estado={estado}")

    def gerar_pis(self) -> str:
        return self._post("acao=gerar_pis")

    def gerar_certidao(self, tipo: str = "nascimento") -> str:
        return self._post(f"acao=gerador_certidao&tipo_certidao={tipo}&pontuacao=S")

    def gerar_ie(self, estado: str = "SP") -> str:
        return self._post(f"acao=gerar_ie&estado={estado}&pontuacao=S")

    def gerar_pessoa(self, estado: str = "SP") -> dict:
        raw = self._post(
            f"acao=gerar_pessoa&sexo=I&pontuacao=S&idade=0"
            f"&cep_estado={estado}&txt_qtde=1&cep_cidade="
        )
        return json.loads(raw)[0]
