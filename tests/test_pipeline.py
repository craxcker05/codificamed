"""Pipeline con LLM e índice falsos (sin red ni modelos)."""

import json

import numpy as np
import pandas as pd

from codificamed.pipeline import analizar
from codificamed.retrieval import Indice

NOTA = "Paciente con HTA. Niega fiebre. Refiere cefalea occipital."


class LLMFalso:
    nombre, modelo = "falso", "falso-1"

    def generar(self, prompt: str, sistema: str | None = None) -> str:
        if "Extrae" in prompt:
            return json.dumps({"menciones": ["cefalea occipital", "fiebre", "inventada"]})
        return '```json\n{"codigo": "R51", "confianza": 0.9}\n```'


def _indice() -> Indice:
    cat = pd.read_parquet("data/processed/catalogo_cie10.parquet")
    cat = pd.concat([cat[cat["codigo"] == "R51"], cat.head(20)]).drop_duplicates("codigo")
    emb = np.eye(len(cat), 8, dtype=np.float32)
    return Indice(emb, cat, lambda t: np.ones((len(t), 8), dtype=np.float32))


def test_pipeline_sugiere_separa_negados_y_descarta():
    res = analizar(NOTA, LLMFalso(), _indice())
    assert [s.codigo for s in res.sugerencias] == ["R51"]
    s = res.sugerencias[0]
    assert NOTA[s.offsets[0]:s.offsets[1]] == s.evidencia == "cefalea occipital"
    assert [(n.evidencia, n.estado) for n in res.negados] == [("fiebre", "negado")]
    assert res.descartadas == ["inventada"]
    assert res.llamadas_llm == 2
