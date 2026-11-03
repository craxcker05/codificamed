"""Pruebas de recuperación con un codificador falso (sin descargar el modelo)."""

import numpy as np
import pandas as pd
import pytest

from codificamed.retrieval import (
    Indice, buscar, huella_catalogo, texto_consulta, texto_pasaje,
)

CATALOGO = pd.DataFrame({
    "codigo": ["J189", "I10", "R509"],
    "codigo_original": ["J18.9", "I10", "R50.9"],
    "descripcion_es": ["Neumonía, no especificada", "Hipertensión esencial", "Fiebre, no especificada"],
})
VECTORES = {"neumonía": [1, 0, 0], "hipertensión": [0, 1, 0], "fiebre": [0, 0, 1]}


def codificador_falso(textos):
    visto.extend(textos)
    out = []
    for t in textos:
        v = np.zeros(3, dtype=np.float32)
        for palabra, vec in VECTORES.items():
            if palabra in t.lower():
                v += np.array(vec, dtype=np.float32)
        out.append(v / (np.linalg.norm(v) or 1))
    return np.array(out)


visto: list[str] = []


def indice() -> Indice:
    emb = codificador_falso([texto_pasaje(d) for d in CATALOGO["descripcion_es"]])
    return Indice(emb.astype(np.float16), CATALOGO, codificador_falso)


def test_prefijos() -> None:
    assert texto_pasaje(" Fiebre ") == "passage: Fiebre"
    assert texto_consulta("dolor   de  cabeza") == "query: dolor de cabeza"


def test_buscar_ordena_por_score_y_devuelve_codigos_del_catalogo() -> None:
    r = buscar("fiebre alta", k=2, indice=indice())
    assert [c.codigo for c in r][0] == "R509"
    assert r[0].score >= r[1].score
    assert {c.codigo for c in r} <= set(CATALOGO["codigo"])
    assert r[0].codigo_original == "R50.9"


def test_la_consulta_usa_prefijo_query() -> None:
    visto.clear()
    buscar("neumonía", k=1, indice=indice())
    assert visto[-1] == "query: neumonía"


def test_k_mayor_que_catalogo() -> None:
    assert len(buscar("fiebre", k=50, indice=indice())) == 3


def test_indice_y_catalogo_deben_coincidir() -> None:
    with pytest.raises(ValueError):
        Indice(np.zeros((2, 3), dtype=np.float16), CATALOGO, codificador_falso)


def test_huella_cambia_si_cambia_el_catalogo() -> None:
    otro = CATALOGO.copy()
    otro.loc[0, "descripcion_es"] = "Otra cosa"
    assert huella_catalogo(CATALOGO) != huella_catalogo(otro)
