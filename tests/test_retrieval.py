"""Pruebas de recuperación con un codificador falso (sin descargar el modelo)."""

import numpy as np
import pandas as pd
import pytest

from codificamed.retrieval import (
    Indice, buscar, get_or_build_consultas, get_or_build_embeddings, huella_catalogo, mejores, puntajes,
    texto_consulta, texto_pasaje,
)

CATALOGO = pd.DataFrame({
    "codigo": ["J189", "I10", "R509"],
    "codigo_original": ["J18.9", "I10", "R50.9"],
    "descripcion_es": ["Neumonía, no especificada", "Hipertensión esencial", "Fiebre, no especificada"],
    "descripcion_en": ["Pneumonia (neumonía)", "Essential (hipertensión)", ""],
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


def contador():
    llamadas = []

    def cod(textos):
        llamadas.append(list(textos))
        return codificador_falso(textos)
    return cod, llamadas


def test_cache_no_recalcula_y_detecta_cambios(tmp_path) -> None:
    cod, llamadas = contador()
    a = get_or_build_embeddings("es", "m/x", catalogo=CATALOGO, codificador=cod, dir_cache=tmp_path)
    assert len(llamadas) == 1 and (tmp_path / "es_m__x.npy").exists() and (tmp_path / "es_m__x.json").exists()
    b = get_or_build_embeddings("es", "m/x", catalogo=CATALOGO, codificador=cod, dir_cache=tmp_path)
    assert len(llamadas) == 1 and np.array_equal(a, b)  # segunda vez: del cache
    otro = CATALOGO.copy()
    otro.loc[0, "descripcion_es"] = "Fiebre reumática"
    get_or_build_embeddings("es", "m/x", catalogo=otro, codificador=cod, dir_cache=tmp_path)
    assert len(llamadas) == 2  # huella distinta: recalcula
    get_or_build_embeddings("es", "m/x", catalogo=otro, codificador=cod, dir_cache=tmp_path, force=True)
    assert len(llamadas) == 3  # --force


def test_descripciones_vacias_quedan_en_cero(tmp_path) -> None:
    cod, _ = contador()
    en = get_or_build_embeddings("en", "m/x", catalogo=CATALOGO, codificador=cod, dir_cache=tmp_path)
    assert not en[2].any() and en[0].any()


def test_modo_combinado_es_el_maximo() -> None:
    es = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float16)
    en = np.array([[0, 0, 1], [0, 1, 0]], dtype=np.float16)
    q = np.array([[0, 0, 1]], dtype=np.float32)
    assert puntajes(q, es, en, "es").tolist() == [[0.0, 0.0]]
    assert puntajes(q, es, en, "combinado").tolist() == [[1.0, 0.0]]
    assert mejores(puntajes(q, es, en, "combinado"), 1).tolist() == [[0]]


def test_modo_en_requiere_embeddings_en() -> None:
    with pytest.raises(ValueError):
        puntajes(np.zeros((1, 3)), np.zeros((2, 3)), None, "en")


def test_consultas_en_cache_incremental(tmp_path) -> None:
    cod, llamadas = contador()
    get_or_build_consultas(["fiebre", "neumonía"], "m/x", codificador=cod, dir_cache=tmp_path)
    r = get_or_build_consultas(["neumonía", "fiebre", "hipertensión"], "m/x", codificador=cod, dir_cache=tmp_path)
    assert llamadas[1] == ["query: hipertensión"] and r.shape[0] == 3


def test_indice_expandir_abreviaturas() -> None:
    visto.clear()
    idx = indice()
    idx.expandir = True
    idx.buscar("HTA", k=1)
    assert visto[-1].lower() == "query: hipertensión arterial"  # mayúscula inicial por inicio de oración
