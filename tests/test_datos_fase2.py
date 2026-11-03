"""Pruebas de la Fase 2: normalización, unicidad del catálogo, casos de evaluación y notas ficticias.

Las pruebas que necesitan los archivos generados por scripts/prepare_data.py
se omiten (skip) si el archivo no existe, para que la suite corra también en un clon limpio.
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from codificamed.codigos import detectar_colisiones, es_formato_valido, normalizar_codigo
from codificamed.preparacion import (
    ColisionError,
    construir_catalogo,
    leer_jsonl,
    validar_notas_ficticias,
)
from config.settings import DIR_PROCESSED, RAIZ_PROYECTO

RUTA_CATALOGO = DIR_PROCESSED / "catalogo_cie10.parquet"
RUTA_CASOS = DIR_PROCESSED / "casos_eval.jsonl"
RUTA_NOTAS = RAIZ_PROYECTO / "data" / "ficticios" / "notas_ficticias.jsonl"
RUTA_ABREVIATURAS = RAIZ_PROYECTO / "config" / "abreviaturas.yaml"
CARPETA_DEV = RAIZ_PROYECTO / "data" / "raw" / "final_dataset_v2_to_publish" / "dev" / "text_files"


def _catalogo_o_saltar() -> pd.DataFrame:
    if not RUTA_CATALOGO.exists():
        pytest.skip("Falta data/processed/catalogo_cie10.parquet (ejecutar scripts/prepare_data.py)")
    return pd.read_parquet(RUTA_CATALOGO)


# --- Normalización ---------------------------------------------------------


@pytest.mark.parametrize(
    ("entrada", "esperado"),
    [("A01.00", "A0100"), ("n44.8", "N448"), (" r52 ", "R52"), ("Z94.81", "Z9481")],
)
def test_normalizar_quita_punto_y_pasa_a_mayusculas(entrada: str, esperado: str) -> None:
    assert normalizar_codigo(entrada) == esperado


def test_formato_valido_acepta_codigos_normalizados() -> None:
    for codigo in ["A0100", "R52", "C4A0", "Z9481", "S6701XA"]:
        assert es_formato_valido(codigo), codigo


def test_formato_invalido_rechaza_puntos_y_minusculas() -> None:
    assert not es_formato_valido("A01.00")
    assert not es_formato_valido("a0100")
    assert not es_formato_valido("12")


def test_colision_detectada_cuando_dos_originales_quedan_iguales() -> None:
    colisiones = detectar_colisiones(["A01.00", "A0100", "B20"])
    assert colisiones == {"A0100": ["A01.00", "A0100"]}


def test_sin_colision_si_los_normalizados_son_distintos() -> None:
    assert detectar_colisiones(["A01.00", "A01.01"]) == {}


def test_construir_catalogo_detiene_ante_colision() -> None:
    fuente = pd.DataFrame(
        {
            "codigo": ["A01.00", "A0100"],
            "descripcion_es": ["x", "y"],
            "descripcion_en": ["x", "y"],
        }
    )
    with pytest.raises(ColisionError):
        construir_catalogo(fuente)


def test_construir_catalogo_conserva_original_y_categoria() -> None:
    fuente = pd.DataFrame(
        {
            "codigo": ["A01.00", "Z94.81"],
            "descripcion_es": ["Fiebre tifoidea", "Trasplante"],
            "descripcion_en": ["Typhoid", ""],
        }
    )
    tabla = construir_catalogo(fuente)
    assert tabla.loc[0, "codigo"] == "A0100"
    assert tabla.loc[0, "codigo_original"] == "A01.00"
    assert tabla.loc[0, "categoria"] == "A01"
    assert tabla.loc[1, "categoria"] == "Z94"


# --- Catálogo generado ------------------------------------------------------


def test_catalogo_codigos_unicos_y_sin_vacios() -> None:
    catalogo = _catalogo_o_saltar()
    assert catalogo["codigo"].is_unique
    assert (catalogo["codigo"].str.len() > 0).all()
    assert (catalogo["descripcion_es"].str.strip() != "").all()


def test_catalogo_formato_y_reversibilidad() -> None:
    catalogo = _catalogo_o_saltar()
    assert catalogo["codigo"].map(es_formato_valido).all()
    recalculado = catalogo["codigo_original"].map(normalizar_codigo)
    assert (recalculado == catalogo["codigo"]).all()
    assert (catalogo["categoria"] == catalogo["codigo"].str[:3]).all()


# --- Casos de evaluación ----------------------------------------------------


def test_casos_eval_tiene_50_casos_sin_duplicados() -> None:
    if not RUTA_CASOS.exists():
        pytest.skip("Falta data/processed/casos_eval.jsonl")
    casos = leer_jsonl(RUTA_CASOS)
    assert len(casos) == 50
    ids = [c["id"] for c in casos]
    assert len(set(ids)) == 50


def test_casos_eval_viene_solo_de_dev() -> None:
    if not RUTA_CASOS.exists():
        pytest.skip("Falta data/processed/casos_eval.jsonl")
    if not CARPETA_DEV.exists():
        pytest.skip("Falta el corpus de dev")
    casos = leer_jsonl(RUTA_CASOS)
    for caso in casos:
        assert (CARPETA_DEV / f"{caso['id']}.txt").exists(), caso["id"]
        assert caso["codigos_gold"], f"{caso['id']} sin códigos gold"
        assert all(c == normalizar_codigo(c) for c in caso["codigos_gold"])


# --- Abreviaturas y notas ficticias ----------------------------------------


def test_abreviaturas_al_menos_60() -> None:
    datos = yaml.safe_load(RUTA_ABREVIATURAS.read_text(encoding="utf-8"))
    entradas = datos["abreviaturas"]
    assert len(entradas) >= 60
    assert all(e["abreviatura"] and e["expansion"] for e in entradas)


def test_notas_ficticias_son_15_con_al_menos_5_negaciones() -> None:
    notas = leer_jsonl(RUTA_NOTAS)
    assert len(notas) == 15
    assert len({n["id"] for n in notas}) == 15
    assert sum(1 for n in notas if n["negados"]) >= 5


def test_cada_codigo_de_las_notas_existe_en_el_catalogo() -> None:
    catalogo = _catalogo_o_saltar()
    notas = leer_jsonl(RUTA_NOTAS)
    errores = validar_notas_ficticias(notas, set(catalogo["codigo"]))
    assert errores == []
