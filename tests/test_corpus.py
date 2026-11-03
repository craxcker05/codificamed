"""Pruebas del módulo corpus con un caso ficticio (no se usan historias reales)."""

from pathlib import Path

import pandas as pd

from codificamed.corpus import (
    codigos_por_caso,
    contar_abreviaturas,
    contar_negaciones,
    contar_palabras,
    hay_negacion_previa,
    leer_anotaciones,
    listar_textos,
    offset_coincide,
    parsear_spans,
)

TEXTO_FICTICIO = "Paciente con HTA. Niega dolor torácico. Tiene DM2 controlada."


def test_contar_palabras_separa_por_espacios() -> None:
    assert contar_palabras("uno dos  tres\ncuatro") == 4


def test_contar_abreviaturas_respeta_limites_de_palabra() -> None:
    conteo = contar_abreviaturas(TEXTO_FICTICIO)
    assert conteo["HTA"] == 1
    assert conteo["DM2"] == 1
    assert "DM" not in conteo  # "DM2" no debe contar como "DM"


def test_contar_negaciones_detecta_niega() -> None:
    conteo = contar_negaciones(TEXTO_FICTICIO)
    assert conteo["niega"] == 1


def test_parsear_spans_con_tramos_discontinuos() -> None:
    assert parsear_spans("2787 2801;2810 2823") == [(2787, 2801), (2810, 2823)]


def test_offset_coincide_con_texto_de_referencia() -> None:
    texto = "Paciente con HTA."
    assert offset_coincide(texto, "HTA", "13 16")
    assert not offset_coincide(texto, "DM", "13 16")


def test_offset_coincide_con_mencion_discontinua() -> None:
    texto = "masa renal izquierda y masa hepática"
    # "masa" + "masa" en dos tramos, unidos por espacio
    assert offset_coincide(texto, "masa masa", "0 4;23 27")


def test_hay_negacion_previa_dentro_de_la_ventana() -> None:
    inicio = TEXTO_FICTICIO.index("dolor")
    assert hay_negacion_previa(TEXTO_FICTICIO, inicio, ventana=60)
    inicio_hta = TEXTO_FICTICIO.index("HTA")
    assert not hay_negacion_previa(TEXTO_FICTICIO, inicio_hta, ventana=60)


def test_codigos_por_caso_incluye_casos_sin_codigo() -> None:
    anot = pd.DataFrame(
        {"id_caso": ["c1", "c1", "c1", "c2"], "codigo": ["n44.8", "N44.8", "r60.9", "a23"]}
    )
    conteo = codigos_por_caso(anot, ["c1", "c2", "c3"])
    assert conteo.to_dict() == {"c1": 2, "c2": 1, "c3": 0}


def test_leer_anotaciones_d_y_x(tmp_path: Path) -> None:
    carpeta = tmp_path / "train"
    carpeta.mkdir()
    (carpeta / "trainD.tsv").write_text("c1\tn44.8\nc1\tr60.9\n", encoding="utf-8")
    (carpeta / "trainX.tsv").write_text(
        "c1\tDIAGNOSTICO\tn44.8\tHTA\t13 16\n", encoding="utf-8"
    )
    d = leer_anotaciones(carpeta, "train", "D")
    x = leer_anotaciones(carpeta, "train", "X")
    assert d is not None and len(d) == 2
    assert x is not None and x.loc[0, "texto_referencia"] == "HTA"
    assert leer_anotaciones(carpeta, "train", "P") is None


def test_listar_textos_ordena_por_id(tmp_path: Path) -> None:
    carpeta = tmp_path / "dev" / "text_files"
    carpeta.mkdir(parents=True)
    (carpeta / "b.txt").write_text("x", encoding="utf-8")
    (carpeta / "a.txt").write_text("y", encoding="utf-8")
    assert list(listar_textos(tmp_path / "dev").keys()) == ["a", "b"]
