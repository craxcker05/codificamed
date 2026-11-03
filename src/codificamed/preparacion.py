"""Preparación de datos de la Fase 2: catálogo CIE-10, cobertura y casos de evaluación.

Reglas del proyecto que este módulo hace cumplir:
- train/ solo para construir el catálogo y ajustar reglas; la evaluación usa dev/.
- test/ nunca se usa (no tiene anotaciones).
- Si la normalización de códigos produce colisiones, se detiene con ColisionError.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from codificamed.codigos import categoria, detectar_colisiones, es_formato_valido, normalizar_codigo

COLUMNAS_CATALOGO_FUENTE: list[str] = ["codigo", "descripcion_es", "descripcion_en"]
SPLITS_EVALUACION_NO_TEST: tuple[str, ...] = ("train", "dev")
SEMILLA_EVAL: int = 42
N_CASOS_EVAL: int = 50


class ColisionError(ValueError):
    """Dos códigos originales distintos quedan iguales tras normalizar."""


class CatalogoInvalido(ValueError):
    """El catálogo fuente no cumple las comprobaciones previas."""


def cargar_catalogo_fuente(ruta: Path) -> pd.DataFrame:
    """Lee el catálogo TSV fuente. NO tiene cabecera: se asignan los nombres de columna."""
    return pd.read_csv(
        ruta,
        sep="\t",
        header=None,
        names=COLUMNAS_CATALOGO_FUENTE,
        dtype=str,
        keep_default_na=False,
        quoting=3,
        encoding="utf-8",
    )


def resumen_catalogo_fuente(catalogo: pd.DataFrame) -> dict[str, Any]:
    """Comprobaciones previas: filas, duplicados, vacíos y códigos con punto."""
    con_punto = catalogo["codigo"].str.contains(".", regex=False)
    return {
        "filas": int(len(catalogo)),
        "duplicados_exactos": int(catalogo.duplicated().sum()),
        "codigo_repetido": int(catalogo["codigo"].duplicated().sum()),
        "codigo_vacio": int((catalogo["codigo"].str.strip() == "").sum()),
        "descripcion_es_vacia": int((catalogo["descripcion_es"].str.strip() == "").sum()),
        "descripcion_en_vacia": int((catalogo["descripcion_en"].str.strip() == "").sum()),
        "codigos_con_punto": int(con_punto.sum()),
        "ejemplos_con_punto": catalogo.loc[con_punto, "codigo"].head(5).tolist(),
    }


def construir_catalogo(catalogo_fuente: pd.DataFrame) -> pd.DataFrame:
    """Normaliza el catálogo y valida reversibilidad y ausencia de colisiones.

    Lanza ColisionError si dos códigos originales distintos quedan iguales.
    Lanza CatalogoInvalido si hay códigos vacíos o duplicados tras normalizar.
    """
    originales = catalogo_fuente["codigo"].tolist()
    colisiones = detectar_colisiones(originales)
    if colisiones:
        muestra = dict(list(colisiones.items())[:10])
        raise ColisionError(f"{len(colisiones)} colisiones tras normalizar. Ejemplos: {muestra}")

    tabla = pd.DataFrame(
        {
            "codigo": [normalizar_codigo(c) for c in originales],
            "codigo_original": originales,
            "descripcion_es": catalogo_fuente["descripcion_es"].tolist(),
            "descripcion_en": catalogo_fuente["descripcion_en"].tolist(),
        }
    )
    if (tabla["codigo"] == "").any():
        raise CatalogoInvalido("Hay códigos vacíos tras normalizar.")
    if tabla["codigo"].duplicated().any():
        raise CatalogoInvalido("Hay códigos duplicados tras normalizar.")
    # Reversibilidad: la columna codigo_original reproduce cada normalizado y no hay ambigüedad.
    recalculado = tabla["codigo_original"].map(normalizar_codigo)
    if not (recalculado == tabla["codigo"]).all():
        raise CatalogoInvalido("Algún codigo_original no reproduce su código normalizado.")
    tabla["categoria"] = tabla["codigo"].map(categoria)
    return tabla.reset_index(drop=True)


def formato_invalido(codigos_normalizados: Iterable[str]) -> list[str]:
    """Códigos normalizados que no cumplen el formato esperado (para reportar)."""
    return sorted({c for c in codigos_normalizados if not es_formato_valido(c)})


def codigos_gold_por_caso(ruta_tsv: Path) -> dict[str, list[str]]:
    """Códigos D gold normalizados y sin repetir por caso, en orden de aparición."""
    df = pd.read_csv(
        ruta_tsv,
        sep="\t",
        header=None,
        names=["id_caso", "codigo"],
        dtype=str,
        keep_default_na=False,
        quoting=3,
        encoding="utf-8",
    )
    resultado: dict[str, list[str]] = {}
    for id_caso, codigo in zip(df["id_caso"], df["codigo"], strict=True):
        lista = resultado.setdefault(id_caso, [])
        normalizado = normalizar_codigo(codigo)
        if normalizado not in lista:
            lista.append(normalizado)
    return resultado


def cobertura(
    gold_por_split: dict[str, dict[str, list[str]]],
    codigos_catalogo: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Cruza códigos gold distintos con el catálogo. Devuelve detalle y resumen por split."""
    filas: list[dict[str, Any]] = []
    resumen: dict[str, Any] = {}
    for split, casos in gold_por_split.items():
        codigos = sorted({c for lista in casos.values() for c in lista})
        for codigo in codigos:
            filas.append(
                {"split": split, "codigo_gold": codigo, "en_catalogo": codigo in codigos_catalogo}
            )
        faltan = [c for c in codigos if c not in codigos_catalogo]
        # Un código ausente es "padre" si el catálogo tiene subcódigos que empiezan igual.
        con_subcodigos = [
            c for c in faltan if any(x.startswith(c) and x != c for x in codigos_catalogo)
        ]
        casos_afectados = [
            id_caso for id_caso, lista in casos.items() if any(c not in codigos_catalogo for c in lista)
        ]
        total_casos = len(casos)
        resumen[split] = {
            "codigos_gold_distintos": len(codigos),
            "en_catalogo": len(codigos) - len(faltan),
            "faltan": len(faltan),
            "faltan_con_subcodigos_en_catalogo": len(con_subcodigos),
            "pct_codigos_faltan": round(100 * len(faltan) / len(codigos), 2) if codigos else 0.0,
            "casos": total_casos,
            "casos_con_codigo_faltante": len(casos_afectados),
            "pct_casos_afectados": round(100 * len(casos_afectados) / total_casos, 2)
            if total_casos
            else 0.0,
            "codigos_faltantes_muestra": faltan[:20],
        }
    return pd.DataFrame(filas), resumen


def seleccionar_casos_eval(
    gold_dev: dict[str, list[str]],
    textos_dev: dict[str, str],
    n: int = N_CASOS_EVAL,
    semilla: int = SEMILLA_EVAL,
) -> list[dict[str, Any]]:
    """Elige n casos de dev con semilla fija. Solo códigos de diagnóstico (pista D)."""
    ids_ordenados = sorted(id_caso for id_caso in gold_dev if id_caso in textos_dev)
    elegidos = random.Random(semilla).sample(ids_ordenados, n)
    return [
        {"id": id_caso, "texto": textos_dev[id_caso], "codigos_gold": gold_dev[id_caso]}
        for id_caso in elegidos
    ]


def escribir_jsonl(ruta: Path, registros: Iterable[dict[str, Any]]) -> int:
    """Escribe una línea JSON por registro, en UTF-8. Devuelve el número de líneas."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with ruta.open("w", encoding="utf-8") as f:
        for registro in registros:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
            n += 1
    return n


def leer_jsonl(ruta: Path) -> list[dict[str, Any]]:
    """Lee un JSONL en UTF-8."""
    with ruta.open("r", encoding="utf-8") as f:
        return [json.loads(linea) for linea in f if linea.strip()]


def validar_notas_ficticias(notas: list[dict[str, Any]], codigos_catalogo: set[str]) -> list[str]:
    """Errores de las notas: códigos fuera del catálogo, evidencias que no son texto literal, etc."""
    errores: list[str] = []
    for nota in notas:
        id_nota = nota.get("id", "?")
        texto = nota.get("texto", "")
        esperados = nota.get("codigos_esperados", [])
        justificacion = nota.get("frase_justificacion", {})
        negados = nota.get("negados", {})
        if set(justificacion) != set(esperados):
            errores.append(f"{id_nota}: frase_justificacion no coincide con codigos_esperados")
        for codigo in esperados:
            if codigo != normalizar_codigo(codigo):
                errores.append(f"{id_nota}: {codigo} no está normalizado")
            if codigo not in codigos_catalogo:
                errores.append(f"{id_nota}: {codigo} no existe en el catálogo")
        for codigo, frase in {**justificacion, **negados}.items():
            if frase not in texto:
                errores.append(f"{id_nota}: la frase de {codigo} no es texto literal de la nota")
        for codigo in negados:
            if codigo not in codigos_catalogo:
                errores.append(f"{id_nota}: negado {codigo} no existe en el catálogo")
            if codigo in esperados:
                errores.append(f"{id_nota}: {codigo} aparece como presente y como negado")
    return errores
