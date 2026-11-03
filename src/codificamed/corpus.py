"""Carga y conteos básicos del corpus CodiEsp (splits train, dev y test).

Estructura de datos (ver README.txt del corpus):
- {split}/text_files/*.txt        texto plano UTF-8, un caso por archivo
- {split}/{split}D.tsv             articleID, código CIE-10 (diagnósticos)
- {split}/{split}P.tsv             articleID, código CIE-10-PCS (procedimientos)
- {split}/{split}X.tsv             articleID, label, código, texto de referencia, posición

El split test no trae anotaciones: solo sus textos (test + background).
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import pandas as pd

SPLITS_ANOTADOS: tuple[str, ...] = ("train", "dev")
SPLITS: tuple[str, ...] = ("train", "dev", "test")
PISTAS: tuple[str, ...] = ("D", "P", "X")

COLUMNAS_D_P: list[str] = ["id_caso", "codigo"]
COLUMNAS_X: list[str] = ["id_caso", "label", "codigo", "texto_referencia", "posicion"]

# Abreviaturas clínicas frecuentes en español (se buscan en mayúsculas, con límites de palabra).
ABREVIATURAS_COMUNES: tuple[str, ...] = (
    "HTA", "DM", "DM2", "IAM", "EPOC", "ICC", "IRC", "AVC", "ACV", "TEP",
    "TVP", "EAP", "ITU", "IVU", "VIH", "VHC", "VHB", "EII", "PCR", "TAC",
    "RMN", "ECG", "EKG", "FC", "FR", "TA", "UCI", "SNC", "HbA1c",
)

# Expresiones de negación y descarte frecuentes en notas clínicas en español.
EXPRESIONES_NEGACION: tuple[str, ...] = (
    r"no", r"niega", r"niegan", r"negó", r"negativo", r"negativa",
    r"sin", r"ausencia de", r"descarta", r"descartado", r"descartada",
    r"tampoco", r"ni", r"libre de", r"no se observa", r"no se evidencia",
    r"no presenta", r"no refiere", r"no existe", r"no hay", r"no se identifica",
)

_PATRON_ABREVIATURAS: dict[str, re.Pattern[str]] = {
    a: re.compile(rf"\b{re.escape(a)}\b") for a in ABREVIATURAS_COMUNES
}
_PATRON_NEGACIONES: dict[str, re.Pattern[str]] = {
    e: re.compile(rf"\b{e}\b", re.IGNORECASE) for e in EXPRESIONES_NEGACION
}
_PATRON_NEGACION_GENERAL: re.Pattern[str] = re.compile(
    r"\b(?:" + "|".join(EXPRESIONES_NEGACION) + r")\b", re.IGNORECASE
)


def ruta_split(raiz: Path, split: str) -> Path:
    """Carpeta de un split dentro del corpus descomprimido."""
    return raiz / split


def listar_textos(carpeta_split: Path) -> dict[str, Path]:
    """Mapa id_caso -> ruta del texto plano, ordenado por id."""
    return {
        p.stem: p for p in sorted((carpeta_split / "text_files").glob("*.txt"))
    }


def leer_texto(ruta: Path) -> str:
    """Lee un caso en UTF-8."""
    return ruta.read_text(encoding="utf-8")


def leer_anotaciones(carpeta_split: Path, split: str, pista: str) -> pd.DataFrame | None:
    """Lee el TSV de una pista (D, P o X). Devuelve None si el split no tiene anotaciones."""
    ruta = carpeta_split / f"{split}{pista}.tsv"
    if not ruta.exists():
        return None
    columnas = COLUMNAS_X if pista == "X" else COLUMNAS_D_P
    return pd.read_csv(
        ruta,
        sep="\t",
        header=None,
        names=columnas,
        dtype=str,
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
    )


def contar_palabras(texto: str) -> int:
    """Número de palabras separadas por espacios en blanco."""
    return len(texto.split())


def codigos_por_caso(anotaciones: pd.DataFrame, ids_caso: list[str]) -> pd.Series:
    """Número de códigos distintos por caso, incluyendo casos sin ningún código."""
    unicos = anotaciones.assign(codigo=anotaciones["codigo"].str.upper())
    conteo = unicos.groupby("id_caso")["codigo"].nunique()
    return conteo.reindex(ids_caso, fill_value=0).astype(int)


def contar_abreviaturas(texto: str) -> Counter[str]:
    """Ocurrencias de cada abreviatura de ABREVIATURAS_COMUNES en el texto."""
    return Counter({a: len(p.findall(texto)) for a, p in _PATRON_ABREVIATURAS.items() if p.search(texto)})


def contar_negaciones(texto: str) -> Counter[str]:
    """Ocurrencias de cada expresión de EXPRESIONES_NEGACION en el texto."""
    return Counter({e: len(p.findall(texto)) for e, p in _PATRON_NEGACIONES.items() if p.search(texto)})


def parsear_spans(posicion: str) -> list[tuple[int, int]]:
    """Convierte '2787 2801;2810 2823' en [(2787, 2801), (2810, 2823)]."""
    spans: list[tuple[int, int]] = []
    for tramo in posicion.split(";"):
        partes = tramo.split()
        if len(partes) == 2:
            spans.append((int(partes[0]), int(partes[1])))
    return spans


def offset_coincide(texto: str, texto_referencia: str, posicion: str) -> bool:
    """True si los tramos de posición, unidos por un espacio, reproducen el texto de referencia.

    Las menciones discontinuas (p. ej. '2787 2801;2810 2823') tienen varios tramos.
    """
    spans = parsear_spans(posicion)
    if not spans:
        return False
    segmentos = [texto[inicio:fin] for inicio, fin in spans]
    return " ".join(segmentos) == texto_referencia


def hay_negacion_previa(texto: str, inicio: int, ventana: int = 60) -> bool:
    """True si alguna expresión de negación aparece en los `ventana` caracteres previos a `inicio`."""
    contexto = texto[max(0, inicio - ventana):inicio]
    return _PATRON_NEGACION_GENERAL.search(contexto) is not None
