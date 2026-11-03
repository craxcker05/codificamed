"""Recuperación semántica de candidatos CIE-10 sobre el catálogo.

- Modelo: intfloat/multilingual-e5-base con prefijos "query: " (menciones) y
  "passage: " (descripciones del catálogo).
- Índice en data/index/: matriz de embeddings normalizados (float16) alineada
  fila a fila con data/processed/catalogo_cie10.parquet. La búsqueda es exacta
  (producto punto = coseno); con ~71k filas no hace falta un índice aproximado.
- Todo candidato proviene del catálogo, así que los códigos devueltos siempre
  existen en él.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
RUTA_CATALOGO = RAIZ / "data" / "processed" / "catalogo_cie10.parquet"
DIR_INDICE = RAIZ / "data" / "index"
MODELO = "intfloat/multilingual-e5-base"
PREFIJO_CONSULTA = "query: "
PREFIJO_PASAJE = "passage: "

# Un codificador recibe textos YA con prefijo y devuelve una matriz (n, d) normalizada.
Codificador = Callable[[Sequence[str]], np.ndarray]


@dataclass(frozen=True)
class Candidato:
    codigo: str  # normalizado, sin punto
    codigo_original: str  # con punto
    descripcion: str
    score: float  # coseno


def crear_codificador(modelo: str = MODELO, lote: int = 64, max_len: int = 128) -> Codificador:
    """Codificador real con sentence-transformers (importación perezosa: pesa)."""
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(modelo, device="cpu")
    st.max_seq_length = max_len

    def codificar(textos: Sequence[str]) -> np.ndarray:
        return st.encode(
            list(textos), batch_size=lote, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ).astype(np.float32)

    return codificar


def texto_pasaje(descripcion_es: str) -> str:
    return PREFIJO_PASAJE + descripcion_es.strip()


def texto_consulta(mencion: str) -> str:
    return PREFIJO_CONSULTA + " ".join(mencion.split())


def huella_catalogo(catalogo: pd.DataFrame) -> str:
    """Huella de códigos y descripciones: detecta un índice desfasado del catálogo."""
    h = hashlib.sha256()
    for c, d in zip(catalogo["codigo"], catalogo["descripcion_es"], strict=True):
        h.update(f"{c}|{d}\n".encode())
    return h.hexdigest()


class Indice:
    """Matriz de embeddings alineada con el catálogo."""

    def __init__(self, embeddings: np.ndarray, catalogo: pd.DataFrame, codificador: Codificador):
        if len(embeddings) != len(catalogo):
            raise ValueError(f"Índice ({len(embeddings)}) y catálogo ({len(catalogo)}) no coinciden")
        self.embeddings = embeddings
        self.catalogo = catalogo.reset_index(drop=True)
        self.codificador = codificador

    def buscar(self, mencion: str, k: int = 10) -> list[Candidato]:
        return self.buscar_lote([mencion], k)[0]

    def buscar_lote(self, menciones: Sequence[str], k: int = 10) -> list[list[Candidato]]:
        if not menciones:
            return []
        q = self.codificador([texto_consulta(m) for m in menciones]).astype(np.float32)
        scores = q @ self.embeddings.T.astype(np.float32)
        k = min(k, scores.shape[1])
        salida: list[list[Candidato]] = []
        for fila in scores:
            top = np.argpartition(-fila, k - 1)[:k]
            top = top[np.argsort(-fila[top], kind="stable")]
            salida.append([self._candidato(int(i), float(fila[i])) for i in top])
        return salida

    def _candidato(self, i: int, score: float) -> Candidato:
        r = self.catalogo.iloc[i]
        return Candidato(r["codigo"], r["codigo_original"], r["descripcion_es"], score)


def cargar_indice(
    dir_indice: Path = DIR_INDICE,
    ruta_catalogo: Path = RUTA_CATALOGO,
    codificador: Codificador | None = None,
) -> Indice:
    """Carga el índice de disco y comprueba que corresponde al catálogo actual."""
    meta_ruta = dir_indice / "meta.json"
    if not meta_ruta.exists():
        raise FileNotFoundError(f"No hay índice en {dir_indice}. Ejecuta: python scripts/build_index.py")
    meta = json.loads(meta_ruta.read_text(encoding="utf-8"))
    catalogo = pd.read_parquet(ruta_catalogo)
    if meta["huella_catalogo"] != huella_catalogo(catalogo):
        raise ValueError("El índice no corresponde al catálogo actual; reconstruye con build_index.py")
    embeddings = np.load(dir_indice / "embeddings.npy")
    return Indice(embeddings, catalogo, codificador or crear_codificador(meta["modelo"]))


@lru_cache(maxsize=1)
def _indice_por_defecto() -> Indice:
    return cargar_indice()


def buscar(mencion: str, k: int = 10, indice: Indice | None = None) -> list[Candidato]:
    """Los k códigos del catálogo más cercanos a la mención, con su score coseno."""
    return (indice or _indice_por_defecto()).buscar(mencion, k)
