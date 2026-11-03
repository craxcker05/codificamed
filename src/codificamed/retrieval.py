"""Recuperación semántica de candidatos CIE-10 sobre el catálogo.

- Modelo: intfloat/multilingual-e5-base con prefijos "query: " (menciones) y
  "passage: " (descripciones del catálogo).
- Cache en disco (data/index/cache/):
    {campo}_{modelo_slug}.npy / .json   embeddings normalizados (float16) del
                                        catálogo por campo ("es" o "en") y su
                                        metadato (ids, huella, modelo, fecha)
    menciones_{modelo_slug}.npy / .json embeddings de consultas ya calculadas
  Lo que está en cache con la huella correcta nunca se recalcula.
- Búsqueda exacta: producto punto en numpy (coseno, vectores normalizados).
- Todo candidato proviene del catálogo, así que los códigos devueltos existen
  en él.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
RUTA_CATALOGO = RAIZ / "data" / "processed" / "catalogo_cie10.parquet"
DIR_INDICE = RAIZ / "data" / "index"
DIR_CACHE = DIR_INDICE / "cache"
MODELO = "intfloat/multilingual-e5-base"
PREFIJO_CONSULTA = "query: "
PREFIJO_PASAJE = "passage: "

CAMPOS: dict[str, str] = {"es": "descripcion_es", "en": "descripcion_en"}
MODOS = ("es", "en", "combinado")  # combinado = máximo de los puntajes es y en

# Variante de recuperación por defecto (ver docs/BITACORA.md, Fase 4 iteración 2).
MODO_POR_DEFECTO = "es"
EXPANDIR_POR_DEFECTO = False

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


def texto_pasaje(descripcion: str) -> str:
    return PREFIJO_PASAJE + descripcion.strip()


def texto_consulta(mencion: str) -> str:
    return PREFIJO_CONSULTA + " ".join(mencion.split())


def slug_modelo(modelo: str) -> str:
    return modelo.replace("/", "__")


def huella_catalogo(catalogo: pd.DataFrame, campo: str = "es") -> str:
    """Huella de códigos y del texto del campo: detecta un cache desfasado."""
    columna = CAMPOS[campo]
    h = hashlib.sha256()
    for c, d in zip(catalogo["codigo"], catalogo[columna].fillna(""), strict=True):
        h.update(f"{c}|{d}\n".encode())
    return h.hexdigest()


# ------------------------------------------------------------------ cache

def _rutas_cache(nombre: str, dir_cache: Path) -> tuple[Path, Path]:
    return dir_cache / f"{nombre}.npy", dir_cache / f"{nombre}.json"


def _guardar_atomico(ruta: Path, matriz: np.ndarray) -> None:
    tmp = ruta.with_suffix(".tmp.npy")
    np.save(tmp, matriz)
    tmp.replace(ruta)


def _meta_valida(meta_ruta: Path, huella: str, modelo: str) -> bool:
    if not meta_ruta.exists():
        return False
    meta = json.loads(meta_ruta.read_text(encoding="utf-8"))
    return meta.get("huella_catalogo") == huella and meta.get("modelo") == modelo


def get_or_build_embeddings(
    campo: str,
    modelo: str = MODELO,
    *,
    catalogo: pd.DataFrame | None = None,
    codificador: Codificador | None = None,
    dir_cache: Path = DIR_CACHE,
    force: bool = False,
    parte: int = 2048,
    lote: int = 64,
    progreso: bool = False,
) -> np.ndarray:
    """Embeddings normalizados del campo ("es"/"en") del catálogo, alineados con sus filas.

    Si el cache existe con la huella y el modelo correctos, se carga. Si no, se
    calcula por partes (reanudable) y se guarda. Las descripciones vacías
    (hay 109 en inglés) quedan con vector cero: puntaje 0, el más bajo posible
    para un par e5.
    """
    if campo not in CAMPOS:
        raise ValueError(f"Campo desconocido {campo!r}; usa {list(CAMPOS)}")
    catalogo = pd.read_parquet(RUTA_CATALOGO) if catalogo is None else catalogo
    huella = huella_catalogo(catalogo, campo)
    nombre = f"{campo}_{slug_modelo(modelo)}"
    ruta_npy, ruta_json = _rutas_cache(nombre, dir_cache)
    if not force and ruta_npy.exists() and _meta_valida(ruta_json, huella, modelo):
        return np.load(ruta_npy)

    dir_cache.mkdir(parents=True, exist_ok=True)
    textos = catalogo[CAMPOS[campo]].fillna("").astype(str).tolist()
    vacios = np.array([not t.strip() for t in textos])
    orden = np.argsort([len(t) for t in textos], kind="stable")  # lotes homogéneos
    trozos = [orden[i:i + parte] for i in range(0, len(orden), parte)]
    partes_dir = dir_cache / f"partes_{nombre}_{huella[:12]}"
    if force and partes_dir.exists():
        shutil.rmtree(partes_dir)
    partes_dir.mkdir(exist_ok=True)

    iterador = enumerate(trozos)
    if progreso:
        from tqdm import tqdm
        iterador = enumerate(tqdm(trozos, desc=f"Embeddings {campo}", unit="parte"))
    t0 = time.time()
    codificar = codificador
    for n, _ in iterador:
        ruta = partes_dir / f"parte_{n:05d}.npy"
        if ruta.exists():
            continue
        if codificar is None:
            codificar = crear_codificador(modelo, lote=lote)
        idx = trozos[n]
        validos = [i for i in idx if not vacios[i]]
        emb = np.zeros((len(idx), 0), dtype=np.float16)  # parte sin textos: se rellena al ensamblar
        if validos:
            calc = codificar([texto_pasaje(textos[i]) for i in validos]).astype(np.float16)
            emb = np.zeros((len(idx), calc.shape[1]), dtype=np.float16)
            posicion = {int(i): j for j, i in enumerate(idx)}
            for fila, i in zip(calc, validos, strict=True):
                emb[posicion[int(i)]] = fila
        _guardar_atomico(ruta, emb)

    cargadas = [np.load(partes_dir / f"parte_{n:05d}.npy") for n in range(len(trozos))]
    dim = max(c.shape[1] for c in cargadas)
    matriz = np.zeros((len(catalogo), dim), dtype=np.float16)
    for idx, c in zip(trozos, cargadas, strict=True):
        if c.shape[1]:
            matriz[idx] = c
    _guardar_atomico(ruta_npy, matriz)
    ruta_json.write_text(json.dumps({
        "campo": campo, "modelo": modelo, "huella_catalogo": huella, "filas": len(catalogo),
        "dimension": int(dim), "ids": catalogo["codigo"].tolist(), "descripciones_vacias": int(vacios.sum()),
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "segundos_esta_ejecucion": round(time.time() - t0, 1),
        "dispositivo": "cpu", "lote": lote,
    }), encoding="utf-8")
    shutil.rmtree(partes_dir, ignore_errors=True)
    return matriz


def migrar_indice_legado(modelo: str = MODELO, dir_indice: Path = DIR_INDICE,
                         dir_cache: Path = DIR_CACHE, segundos_originales: float | None = None) -> bool:
    """Convierte el índice previo (data/index/embeddings.npy, campo es) al cache.

    Solo si su huella coincide con el catálogo actual; es un traslado, no un
    recálculo. Devuelve True si migró.
    """
    legado_npy, legado_meta = dir_indice / "embeddings.npy", dir_indice / "meta.json"
    ruta_npy, ruta_json = _rutas_cache(f"es_{slug_modelo(modelo)}", dir_cache)
    if ruta_npy.exists() or not (legado_npy.exists() and legado_meta.exists()):
        return False
    catalogo = pd.read_parquet(RUTA_CATALOGO)
    meta = json.loads(legado_meta.read_text(encoding="utf-8"))
    huella = huella_catalogo(catalogo, "es")
    if meta.get("huella_catalogo") != huella or meta.get("modelo") != modelo:
        return False
    dir_cache.mkdir(parents=True, exist_ok=True)
    matriz = np.load(legado_npy)
    _guardar_atomico(ruta_npy, matriz)
    ruta_json.write_text(json.dumps({
        "campo": "es", "modelo": modelo, "huella_catalogo": huella, "filas": len(catalogo),
        "dimension": int(matriz.shape[1]), "ids": catalogo["codigo"].tolist(),
        "descripciones_vacias": 0, "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "segundos_esta_ejecucion": segundos_originales, "dispositivo": "cpu", "lote": 64,
        "origen": "migrado de data/index/embeddings.npy (índice previo de la etapa 4)",
    }), encoding="utf-8")
    legado_npy.unlink()
    legado_meta.unlink()
    shutil.rmtree(dir_indice / "partes", ignore_errors=True)
    return True


def get_or_build_consultas(
    textos: Sequence[str], modelo: str = MODELO, *, codificador: Codificador | None = None,
    dir_cache: Path = DIR_CACHE,
) -> np.ndarray:
    """Embeddings de consultas (sin prefijo en `textos`), con cache incremental por texto."""
    nombre = f"menciones_{slug_modelo(modelo)}"
    ruta_npy, ruta_json = _rutas_cache(nombre, dir_cache)
    claves: list[str] = []
    matriz = np.zeros((0, 0), dtype=np.float16)
    if ruta_npy.exists() and ruta_json.exists():
        meta = json.loads(ruta_json.read_text(encoding="utf-8"))
        if meta.get("modelo") == modelo:
            claves, matriz = meta["textos"], np.load(ruta_npy)
    normal = [" ".join(t.split()) for t in textos]
    indice = {t: i for i, t in enumerate(claves)}
    faltan = list(dict.fromkeys(t for t in normal if t not in indice))
    if faltan:
        codificar = codificador or crear_codificador(modelo)
        nuevos = codificar([PREFIJO_CONSULTA + t for t in faltan]).astype(np.float16)
        matriz = nuevos if matriz.size == 0 else np.vstack([matriz, nuevos])
        for t in faltan:
            indice[t] = len(claves)
            claves.append(t)
        dir_cache.mkdir(parents=True, exist_ok=True)
        _guardar_atomico(ruta_npy, matriz)
        ruta_json.write_text(json.dumps({"modelo": modelo, "textos": claves,
                                         "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds")}),
                             encoding="utf-8")
    return matriz[[indice[t] for t in normal]]


# ------------------------------------------------------------------ búsqueda

def puntajes(q: np.ndarray, emb_es: np.ndarray, emb_en: np.ndarray | None, modo: str) -> np.ndarray:
    """Matriz (consultas, catálogo) de cosenos según el modo: es, en o combinado (máximo)."""
    q = q.astype(np.float32)
    if modo == "es":
        return q @ emb_es.T.astype(np.float32)
    if emb_en is None:
        raise ValueError(f"El modo {modo!r} requiere embeddings 'en'")
    s_en = q @ emb_en.T.astype(np.float32)
    return s_en if modo == "en" else np.maximum(q @ emb_es.T.astype(np.float32), s_en)


def mejores(scores: np.ndarray, k: int) -> np.ndarray:
    """Índices de las k mejores columnas por fila, ordenados por puntaje (desempate estable)."""
    k = min(k, scores.shape[1])
    top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    orden = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1, kind="stable")
    return np.take_along_axis(top, orden, axis=1)


class Indice:
    """Matrices de embeddings alineadas con el catálogo."""

    def __init__(
        self, embeddings: np.ndarray, catalogo: pd.DataFrame, codificador: Codificador,
        embeddings_en: np.ndarray | None = None, modo: str = MODO_POR_DEFECTO,
        expandir: bool = EXPANDIR_POR_DEFECTO,
    ):
        for m in (embeddings, embeddings_en):
            if m is not None and len(m) != len(catalogo):
                raise ValueError(f"Índice ({len(m)}) y catálogo ({len(catalogo)}) no coinciden")
        if modo not in MODOS:
            raise ValueError(f"Modo desconocido {modo!r}; usa {MODOS}")
        self.embeddings, self.embeddings_en = embeddings, embeddings_en
        self.catalogo = catalogo.reset_index(drop=True)
        self.codificador, self.modo, self.expandir = codificador, modo, expandir

    def _consulta(self, mencion: str) -> str:
        if self.expandir:
            from codificamed.preprocess import preprocesar
            mencion = preprocesar(mencion).texto
        return texto_consulta(mencion)

    def buscar(self, mencion: str, k: int = 10) -> list[Candidato]:
        return self.buscar_lote([mencion], k)[0]

    def buscar_lote(self, menciones: Sequence[str], k: int = 10) -> list[list[Candidato]]:
        if not menciones:
            return []
        q = self.codificador([self._consulta(m) for m in menciones])
        scores = puntajes(q, self.embeddings, self.embeddings_en, self.modo)
        top = mejores(scores, k)
        return [[self._candidato(int(i), float(fila[i])) for i in idx] for fila, idx in zip(scores, top, strict=True)]

    def _candidato(self, i: int, score: float) -> Candidato:
        r = self.catalogo.iloc[i]
        return Candidato(r["codigo"], r["codigo_original"], r["descripcion_es"], score)


def cargar_indice(
    modo: str = MODO_POR_DEFECTO,
    expandir: bool = EXPANDIR_POR_DEFECTO,
    ruta_catalogo: Path = RUTA_CATALOGO,
    codificador: Codificador | None = None,
    dir_cache: Path = DIR_CACHE,
) -> Indice:
    """Carga los embeddings desde el cache (falla si no se construyó con build_index.py)."""
    catalogo = pd.read_parquet(ruta_catalogo)
    matrices: dict[str, np.ndarray] = {}
    for campo in ("es", "en") if modo != "es" else ("es",):
        ruta_npy, ruta_json = _rutas_cache(f"{campo}_{slug_modelo(MODELO)}", dir_cache)
        if not (ruta_npy.exists() and _meta_valida(ruta_json, huella_catalogo(catalogo, campo), MODELO)):
            raise FileNotFoundError(
                f"No hay cache válido del campo {campo!r} en {dir_cache}. Ejecuta: python scripts/build_index.py")
        matrices[campo] = np.load(ruta_npy)
    return Indice(matrices["es"], catalogo, codificador or crear_codificador(MODELO),
                  matrices.get("en"), modo, expandir)


@lru_cache(maxsize=1)
def _indice_por_defecto() -> Indice:
    return cargar_indice()


def buscar(mencion: str, k: int = 10, indice: Indice | None = None) -> list[Candidato]:
    """Los k códigos del catálogo más cercanos a la mención, con su score coseno."""
    return (indice or _indice_por_defecto()).buscar(mencion, k)
