"""Construye el índice de embeddings del catálogo CIE-10 en data/index/.

Reanudable: guarda partes de --parte filas en data/index/partes/ y las reutiliza
si el script se interrumpe. Las filas se procesan ordenadas por longitud de
descripción (lotes más homogéneos = más rápido en CPU); el orden es
determinista.

Salidas: data/index/embeddings.npy (float16, filas alineadas con el catálogo) y
data/index/meta.json (modelo, huella del catálogo, dimensiones).

Uso: python scripts/build_index.py [--limite N] [--parte 2048] [--lote 64]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from codificamed.retrieval import (  # noqa: E402
    DIR_INDICE, MODELO, RUTA_CATALOGO, crear_codificador, huella_catalogo, texto_pasaje,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int, default=None, help="solo las primeras N filas (prueba de humo)")
    ap.add_argument("--parte", type=int, default=2048)
    ap.add_argument("--lote", type=int, default=64)
    args = ap.parse_args()

    catalogo = pd.read_parquet(RUTA_CATALOGO)
    if args.limite:
        catalogo = catalogo.head(args.limite)
        destino = DIR_INDICE / "prueba"
    else:
        destino = DIR_INDICE
    huella = huella_catalogo(catalogo)
    partes_dir = destino / "partes" / huella[:12]
    partes_dir.mkdir(parents=True, exist_ok=True)

    textos = [texto_pasaje(d) for d in catalogo["descripcion_es"]]
    orden = np.argsort([len(t) for t in textos], kind="stable")
    trozos = [orden[i:i + args.parte] for i in range(0, len(orden), args.parte)]

    codificar = None
    t0 = time.time()
    for n, idx in enumerate(tqdm(trozos, desc="Embeddings", unit="parte")):
        ruta = partes_dir / f"parte_{n:05d}.npy"
        if ruta.exists():
            continue
        if codificar is None:
            codificar = crear_codificador(MODELO, lote=args.lote)
        emb = codificar([textos[i] for i in idx]).astype(np.float16)
        np.save(ruta.with_suffix(".tmp.npy"), emb)
        ruta.with_suffix(".tmp.npy").replace(ruta)

    dim = np.load(partes_dir / "parte_00000.npy").shape[1]
    matriz = np.zeros((len(catalogo), dim), dtype=np.float16)
    for n, idx in enumerate(trozos):
        matriz[idx] = np.load(partes_dir / f"parte_{n:05d}.npy")
    np.save(destino / "embeddings.npy", matriz)
    (destino / "meta.json").write_text(json.dumps({
        "modelo": MODELO, "filas": len(catalogo), "dimension": int(dim),
        "huella_catalogo": huella, "prefijo_pasaje": "passage: ",
    }, indent=2), encoding="utf-8")
    print(f"Índice listo: {matriz.shape} en {destino} ({time.time() - t0:.0f} s esta ejecución)")


if __name__ == "__main__":
    main()
