"""Construye los embeddings del catálogo CIE-10 en data/index/cache/.

Solo calcula lo que falta: si el cache de un campo existe con la huella del
catálogo y el modelo correctos, lo reutiliza. --force recalcula.
El cálculo es reanudable por partes y muestra progreso. Si queda un índice
previo en data/index/embeddings.npy (campo es, etapa 4), se traslada al cache
sin recalcular.

Uso: python scripts/build_index.py [--campos es en] [--force] [--lote 64]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from codificamed.retrieval import (  # noqa: E402
    CAMPOS, DIR_CACHE, MODELO, get_or_build_embeddings, migrar_indice_legado, slug_modelo,
)

# Tiempo medido de la construcción original del campo es (log de la etapa 4).
SEGUNDOS_INDICE_PREVIO_ES = 2086.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campos", nargs="+", default=list(CAMPOS), choices=list(CAMPOS))
    ap.add_argument("--force", action="store_true", help="recalcula aunque haya cache válido")
    ap.add_argument("--lote", type=int, default=64)
    args = ap.parse_args()

    if not args.force and migrar_indice_legado(segundos_originales=SEGUNDOS_INDICE_PREVIO_ES):
        print("Índice previo (es) trasladado al cache sin recalcular.")
    for campo in args.campos:
        t0 = time.time()
        matriz = get_or_build_embeddings(campo, MODELO, force=args.force, lote=args.lote, progreso=True)
        meta = json.loads((DIR_CACHE / f"{campo}_{slug_modelo(MODELO)}.json").read_text(encoding="utf-8"))
        print(f"[{campo}] {matriz.shape} listo en {time.time() - t0:.0f} s "
              f"(construcción registrada: {meta.get('segundos_esta_ejecucion')} s)")


if __name__ == "__main__":
    main()
