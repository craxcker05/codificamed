"""Experimento de variantes de recuperación (Fase 4, iteración 2).

Usa las mismas 589 menciones oráculo y los gold de los 50 casos de
data/processed/casos_eval.jsonl (dev). Los embeddings del catálogo y de las
menciones salen del cache de data/index/cache/ (ejecuta antes
scripts/build_index.py); aquí no se recalcula nada del catálogo.

Variantes: campo es (k=10,20,30), en (k=10,20), combinado = máximo de los
puntajes es y en (k=10,20); cada una sin y con expansión de abreviaturas.

Selección: mayor recall@10 exacto sobre todos los gold; si la diferencia con
la mejor es < 0,01, se elige la más simple (es < en < combinado; sin expansión
antes que con expansión) y, a igual variante, el k menor.

Salida: results/recall_variantes.json (no toca results/recall_retrieval.json).

Uso: python scripts/experiment_recall.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from eval_retrieval import menciones_por_caso, metricas  # noqa: E402

from codificamed.preprocess import preprocesar  # noqa: E402
from codificamed.retrieval import (  # noqa: E402
    DIR_CACHE, MODELO, RUTA_CATALOGO, get_or_build_consultas, get_or_build_embeddings, mejores, puntajes,
)

KMAX = 30
VARIANTES: list[tuple[str, tuple[int, ...]]] = [
    ("es", (10, 20, 30)), ("en", (10, 20)), ("combinado", (10, 20)),
]
SIMPLICIDAD = {"es": 0, "en": 1, "combinado": 2}
UMBRAL_EMPATE = 0.01


def main() -> None:
    casos = [json.loads(linea) for linea in (RAIZ / "data/processed/casos_eval.jsonl").read_text(encoding="utf-8").splitlines()]
    menciones = menciones_por_caso({c["id"] for c in casos})
    unicas = sorted({m for ms in menciones.values() for m, _ in ms})

    catalogo = pd.read_parquet(RUTA_CATALOGO)
    codigos = catalogo["codigo"].to_numpy()
    emb = {c: get_or_build_embeddings(c, MODELO, catalogo=catalogo) for c in ("es", "en")}  # solo carga

    entradas: list[dict] = []
    for expandir in (False, True):
        textos = [preprocesar(m).texto if expandir else m for m in unicas]
        t0 = time.time()
        q = get_or_build_consultas(textos, MODELO)  # cache incremental
        t_consultas = time.time() - t0
        for modo, ks in VARIANTES:
            t0 = time.time()
            scores = puntajes(q, emb["es"], emb["en"], modo)
            top = mejores(scores, KMAX)
            cand = {m: [str(c) for c in codigos[fila]] for m, fila in zip(unicas, top, strict=True)}
            t_busqueda = time.time() - t0
            for k in ks:
                t0 = time.time()
                res, _ = metricas(cand, casos, menciones, (k,) if k == 10 else (10, k))
                entradas.append({
                    "variante": f"{modo}|{'con' if expandir else 'sin'}_expansion|k={k}",
                    "campo": modo, "expansion": expandir, "k": k,
                    "tiempo_busqueda_s": round(t_busqueda, 3), "tiempo_metricas_s": round(time.time() - t0, 3),
                    "tiempo_embeddings_consultas_s": round(t_consultas, 3),
                    "nota_tiempo": "búsqueda y embeddings de consultas compartidos por todas las k de la misma variante",
                    "metricas": res,
                })
            del scores

    # Selección entre variantes (k=10: el criterio es recall@10)
    def r10(e: dict) -> float:
        return e["metricas"]["todos_los_gold"]["exacto"]["recall@10"]

    base = [e for e in entradas if e["k"] == 10]
    mejor = max(r10(e) for e in base)
    empatadas = [e for e in base if mejor - r10(e) < UMBRAL_EMPATE]
    elegida = min(empatadas, key=lambda e: (SIMPLICIDAD[e["campo"]], e["expansion"], e["k"]))

    salida = {
        "casos": len(casos), "menciones_oraculo": sum(len(v) for v in menciones.values()),
        "menciones_unicas": len(unicas), "modelo": MODELO,
        "regla_seleccion": f"mayor recall@10 exacto (todos los gold); empate si diferencia < {UMBRAL_EMPATE}: "
                           "más simple (es < en < combinado; sin expansión primero) y k menor",
        "mejor_recall10_exacto": mejor,
        "empatadas": [e["variante"] for e in empatadas],
        "elegida": {k: elegida[k] for k in ("variante", "campo", "expansion", "k")},
        "variantes": entradas,
    }
    (RAIZ / "results" / "recall_variantes.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'variante':38s} {'ex@10':>6s} {'cat@10':>7s} {'ex@k':>6s} {'cat@k':>6s} {'ex@10 emit':>10s}")
    for e in entradas:
        m, k = e["metricas"], e["k"]
        print(f"{e['variante']:38s} {m['todos_los_gold']['exacto']['recall@10']:6.4f} "
              f"{m['todos_los_gold']['categoria']['recall@10']:7.4f} "
              f"{m['todos_los_gold']['exacto'][f'recall@{k}']:6.4f} {m['todos_los_gold']['categoria'][f'recall@{k}']:6.4f} "
              f"{m['solo_emitibles']['exacto']['recall@10']:10.4f}")
    print("ELEGIDA:", salida["elegida"], "| empatadas:", salida["empatadas"])


if __name__ == "__main__":
    main()
