"""Recall@k de la recuperación sobre los 50 casos de evaluación (dev, semilla 42).

Menciones de consulta: las menciones DIAGNOSTICO de la pista X de cada caso
(devX.tsv), es decir, menciones "oráculo". Así se mide la recuperación aislada
de la extracción de menciones; con menciones extraídas automáticamente el
recall será menor o igual.

Para cada código gold D del caso: ¿aparece entre los k candidatos de alguna
mención del caso? Se reporta:
- exacto: código normalizado idéntico;
- categoría: coincide en los 3 primeros caracteres (criterio de éxito de F1);
- sobre todos los gold y solo sobre los emitibles (existen en el catálogo);
- con y sin expansión de abreviaturas en la consulta;
- a nivel de mención: el código X de la mención aparece en sus propios candidatos.

Salida: results/recall_retrieval.json, results/recall_retrieval_detalle.csv

Uso: python scripts/eval_retrieval.py [--k 10]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from codificamed.codigos import categoria, normalizar_codigo  # noqa: E402
from codificamed.corpus import leer_anotaciones  # noqa: E402
from codificamed.preprocess import preprocesar  # noqa: E402
from codificamed.retrieval import RUTA_CATALOGO, cargar_indice  # noqa: E402

CORPUS_DEV = RAIZ / "data" / "raw" / "final_dataset_v2_to_publish" / "dev"
KS = (1, 5, 10, 20)


def menciones_por_caso(ids: set[str]) -> dict[str, list[tuple[str, str]]]:
    """id_caso -> [(texto_mención, código_normalizado)] únicos, solo DIAGNOSTICO."""
    x = leer_anotaciones(CORPUS_DEV, "dev", "X")
    x = x[(x["label"] == "DIAGNOSTICO") & (x["id_caso"].isin(ids))]
    salida: dict[str, list[tuple[str, str]]] = {i: [] for i in ids}
    for fila in x.itertuples():
        par = (" ".join(str(fila.texto_referencia).split()), normalizar_codigo(fila.codigo))
        if par not in salida[fila.id_caso]:
            salida[fila.id_caso].append(par)
    return salida


def metricas(cand: dict[str, list[str]], casos: list[dict], menciones: dict, ks: tuple[int, ...]) -> dict:
    """Recall@k exacto y de categoría (todos los gold / solo emitibles) y a nivel de mención.

    `cand`: mención -> códigos candidatos ordenados (al menos max(ks) por mención).
    """
    catalogo = set(pd.read_parquet(RUTA_CATALOGO)["codigo"])
    filas = []
    for caso in casos:
        ms = menciones[caso["id"]]
        for gold in caso["codigos_gold"]:
            rango_exacto = rango_cat = None  # mejor rango (1-based) entre las menciones del caso
            for m, _ in ms:
                for r, c in enumerate(cand[m], start=1):
                    if c == gold and (rango_exacto is None or r < rango_exacto):
                        rango_exacto = r
                    if categoria(c) == categoria(gold) and (rango_cat is None or r < rango_cat):
                        rango_cat = r
            filas.append({"id": caso["id"], "gold": gold, "emitible": gold in catalogo,
                          "rango_exacto": rango_exacto, "rango_categoria": rango_cat})
    df = pd.DataFrame(filas)

    def recall(sub: pd.DataFrame, col: str, k: int) -> float | None:
        return round(float((sub[col].notna() & (sub[col] <= k)).mean()), 4) if len(sub) else None

    res: dict = {}
    for nombre, sub in (("todos_los_gold", df), ("solo_emitibles", df[df["emitible"]])):
        res[nombre] = {"n_codigos": int(len(sub)),
                       "exacto": {f"recall@{k}": recall(sub, "rango_exacto", k) for k in ks},
                       "categoria": {f"recall@{k}": recall(sub, "rango_categoria", k) for k in ks}}
    res["nivel_mencion"] = {}
    for k in ks:
        n = ok = ok_cat = n_em = ok_em = 0
        for ms in menciones.values():
            for m, cod in ms:
                n += 1
                codigos = cand[m][:k]
                hit = cod in codigos
                ok += hit
                ok_cat += any(categoria(c) == categoria(cod) for c in codigos)
                if cod in catalogo:
                    n_em += 1
                    ok_em += hit
        res["nivel_mencion"][f"@{k}"] = {
            "menciones": n, "exacto": round(ok / n, 4), "categoria": round(ok_cat / n, 4),
            "menciones_con_codigo_emitible": n_em, "exacto_solo_emitibles": round(ok_em / n_em, 4)}
    return res, df


def evaluar(indice, casos, menciones, expandir: bool, kmax: int):
    """Compatibilidad con la etapa 4: expande (opcional), recupera y mide con KS."""
    unicas = sorted({m for ms in menciones.values() for m, _ in ms})
    consulta = {m: (preprocesar(m).texto if expandir else m) for m in unicas}
    lotes = indice.buscar_lote([consulta[m] for m in unicas], kmax)
    cand = {m: [c.codigo for c in lote] for m, lote in zip(unicas, lotes, strict=True)}
    return metricas(cand, casos, menciones, tuple(k for k in KS if k <= kmax))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20, help="máximo k a recuperar (se reporta @1,5,10,20)")
    ap.add_argument("--sobrescribir", action="store_true",
                    help="permite reescribir results/recall_retrieval.json (línea base de la etapa 4)")
    args = ap.parse_args()
    if (RAIZ / "results" / "recall_retrieval.json").exists() and not args.sobrescribir:
        sys.exit("results/recall_retrieval.json es la línea base; usa scripts/experiment_recall.py "
                 "o --sobrescribir si de verdad quieres reemplazarla.")

    casos = [json.loads(linea) for linea in (RAIZ / "data/processed/casos_eval.jsonl").read_text(encoding="utf-8").splitlines()]
    menciones = menciones_por_caso({c["id"] for c in casos})
    indice = cargar_indice()

    resultado = {"casos": len(casos), "menciones_totales": sum(len(v) for v in menciones.values()),
                 "casos_sin_menciones": sum(1 for v in menciones.values() if not v),
                 "tipo_menciones": "oraculo (DIAGNOSTICO de devX.tsv)", "modelo": "intfloat/multilingual-e5-base"}
    for nombre, expandir in (("sin_expansion", False), ("con_expansion_abreviaturas", True)):
        res, df = evaluar(indice, casos, menciones, expandir, args.k)
        resultado[nombre] = res
        df.to_csv(RAIZ / "results" / f"recall_retrieval_detalle_{nombre}.csv", index=False)
    (RAIZ / "results" / "recall_retrieval.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
