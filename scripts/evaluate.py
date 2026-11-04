"""Evalúa el pipeline sobre los casos de dev (y las notas ficticias).

Reanudable: las predicciones se guardan por nota en results/preds_{proveedor}.jsonl
y las notas ya hechas no se repiten. Salida: results/metricas_{proveedor}.json.

Uso: python scripts/evaluate.py --provider ollama|gemini --n 50 [--sin-negacion]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ))

from config.settings import DIR_PROCESSED, DIR_RESULTS, get_settings  # noqa: E402

FICTICIAS = RAIZ / "data" / "ficticios" / "notas_ficticias.jsonl"


def leer_jsonl(ruta: Path) -> list[dict]:
    with ruta.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r) if p + r else 0.0, 4)}


def metricas(preds: list[dict], campo_gold: str, nivel: str) -> dict:
    """Micro y macro (promedio por nota) a código completo o categoría de 3 caracteres."""
    cortar = (lambda c: c[:3]) if nivel == "categoria" else (lambda c: c)
    TP = FP = FN = 0
    por_nota = []
    for p in preds:
        g, s = {cortar(c) for c in p[campo_gold]}, {cortar(c) for c in p["codigos"]}
        tp, fp, fn = len(g & s), len(s - g), len(g - s)
        TP, FP, FN = TP + tp, FP + fp, FN + fn
        por_nota.append(prf(tp, fp, fn))
    macro = {k: round(sum(m[k] for m in por_nota) / len(por_nota), 4) for k in ("precision", "recall", "f1")}
    return {"micro": prf(TP, FP, FN), "macro": macro}


def correr(notas: list[dict], destino: Path, usar_negacion: bool, proveedor) -> list[dict]:
    """Ejecuta el pipeline nota a nota, guardando cada resultado (reanudable)."""
    from codificamed.pipeline import analizar

    hechas = {p["id"]: p for p in leer_jsonl(destino)} if destino.exists() else {}
    for i, n in enumerate(notas, 1):
        if n["id"] in hechas:
            continue
        print(f"[{i}/{len(notas)}] {n['id']}", flush=True)
        try:
            r = analizar(n["texto"], proveedor, usar_negacion=usar_negacion)
            reg = {"id": n["id"], "codigos": sorted({s.codigo for s in r.sugerencias}),
                   "negados": [s.evidencia for s in r.negados], "segundos": r.tiempo_total_s,
                   "llamadas": r.llamadas_llm, "error": None}
        except Exception as e:  # noqa: BLE001 - una nota fallida no detiene la corrida
            reg = {"id": n["id"], "codigos": [], "negados": [], "segundos": 0.0,
                   "llamadas": 0, "error": str(e)[:200]}
        with destino.open("a", encoding="utf-8") as f:
            f.write(json.dumps(reg, ensure_ascii=False) + "\n")
        hechas[n["id"]] = reg
    return [hechas[n["id"]] for n in notas]


def resumen_conjunto(notas: list[dict], preds: list[dict], campo_gold: str) -> dict:
    gold = {n["id"]: n[campo_gold] for n in notas}
    for p in preds:
        p["gold"] = gold[p["id"]]
    ok = [p for p in preds if not p["error"]]
    return {
        "notas": len(preds), "con_error": len(preds) - len(ok),
        "codigo_completo": metricas(ok, "gold", "codigo"),
        "categoria_3": metricas(ok, "gold", "categoria"),
        "segundos_medio_por_nota": round(sum(p["segundos"] for p in ok) / max(len(ok), 1), 2),
        "costo_estimado_usd": 0.0,  # Ollama local y tier gratuito de Gemini
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["ollama", "gemini"], required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--n-fic", type=int, default=15, help="notas ficticias a evaluar")
    ap.add_argument("--solo-ficticias", action="store_true")
    ap.add_argument("--sin-negacion", action="store_true", help="ablación (solo ficticias)")
    a = ap.parse_args()

    from codificamed.codigos import normalizar_codigo
    from codificamed.llm import crear_proveedor

    ajustes = get_settings().model_copy(update={"llm_provider": a.provider})
    proveedor = crear_proveedor(ajustes)
    DIR_RESULTS.mkdir(exist_ok=True)

    ficticias = leer_jsonl(FICTICIAS)[: a.n_fic]
    for n in ficticias:
        n["gold"] = [normalizar_codigo(c) for c in n["codigos_esperados"]]
    negados_gold = {n["id"]: {normalizar_codigo(c) for c in n["negados"]} for n in ficticias}

    def eval_fic(sufijo: str, usar_negacion: bool) -> dict:
        preds = correr(ficticias, DIR_RESULTS / f"preds_{a.provider}_ficticias{sufijo}.jsonl",
                       usar_negacion, proveedor)
        res = resumen_conjunto(ficticias, preds, "gold")
        # negados sugeridos como confirmados: código negado que aparece entre las sugerencias
        total = sum(len(v) for v in negados_gold.values())
        mal = sum(len(set(p["codigos"]) & negados_gold[p["id"]]) for p in preds)
        res["negados_totales"], res["negados_sugeridos_como_confirmados"] = total, mal
        return res

    if a.sin_negacion:
        salida = DIR_RESULTS / f"metricas_{a.provider}_sin_negacion.json"
        out = {"proveedor": a.provider, "modelo": proveedor.modelo,
               "ficticias_sin_negacion": eval_fic("_sin_negacion", False)}
    else:
        out = {"proveedor": a.provider, "modelo": proveedor.modelo,
               "ficticias": eval_fic("", True)}
        salida = DIR_RESULTS / f"metricas_{a.provider}.json"
        if not a.solo_ficticias:
            casos = leer_jsonl(DIR_PROCESSED / "casos_eval.jsonl")[: a.n]
            preds = correr(casos, DIR_RESULTS / f"preds_{a.provider}.jsonl", True, proveedor)
            out["dev"] = resumen_conjunto(casos, preds, "codigos_gold")
    salida.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
