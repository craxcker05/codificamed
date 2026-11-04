"""Compara Ollama vs Gemini a partir de results/metricas_*.json.

Salida: results/comparacion.md, results/comparacion_f1.png y results/errores_ejemplos.md.

Uso: python scripts/compare.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
R = RAIZ / "results"
PROVEEDORES = ["ollama", "gemini"]


def cargar(ruta: Path) -> dict | None:
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


def main() -> None:
    m = {p: cargar(R / f"metricas_{p}.json") for p in PROVEEDORES}
    abl = {p: cargar(R / f"metricas_{p}_sin_negacion.json") for p in PROVEEDORES}
    m = {p: v for p, v in m.items() if v}
    filas = ["| Proveedor | Modelo | F1 micro código | F1 micro categoría | F1 macro categoría "
             "| s/nota | Negados sugeridos como confirmados (ficticias) |", "|---|---|---|---|---|---|---|"]
    barras: dict[str, list[float]] = {}
    for p, v in m.items():
        d = v.get("dev")
        if not d:
            continue
        f = v["ficticias"]
        filas.append(f"| {p} | {v['modelo']} | {d['codigo_completo']['micro']['f1']} "
                     f"| {d['categoria_3']['micro']['f1']} | {d['categoria_3']['macro']['f1']} "
                     f"| {d['segundos_medio_por_nota']} | {f['negados_sugeridos_como_confirmados']}"
                     f" de {f['negados_totales']} |")
        barras[p] = [d["codigo_completo"]["micro"]["f1"], d["categoria_3"]["micro"]["f1"]]
    filas += ["", "Ablación (notas ficticias evaluadas): negados sugeridos como confirmados", "",
              "| Proveedor | Con negación | Sin negación |", "|---|---|---|"]
    for p in m:
        a = abl.get(p)
        if a:
            filas.append(f"| {p} | {m[p]['ficticias']['negados_sugeridos_como_confirmados']} "
                         f"| {a['ficticias_sin_negacion']['negados_sugeridos_como_confirmados']} |")
    (R / "comparacion.md").write_text("\n".join(filas) + "\n", encoding="utf-8")

    if barras:
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, (p, vals) in enumerate(barras.items()):
            ax.bar([x + i * 0.35 for x in range(2)], vals, 0.35, label=p)
        ax.set_xticks([0.17, 1.17], ["Código completo", "Categoría (3 car.)"])
        ax.set_ylabel("F1 micro (50 casos dev)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(R / "comparacion_f1.png", dpi=120)

    # 5 ejemplos de error (el proveedor con más predicciones disponibles)
    casos = {json.loads(l)["id"]: json.loads(l) for l in
             (RAIZ / "data/processed/casos_eval.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    out = ["# Ejemplos de error (50 casos dev)", ""]
    for p in m:
        ruta = R / f"preds_{p}.jsonl"
        if not ruta.exists():
            continue
        preds = [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]
        peores = sorted((x for x in preds if not x["error"]),
                        key=lambda x: len(set(x["codigos"]) ^ set(casos[x["id"]]["codigos_gold"])),
                        reverse=True)[:5]
        out.append(f"## {p}")
        for x in peores:
            g = set(casos[x["id"]]["codigos_gold"])
            s = set(x["codigos"])
            out += [f"- `{x['id']}`: acertados {sorted(g & s)}; faltantes {sorted(g - s)}; "
                    f"de más {sorted(s - g)}"]
        out.append("")
    (R / "errores_ejemplos.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(filas))


if __name__ == "__main__":
    main()
