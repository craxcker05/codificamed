"""Mide la negación sobre las frases de tests/casos_negacion.py.

Salida: results/negacion_tests.json (exactitud global, precisión/exhaustividad
de la clase NEGADO y matriz de confusión).

Uso: python scripts/eval_negacion.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ))

from codificamed.negation import clasificar_frase  # noqa: E402
from tests.casos_negacion import CASOS, evaluar_ficticias  # noqa: E402


def main() -> None:
    confusion: Counter[tuple[str, str]] = Counter()
    errores = []
    for frase, mencion, esperado in CASOS:
        obtenido = clasificar_frase(frase, mencion).estado.value
        confusion[(esperado, obtenido)] += 1
        if obtenido != esperado:
            errores.append({"frase": frase, "mencion": mencion, "esperado": esperado, "obtenido": obtenido})

    fic = evaluar_ficticias()
    total = len(CASOS)
    aciertos = sum(v for (e, o), v in confusion.items() if e == o)
    tp = confusion[("negado", "negado")]
    fp = sum(v for (e, o), v in confusion.items() if o == "negado" and e != "negado")
    fn = sum(v for (e, o), v in confusion.items() if e == "negado" and o != "negado")
    resultado = {
        "frases": total,
        "aciertos": aciertos,
        "exactitud": round(aciertos / total, 4),
        "negado": {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "exhaustividad": round(tp / (tp + fn), 4) if tp + fn else None,
        },
        "confusion": {f"{e}->{o}": v for (e, o), v in sorted(confusion.items())},
        "errores": errores,
        "notas_ficticias": {
            "evaluados": len(fic),
            "aciertos": sum(f[2] == f[3] for f in fic),
            "errores": [f for f in fic if f[2] != f[3]],
        },
        "nota": "Frases escritas junto con las reglas: mide regresión, no desempeño en notas reales.",
    }
    salida = RAIZ / "results" / "negacion_tests.json"
    salida.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in resultado.items() if k not in ("errores",)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
