"""Fase 2: catálogo CIE-10, cobertura frente a CodiEsp-D y casos de evaluación.

Entradas:
- data/processed/catalogo_fuente.tsv (sin cabecera; códigos con punto)
- data/raw/final_dataset_v2_to_publish/{train,dev}/ (anotaciones D y textos)
- config/abreviaturas.yaml y data/ficticios/notas_ficticias.jsonl (validación)

Salidas:
- data/processed/catalogo_cie10.parquet
- data/processed/casos_eval.jsonl (50 casos de dev, semilla 42)
- results/catalogo_resumen.json, results/cobertura_catalogo.csv,
  results/cobertura_resumen.json, results/notas_ficticias_validacion.json

Uso: python scripts/prepare_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_PROYECTO / "src"))
sys.path.insert(0, str(RAIZ_PROYECTO))

from config.settings import DIR_PROCESSED, DIR_RAW, DIR_RESULTS  # noqa: E402
from codificamed.corpus import leer_texto, listar_textos, ruta_split  # noqa: E402
from codificamed.preparacion import (  # noqa: E402
    SPLITS_EVALUACION_NO_TEST,
    cargar_catalogo_fuente,
    codigos_gold_por_caso,
    construir_catalogo,
    cobertura,
    escribir_jsonl,
    formato_invalido,
    leer_jsonl,
    resumen_catalogo_fuente,
    seleccionar_casos_eval,
    validar_notas_ficticias,
)

RAIZ_CODIESP = DIR_RAW / "final_dataset_v2_to_publish"
RUTA_CATALOGO_FUENTE = DIR_PROCESSED / "catalogo_fuente.tsv"
RUTA_CATALOGO_PARQUET = DIR_PROCESSED / "catalogo_cie10.parquet"
RUTA_CASOS_EVAL = DIR_PROCESSED / "casos_eval.jsonl"
RUTA_ABREVIATURAS = RAIZ_PROYECTO / "config" / "abreviaturas.yaml"
RUTA_NOTAS = RAIZ_PROYECTO / "data" / "ficticios" / "notas_ficticias.jsonl"
MIN_ABREVIATURAS = 60


def paso(numero: int, total: int, mensaje: str) -> None:
    """Muestra el progreso de una etapa."""
    print(f"[{numero}/{total}] {mensaje}", flush=True)


def guardar_json(ruta: Path, datos: Any) -> None:
    """Guarda un JSON en UTF-8 dentro de results/."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    """Ejecuta las etapas de la Fase 2."""
    total = 7
    paso(1, total, "Comprobaciones previas del catálogo fuente")
    fuente = cargar_catalogo_fuente(RUTA_CATALOGO_FUENTE)
    resumen_fuente = resumen_catalogo_fuente(fuente)
    print(json.dumps(resumen_fuente, ensure_ascii=False, indent=2), flush=True)

    paso(2, total, "Normalizando códigos (verificando reversibilidad y colisiones)")
    try:
        catalogo = construir_catalogo(fuente)
    except ValueError as error:
        raise SystemExit(f"DETENIDO: {error}") from error
    invalidos = formato_invalido(catalogo["codigo"])
    resumen_catalogo = {
        **resumen_fuente,
        "filas_catalogo_cie10": int(len(catalogo)),
        "colisiones": 0,
        "codigos_formato_invalido": len(invalidos),
        "ejemplos_formato_invalido": invalidos[:10],
        "categorias_distintas": int(catalogo["categoria"].nunique()),
    }
    guardar_json(DIR_RESULTS / "catalogo_resumen.json", resumen_catalogo)
    RUTA_CATALOGO_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    catalogo.to_parquet(RUTA_CATALOGO_PARQUET, index=False)
    print(f"    catálogo: {len(catalogo)} filas -> {RUTA_CATALOGO_PARQUET.name}", flush=True)

    paso(3, total, "Códigos gold de train y dev, normalizados, y cruce con el catálogo")
    gold: dict[str, dict[str, list[str]]] = {}
    for split in SPLITS_EVALUACION_NO_TEST:
        gold[split] = codigos_gold_por_caso(ruta_split(RAIZ_CODIESP, split) / f"{split}D.tsv")
    detalle, resumen_cob = cobertura(gold, set(catalogo["codigo"]))
    for split in SPLITS_EVALUACION_NO_TEST:
        crudos = pd.read_csv(
            ruta_split(RAIZ_CODIESP, split) / f"{split}D.tsv",
            sep="\t",
            header=None,
            names=["id_caso", "codigo"],
            dtype=str,
            keep_default_na=False,
            quoting=3,
        )["codigo"].str.upper()
        distintos = set(crudos)
        resumen_cob[split]["codigos_gold_con_punto"] = sum("." in c for c in distintos)
        resumen_cob[split]["codigos_gold_sin_punto"] = sum("." not in c for c in distintos)
    detalle.to_csv(DIR_RESULTS / "cobertura_catalogo.csv", index=False)
    guardar_json(DIR_RESULTS / "cobertura_resumen.json", resumen_cob)
    for split, datos in resumen_cob.items():
        print(
            f"    {split}: {datos['codigos_gold_distintos']} códigos, "
            f"{datos['faltan']} fuera del catálogo ({datos['pct_codigos_faltan']} %), "
            f"{datos['casos_con_codigo_faltante']} casos afectados "
            f"({datos['pct_casos_afectados']} %)",
            flush=True,
        )

    paso(4, total, "Casos de evaluación: 50 de dev, semilla 42, solo códigos D")
    textos_dev = {
        id_caso: leer_texto(ruta)
        for id_caso, ruta in listar_textos(ruta_split(RAIZ_CODIESP, "dev")).items()
    }
    casos = seleccionar_casos_eval(gold["dev"], textos_dev)
    n_escritos = escribir_jsonl(RUTA_CASOS_EVAL, casos)
    print(f"    {n_escritos} casos -> {RUTA_CASOS_EVAL.name}", flush=True)
    codigos_cat = set(catalogo["codigo"])
    afectados_eval = [c["id"] for c in casos if any(x not in codigos_cat for x in c["codigos_gold"])]
    resumen_cob["casos_eval"] = {
        "casos": len(casos),
        "casos_con_codigo_faltante": len(afectados_eval),
        "pct_casos_afectados": round(100 * len(afectados_eval) / len(casos), 2),
        "ids_afectados": afectados_eval,
    }
    guardar_json(DIR_RESULTS / "cobertura_resumen.json", resumen_cob)
    print(f"    casos_eval: {len(afectados_eval)} de {len(casos)} con algún código fuera del catálogo", flush=True)

    paso(5, total, "Validando config/abreviaturas.yaml")
    datos_abrev = yaml.safe_load(RUTA_ABREVIATURAS.read_text(encoding="utf-8"))
    n_abrev = len(datos_abrev.get("abreviaturas", []))
    if n_abrev < MIN_ABREVIATURAS:
        raise SystemExit(f"DETENIDO: solo {n_abrev} abreviaturas (mínimo {MIN_ABREVIATURAS}).")
    print(f"    {n_abrev} abreviaturas", flush=True)

    paso(6, total, "Validando notas ficticias contra el catálogo")
    notas = leer_jsonl(RUTA_NOTAS)
    errores = validar_notas_ficticias(notas, set(catalogo["codigo"]))
    con_negacion = sum(1 for n in notas if n.get("negados"))
    guardar_json(
        DIR_RESULTS / "notas_ficticias_validacion.json",
        {
            "notas": len(notas),
            "notas_con_negacion": con_negacion,
            "errores": errores,
        },
    )
    print(f"    {len(notas)} notas, {con_negacion} con negación, {len(errores)} errores", flush=True)

    paso(7, total, "Listo")
    if errores:
        raise SystemExit("Errores en notas ficticias:\n" + "\n".join(errores))
    print("Fase 2 (datos) preparada.", flush=True)


if __name__ == "__main__":
    main()
