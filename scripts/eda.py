"""EDA del corpus CodiEsp (subtarea D: diagnósticos CIE-10).

Explora data/raw/ y guarda tablas, gráficos y un resumen JSON en results/eda/.
Todas las cifras de docs/BITACORA.md deben salir de estos archivos.

Uso:
    python scripts/eda.py
    python scripts/eda.py --raiz "ruta/al/final_dataset_v2_to_publish"
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_PROYECTO / "src"))
sys.path.insert(0, str(RAIZ_PROYECTO))

from config.settings import DIR_EDA, DIR_RAW  # noqa: E402
from codificamed.corpus import (  # noqa: E402
    ABREVIATURAS_COMUNES,
    EXPRESIONES_NEGACION,
    SPLITS,
    SPLITS_ANOTADOS,
    codigos_por_caso,
    contar_abreviaturas,
    contar_negaciones,
    contar_palabras,
    hay_negacion_previa,
    leer_anotaciones,
    leer_texto,
    listar_textos,
    offset_coincide,
    parsear_spans,
    ruta_split,
)

RAIZ_CODIESP_DEFECTO = DIR_RAW / "final_dataset_v2_to_publish"
TOP_N = 30
VENTANA_NEGACION = 60


def paso(numero: int, total: int, mensaje: str) -> None:
    """Muestra el progreso de una etapa."""
    print(f"[{numero}/{total}] {mensaje}", flush=True)


def textos_por_split(raiz: Path) -> dict[str, dict[str, str]]:
    """Carga los textos de cada split indexados por id de caso."""
    resultado: dict[str, dict[str, str]] = {}
    for split in SPLITS:
        rutas = listar_textos(ruta_split(raiz, split))
        resultado[split] = {}
        for i, (id_caso, ruta) in enumerate(rutas.items(), start=1):
            resultado[split][id_caso] = leer_texto(ruta)
            if i % 1000 == 0:
                print(f"    {split}: {i}/{len(rutas)} textos leídos", flush=True)
        print(f"    {split}: {len(rutas)} textos leídos", flush=True)
    return resultado


def tabla_casos_por_split(
    textos: dict[str, dict[str, str]], anotaciones: dict[str, dict[str, pd.DataFrame]]
) -> pd.DataFrame:
    """Casos por split con filas de anotación por pista."""
    filas = []
    for split in SPLITS:
        fila = {"split": split, "textos": len(textos[split])}
        for pista in ("D", "P", "X"):
            df = anotaciones.get(split, {}).get(pista)
            fila[f"filas_{pista}"] = len(df) if df is not None else None
            fila[f"casos_con_{pista}"] = int(df["id_caso"].nunique()) if df is not None else None
        filas.append(fila)
    return pd.DataFrame(filas)


def tabla_longitud(textos: dict[str, dict[str, str]]) -> pd.DataFrame:
    """Estadísticos de longitud en palabras y caracteres por split."""
    filas = []
    for split in SPLITS:
        palabras = pd.Series({k: contar_palabras(v) for k, v in textos[split].items()})
        caracteres = pd.Series({k: len(v) for k, v in textos[split].items()})
        for metrica, serie in (("palabras", palabras), ("caracteres", caracteres)):
            filas.append(
                {
                    "split": split,
                    "metrica": metrica,
                    "media": round(serie.mean(), 1),
                    "mediana": float(serie.median()),
                    "minimo": int(serie.min()),
                    "maximo": int(serie.max()),
                    "desv_std": round(serie.std(), 1),
                }
            )
    return pd.DataFrame(filas)


def tabla_codigos_por_caso(
    textos: dict[str, dict[str, str]], anotaciones: dict[str, dict[str, pd.DataFrame]]
) -> pd.DataFrame:
    """Códigos distintos por caso (media, mediana, rango y casos sin código)."""
    filas = []
    for split in SPLITS_ANOTADOS:
        for pista in ("D", "P"):
            df = anotaciones[split][pista]
            conteo = codigos_por_caso(df, list(textos[split].keys()))
            filas.append(
                {
                    "split": split,
                    "pista": pista,
                    "media": round(conteo.mean(), 2),
                    "mediana": float(conteo.median()),
                    "minimo": int(conteo.min()),
                    "maximo": int(conteo.max()),
                    "casos_sin_codigo": int((conteo == 0).sum()),
                    "total_casos": len(conteo),
                }
            )
    return pd.DataFrame(filas)


def tabla_top_codigos_d(anotaciones: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """Top 30 códigos D en train+dev, normalizados a mayúsculas."""
    df = pd.concat([anotaciones[s]["D"] for s in SPLITS_ANOTADOS], ignore_index=True)
    df["codigo"] = df["codigo"].str.upper()
    frec = df.groupby("codigo").agg(
        frecuencia=("codigo", "size"), casos=("id_caso", "nunique")
    )
    frec = frec.sort_values(["frecuencia", "casos"], ascending=False).head(TOP_N)
    frec.insert(0, "rank", range(1, len(frec) + 1))
    return frec.reset_index()


def conteos_por_documento(
    textos: dict[str, dict[str, str]], contador
) -> dict[str, dict[str, Counter]]:
    """Calcula una sola vez el Counter de cada documento (evita repetir regex por expresión)."""
    return {
        split: {id_caso: contador(texto) for id_caso, texto in textos[split].items()}
        for split in SPLITS
    }


def tabla_conteo_textual(
    conteos: dict[str, dict[str, Counter]], claves: tuple[str, ...]
) -> pd.DataFrame:
    """Ocurrencias por expresión y por split, con total y número de documentos."""
    filas = []
    for clave in claves:
        fila = {"expresion": clave}
        total = 0
        docs = 0
        for split in SPLITS:
            ocurrencias = 0
            for conteo in conteos[split].values():
                n = conteo.get(clave, 0)
                ocurrencias += n
                docs += 1 if n else 0
            fila[f"ocurrencias_{split}"] = ocurrencias
            total += ocurrencias
        fila["ocurrencias_total"] = total
        fila["documentos_con_expresion"] = docs
        filas.append(fila)
    return pd.DataFrame(filas).sort_values("ocurrencias_total", ascending=False)


def analizar_x_diagnosticos(
    textos: dict[str, dict[str, str]], anotaciones: dict[str, dict[str, pd.DataFrame]]
) -> dict[str, int]:
    """Verifica offsets y negación previa en las menciones DIAGNOSTICO de train+dev."""
    total = offset_ok = con_negacion = sin_texto = discontinuas = 0
    for split in SPLITS_ANOTADOS:
        df = anotaciones[split]["X"]
        df = df[df["label"].str.upper() == "DIAGNOSTICO"]
        for fila in df.itertuples(index=False):
            texto = textos[split].get(fila.id_caso)
            total += 1
            if texto is None:
                sin_texto += 1
                continue
            spans = parsear_spans(fila.posicion)
            discontinuas += 1 if len(spans) > 1 else 0
            if offset_coincide(texto, fila.texto_referencia, fila.posicion):
                offset_ok += 1
            if hay_negacion_previa(texto, spans[0][0], VENTANA_NEGACION):
                con_negacion += 1
    return {
        "menciones_diagnostico": total,
        "menciones_discontinuas": discontinuas,
        "offset_coincide": offset_ok,
        "con_negacion_previa": con_negacion,
        "caso_sin_texto": sin_texto,
    }


def tabla_labels_x(anotaciones: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """Distribución de labels de la pista X por split."""
    filas = []
    for split in SPLITS_ANOTADOS:
        conteo = anotaciones[split]["X"]["label"].str.upper().value_counts()
        for label, n in conteo.items():
            filas.append({"split": split, "label": label, "menciones": int(n)})
    return pd.DataFrame(filas)


def graficar(
    casos: pd.DataFrame,
    longitud: pd.DataFrame,
    codigos: dict[str, list[int]],
    top: pd.DataFrame,
    abrev: pd.DataFrame,
    neg: pd.DataFrame,
    carpeta: Path,
) -> None:
    """Genera los gráficos PNG del EDA."""
    carpeta.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(casos["split"], casos["textos"], color="#4C72B0")
    ax.set_title("Casos (textos) por split")
    ax.set_ylabel("Casos")
    fig.tight_layout()
    fig.savefig(carpeta / "casos_por_split.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    palabras = longitud[longitud["metrica"] == "palabras"].set_index("split")
    ax.bar(palabras.index, palabras["media"], color="#55A868")
    ax.set_title("Longitud media de los casos (palabras) por split")
    ax.set_ylabel("Palabras (media)")
    fig.tight_layout()
    fig.savefig(carpeta / "longitud_media_palabras.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for split, valores in codigos.items():
        ax.hist(valores, bins=range(0, max(valores) + 2), alpha=0.5, label=split)
    ax.set_title("Códigos D distintos por caso")
    ax.set_xlabel("Códigos por caso")
    ax.set_ylabel("Casos")
    ax.legend()
    fig.tight_layout()
    fig.savefig(carpeta / "codigos_por_caso_D.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(top["codigo"][::-1], top["frecuencia"][::-1], color="#C44E52")
    ax.set_title(f"Top {TOP_N} códigos D (train + dev)")
    ax.set_xlabel("Frecuencia")
    fig.tight_layout()
    fig.savefig(carpeta / "top30_codigos_D.png", dpi=120)
    plt.close(fig)

    for nombre, tabla, titulo in (
        ("abreviaturas.png", abrev, "Abreviaturas clínicas (ocurrencias, todos los splits)"),
        ("negacion.png", neg, "Expresiones de negación (ocurrencias, todos los splits)"),
    ):
        fig, ax = plt.subplots(figsize=(8, 6))
        t = tabla.head(15).iloc[::-1]
        ax.barh(t["expresion"], t["ocurrencias_total"], color="#8172B2")
        ax.set_title(titulo)
        fig.tight_layout()
        fig.savefig(carpeta / nombre, dpi=120)
        plt.close(fig)


def cargar_anotaciones(raiz: Path) -> dict[str, dict[str, pd.DataFrame]]:
    """Lee los TSV de D, P y X de cada split que tenga anotaciones."""
    anot: dict[str, dict[str, pd.DataFrame]] = {}
    for split in SPLITS:
        anot[split] = {}
        for pista in ("D", "P", "X"):
            df = leer_anotaciones(ruta_split(raiz, split), split, pista)
            if df is not None:
                anot[split][pista] = df
    return anot


def main() -> None:
    """Punto de entrada: EDA completo de CodiEsp."""
    parser = argparse.ArgumentParser(description="EDA del corpus CodiEsp")
    parser.add_argument("--raiz", type=Path, default=RAIZ_CODIESP_DEFECTO)
    parser.add_argument("--salida", type=Path, default=DIR_EDA)
    args = parser.parse_args()
    raiz: Path = args.raiz
    salida: Path = args.salida
    if not raiz.exists():
        raise SystemExit(f"No existe el corpus en {raiz}. Ver README.txt del dataset.")
    salida.mkdir(parents=True, exist_ok=True)

    total_pasos = 7
    paso(1, total_pasos, "Leyendo textos de train, dev y test")
    textos = textos_por_split(raiz)

    paso(2, total_pasos, "Leyendo anotaciones D, P y X")
    anotaciones = cargar_anotaciones(raiz)

    paso(3, total_pasos, "Casos por split y longitud de los textos")
    casos = tabla_casos_por_split(textos, anotaciones)
    longitud = tabla_longitud(textos)

    paso(4, total_pasos, "Códigos por caso y top 30 códigos D")
    codigos_tabla = tabla_codigos_por_caso(textos, anotaciones)
    top = tabla_top_codigos_d(anotaciones)
    labels = tabla_labels_x(anotaciones)

    paso(5, total_pasos, "Abreviaturas comunes")
    abrev = tabla_conteo_textual(
        conteos_por_documento(textos, contar_abreviaturas), ABREVIATURAS_COMUNES
    )

    paso(6, total_pasos, "Expresiones de negación y verificación de offsets X")
    neg = tabla_conteo_textual(
        conteos_por_documento(textos, contar_negaciones), EXPRESIONES_NEGACION
    )
    x_diag = analizar_x_diagnosticos(textos, anotaciones)

    paso(7, total_pasos, "Guardando tablas, gráficos y resumen en results/eda/")
    casos.to_csv(salida / "casos_por_split.csv", index=False)
    longitud.to_csv(salida / "longitud_textos.csv", index=False)
    codigos_tabla.to_csv(salida / "codigos_por_caso.csv", index=False)
    top.to_csv(salida / "top30_codigos_D.csv", index=False)
    labels.to_csv(salida / "labels_X.csv", index=False)
    abrev.to_csv(salida / "abreviaturas.csv", index=False)
    neg.to_csv(salida / "negacion.csv", index=False)

    codigos_por_split = {
        s: codigos_por_caso(anotaciones[s]["D"], list(textos[s].keys())).tolist()
        for s in SPLITS_ANOTADOS
    }
    graficar(casos, longitud, codigos_por_split, top, abrev, neg, salida)

    resumen = {
        "corpus_raiz": str(raiz.name),
        "casos_por_split": casos.to_dict(orient="records"),
        "longitud_textos": longitud.to_dict(orient="records"),
        "codigos_por_caso": codigos_tabla.to_dict(orient="records"),
        "top30_codigos_D": top.to_dict(orient="records"),
        "labels_X": labels.to_dict(orient="records"),
        "abreviaturas_top15": abrev.head(15).to_dict(orient="records"),
        "negacion_top15": neg.head(15).to_dict(orient="records"),
        "x_diagnosticos_train_dev": {**x_diag, "ventana_caracteres": VENTANA_NEGACION},
        "codigos_D_distintos_train_dev": int(
            pd.concat([anotaciones[s]["D"] for s in SPLITS_ANOTADOS])["codigo"].str.upper().nunique()
        ),
        "codigos_D_con_minusculas": int(
            sum(
                (anotaciones[s]["D"]["codigo"] != anotaciones[s]["D"]["codigo"].str.upper()).sum()
                for s in SPLITS_ANOTADOS
            )
        ),
    }
    (salida / "resumen.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Listo. Resultados en {salida}", flush=True)


if __name__ == "__main__":
    main()
