"""Resume el feedback y la latencia registrados en logs/.

Muestra tasa de aceptación por proveedor y por código, latencia media y tasa de error,
y avisa si se cruzan los umbrales de alerta (ver Fase 6 de la bitácora).

Uso: python scripts/monitor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
LOGS = RAIZ / "logs"

# Umbrales de alerta (documentados en docs/BITACORA.md, Fase 6)
UMBRAL_ACEPTACION = 0.60
UMBRAL_LATENCIA_S = 120.0
UMBRAL_ERRORES = 0.10
MIN_DECISIONES = 5  # no alertar con muestras muy pequeñas


def resumen_feedback(fb: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fb = fb.assign(aceptado=(fb["decision"] == "aceptado").astype(int))
    por_prov = fb.groupby("proveedor")["aceptado"].agg(decisiones="count", aceptacion="mean")
    por_cod = (fb.groupby(["proveedor", "codigo"])["aceptado"]
               .agg(decisiones="count", aceptacion="mean").sort_values("decisiones", ascending=False))
    return por_prov.round(3), por_cod.round(3)


def alertas(por_prov: pd.DataFrame | None, lat: pd.DataFrame | None) -> list[str]:
    avisos = []
    if por_prov is not None:
        for p, f in por_prov.iterrows():
            if f["decisiones"] >= MIN_DECISIONES and f["aceptacion"] < UMBRAL_ACEPTACION:
                avisos.append(f"{p}: aceptación {f['aceptacion']:.2f} < {UMBRAL_ACEPTACION}")
    if lat is not None:
        for p, g in lat.groupby("proveedor"):
            ok = g[g["error"].isna()]
            if len(ok) and ok["segundos"].mean() > UMBRAL_LATENCIA_S:
                avisos.append(f"{p}: latencia media {ok['segundos'].mean():.0f} s > {UMBRAL_LATENCIA_S:.0f} s")
            if g["error"].notna().mean() > UMBRAL_ERRORES:
                avisos.append(f"{p}: tasa de error {g['error'].notna().mean():.2f} > {UMBRAL_ERRORES}")
    return avisos


def main() -> None:
    fb_ruta, lat_ruta = LOGS / "feedback.csv", LOGS / "latencia.csv"
    por_prov = lat = None
    if fb_ruta.exists():
        por_prov, por_cod = resumen_feedback(pd.read_csv(fb_ruta))
        print("== Aceptación por proveedor ==\n", por_prov, "\n")
        print("== Aceptación por código ==\n", por_cod.head(20), "\n")
    else:
        print("Aún no hay feedback en logs/feedback.csv")
    if lat_ruta.exists():
        lat = pd.read_csv(lat_ruta)
        print("== Latencia y errores por proveedor ==")
        print(lat.groupby("proveedor").agg(notas=("segundos", "count"),
              segundos_medio=("segundos", "mean"), con_error=("error", lambda e: e.notna().sum())).round(2))
    else:
        print("Aún no hay registro de latencia en logs/latencia.csv")
    for a in alertas(por_prov, lat):
        print("ALERTA:", a)
    sys.exit(0)


if __name__ == "__main__":
    main()
