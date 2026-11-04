"""Registro de feedback del codificador y de latencia/errores (carpeta logs/).

Nunca se guarda el texto de la nota: solo un identificador (id del ejemplo o hash corto).
"""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime
from pathlib import Path

from config.settings import RAIZ_PROYECTO

DIR_LOGS: Path = RAIZ_PROYECTO / "logs"
CAMPOS_FEEDBACK = ["fecha", "nota_id", "codigo", "proveedor", "decision"]
CAMPOS_LATENCIA = ["fecha", "nota_id", "proveedor", "modelo", "segundos", "llamadas_llm",
                   "n_sugerencias", "error"]


def id_nota(texto: str, id_ejemplo: str | None = None) -> str:
    """Id del ejemplo ficticio, o hash corto del texto (el texto no se guarda)."""
    return id_ejemplo or hashlib.sha256(texto.encode("utf-8")).hexdigest()[:10]


def _agregar(nombre: str, campos: list[str], fila: dict, dir_logs: Path) -> None:
    dir_logs.mkdir(parents=True, exist_ok=True)
    ruta = dir_logs / nombre
    nuevo = not ruta.exists()
    with ruta.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        if nuevo:
            w.writeheader()
        w.writerow({"fecha": datetime.now().isoformat(timespec="seconds"), **fila})


def registrar_decision(nota_id: str, codigo: str, proveedor: str, decision: str,
                       dir_logs: Path = DIR_LOGS) -> None:
    if decision not in ("aceptado", "rechazado"):
        raise ValueError(f"Decisión inválida: {decision}")
    _agregar("feedback.csv", CAMPOS_FEEDBACK, {"nota_id": nota_id, "codigo": codigo,
             "proveedor": proveedor, "decision": decision}, dir_logs)


def registrar_latencia(nota_id: str, proveedor: str, modelo: str, segundos: float,
                       llamadas: int, n_sugerencias: int, error: str = "",
                       dir_logs: Path = DIR_LOGS) -> None:
    _agregar("latencia.csv", CAMPOS_LATENCIA, {
        "nota_id": nota_id, "proveedor": proveedor, "modelo": modelo, "segundos": segundos,
        "llamadas_llm": llamadas, "n_sugerencias": n_sugerencias, "error": error[:200]}, dir_logs)
