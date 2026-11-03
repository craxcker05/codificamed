"""Interfaz común de los proveedores de LLM y utilidades compartidas.

Todos los proveedores usan temperatura 0 (ver config.settings.TEMPERATURA_LLM).
Los errores se envuelven en ErrorProveedor para que el resto del código no
dependa de las excepciones de cada SDK.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol, TypeVar

T = TypeVar("T")


class ErrorProveedor(RuntimeError):
    """Fallo al generar con un proveedor de LLM. Nunca incluye claves de API."""


class LLMProvider(Protocol):
    """Contrato mínimo que debe cumplir cualquier proveedor de LLM."""

    nombre: str
    modelo: str

    def generar(self, prompt: str, sistema: str | None = None) -> str:
        """Devuelve el texto generado para el prompt (temperatura 0)."""
        ...


def con_reintentos(
    operacion: Callable[[], T],
    es_reintentable: Callable[[Exception], bool],
    intentos: int = 5,
    espera_base: float = 2.0,
    dormir: Callable[[float], None] = time.sleep,
) -> T:
    """Ejecuta `operacion` reintentando con espera exponencial ante errores transitorios.

    Sirve para la cuota del tier gratuito (HTTP 429) y errores 503 de disponibilidad.
    """
    for intento in range(1, intentos + 1):
        try:
            return operacion()
        except Exception as error:  # noqa: BLE001 - se re-lanza si no es reintentable
            if intento == intentos or not es_reintentable(error):
                raise
            dormir(espera_base * (2 ** (intento - 1)))
    raise AssertionError("inalcanzable")  # pragma: no cover
