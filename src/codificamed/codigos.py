"""Normalización de códigos CIE-10 (sin punto, en mayúsculas).

Ejemplo: 'A01.00' -> 'A0100', 'n44.8' -> 'N448'.

La transformación NO es reversible por sí sola: 'A0100' podría venir de
'A01.00' o de 'A010.0'. La reversibilidad se garantiza con la tabla del
catálogo (columna codigo_original) y comprobando que no hay colisiones.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

PATRON_NORMALIZADO: re.Pattern[str] = re.compile(r"^[A-Z][0-9A-Z]{2}[0-9A-Z]*$")


def normalizar_codigo(codigo: str) -> str:
    """Quita espacios y puntos y pasa a mayúsculas."""
    return codigo.strip().replace(".", "").upper()


def categoria(codigo_normalizado: str) -> str:
    """Categoría CIE-10: los tres primeros caracteres."""
    return codigo_normalizado[:3]


def es_formato_valido(codigo_normalizado: str) -> bool:
    """True si el código normalizado tiene letra inicial, al menos 3 caracteres y solo A-Z/0-9."""
    return PATRON_NORMALIZADO.match(codigo_normalizado) is not None


def detectar_colisiones(originales: Iterable[str]) -> dict[str, list[str]]:
    """Agrupa códigos originales distintos que quedan iguales tras normalizar.

    Devuelve solo los grupos con más de un original (las colisiones).
    """
    grupos: dict[str, set[str]] = defaultdict(set)
    for original in originales:
        grupos[normalizar_codigo(original)].add(original)
    return {clave: sorted(valores) for clave, valores in grupos.items() if len(valores) > 1}
