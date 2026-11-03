"""Preprocesado de notas clínicas: normalización, abreviaturas y oraciones.

Principio: el texto procesado puede diferir del original (se expanden
abreviaturas), pero **cada carácter procesado conserva el rango del original
del que procede**. Así, cualquier evidencia hallada en el texto procesado se
traduce con `TextoProcesado.span_original` a texto literal de la nota.

- La normalización conserva la longitud (1 carácter -> 1 carácter).
- La expansión de una abreviatura asigna a todos sus caracteres el rango de la
  abreviatura original.
- Abreviaturas ambiguas (TA, FC, FR, PA, d, h, a., IC, CC) solo se expanden en
  contexto seguro; si no, se dejan tal cual (ver `_CONTEXTUALES`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

RUTA_ABREVIATURAS = Path(__file__).resolve().parents[2] / "config" / "abreviaturas.yaml"

# Normalización 1:1 (nunca cambia la longitud del texto).
_REEMPLAZOS: dict[str, str] = {
    " ": " ", "​": " ", "‌": " ", "‍": " ", "﻿": " ",
    "\t": " ", "\r": " ", "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-",
}
_TABLA = str.maketrans(_REEMPLAZOS)

# Abreviaturas de >= 3 letras que sí son palabras o riesgosas en minúscula:
# solo se aceptan con la capitalización exacta del diccionario.
_SENSIBLES_A_MAYUSCULAS: frozenset[str] = frozenset({"ECO", "ASA", "EDA", "URG", "BUN"})

# Solo se expanden si el contexto lo confirma.
#   "numero_despues": va seguida de un número (signos vitales: "TA 150/95").
#   "numero_antes": va precedida de un número ("3 d", "6 h", "58 a.").
#   "nunca": demasiado ambigua; no se expande (el yaml pide revisión clínica).
_CONTEXTUALES: dict[str, str] = {
    "TA": "numero_despues", "FC": "numero_despues", "FR": "numero_despues",
    "PA": "numero_despues", "T°": "numero_despues",
    "d": "numero_antes", "h": "numero_antes", "a.": "numero_antes",
    "IC": "nunca", "CC": "nunca",
}

# Palabras tras las que un punto NO termina oración.
_PROTEGIDAS: frozenset[str] = frozenset(
    {"dr", "dra", "sr", "sra", "srta", "etc", "aprox", "vs", "ej", "tel", "num", "núm",
     "mg", "ml", "cm", "mm", "kg", "ud", "uds", "fig", "pág", "pag", "hrs", "min", "no",
     "cap", "aprox", "ca", "ref"}
)


@dataclass(frozen=True)
class Oracion:
    """Oración del texto procesado, con sus offsets en ambos textos."""

    texto: str
    ini: int
    fin: int
    ini_orig: int
    fin_orig: int


@dataclass
class TextoProcesado:
    """Texto normalizado/expandido con el mapa de offsets hacia el original."""

    original: str
    texto: str
    _orig_ini: list[int] = field(repr=False)
    _orig_fin: list[int] = field(repr=False)

    def span_original(self, ini: int, fin: int) -> tuple[int, int]:
        """Traduce [ini, fin) del texto procesado a [ini, fin) del original."""
        if not (0 <= ini < fin <= len(self.texto)):
            raise ValueError(f"Rango inválido [{ini}, {fin}) para texto de {len(self.texto)}")
        return min(self._orig_ini[ini:fin]), max(self._orig_fin[ini:fin])

    def evidencia(self, ini: int, fin: int) -> str:
        """Texto literal de la nota correspondiente a [ini, fin) del procesado."""
        a, b = self.span_original(ini, fin)
        return self.original[a:b]

    def oraciones(self) -> list[Oracion]:
        return segmentar_oraciones(self)


# ---------------------------------------------------------------- abreviaturas

@lru_cache(maxsize=1)
def cargar_abreviaturas(ruta: Path = RUTA_ABREVIATURAS) -> dict[str, str]:
    """Lee config/abreviaturas.yaml -> {abreviatura: expansión}."""
    datos = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    return {e["abreviatura"]: e["expansion"] for e in datos["abreviaturas"]}


def _distingue_mayusculas(abrev: str) -> bool:
    """Las abreviaturas cortas o riesgosas exigen la capitalización exacta."""
    return len(abrev) < 3 or abrev in _SENSIBLES_A_MAYUSCULAS


def _compilar(abreviaturas: dict[str, str]) -> tuple[re.Pattern[str], dict[str, str]]:
    """Un solo patrón con las más largas primero. Devuelve (patrón, clave->abrev)."""
    claves: dict[str, str] = {}
    partes: list[str] = []
    for abrev in sorted(abreviaturas, key=len, reverse=True):
        if _distingue_mayusculas(abrev):
            partes.append(f"(?-i:{re.escape(abrev)})")
            claves[abrev] = abrev
        else:
            partes.append(re.escape(abrev))
            claves[abrev.lower()] = abrev
    patron = re.compile(r"(?<![\w])(?:" + "|".join(partes) + r")(?![\w])", re.IGNORECASE)
    return patron, claves


def _contexto_ok(abrev: str, original: str, ini: int, fin: int) -> bool:
    regla = _CONTEXTUALES.get(abrev)
    if regla is None:
        return True
    if regla == "nunca":
        return False
    if regla == "numero_antes":
        return bool(re.search(r"\d\s?$", original[max(0, ini - 3):ini]))
    return bool(re.match(r"\s?\d", original[fin:fin + 3]))  # numero_despues


def _inicio_de_oracion(texto: str, pos: int) -> bool:
    previo = texto[:pos].rstrip(" ")
    return previo == "" or previo[-1] in ".?!:\n"


def normalizar(texto: str) -> str:
    """Normalización que conserva la longitud: espacios raros, comillas, guiones."""
    return texto.translate(_TABLA)


def preprocesar(texto: str, abreviaturas: dict[str, str] | None = None) -> TextoProcesado:
    """Normaliza, expande abreviaturas y devuelve el texto con su mapa de offsets."""
    abreviaturas = cargar_abreviaturas() if abreviaturas is None else abreviaturas
    base = normalizar(texto)
    patron, claves = _compilar(abreviaturas)

    salida: list[str] = []
    o_ini: list[int] = []
    o_fin: list[int] = []
    cursor = 0

    def copiar(a: int, b: int) -> None:
        for i in range(a, b):
            salida.append(base[i])
            o_ini.append(i)
            o_fin.append(i + 1)

    for m in patron.finditer(base):
        abrev = claves[m.group(0)] if m.group(0) in claves else claves.get(m.group(0).lower())
        if abrev is None or not _contexto_ok(abrev, base, m.start(), m.end()):
            continue
        copiar(cursor, m.start())
        expansion = abreviaturas[abrev]
        if _inicio_de_oracion(base, m.start()) and m.group(0)[0].isupper():
            expansion = expansion[0].upper() + expansion[1:]  # conserva el límite de oración
        for ch in expansion:
            salida.append(ch)
            o_ini.append(m.start())
            o_fin.append(m.end())
        cursor = m.end()
    copiar(cursor, len(base))

    return TextoProcesado(original=texto, texto="".join(salida), _orig_ini=o_ini, _orig_fin=o_fin)


# ---------------------------------------------------------------- oraciones

def _es_fin_de_oracion(texto: str, pos: int) -> bool:
    """¿El signo en `pos` ('.', '?', '!') cierra una oración?"""
    if texto[pos] in "?!":
        return True
    resto = texto[pos + 1:]
    if resto and not resto[0].isspace():
        return False  # "3.5", "S.A.", "etc.," ...
    previo = re.search(r"([\wáéíóúüñÁÉÍÓÚÜÑ]+)$", texto[:pos])
    palabra = previo.group(1) if previo else ""
    if palabra.lower() in _PROTEGIDAS:
        return False
    if len(palabra) == 1 and palabra.isalpha():  # iniciales: "J. Pérez"
        return False
    if palabra.isdigit():
        inicio = texto[:pos - len(palabra)].rstrip(" ")
        if inicio == "" or inicio.endswith("\n"):  # numeración "1. Cefalea"
            return False
    siguiente = resto.lstrip()
    if siguiente and siguiente[0].islower():
        return False  # "aprox. cinco": la oración continúa
    return True


def segmentar_oraciones(tp: TextoProcesado) -> list[Oracion]:
    """Divide en oraciones por puntuación final y líneas en blanco.

    Un salto de línea simple solo corta si la línea anterior terminó en
    puntuación final o dos puntos; si no, se trata como un espacio (texto con
    saltos de línea de maquetación).
    """
    t = tp.texto
    cortes: list[int] = []  # índice exclusivo del fin de cada oración
    i = 0
    while i < len(t):
        c = t[i]
        if c in ".?!":
            j = i
            while j + 1 < len(t) and t[j + 1] in ".?!":
                j += 1
            if _es_fin_de_oracion(t, j):
                cortes.append(j + 1)
            i = j + 1
            continue
        if c == "\n":
            if re.match(r"\n\s*\n", t[i:]):
                cortes.append(i)
            else:
                previo = t[:i].rstrip(" ")
                if previo.endswith(":"):
                    cortes.append(i)
        i += 1
    cortes.append(len(t))

    oraciones: list[Oracion] = []
    inicio = 0
    for corte in sorted(set(cortes)):
        tramo = t[inicio:corte]
        lstrip = len(tramo) - len(tramo.lstrip())
        rstrip = len(tramo.rstrip())
        if rstrip > lstrip:
            a, b = inicio + lstrip, inicio + rstrip
            ia, fb = tp.span_original(a, b)
            oraciones.append(Oracion(texto=t[a:b], ini=a, fin=b, ini_orig=ia, fin_orig=fb))
        inicio = corte
    return oraciones
