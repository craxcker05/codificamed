"""Detección de negación e incertidumbre en español (estilo NegEx/ConText).

Dada una oración y el rango de una mención clínica, devuelve su estado:
AFIRMADO, NEGADO o SOSPECHA.

Reglas:
- Disparadores previos ("niega", "sin", "se descarta"...) actúan hacia
  adelante; posteriores ("descartada", "negativo") hacia atrás; algunos
  ("probable") en ambos sentidos.
- Cada disparador tiene una ventana de alcance en palabras.
- El alcance se corta con terminadores (pero, sin embargo, aunque...), con
  puntuación fuerte (. ; salto de línea) y con verbos afirmativos
  ("refiere", "presenta"...).
- Pseudo-negaciones ("sin embargo", "no solo", "no hay duda") no disparan.
- Gana el disparador más cercano a la mención. Las coincidencias se resuelven
  por la forma más larga ("no se descarta" > "descarta").
- El análisis es sobre una copia en minúsculas y sin tildes de la misma
  longitud, por lo que los offsets coinciden con la oración.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum

from codificamed.preprocess import TextoProcesado


class Estado(str, Enum):
    AFIRMADO = "afirmado"
    NEGADO = "negado"
    SOSPECHA = "sospecha"


@dataclass(frozen=True)
class Disparador:
    texto: str
    estado: Estado | None  # None = pseudo-negación
    direccion: str  # "pre", "post" o "ambos"
    ventana: int  # palabras de alcance


@dataclass(frozen=True)
class ResultadoNegacion:
    estado: Estado
    disparador: str | None = None
    ini_disparador: int | None = None
    fin_disparador: int | None = None


N, S = Estado.NEGADO, Estado.SOSPECHA

_DISPARADORES: list[Disparador] = [
    # Negación previa
    *[Disparador(t, N, "pre", 8) for t in (
        "niega", "niegan", "negó", "nego", "no refiere", "no refieren", "no presenta",
        "no presentan", "no presentó", "no tiene", "no tienen", "no hay", "no existe",
        "no existen", "no se evidencia", "no se observa", "no se aprecia", "no se palpa",
        "no se detecta", "no hay evidencia de", "ausencia de", "carece de", "libre de",
        "negativo para", "negativa para", "negativos para", "negativas para",
        "se descarta", "se descartó", "descarta", "descartó", "descartan", "sin evidencia de",
        "sin signos de", "sin datos de", "sin antecedentes de", "nunca ha tenido", "tampoco")],
    *[Disparador(t, N, "pre", 6) for t in ("no", "sin", "s/", "ni")],
    # Negación posterior
    *[Disparador(t, N, "post", 4) for t in (
        "descartado", "descartada", "descartados", "descartadas", "negado", "negada",
        "negados", "negadas", "ausente", "ausentes", "negativo", "negativa", "negativos",
        "negativas", "no evidenciable", "no se evidencia", "no se observa")],
    # Sospecha / incertidumbre
    *[Disparador(t, S, "pre", 8) for t in (
        "sospecha de", "sospecha", "se sospecha", "sospechamos", "sospechoso de",
        "compatible con", "sugestivo de", "sugerente de", "podria tratarse de",
        "podria ser", "no se descarta", "no descarta", "no se puede descartar",
        "para descartar", "a descartar", "por descartar", "descartar", "rule out", "r/o",
        "probablemente", "posiblemente")],
    *[Disparador(t, S, "ambos", 4) for t in ("probable", "posible")],
    *[Disparador(t, S, "post", 4) for t in (
        "en estudio", "por confirmar", "a descartar", "por descartar", "sospechoso")],
    # Pseudo-negaciones: consumen el texto para que "no"/"sin" no se disparen.
    *[Disparador(t, None, "pre", 0) for t in (
        "sin embargo", "no solo", "no solamente", "no hay duda", "sin duda",
        "no cabe duda", "no obstante", "no solo")],
]

_TERMINADORES = re.compile(
    r"(?<![a-z0-9])(?:pero|aunque|sin embargo|no obstante|excepto|salvo|sino|"
    r"mientras que|ahora bien|refiere|refieren|presenta|presentan|cursa|acude|padece|"
    r"tiene|tienen|ingresa|consulta por|se evidencia|se observa)(?![a-z0-9])"
)
_PUNTUACION_FUERTE = re.compile(r"[.;\n!?]")
_PALABRA = re.compile(r"\w+")


def _plegar(texto: str) -> str:
    """Minúsculas y sin tildes, carácter a carácter, conservando la longitud."""
    salida = []
    for c in texto:
        base = unicodedata.normalize("NFD", c)[0].lower()
        salida.append(base if len(base) == 1 else c)
    return "".join(salida)


def _construir_indice() -> tuple[re.Pattern[str], dict[str, list[Disparador]]]:
    por_forma: dict[str, list[Disparador]] = {}
    for d in _DISPARADORES:
        por_forma.setdefault(_plegar(d.texto), []).append(d)
    formas = sorted(por_forma, key=len, reverse=True)
    alternativas = [re.escape(f).replace(r"\ ", r"\s+") for f in formas]
    patron = re.compile(r"(?<![a-z0-9])(?:" + "|".join(alternativas) + r")(?![a-z0-9])")
    return patron, por_forma


_PATRON, _POR_FORMA = _construir_indice()


def _palabras(tramo: str) -> int:
    return len(_PALABRA.findall(tramo))


def _alcanza(tramo: str, ventana: int, hacia_atras: bool = False) -> bool:
    """¿Llega el alcance del disparador a través de `tramo` (texto entre ambos)?

    Un disparador posterior no cruza comas: "faringitis, probable etiología
    viral" califica a la etiología, no a la faringitis.
    """
    if _palabras(tramo) > ventana:
        return False
    if hacia_atras and "," in tramo:
        return False
    if _PUNTUACION_FUERTE.search(tramo) or _TERMINADORES.search(tramo):
        return False
    return True


def clasificar(oracion: str, ini: int, fin: int) -> ResultadoNegacion:
    """Estado de la mención oracion[ini:fin] dentro de su oración."""
    if not (0 <= ini < fin <= len(oracion)):
        raise ValueError(f"Rango inválido [{ini}, {fin}) para oración de {len(oracion)}")
    t = _plegar(oracion)

    candidatos: list[tuple[int, Disparador, int, int]] = []  # (distancia, d, ini, fin)
    for m in _PATRON.finditer(t):
        forma = re.sub(r"\s+", " ", m.group(0))
        if m.start() < fin and m.end() > ini:
            continue  # el disparador se solapa con la mención
        for d in _POR_FORMA[forma]:
            if d.estado is None:
                continue
            if m.end() <= ini and d.direccion in ("pre", "ambos"):
                if _alcanza(t[m.end():ini], d.ventana):
                    candidatos.append((ini - m.end(), d, m.start(), m.end()))
            elif m.start() >= fin and d.direccion in ("post", "ambos"):
                if _alcanza(t[fin:m.start()], d.ventana, hacia_atras=True):
                    candidatos.append((m.start() - fin, d, m.start(), m.end()))

    if not candidatos:
        return ResultadoNegacion(Estado.AFIRMADO)
    _, d, a, b = min(candidatos, key=lambda c: c[0])
    return ResultadoNegacion(d.estado, oracion[a:b], a, b)  # type: ignore[arg-type]


def clasificar_frase(frase: str, mencion: str) -> ResultadoNegacion:
    """Atajo: busca `mencion` (sin distinguir mayúsculas ni tildes) en `frase`."""
    pos = _plegar(frase).find(_plegar(mencion))
    if pos < 0:
        raise ValueError(f"La mención {mencion!r} no está en la frase {frase!r}")
    return clasificar(frase, pos, pos + len(mencion))


def clasificar_en_texto(tp: TextoProcesado, ini: int, fin: int) -> ResultadoNegacion:
    """Estado de la mención tp.texto[ini:fin], usando solo su oración."""
    for o in tp.oraciones():
        if o.ini <= ini and fin <= o.fin:
            return clasificar(o.texto, ini - o.ini, fin - o.ini)
    return ResultadoNegacion(Estado.AFIRMADO)  # mención que cruza oraciones
