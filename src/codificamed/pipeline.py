"""Pipeline: nota -> menciones (LLM) -> negación -> candidatos (RAG) -> código (LLM)."""

from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from typing import TypeVar

import pandas as pd
from pydantic import BaseModel, ValidationError

from codificamed.llm import LLMProvider, crear_proveedor
from codificamed.llm.base import ErrorProveedor
from codificamed.negation import Estado, clasificar_en_texto
from codificamed.preprocess import preprocesar
from codificamed.retrieval import Indice, buscar
from codificamed.schemas import (
    CandidatoOut,
    RespuestaCodigo,
    RespuestaMenciones,
    ResultadoNota,
    Sugerencia,
)
from config.settings import DIR_PROCESSED

M = TypeVar("M", bound=BaseModel)
REINTENTOS_JSON = 2

SISTEMA = "Eres un asistente de codificación CIE-10 en español. Respondes SOLO con JSON válido."

PROMPT_MENCIONES = """Extrae de la nota clínica las menciones de diagnósticos, enfermedades, síntomas y hallazgos \
(incluye las que la nota niega o descarta). Copia cada mención TAL CUAL aparece en la nota, sin cambiar palabras. \
Usa fragmentos cortos (2 a 6 palabras). No incluyas procedimientos, medicamentos ni valores de laboratorio.
Responde: {{"menciones": ["...", "..."]}}

NOTA:
{nota}"""

PROMPT_CODIGO = """Mención clínica: "{mencion}"
Contexto: "{oracion}"

Candidatos CIE-10:
{lista}

Elige el código que mejor corresponde a la mención. Si ninguno corresponde, usa null.
Responde: {{"codigo": "<código exactamente como aparece arriba, o null>", "confianza": <0 a 1>}}"""


class _Llamador:
    """Envuelve al proveedor: JSON validado con reintentos y contadores de tiempo."""

    def __init__(self, proveedor: LLMProvider) -> None:
        self.proveedor, self.llamadas, self.tiempo = proveedor, 0, 0.0

    def json(self, prompt: str, modelo: type[M]) -> M | None:
        for _ in range(REINTENTOS_JSON + 1):
            t0 = time.perf_counter()
            try:
                bruto = self.proveedor.generar(prompt, SISTEMA)
            finally:
                self.llamadas += 1
                self.tiempo += time.perf_counter() - t0
            try:
                return modelo.model_validate(json.loads(_extraer_json(bruto)))
            except (ValueError, ValidationError):
                continue
        return None


def _extraer_json(texto: str) -> str:
    """Devuelve el primer objeto {...} del texto (los LLM a veces añaden ```json)."""
    m = re.search(r"\{.*\}", texto, re.DOTALL)
    return m.group(0) if m else texto


def _encontrar(texto: str, mencion: str) -> tuple[int, int] | None:
    mencion = mencion.strip()
    if not mencion:
        return None
    m = re.search(re.escape(mencion), texto, re.IGNORECASE)
    return (m.start(), m.end()) if m else None


@lru_cache(maxsize=1)
def _codigos_catalogo() -> frozenset[str]:
    df = pd.read_parquet(DIR_PROCESSED / "catalogo_cie10.parquet", columns=["codigo"])
    return frozenset(df["codigo"])


def analizar(
    nota: str, proveedor: LLMProvider | None = None, indice: Indice | None = None,
    k: int = 10, usar_negacion: bool = True,
) -> ResultadoNota:
    """Sugiere códigos CIE-10 para la nota. Los negados se devuelven aparte."""
    t_ini = time.perf_counter()
    proveedor = proveedor or crear_proveedor()
    ll = _Llamador(proveedor)
    tp = preprocesar(nota)
    res = ResultadoNota(proveedor=proveedor.nombre, modelo=proveedor.modelo)

    r = ll.json(PROMPT_MENCIONES.format(nota=tp.texto), RespuestaMenciones)
    vistas: set[tuple[int, int]] = set()
    for mencion in (r.menciones if r else []):
        span = _encontrar(tp.texto, mencion)
        if span is None:  # no es literal de la nota: se descarta
            res.descartadas.append(mencion)
            continue
        if span in vistas:
            continue
        vistas.add(span)
        estado = clasificar_en_texto(tp, *span).estado if usar_negacion else Estado.AFIRMADO
        evidencia, offsets = tp.evidencia(*span), tp.span_original(*span)
        if estado is not Estado.AFIRMADO:  # sin recuperación: solo se informa aparte
            res.negados.append(Sugerencia(
                codigo="", codigo_original="", descripcion="", evidencia=evidencia,
                offsets=offsets, estado=estado.value, confianza=1.0))
            continue
        cands = buscar(tp.texto[span[0]:span[1]], k, indice)
        lista = "\n".join(f"- {c.codigo_original}: {c.descripcion}" for c in cands)
        oracion = next((o.texto for o in tp.oraciones() if o.ini <= span[0] < o.fin), "")
        elec = ll.json(PROMPT_CODIGO.format(mencion=mencion, oracion=oracion, lista=lista), RespuestaCodigo)
        elegido = next((c for c in cands if elec and elec.codigo
                        and c.codigo_original.upper() == elec.codigo.strip().upper()), None)
        if elegido is None or elec is None:
            res.descartadas.append(mencion)
            continue
        res.sugerencias.append(Sugerencia(
            codigo=elegido.codigo, codigo_original=elegido.codigo_original,
            descripcion=elegido.descripcion, evidencia=evidencia, offsets=offsets,
            estado=Estado.AFIRMADO.value, confianza=elec.confianza,
            candidatos=[CandidatoOut(codigo=c.codigo, descripcion=c.descripcion, score=c.score) for c in cands]))

    res.sugerencias = _sin_repetidos(res.sugerencias)
    res.llamadas_llm, res.tiempo_llm_s = ll.llamadas, round(ll.tiempo, 2)
    res.tiempo_total_s = round(time.perf_counter() - t_ini, 2)
    _validar(res, nota)
    return res


def _sin_repetidos(sugs: list[Sugerencia]) -> list[Sugerencia]:
    """Un código aparece una vez: se conserva el de mayor confianza."""
    mejor: dict[str, Sugerencia] = {}
    for s in sugs:
        if s.codigo not in mejor or s.confianza > mejor[s.codigo].confianza:
            mejor[s.codigo] = s
    return list(mejor.values())


def _validar(res: ResultadoNota, nota: str) -> None:
    """Post-proceso duro: código en el catálogo y evidencia literal de la nota."""
    catalogo = _codigos_catalogo()
    for s in [*res.sugerencias, *res.negados]:
        if (s.codigo and s.codigo not in catalogo) or nota[s.offsets[0]:s.offsets[1]] != s.evidencia:
            raise ErrorProveedor(f"Sugerencia inválida: {s.codigo} / {s.evidencia!r}")
