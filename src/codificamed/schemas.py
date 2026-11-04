"""Esquemas pydantic: respuestas JSON del LLM y salida del pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RespuestaMenciones(BaseModel):
    """Respuesta del LLM al extraer menciones diagnósticas."""

    menciones: list[str] = Field(default_factory=list)


class RespuestaCodigo(BaseModel):
    """Respuesta del LLM al elegir un código entre los candidatos."""

    codigo: str | None = None
    confianza: float = Field(default=0.5, ge=0, le=1)


class CandidatoOut(BaseModel):
    codigo: str
    descripcion: str
    score: float


class Sugerencia(BaseModel):
    """Un código sugerido con su evidencia literal de la nota."""

    codigo: str  # sin punto
    codigo_original: str  # con punto
    descripcion: str
    evidencia: str  # texto literal de la nota
    offsets: tuple[int, int]  # [ini, fin) en la nota original
    estado: str  # afirmado | negado | sospecha
    confianza: float
    candidatos: list[CandidatoOut] = Field(default_factory=list)


class ResultadoNota(BaseModel):
    sugerencias: list[Sugerencia] = Field(default_factory=list)  # afirmadas
    negados: list[Sugerencia] = Field(default_factory=list)  # negadas o en sospecha (sin código)
    descartadas: list[str] = Field(default_factory=list)  # menciones sin código o no literales
    proveedor: str = ""
    modelo: str = ""
    llamadas_llm: int = 0
    tiempo_llm_s: float = 0.0
    tiempo_total_s: float = 0.0
