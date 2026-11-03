"""Proveedor externo Gemini (API gratuita de Google AI Studio).

Solo debe recibir CodiEsp y notas ficticias (ver CLAUDE.md).
La clave se lee de GEMINI_API_KEY y nunca se imprime ni se registra.
"""

from __future__ import annotations

from typing import Any

from codificamed.llm.base import ErrorProveedor, con_reintentos
from config.settings import TEMPERATURA_LLM

# Fragmentos de error transitorio: cuota agotada (429) o servicio no disponible (503).
_MARCAS_TRANSITORIAS: tuple[str, ...] = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")


def es_error_transitorio(error: Exception) -> bool:
    """True si el error de la API es de cuota o disponibilidad, y conviene reintentar."""
    texto = str(error)
    return any(marca in texto for marca in _MARCAS_TRANSITORIAS)


class GeminiProvider:
    """Genera texto con un modelo de Gemini mediante google-genai."""

    nombre = "gemini"

    def __init__(self, modelo: str, api_key: str, cliente: Any | None = None) -> None:
        self.modelo = modelo
        if cliente is None:
            if not api_key:
                raise ErrorProveedor("Falta GEMINI_API_KEY en .env para usar el proveedor gemini.")
            from google import genai

            cliente = genai.Client(api_key=api_key)
        self._cliente = cliente

    def generar(self, prompt: str, sistema: str | None = None) -> str:
        """Devuelve la respuesta de Gemini con temperatura 0, reintentando ante cuota."""
        from google.genai import types

        configuracion = types.GenerateContentConfig(
            temperature=TEMPERATURA_LLM,
            system_instruction=sistema,
        )

        def llamar() -> str:
            respuesta = self._cliente.models.generate_content(
                model=self.modelo,
                contents=prompt,
                config=configuracion,
            )
            return str(respuesta.text or "")

        try:
            return con_reintentos(llamar, es_reintentable=es_error_transitorio)
        except Exception as error:  # noqa: BLE001
            raise ErrorProveedor(
                f"Gemini falló con el modelo {self.modelo}: {type(error).__name__}"
            ) from None
