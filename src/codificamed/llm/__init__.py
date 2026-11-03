"""Fábrica de proveedores de LLM. El proveedor se elige solo con LLM_PROVIDER."""

from __future__ import annotations

from codificamed.llm.base import ErrorProveedor, LLMProvider
from config.settings import Settings, get_settings


def crear_proveedor(ajustes: Settings | None = None) -> LLMProvider:
    """Crea el proveedor indicado por LLM_PROVIDER (ollama por defecto, o gemini)."""
    ajustes = ajustes or get_settings()
    if ajustes.llm_provider == "ollama":
        from codificamed.llm.ollama_provider import OllamaProvider

        return OllamaProvider(modelo=ajustes.ollama_model, host=ajustes.ollama_host)
    if ajustes.llm_provider == "gemini":
        from codificamed.llm.gemini_provider import GeminiProvider

        clave = ajustes.gemini_api_key.get_secret_value() if ajustes.gemini_api_key else ""
        return GeminiProvider(modelo=ajustes.gemini_model, api_key=clave)
    raise ErrorProveedor(f"Proveedor no soportado: {ajustes.llm_provider}")


__all__ = ["ErrorProveedor", "LLMProvider", "crear_proveedor"]
