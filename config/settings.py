"""Configuración central del proyecto.

Lee variables de entorno y el archivo .env (ver .env.example).
Las claves de API se guardan como SecretStr para que no aparezcan en logs.
El proveedor de LLM se cambia únicamente con LLM_PROVIDER.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

RAIZ_PROYECTO: Path = Path(__file__).resolve().parents[1]
DIR_RAW: Path = RAIZ_PROYECTO / "data" / "raw"
DIR_PROCESSED: Path = RAIZ_PROYECTO / "data" / "processed"
DIR_INDEX: Path = RAIZ_PROYECTO / "data" / "index"
DIR_RESULTS: Path = RAIZ_PROYECTO / "results"
DIR_EDA: Path = DIR_RESULTS / "eda"
DIR_DOCS: Path = RAIZ_PROYECTO / "docs"

# Temperatura fija para todos los LLM (regla del proyecto, no configurable).
TEMPERATURA_LLM: float = 0.0


class Settings(BaseSettings):
    """Parámetros del prototipo cargados desde el entorno o .env."""

    model_config = SettingsConfigDict(
        env_file=RAIZ_PROYECTO / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: Literal["ollama", "gemini"] = Field(
        default="ollama", alias="LLM_PROVIDER"
    )

    ollama_host: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST")
    ollama_model: str = Field(default="qwen2.5:7b-instruct", alias="OLLAMA_MODEL")

    gemini_api_key: SecretStr | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")

    embedding_model: str = Field(
        default="intfloat/multilingual-e5-base", alias="EMBEDDING_MODEL"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Devuelve la configuración una sola vez por proceso."""
    return Settings()
