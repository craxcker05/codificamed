"""Proveedor local con Ollama (por defecto)."""

from __future__ import annotations

from ollama import Client

from codificamed.llm.base import ErrorProveedor
from config.settings import TEMPERATURA_LLM


class OllamaProvider:
    """Genera texto con un modelo servido por Ollama en la máquina local."""

    nombre = "ollama"

    def __init__(self, modelo: str, host: str, cliente: Client | None = None) -> None:
        self.modelo = modelo
        self._cliente = cliente or Client(host=host)

    def generar(self, prompt: str, sistema: str | None = None) -> str:
        """Devuelve la respuesta del modelo con temperatura 0."""
        mensajes = []
        if sistema:
            mensajes.append({"role": "system", "content": sistema})
        mensajes.append({"role": "user", "content": prompt})
        try:
            respuesta = self._cliente.chat(
                model=self.modelo,
                messages=mensajes,
                options={"temperature": TEMPERATURA_LLM},
            )
        except Exception as error:  # noqa: BLE001
            raise ErrorProveedor(f"Ollama falló con el modelo {self.modelo}: {error}") from error
        return str(respuesta["message"]["content"])
