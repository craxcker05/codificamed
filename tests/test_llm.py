"""Pruebas de la capa de LLM sin red: clientes falsos y comprobación de reintentos."""

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from codificamed.llm import ErrorProveedor, crear_proveedor
from codificamed.llm.base import con_reintentos
from codificamed.llm.gemini_provider import GeminiProvider, es_error_transitorio
from codificamed.llm.ollama_provider import OllamaProvider
from config.settings import TEMPERATURA_LLM, Settings


class ClienteOllamaFalso:
    """Imita el método chat() del cliente de Ollama y guarda la última llamada."""

    def __init__(self) -> None:
        self.llamada: dict[str, Any] = {}

    def chat(self, **kwargs: Any) -> dict[str, Any]:
        self.llamada = kwargs
        return {"message": {"content": "respuesta local"}}


class ClienteGeminiFalso:
    """Imita client.models.generate_content() de google-genai."""

    def __init__(self, fallos: list[Exception] | None = None) -> None:
        self._fallos = list(fallos or [])
        self.llamadas: list[dict[str, Any]] = []
        self.models = SimpleNamespace(generate_content=self._generar)

    def _generar(self, **kwargs: Any) -> SimpleNamespace:
        self.llamadas.append(kwargs)
        if self._fallos:
            raise self._fallos.pop(0)
        return SimpleNamespace(text="respuesta externa")


def test_temperatura_del_proyecto_es_cero() -> None:
    assert TEMPERATURA_LLM == 0.0


def test_ollama_usa_temperatura_cero_y_sistema() -> None:
    cliente = ClienteOllamaFalso()
    proveedor = OllamaProvider(modelo="m", host="http://x", cliente=cliente)
    assert proveedor.generar("hola", sistema="eres experto") == "respuesta local"
    assert cliente.llamada["options"] == {"temperature": 0.0}
    assert cliente.llamada["messages"][0] == {"role": "system", "content": "eres experto"}


def test_gemini_sin_clave_falla_con_mensaje_claro() -> None:
    with pytest.raises(ErrorProveedor, match="GEMINI_API_KEY"):
        GeminiProvider(modelo="g", api_key="")


def test_gemini_genera_con_cliente_falso() -> None:
    pytest.importorskip("google.genai")
    cliente = ClienteGeminiFalso()
    proveedor = GeminiProvider(modelo="g", api_key="x", cliente=cliente)
    assert proveedor.generar("hola") == "respuesta externa"
    config = cliente.llamadas[0]["config"]
    assert config.temperature == 0.0


def test_gemini_reintenta_ante_cuota() -> None:
    pytest.importorskip("google.genai")
    fallo = RuntimeError("429 RESOURCE_EXHAUSTED")
    cliente = ClienteGeminiFalso(fallos=[fallo, fallo])
    proveedor = GeminiProvider(modelo="g", api_key="x", cliente=cliente)
    assert proveedor.generar("hola") == "respuesta externa"
    assert len(cliente.llamadas) == 3


def test_gemini_no_oculta_la_clave_en_el_error() -> None:
    pytest.importorskip("google.genai")
    cliente = ClienteGeminiFalso(fallos=[ValueError("clave AIzaSECRETO inválida")])
    proveedor = GeminiProvider(modelo="g", api_key="AIzaSECRETO", cliente=cliente)
    with pytest.raises(ErrorProveedor) as info:
        proveedor.generar("hola")
    assert "AIzaSECRETO" not in str(info.value)


def test_es_error_transitorio_distingue_cuota_de_error_de_entrada() -> None:
    assert es_error_transitorio(RuntimeError("429 Too Many Requests"))
    assert es_error_transitorio(RuntimeError("503 UNAVAILABLE"))
    assert not es_error_transitorio(RuntimeError("400 INVALID_ARGUMENT"))


def test_con_reintentos_espera_de_forma_exponencial() -> None:
    esperas: list[float] = []
    intentos = iter([RuntimeError("429"), RuntimeError("429"), "ok"])

    def operacion() -> str:
        valor = next(intentos)
        if isinstance(valor, Exception):
            raise valor
        return valor

    resultado = con_reintentos(
        operacion, es_reintentable=es_error_transitorio, espera_base=1.0, dormir=esperas.append
    )
    assert resultado == "ok"
    assert esperas == [1.0, 2.0]


def test_crear_proveedor_gemini_sin_clave_falla() -> None:
    ajustes = Settings(_env_file=None, LLM_PROVIDER="gemini")
    with pytest.raises(ErrorProveedor):
        crear_proveedor(ajustes)


def test_crear_proveedor_ollama_por_defecto() -> None:
    ajustes = Settings(_env_file=None)
    proveedor = crear_proveedor(ajustes)
    assert proveedor.nombre == "ollama"


def test_proveedores_antiguos_no_son_validos() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, LLM_PROVIDER="anthropic")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, LLM_PROVIDER="openai")
