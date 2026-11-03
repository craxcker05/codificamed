# CodificaMed — instrucciones para el agente

## Objetivo
Prototipo que recibe una nota clínica en español (texto libre) y sugiere códigos
CIE-10 de diagnóstico, cada uno con la frase de la nota que lo justifica.
Debe distinguir diagnósticos confirmados de negados o descartados.
Proyecto académico (ESPOCH, Base de Conocimiento), metodología CRISP-ML(Q).

## Reglas no negociables
- Prohibido usar historias clínicas reales. Solo CodiEsp y notas ficticias.
- Nunca escribas claves de API en código, logs, tests ni commits. Solo en .env.
- El proveedor de LLM se cambia SOLO con LLM_PROVIDER en .env.
  Proveedores soportados: `ollama` (local, por defecto) y `gemini` (externo,
  API gratuita de Google AI Studio). No usar anthropic ni openai.
- Al proveedor externo (gemini) solo se le envían CodiEsp y notas ficticias.
  El tier gratuito puede usar los datos para mejorar productos de Google.
- Ninguna cifra en docs/ que no salga de un archivo en results/.
- Todo código sugerido debe existir en el catálogo y su evidencia debe ser
  texto literal de la nota.
- Trabaja solo la etapa que se te pide; al terminar, resume cambios y cómo verificarlos.

## Stack
Python 3.11, uv, pydantic, sentence-transformers (multilingual-e5-base),
chromadb o faiss, ollama, google-genai (Gemini), streamlit, scikit-learn, pytest.

## Convenciones
- Código y comentarios en español o inglés, pero UI y documentación en español.
- Funciones pequeñas con type hints; esquemas en src/codificamed/schemas.py.
- Temperatura 0 en todos los LLM; salidas en JSON validado con pydantic.
- Scripts largos deben ser reanudables y mostrar progreso.
- Tras cada etapa: pytest en verde y una entrada en docs/BITACORA.md con
  fase CRISP-ML(Q), qué se hizo, decisiones, resultados y pendientes.

## Comandos
- Tests: pytest -q
- Índice: python scripts/build_index.py
- Evaluar: python scripts/evaluate.py --provider ollama|gemini --n 50
- App: streamlit run app/streamlit_app.py