# Guion del video — CodificaMed

Duración sugerida: 5–6 minutos. Antes de grabar: `uv run streamlit run app/streamlit_app.py`, proveedor `gemini` (responde en unos 10–30 s; con `ollama` tarda cerca de 2–3 min por nota, medido: 168 s). Si recortas la espera en el montaje, dilo en voz: «aquí acorto la espera».

## 1. Presentación (0:00–0:30)
- Nombre, carrera, materia (Base de Conocimiento, ESPOCH).
- «CodificaMed sugiere códigos CIE-10 a partir de una nota clínica en español y cita la frase que lo justifica. Es un prototipo académico; solo usa notas ficticias y CodiEsp.»

## 2. Problema (0:30–1:15)
- La codificación CIE-10 es manual y lenta.
- Riesgo: codificar como presente algo que la nota niega («niega fiebre»).
- Objetivo: códigos + evidencia literal + distinguir confirmados de negados.

## 3. Arquitectura (1:15–2:30) — mostrar el diagrama o `pipeline.py`
1. Preprocesado: normalización y expansión de abreviaturas con mapa de offsets.
2. El LLM extrae menciones (se descartan las que no son literales de la nota).
3. Reglas de negación: afirmado / negado / sospecha.
4. Recuperación semántica (multilingual-e5-base) de 10 candidatos del catálogo CIE-10 (71 486 códigos).
5. El LLM elige un código entre los candidatos o ninguno.
6. Validación: el código existe en el catálogo y la evidencia es texto literal.
- Cambio de proveedor solo con `LLM_PROVIDER` en `.env` (Ollama local o Gemini).

## 4. Demo en la app (2:30–4:15)
1. Cargar la nota ficticia FIC-01 con «Cargar ejemplo» y pulsar «Codificar».
2. Mientras procesa: «el sistema hace varias llamadas al LLM; la app muestra el tiempo real medido».
3. Señalar: evidencia en verde (cefalea occipital → R51) y «dolor torácico» tachado en gris, listado como negado, sin código.
4. Pulsar «Aceptar» en un código y «Rechazar» en otro; mostrar `logs/feedback.csv` (sin texto de la nota).
5. Opcional: ejecutar `python scripts/monitor.py`.
- Comentar con honestidad que el modelo puede omitir diagnósticos (p. ej. HTA/DM2 en algunas ejecuciones).

## 5. Resultados (4:15–5:15) — cifras de `results/`, ver `docs/informe/datos_informe.md`
- La evaluación fue parcial por falta de tiempo: 6 casos de dev y 6 notas ficticias con Gemini; 5 ficticias con Ollama.
- Gemini, dev (6 casos): F1 de categoría 0.348 (precisión 0.696, exhaustividad 0.232). No alcanza el criterio de 0.50.
- Ollama: 168 s por nota (criterio ≤ 120 s, no cumplido); Gemini: 11 s por nota ficticia.
- Negados sugeridos como confirmados: 0 de 4 en las notas evaluadas.
- Recuperación (50 casos, menciones oráculo): recall de categoría @10 = 0.653.

## 6. Limitaciones y cierre (5:15–6:00)
- Evaluación pequeña; sin validación clínica de abreviaturas ni de negación; el catálogo no contiene algunos códigos padre del gold.
- Trabajo futuro: evaluar con los 50 casos, mejorar la extracción de menciones, recuperación híbrida.
- Cierre: «Gracias; el código y la bitácora están en el repositorio».
