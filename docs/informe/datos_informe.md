# Datos para el informe ejecutivo

Todas las cifras salen de archivos de `results/` (se indica la fuente). Las evaluaciones de LLM son **parciales** por falta de tiempo (ver sección 5); decláralo en el informe.

## 1. Problema y objetivo
- Codificar diagnósticos CIE-10 es manual y costoso; un error grave es codificar como presente un diagnóstico negado.
- Objetivo: dada una nota en español, sugerir códigos CIE-10 con la frase literal que los justifica y separar confirmados de negados. Prototipo académico, solo CodiEsp y notas ficticias.
- Criterios de éxito (fijados antes de evaluar, `docs/BITACORA.md`): F1 de categoría ≥ 0,50 en 50 casos dev; 0 negados sugeridos como confirmados en 15 ficticias; ≤ 120 s por nota con qwen2.5:7b local.

## 2. Datos (`results/eda/resumen.json`, `results/cobertura_resumen.json`, `results/catalogo_resumen.json`)
- CodiEsp: train 500 textos, dev 250, test 3001 (sin anotaciones). Media ≈ 349 palabras (train).
- Pista D, train+dev: 2194 códigos distintos; media de 11,32 códigos por caso en train.
- Catálogo CIE-10: 71 486 códigos, 1910 categorías de 3 caracteres.
- Cobertura: 182 de 1767 códigos gold de train y 95 de 1158 de dev no existen en el catálogo (todos son códigos padre). En los 50 casos de evaluación, 13 (26 %) tienen algún código gold fuera del catálogo.
- 15 notas ficticias inventadas (9 con negación), validadas (`notas_ficticias_validacion.json`).

## 3. Fases CRISP-ML(Q) — qué se hizo
1. Entendimiento: problema, criterios, EDA, riesgos.
2. Preparación: normalización de códigos (0 colisiones), catálogo parquet, 50 casos de evaluación (semilla 42), 87 abreviaturas.
3. Modelado: preprocesado con offsets, negación basada en reglas, recuperación con embeddings, pipeline con LLM, proveedores Ollama y Gemini.
4. Evaluación: recuperación (completa) y pipeline (parcial).
5. Despliegue: app Streamlit mínima.
6. Monitoreo: feedback Aceptar/Rechazar, logs de latencia y errores, `scripts/monitor.py` (umbrales en la bitácora).

## 4. Arquitectura (para el diagrama)
Nota → preprocesado (abreviaturas + offsets) → LLM extrae menciones → reglas de negación (afirmado/negado/sospecha) → recuperación de 10 candidatos (multilingual-e5-base, búsqueda exacta con numpy) solo para las afirmadas → LLM elige código o ninguno → validación (código en catálogo, evidencia literal) → salida: sugerencias, negados aparte, y feedback del codificador en `logs/`.

## 5. Resultados medidos

### Negación por reglas (`results/negacion_tests.json`)
- 44 de 44 frases correctas; 27 de 27 menciones en las notas ficticias. Frases escritas junto con las reglas: no es una estimación de desempeño real.

### Recuperación (`results/recall_retrieval.json`, `results/recall_variantes.json`; 50 casos, 521 códigos gold, menciones oráculo)
| Criterio | @1 | @5 | @10 | @20 |
|---|---|---|---|---|
| Código exacto | 0,167 | 0,288 | 0,374 | 0,472 |
| Categoría (3 car.) | 0,370 | 0,557 | 0,653 | 0,749 |
- Ni el campo en inglés, ni combinarlo, ni expandir abreviaturas mejoraron el recall@10 en ≥ 0,01.

### Pipeline completo — evaluación PARCIAL (`results/metricas_*.json`, `results/comparacion.md`)
| Proveedor / modelo | Notas ficticias | Casos dev | F1 micro categoría (dev) | F1 micro código (dev) | s/nota (dev) | s/nota (ficticias) | Negados como confirmados |
|---|---|---|---|---|---|---|---|
| Gemini / gemini-3.5-flash-lite | 6 | 6 | 0,3478 | 0,2653 | 57,17 | 11,03 | 0 de 4 |
| Ollama / qwen2.5:7b-instruct | 5 | 0 (no evaluado) | — | — | — | 168,04 | 0 de 4 |

- Gemini, dev: precisión de categoría 0,6957; exhaustividad 0,2319; F1 macro de categoría 0,3273.
- Gemini, ficticias: F1 micro 0,375. Ollama, ficticias (5 notas): F1 micro 0,1.
- Ablación sin negación (Gemini, 6 ficticias): negados como confirmados 0 de 4 (igual que con negación); F1 micro 0,3529 sin negación frente a 0,375 con negación. Muestra demasiado pequeña para concluir.
- Cumplimiento de criterios: F1 de categoría (0,3478 en 6 casos) no alcanza 0,50; tiempo de Ollama (168 s en 5 notas) supera 120 s; negados: sin incumplimientos en lo evaluado.
- Costo: 0 USD (Ollama local y tier gratuito de Gemini).

## 6. Conclusiones sugeridas (reescribir con tu voz)
- La recuperación acota el desempeño: el recall de categoría @10 es 0,653 incluso con menciones perfectas.
- El pipeline separa negados y exige evidencia literal; en lo evaluado no sugirió negados como confirmados.
- La exhaustividad es baja; el desempeño con LLM aún no cumple el criterio de F1 y requiere evaluación completa.
- Gemini es mucho más rápido que el modelo local en estas pruebas (11 s frente a 168 s en notas ficticias).

## 7. Limitaciones (para el informe)
- Evaluación del pipeline con 6 casos dev y 5–6 notas ficticias, no 50 y 15; sin comparación Ollama vs Gemini sobre dev.
- Los mismos 50 casos de dev se usaron para elegir la variante de recuperación y serían los de la evaluación final.
- La negación no tiene referencia en CodiEsp; las frases de prueba se escribieron con las reglas.
- Abreviaturas y reglas sin validación clínica; techo de recuperación por códigos padre fuera del catálogo.
- Gemini gratuito puede usar los datos enviados: solo se enviaron notas ficticias y CodiEsp.
- Sin tokens registrados; sin datos de uso real de feedback (los umbrales de monitoreo son propuestos).

## 8. Datos de portada (completa tú)
Nombre, código, carrera, materia (Base de Conocimiento), fecha, docente.
