# Bitácora CodificaMed

Registro por fase CRISP-ML(Q). Cada cifra citada aquí sale de un archivo en `results/`.

---

## Fase 1 — Entendimiento del negocio y de los datos

**Fecha:** 2026-11-02
**Alcance de esta entrada:** problema de negocio, objetivo, criterios de éxito propuestos, riesgos y hallazgos del EDA de CodiEsp (subtarea D). No se ha entrenado ni evaluado ningún modelo.

### 1. Problema de negocio

La codificación de diagnósticos con CIE-10 es un trabajo manual que consume tiempo del personal clínico y de documentación. Un asistente que, dada una nota clínica en español, sugiera códigos CIE-10 de diagnóstico **con la frase literal de la nota que los justifica** puede acelerar esa tarea y facilitar la revisión humana.

Un error frecuente y costoso es codificar como presente un diagnóstico que la nota **niega o descarta** ("niega fiebre", "descarta neumonía"). El sistema debe distinguir diagnósticos confirmados de negados o descartados.

Es un prototipo académico (ESPOCH, Base de Conocimiento). **No es una herramienta clínica** y no se usa con historias clínicas reales: solo se emplean CodiEsp y notas ficticias.

### 2. Objetivo

Construir un prototipo que, a partir de texto libre en español, proponga códigos CIE-10 de diagnóstico, cada uno con:

- la **evidencia**: fragmento de la nota, copiado literalmente;
- el **estado**: confirmado, negado o descartado.

Todo código sugerido debe existir en el catálogo CIE-10, y toda evidencia debe ser texto literal de la nota.

### 3. Criterios de éxito propuestos

Los umbrales numéricos **no se fijan todavía**: se decidirán con la línea base de la Fase 4 (evaluación), para no inventar cifras antes de medir.

| Criterio | Métrica propuesta | Tipo |
|---|---|---|
| Códigos correctos | Precisión, exhaustividad y F1 a nivel de código por caso, sobre el split *dev* de CodiEsp-D | Cuantitativo, umbral pendiente |
| Evidencia fiel | Proporción de evidencias que son texto literal de la nota | Requisito duro: debe ser 100 % |
| Código válido | Proporción de códigos sugeridos que existen en el catálogo | Requisito duro: debe ser 100 % |
| Negados excluidos | Proporción de diagnósticos negados que el sistema reporta como presentes | Cuantitativo, umbral pendiente |
| Comparación de proveedores | Diferencia de F1 entre LLM local (Ollama) y API, con el mismo protocolo | Descriptivo |

## Criterios de éxito (fijados el 2026-11-02, antes de evaluar)
- F1 categoría CIE-10 (3 caracteres) ≥ 0,50 en 50 casos de dev/ (semilla 42)
- 0 diagnósticos negados sugeridos como confirmados en las 15 notas ficticias
- Tiempo medio por nota ≤ 120 s con qwen2.5:7b-instruct en local

### 4. Datos y decisiones de esta fase

- **Corpus:** CodiEsp (`data/raw/final_dataset_v2_to_publish/`), no versionado en git (`.gitignore`).
- **Particiones:** `train` y `dev` tienen anotaciones de las pistas D (diagnósticos), P (procedimientos) y X (menciones con texto de referencia). La carpeta `test` **no tiene anotaciones** y contiene los textos del test junto con los del background, sin separación por archivo.
- **Decisión:** el desarrollo y la evaluación con etiquetas se hacen sobre `train` y `dev`. `test` queda reservado para la evaluación final y no se usa para ajustar nada.
- **Decisión de alcance:** se trabaja solo la pista D. La pista P se describe en el EDA, pero no entra en el prototipo.
- **Entorno:** Python 3.12 instalado (el proyecto exige 3.11 o superior y menor que 3.13). Dependencias gestionadas con `uv`. Las dependencias pesadas (torch vía sentence-transformers, chromadb, LLM, streamlit) son extras opcionales (`ml`, `llm`, `app`), para que el EDA se ejecute sin instalar torch.

### 5. Hallazgos del EDA

Fuentes: `results/eda/resumen.json` y las tablas CSV de `results/eda/`. Generado con `python scripts/eda.py`.

**5.1 Casos por split** (`casos_por_split.csv`)

- `train`: 500 textos; 5661 filas D, 1550 filas P y 9211 filas X.
- `dev`: 250 textos; 2683 filas D, 817 filas P y 4485 filas X.
- `test` (carpeta): 3001 textos, sin anotaciones.
- Todos los casos de `train` y `dev` tienen al menos un código D. Casos con P: 435 de 500 en `train` y 222 de 250 en `dev` (`codigos_por_caso.csv`).

**5.2 Longitud de los textos** (`longitud_textos.csv`, `longitud_media_palabras.png`)

- `train`: media de 349.0 palabras, mediana de 314.5, rango de 73 a 1172.
- `dev`: media de 352.3 palabras, mediana de 327.5, rango de 69 a 956.
- `test` (carpeta): media de 374.1 palabras, mediana de 315.0, rango de 20 a 4901. El máximo es un caso atípico muy largo.

Implicación: las notas son lo bastante largas para que el contexto de un LLM local sea un factor a vigilar.

**5.3 Códigos por caso** (`codigos_por_caso.csv`, `codigos_por_caso_D.png`)

- Pista D, `train`: media de 11.32 códigos distintos por caso, mediana de 10, rango de 1 a 41.
- Pista D, `dev`: media de 10.73, mediana de 10, rango de 1 a 33.
- Ningún caso de `train` o `dev` carece de códigos D.
- Pista P: en `train`, 65 de 500 casos no tienen ningún código; en `dev`, 28 de 250.

Implicación: el prototipo debe sugerir muchos códigos por nota, no uno o dos.

**5.4 Top 30 códigos D** (`top30_codigos_D.csv`, `top30_codigos_D.png`)

- Los códigos más frecuentes en `train` + `dev` son R52 (163 casos), R69 (150), R50.9 (142), I10 (116) y R59.9 (95).
- 15 de los 30 códigos del top pertenecen al capítulo R (síntomas y hallazgos clínicos y de laboratorio). Es decir, una parte grande de la señal está en signos y síntomas, no en enfermedades definidas.
- En `train` + `dev` hay 2194 códigos D distintos (`resumen.json`).

Implicación: la cola larga de códigos poco frecuentes será el principal reto para la recuperación desde el catálogo.

**5.5 Formato de los códigos**

- Las 8344 filas D de `train` + `dev` (5661 + 2683, `casos_por_split.csv`) están en minúsculas (`codigos_D_con_minusculas` en `resumen.json`).
- Decisión: normalizar a mayúsculas antes de comparar con el catálogo.

**5.6 Etiquetas de la pista X** (`labels_X.csv`)

- Solo aparecen dos etiquetas: DIAGNOSTICO y PROCEDIMIENTO.
- `train`: 7239 DIAGNOSTICO y 1972 PROCEDIMIENTO. `dev`: 3439 DIAGNOSTICO y 1046 PROCEDIMIENTO.

**5.7 Calidad de los offsets** (`resumen.json`, bloque `x_diagnosticos_train_dev`)

- Menciones DIAGNOSTICO en `train` + `dev`: 10678. De ellas, 1556 son discontinuas (varios tramos de texto).
- El texto de referencia coincide con los tramos indicados en 10675 menciones. Las 3 restantes son casos límite: la referencia incluye un signo de puntuación o "%" que no aparece en el tramo indicado.
- Hallazgo metodológico: al verificar offsets hay que unir los tramos discontinuos. Comprobar solo el primer tramo da una cifra engañosa (en una primera versión del script así se midió y se corrigió).

**5.8 Abreviaturas comunes** (`abreviaturas.csv`, `abreviaturas.png`)

Las más frecuentes en todo el corpus (train + dev + test):

- TAC: 1213 ocurrencias.
- PCR: 506.
- VIH: 331.
- UCI: 253.
- RMN: 214.
- ECG: 213.
- VHC: 160.
- HTA: 95.
- TA: 82. Puede tener más de un significado en clínica; requiere revisión manual antes de expandirla.

Implicación: el preprocesado debe expandir abreviaturas, y algunas (TA, FR, FC) necesitan contexto para no confundirse.

**5.9 Expresiones de negación** (`negacion.csv`, `negacion.png`)

Ocurrencias en todo el corpus:

- "no": 7773.
- "sin": 7078.
- "ni": 2177.
- "negativo": 725.
- "ausencia de": 646.
- "negativa": 530.
- "no presenta": 160.
- "niega": 59.
- "descarta": 73.

Hallazgo sobre la pista X: en las 10678 menciones DIAGNOSTICO de `train` + `dev`, 1291 tienen alguna expresión de negación en los 60 caracteres previos (`con_negacion_previa`).

**Limitaciones de este hallazgo:**

- "no", "sin" y "ni" son ambiguos ("sin embargo", "ni" como conjunción). Los conteos son una aproximación, no una medida de negación.
- La ventana de 60 caracteres no respeta límites de oración.
- Los campos del TSV X (según el README del corpus) no incluyen un indicador de negación. **No hay etiqueta de referencia para medir la detección de negados con CodiEsp.** Esta es la principal limitación para cumplir el objetivo de distinguir negados.

### 6. Riesgos

| Riesgo | Impacto | Mitigación propuesta |
|---|---|---|
| CodiEsp no etiqueta negación: no hay referencia directa para medir la exclusión de negados | Alto | Construir un subconjunto de evaluación propio con notas ficticias y negaciones controladas; documentar la limitación en el informe |
| Cola larga de códigos (2194 códigos D distintos, muchos poco frecuentes) | Alto | Recuperación semántica desde el catálogo con embeddings; medir la cobertura del catálogo en la Fase 2 |
| Versión del catálogo CIE-10 usada por CodiEsp puede no coincidir con la del catálogo del proyecto | Medio | Verificar en la Fase 2 cuántos códigos de `train` + `dev` existen en el catálogo elegido |
| Formato de los códigos en minúsculas y con/sin punto | Medio | Normalizar (mayúsculas) y comparar con una tabla de equivalencias; ya identificado en 5.5 |
| Abreviaturas ambiguas (TA, FR, FC) | Medio | Expandir con diccionario revisado; no expandir sin contexto |
| Notas largas (media de 349 a 374 palabras, con un caso de 4901) | Medio | Trocear por oraciones; limitar el contexto que recibe el LLM local |
| Uso indebido con historias clínicas reales | Alto (ético y legal) | Regla del proyecto: solo CodiEsp y notas ficticias; claves solo en `.env`, nunca en el código |
| Dependencia de APIs externas con costo y privacidad | Medio | Proveedor local (Ollama) por defecto; el cambio de proveedor solo por `LLM_PROVIDER` |

### 7. Pendientes para la Fase 2 (preparación de datos)

- Verificar la cobertura del catálogo CIE-10 frente a los códigos D de `train` y `dev`.
- Implementar la normalización de códigos y la segmentación en oraciones.
- Construir `data/ficticios/` con notas inventadas que incluyan diagnósticos negados.
- Decidir el umbral de los criterios de éxito tras la línea base.

### 8. Verificación de esta etapa

- `uv run pytest -q`: 10 pruebas en verde (módulo `corpus` y configuración).
- `uv run python scripts/eda.py`: regenera `results/eda/` completo.

---

## Preparación previa a la Fase 2 — proveedores de LLM

**Fecha:** 2026-11-02
**Fase CRISP-ML(Q):** Entendimiento (ajuste de entorno; no es una fase de modelado)

### Qué se hizo

- Proveedores soportados: `ollama` (local, por defecto) y `gemini` (externo, API gratuita de Google AI Studio). Se eliminaron anthropic y openai de la configuración, de `.env.example`, de `pyproject.toml` y de CLAUDE.md.
- `config/settings.py`: `LLM_PROVIDER` acepta solo `ollama` o `gemini`. Las claves se leen como `SecretStr`.
- `src/codificamed/llm/`:
  - `base.py`: interfaz `LLMProvider`, error `ErrorProveedor` y reintentos con espera exponencial (`con_reintentos`).
  - `ollama_provider.py`: temperatura 0, mensaje de sistema opcional.
  - `gemini_provider.py`: SDK `google-genai`; reintenta ante cuota (429, RESOURCE_EXHAUSTED) y 503; los errores no incluyen la clave.
  - `__init__.py`: fábrica `crear_proveedor()` que elige según `LLM_PROVIDER`.
- `tests/test_llm.py`: pruebas con clientes falsos, sin red.

### Decisiones

- Al proveedor externo (gemini) solo se envían CodiEsp y notas ficticias. El tier gratuito puede usar los datos para mejorar productos de Google (registrado en CLAUDE.md).
- Ollama es el proveedor por defecto y el de referencia; Gemini sirve para comparar.
- Modelo Gemini por defecto en código: `gemini-2.5-flash`. El `.env` local usa `gemini-3.5-flash-lite`, que fue el que respondió en la prueba. Se configura en `GEMINI_MODEL`.
- La temperatura 0 es fija en código (`TEMPERATURA_LLM`), no configurable.

### Verificación

- `uv run pytest -q`: 21 pruebas en verde.
- Prueba real mínima con Gemini (prompt sin datos clínicos: "Responde solo con la palabra: ok"): respondió `ok`.
- Prueba real con Ollama (`qwen2.5:7b-instruct`, mismo prompt): respondió `ok`. Al principio fallaba porque la variable de entorno del sistema `OLLAMA_HOST=0.0.0.0` (usada para que el servidor escuche en todas las interfaces) tiene prioridad sobre `.env` en pydantic-settings y el cliente no puede conectarse a esa dirección. Para la prueba se sobreescribió con `OLLAMA_HOST=http://localhost:11434` solo en el proceso.

### Pendientes

- Decidir cómo resolver el conflicto de `OLLAMA_HOST`: cambiar la variable del sistema a `http://localhost:11434` o renombrar la variable del cliente (p. ej. `CODIFICAMED_OLLAMA_HOST`). Hasta entonces, las ejecuciones desde el proyecto necesitan la sobreescritura.
- Confirmar que la clave de Gemini no se haya compartido fuera de `.env`. Si se compartió en algún chat, rotarla en Google AI Studio.
- Evaluación con reintentos y reanudación: `scripts/evaluate.py` (Fase 4).

---

## Fase 2 — Preparación de datos

**Fecha:** 2026-11-02
**Fase CRISP-ML(Q):** Preparación de los datos (Data Preparation)
**Fuentes de cifras:** `results/catalogo_resumen.json`, `results/cobertura_resumen.json`, `results/cobertura_catalogo.csv`, `results/notas_ficticias_validacion.json`.

### 1. Comprobaciones previas del catálogo fuente

Archivo: `data/processed/catalogo_fuente.tsv` (`results/catalogo_resumen.json`).

- **Filas:** 71486. Sin duplicados exactos y sin códigos repetidos. Sin códigos vacíos y sin descripciones en español vacías. Descripciones en inglés vacías: 109.
- **Cabecera: no existe.** El archivo no tiene la línea de nombres de columna que indicaba la especificación; la primera fila ya es un dato (`A00.0`). Se leyó sin cabecera y se asignaron las columnas `codigo`, `descripcion_es`, `descripcion_en`. Las 71486 líneas tienen exactamente 3 campos.
- **Códigos con punto:** 71259 de 71486. Ejemplos: `A00.0`, `A00.1`, `A00.9`, `A01.00`, `A01.01`.
- **Codificación:** UTF-8 válido, sin BOM.

### 2. Códigos gold de CodiEsp-D (`results/cobertura_resumen.json`)

| Split | Códigos distintos | Con punto | Sin punto |
|---|---|---|---|
| train | 1767 | 1688 | 79 |
| dev | 1158 | 1110 | 48 |

Los códigos gold están en minúsculas en el TSV. Se normalizan a mayúsculas y sin punto.

### 3. Normalización: transformación, reversibilidad y colisiones

- **Transformación:** quitar el punto y pasar a mayúsculas (`A01.00` → `A0100`, `n44.8` → `N448`).
- **Colisiones:** 0. Ningún par de códigos originales distintos queda igual tras normalizar.
- **Reversibilidad:** la transformación **no es reversible por sí sola** (`A0100` podría venir de `A01.00` o de `A010.0`). Se garantiza con la columna `codigo_original` del catálogo: se comprobó que normalizar cada `codigo_original` reproduce exactamente su `codigo` en las 71486 filas.
- **Formato:** los 71486 códigos normalizados cumplen el patrón esperado (letra inicial, 3 o más caracteres alfanuméricos). Categorías distintas (3 primeros caracteres): 1910.
- **Catálogo generado:** `data/processed/catalogo_cie10.parquet`, columnas `codigo`, `codigo_original`, `descripcion_es`, `descripcion_en`, `categoria`.

### 4. Cobertura del catálogo frente a CodiEsp-D

Detalle por código en `results/cobertura_catalogo.csv`; resumen en `results/cobertura_resumen.json`.

| Split | Códigos gold distintos | Fuera del catálogo | % códigos | Casos | Casos con algún código fuera | % casos |
|---|---|---|---|---|---|---|
| train | 1767 | 182 | 10.3 % | 500 | 123 | 24.6 % |
| dev | 1158 | 95 | 8.2 % | 250 | 56 | 22.4 % |

**Hallazgo sobre los códigos que faltan:** en train y dev, **todos** los códigos ausentes (182 y 95) son códigos padre: el catálogo no los contiene, pero sí contiene subcódigos que empiezan igual. Ejemplo: el gold `C349` no existe en el catálogo, que solo tiene subcódigos como `C34.90` o `C34.91`. El problema es de **granularidad** (el gold usa un código menos específico), no de códigos inventados ni de una categoría ausente.

**Limitación:** de los 1767 códigos gold distintos de train y 1158 de dev, respectivamente 182 y 95 no se pueden emitir tal cual desde este catálogo. Esto fija un techo de recuperación para el sistema. No se han inventado ni descartado en silencio: quedan en el detalle y en el resumen.

### 5. Casos de evaluación (`data/processed/casos_eval.jsonl`)

- 50 casos seleccionados de **dev** con semilla 42, con `{id, texto, codigos_gold}`. `codigos_gold` contiene solo códigos de diagnóstico (pista D), normalizados y sin repetir.
- Sin duplicados. Todos los ids existen en `dev/text_files`. Ningún caso de `test/`.
- **13 de 50 casos (26.0 %)** tienen al menos un código gold fuera del catálogo (`results/cobertura_resumen.json`, bloque `casos_eval`). Para estos casos, ningún sistema que emita solo códigos del catálogo puede acertar todo el conjunto.
- `codigos_gold` conserva los códigos fuera del catálogo: no se eliminan, para que la métrica refleje el techo real.

### 6. Configuración y notas ficticias

- **`config/abreviaturas.yaml`:** 87 abreviaturas con su expansión (mínimo exigido: 60). Incluye vocabulario de historias ecuatorianas (`pcte`, `APP`, `EGO`, `BH`, `IESS`, `tto`, `s/`, `c/`). **Pendiente:** revisión por un clínico antes de usarlas en producción; `TA`, `FC`, `FR`, `IC` tienen más de un significado.
- **`data/ficticios/notas_ficticias.jsonl`:** 15 notas **inventadas**, con abreviaturas, errores de tipeo y negaciones. Cada nota tiene `id`, `texto`, `codigos_esperados` (sin punto), `frase_justificacion` (un fragmento literal por código) y `negados`.
- **Notas con negación:** 9 de 15 (`results/notas_ficticias_validacion.json`).
- **Validación:** 0 errores. Todos los códigos esperados y negados existen en el catálogo, y todas las justificaciones son texto literal de la nota.
- **Campo añadido:** `negados` (no estaba en la especificación). Se usa para evaluar la exclusión de diagnósticos negados. Cada código negado debe ser texto literal y no puede aparecer también como presente.
- **Corrección durante la validación:** la primera versión de FIC-07 usaba `I26.9`, que no existe en el catálogo (solo hay subcódigos). Se cambió por `I26.99` ("Otra embolia pulmonar sin cor pulmonale agudo"), que sí existe.

### 7. Decisiones

- **Ruta del corpus:** la especificación indicaba `data/raw/codiesp/`. La ruta real del proyecto es `data/raw/final_dataset_v2_to_publish/`, que es la que usan `config/settings.py` y `scripts/eda.py`. Se usó esa ruta.
- **Dependencia añadida:** `pyarrow`, necesaria para escribir el parquet.
- **Uso de datos:** `train/` solo para construir el catálogo y las estadísticas de cobertura. La evaluación usa `dev/`. `test/` no se usó.
- **Catálogo fuente sin modificar:** `data/processed/catalogo_fuente.tsv` no se tocó.
- **Sin mapeo automático de padre a hijo:** no se reemplazó ningún código gold por un subcódigo. Eso sería una decisión de evaluación, no de preparación.

### 8. Limitaciones

- Techo de recuperación: 182 códigos de train y 95 de dev no son emitibles como aparecen en el gold.
- Evaluación pequeña: 50 casos de dev, y 13 de ellos con códigos fuera del catálogo.
- Notas ficticias: 15 notas, 9 con negación. Sirven para validar el pipeline, no para estimar métricas.
- Abreviaturas sin validación clínica.

### 9. Pendientes

- **Fase 3 (modelado):** definir la regla de evaluación para códigos padre: exigir código exacto, o aceptar la categoría como acierto. Debe fijarse antes de evaluar.
- Decidir si las métricas se reportan sobre todos los códigos gold o solo sobre los emitibles (ambas, idealmente).
- Revisión clínica de `config/abreviaturas.yaml`.
- Confirmar la fuente oficial del catálogo fuente y su versión de CIE-10.

### 10. Verificación

- `uv run python scripts/prepare_data.py`: termina con código 0 y regenera todos los archivos de esta fase.
- `uv run pytest -q`: 38 pruebas en verde (normalización, colisiones, reversibilidad, unicidad del catálogo, 50 casos sin duplicados, solo dev, abreviaturas ≥ 60, notas ficticias ≥ 5 con negación, códigos de las notas en el catálogo).

---

## Fase 3 (parte 1) — Preprocesado y negación

**Fecha:** 2026-11-02
**Fase CRISP-ML(Q):** Modelado (componentes basados en reglas, previos al LLM)
**Fuentes de cifras:** `results/negacion_tests.json` (generado con `python scripts/eval_negacion.py`).

### Qué se hizo

- `src/codificamed/preprocess.py`:
  - Normalización que conserva la longitud (espacios raros, comillas, guiones).
  - Expansión de las abreviaturas de `config/abreviaturas.yaml` con mapa de offsets: cada carácter del texto procesado conserva el rango del original, y `TextoProcesado.evidencia(ini, fin)` devuelve texto literal de la nota.
  - Abreviaturas ambiguas: `TA`, `FC`, `FR`, `PA`, `T°` solo se expanden si les sigue un número; `d`, `h`, `a.` solo si les precede un número; `IC` y `CC` nunca. Abreviaturas cortas o riesgosas (`ECO`, `ASA`, `EDA`, `URG`, `BUN` y las de menos de 3 letras) exigen la capitalización exacta.
  - Segmentación en oraciones (no corta en abreviaturas, decimales, iniciales ni numeración; un salto de línea simple de maquetación no corta).
- `src/codificamed/negation.py`: reglas tipo NegEx/ConText con disparadores previos, posteriores y bidireccionales, ventana de alcance en palabras, terminadores (pero, sin embargo, verbos afirmativos...), pseudo-negaciones y desempate por el disparador más cercano. Estados: `afirmado`, `negado`, `sospecha`.
- Tests: `tests/test_preprocess.py`, `tests/test_negation.py`, frases en `tests/casos_negacion.py` (44 frases, incluidas las 5 exigidas) y comprobación extremo a extremo sobre las notas ficticias.

### Resultados (`results/negacion_tests.json`)

- Frases de prueba: 44 de 44 correctas; clase NEGADO: precisión 1.0 y exhaustividad 1.0 (22 verdaderos positivos).
- Notas ficticias (27 menciones: justificaciones esperadas como afirmadas y diagnósticos de `negados` como negados): 27 de 27.

**Esta cifra no es una estimación de desempeño real.** Las frases se escribieron junto con las reglas, y las 15 notas ficticias también eran conocidas al afinar. Antes de corregir, la comprobación sobre las notas ficticias dio 25 de 27; los dos fallos eran reales y se corrigieron: un "probable" posterior que cruzaba una coma y calificaba a otro diagnóstico (FIC-02), y la ausencia del disparador "tampoco" (FIC-13). Esos dos casos se añadieron a las frases de prueba. Sobre notas CodiEsp no hay referencia de negación (ver Fase 1, 5.9), así que el desempeño real sigue sin medirse.

### Decisiones

- La negación opera sobre el texto procesado; la evidencia final se recupera siempre con `span_original` para que sea literal.
- `sospecha` ("probable", "compatible con", "para descartar", "no se descarta") se separa de `negado`: el sistema no debe sugerirlo como confirmado.
- El estado `Estado` vive en `negation.py`; aún no existe `schemas.py`, y se moverá allí al definir los esquemas de salida.

### Limitaciones y pendientes

- Sin etiquetas de referencia reales: construir un conjunto de negación anotado a mano (p. ej. con menciones X de CodiEsp) para medir de verdad.
- Reglas conocidas débiles: "no" genérico con ventana de 6 palabras puede negar de más ("no mejora con tratamiento, neumonía"); no hay doble negación ni negación con "sin" + "descartar".
- La normalización no convierte texto Unicode descompuesto (NFD) para no alterar los offsets.
- Revisión clínica de las abreviaturas y de las listas de disparadores.

### Verificación

- `uv run pytest -q`: todas las pruebas en verde.
- `uv run python scripts/eval_negacion.py`: regenera `results/negacion_tests.json`.

---

## Fase 3 (parte 2) — Recuperación (RAG) sobre el catálogo

**Fecha:** 2026-11-02
**Fase CRISP-ML(Q):** Modelado (recuperación de candidatos)
**Fuentes de cifras:** `results/recall_retrieval.json` y `results/recall_retrieval_detalle_*.csv` (generados con `python scripts/eval_retrieval.py`).

### Qué se hizo

- `src/codificamed/retrieval.py`: embeddings con `intfloat/multilingual-e5-base` (prefijos `query: ` y `passage: `), búsqueda exacta por coseno y `buscar(mencion, k=10)` que devuelve `Candidato(codigo, codigo_original, descripcion, score)`. Todo candidato sale del catálogo. El índice guarda una huella del catálogo y se rechaza si no coincide.
- `scripts/build_index.py`: construye `data/index/embeddings.npy` (71486 × 768, float16) y `meta.json`. Reanudable por partes, con barra de progreso. En CPU tardó unos 45 minutos.
- `scripts/eval_retrieval.py`: mide el recall de la recuperación.
- Tests con codificador falso (`tests/test_retrieval.py`), sin descargar el modelo.

### Decisiones

- **Sin chromadb:** con 71486 filas la búsqueda exacta con numpy es suficiente y determinista. Se quitó `chromadb` del extra `ml` de `pyproject.toml`. `uv.lock` no se regeneró.
- **Texto indexado:** solo `descripcion_es`.
- **Menciones de la evaluación:** son las menciones DIAGNOSTICO de la pista X (`devX.tsv`) de los 50 casos (589 menciones). Son menciones "oráculo": miden la recuperación aislada de la extracción. Con menciones extraídas automáticamente el recall será igual o menor.

### Resultados (`results/recall_retrieval.json`, 521 códigos gold en 50 casos)

Pregunta medida: ¿el código gold aparece entre los k candidatos de alguna mención del caso?

| Criterio | recall@1 | @5 | @10 | @20 |
|---|---|---|---|---|
| Código exacto, todos los gold (sin expansión) | 0.167 | 0.288 | 0.374 | 0.472 |
| Categoría (3 caracteres), todos los gold (sin expansión) | 0.370 | 0.557 | 0.653 | 0.749 |
| Código exacto, solo gold emitibles (490) | 0.178 | 0.306 | 0.398 | 0.502 |

- Expandir abreviaturas en la consulta casi no cambia nada: categoría @10 pasa de 0.6526 a 0.6545 y el exacto @10 queda igual (0.3743).
- A nivel de mención (el código de la mención entre sus propios 10 candidatos): exacto 0.338 y categoría 0.569 sobre 589 menciones.
- Cota superior de las menciones: los 521 códigos gold tienen al menos una mención X con ese mismo código, así que el límite del recall con menciones oráculo es 100 %; el recall observado no se explica por menciones que falten.

### Lectura de los fallos (revisión de 14 menciones sin acierto)

- Menciones vagas cuyo código depende del contexto: "enfermedad" → R69, "masa renal" / "riñón masa" → N28.89, "adenopatías" → R59.9.
- Descripciones del catálogo muy genéricas o con otro nivel de especificidad: "infarto agudo de miocardio" → gold I21.3, el recuperador pone antes I25.2 y otros.
- Mención más larga que la descripción: "tumoración en la cara lateral derecha de la lengua" → D49.0.
- El modelo confunde anatomía parecida ("masa tiroideo" devolvió "Artritis tifoidea").

### Implicaciones para el criterio de éxito (F1 de categoría ≥ 0,50)

El recall de categoría @10 con menciones oráculo es 0.653. El umbral de F1 de categoría no se cambia, pero la recuperación por sí sola deja un margen estrecho: el LLM debe elegir bien entre candidatos y el techo de recuperación ya está por debajo de 0.7. Mejoras a evaluar, una a la vez y midiendo en `results/`: recuperación híbrida (léxica + semántica), recuperar con el contexto de la oración además de la mención, reranking con el LLM y k mayor (el recall de categoría sube a 0.749 con k=20).

### Limitaciones

- 50 casos de dev (semilla 42). Es la misma muestra que usará la evaluación final: no ajustar el recuperador mirando estos fallos de forma que sobreajuste.
- Las menciones oráculo y las del sistema real serán distintas.
- El modelo se descargó sin token de Hugging Face (aviso de límites de tasa, sin consecuencias).

### Verificación

- `uv run pytest -q`: en verde.
- `uv run python scripts/build_index.py`: construye o reanuda el índice.
- `uv run python scripts/eval_retrieval.py`: regenera `results/recall_retrieval*.`


---

## Fase 4, iteración 2 — Variantes de recuperación y cache de embeddings

**Fecha:** 2026-11-03
**Fase CRISP-ML(Q):** Evaluación (iteración sobre la recuperación)
**Fuentes de cifras:** `results/recall_variantes.json` (generado con `python scripts/experiment_recall.py`) y los metadatos `data/index/cache/*.json`. La línea base `results/recall_retrieval.json` no se modificó.

### Diagnóstico del índice anterior (paso 0)

- 71486 descripciones codificadas en CPU (`torch.cuda.is_available()` = False, 10 hilos de torch), lote 64, partes de 2048. Tiempo total: 2086 s (34,8 min). Había además una copia intermedia en `data/index/partes/`.

### Cambios

- `retrieval.py`: cache en `data/index/cache/{campo}_{modelo_slug}.npy` y `.json` (ids del catálogo, huella, modelo, fecha, tiempo, dispositivo). `get_or_build_embeddings(campo, modelo)` carga el cache si la huella y el modelo coinciden y solo calcula si falta. Las consultas (menciones) se cachean aparte, de forma incremental por texto. La búsqueda es producto punto en numpy. Modos `es`, `en` y `combinado` (máximo de los dos puntajes).
- `build_index.py`: calcula solo lo que falta; `--force` recalcula. El índice de la etapa 4 se **trasladó** al cache del campo `es` sin recalcular (la huella coincidía); el archivo y la carpeta de partes antiguos se eliminaron.
- `eval_retrieval.py`: la lógica de métricas pasó a una función reutilizable. Ahora se niega a reescribir `results/recall_retrieval.json` sin `--sobrescribir`. El formato de salida del nivel mención cambió (por k), pero el archivo de la línea base no se tocó.
- Tests: 67 en verde (cache, huella, `--force`, vacíos, modo combinado, consultas incrementales).

### Tiempos de construcción de cada cache

| Cache | Tiempo | Nota |
|---|---|---|
| `es` | 2086 s (34,8 min) | Construido en la etapa 4 y trasladado sin recalcular; la cifra viene del log de aquella ejecución. |
| `en` | 2138,7 s (35,6 min) | Calculado una vez en esta iteración, CPU, lote 64. 109 descripciones en inglés vacías: vector cero (puntaje 0). |
| Menciones | 69,9 s (488 menciones únicas sin expansión) y 4,4 s (con expansión, la mayoría ya en cache) | `data/index/cache/menciones_*.npy` |

Búsqueda por variante (una pasada para las 488 menciones únicas): entre 0,49 s y 2,8 s según el modo (campo `combinado` con expansión: 2,8 s).

### Resultados de las variantes (50 casos de dev, 521 códigos gold, 589 menciones oráculo)

El "exacto@10" y el "categoría@10" son siempre el criterio de selección; "@k" es la profundidad propia de la fila.

| Campo | Abreviaturas | k | exacto@10 | categoría@10 | exacto@k | categoría@k | exacto@10 solo emitibles | mención exacto@k | mención categoría@k |
|---|---|---|---|---|---|---|---|---|---|
| es | sin expansión | 10 | 0,3743 | 0,6526 | 0,3743 | 0,6526 | 0,3980 | 0,3379 | 0,5688 |
| es | sin expansión | 20 | 0,3743 | 0,6526 | 0,4722 | 0,7486 | 0,3980 | 0,4261 | 0,6570 |
| es | sin expansión | 30 | 0,3743 | 0,6526 | 0,5298 | 0,7965 | 0,3980 | 0,4822 | 0,7267 |
| en | sin expansión | 10 | 0,3244 | 0,6065 | 0,3244 | 0,6065 | 0,3449 | 0,2988 | 0,5229 |
| en | sin expansión | 20 | 0,3244 | 0,6065 | 0,4549 | 0,7486 | 0,3449 | 0,4126 | 0,6655 |
| combinado | sin expansión | 10 | 0,3800 | 0,6622 | 0,3800 | 0,6622 | 0,4041 | 0,3447 | 0,5671 |
| combinado | sin expansión | 20 | 0,3800 | 0,6622 | 0,4933 | 0,7697 | 0,4041 | 0,4414 | 0,6723 |
| es | con expansión | 10 | 0,3743 | 0,6545 | 0,3743 | 0,6545 | 0,3980 | 0,3379 | 0,5688 |
| es | con expansión | 20 | 0,3743 | 0,6545 | 0,4722 | 0,7505 | 0,3980 | 0,4261 | 0,6570 |
| es | con expansión | 30 | 0,3743 | 0,6545 | 0,5298 | 0,7985 | 0,3980 | 0,4822 | 0,7267 |
| en | con expansión | 10 | 0,3301 | 0,6084 | 0,3301 | 0,6084 | 0,3510 | 0,3056 | 0,5280 |
| en | con expansión | 20 | 0,3301 | 0,6084 | 0,4645 | 0,7486 | 0,3510 | 0,4194 | 0,6706 |
| combinado | con expansión | 10 | 0,3800 | 0,6622 | 0,3800 | 0,6622 | 0,4041 | 0,3447 | 0,5671 |
| combinado | con expansión | 20 | 0,3800 | 0,6622 | 0,4933 | 0,7697 | 0,4041 | 0,4414 | 0,6723 |

La fila `es`, sin expansión, k = 10 reproduce la línea base de la etapa 4 (exacto@10 = 0,3743; categoría@10 = 0,6526).

### Variante elegida y regla

- **Regla:** mayor recall@10 exacto sobre todos los gold; si la diferencia con la mejor es menor que 0,01, se elige la más simple (es < en < combinado; sin expansión antes que con expansión) y el k menor.
- **Resultado:** el mejor valor fue 0,3800 (`combinado`), pero `es` queda a 0,0057 (0,3743), dentro del umbral de empate. Empataron cuatro variantes (`es` y `combinado`, con y sin expansión). Se elige **`es`, sin expansión de abreviaturas, k = 10**, es decir, la configuración que ya tenía la etapa 4.
- **Consecuencia:** en esta iteración **no mejoró el recall@10**. Ni el campo en inglés, ni combinarlo, ni expandir abreviaturas dan una mejora de al menos 0,01. El inglés solo (0,3244 y 0,3301) es peor que el español.
- El recall sí sube al aumentar k (es: exacto 0,3743 con k = 10, 0,4722 con k = 20, 0,5298 con k = 30; categoría 0,6526, 0,7486 y 0,7965), pero eso mide otra cosa: con más candidatos el LLM de la etapa siguiente debe elegir entre más opciones. Quedó fuera de la regla de selección, que usa @10.
- No se necesitó modificar los valores por defecto del código (`MODO_POR_DEFECTO = "es"`, `EXPANDIR_POR_DEFECTO = False`).

### Limitaciones

- **La selección se hizo sobre los mismos 50 casos de dev que se usarán para la evaluación final.** El sesgo es pequeño (se compararon pocas variantes y la elegida es la de partida), pero debe declararse en el informe.
- Las métricas usan **menciones oráculo** (anotaciones de la pista X). El recall real del pipeline (etapa 5), con menciones extraídas automáticamente, será menor.
- La diferencia entre `combinado` y `es` (0,0057) no es concluyente con 521 códigos: es del orden de 3 códigos.
- La expansión de abreviaturas pone en mayúscula inicial la mención si empieza la oración (efecto del preprocesado); no se estudió su efecto aislado.

### Pendientes

- `scripts/evaluate.py` y el pipeline completo no existen aún (etapa 5); cuando se creen deberán llamar a `cargar_indice()` y `buscar()` sin argumentos (ya usan la variante elegida).
- Si se quiere subir el recall@10, las ideas de la iteración anterior siguen abiertas (recuperación léxica híbrida, contexto de la oración, reranking), porque los campos y la expansión no ayudan.

### Verificación

- `uv run pytest -q`: en verde.
- `uv run python scripts/build_index.py`: no recalcula nada si el cache es válido.
- `uv run python scripts/experiment_recall.py`: regenera `results/recall_variantes.json`.


---

## Etapa 5 — Pipeline completo (Fase 3 de CRISP-ML(Q): Modelado)

**Fecha:** 2026-11-03

### Qué se hizo

- `src/codificamed/schemas.py`: esquemas pydantic (respuestas del LLM y salida `ResultadoNota`).
- `src/codificamed/pipeline.py`: `analizar(nota, proveedor=None, indice=None, k=10, usar_negacion=True)`:
  1. preprocesa la nota; 2. el LLM extrae menciones (se descartan las que no son literales);
  3. `negation.py` marca afirmado/negado/sospecha; 4. recupera 10 candidatos solo para las afirmadas;
  5. el LLM elige un código entre los candidatos o `null`; 6. post-proceso: código en el catálogo y
  evidencia literal de la nota (si no, error). Los negados y sospechas salen aparte, sin código.
  JSON validado con pydantic y hasta 2 reintentos si es inválido. Se registran llamadas y tiempo.
- `src/codificamed/__main__.py`: `python -m codificamed "nota"`.
- `tests/test_pipeline.py`: LLM e índice falsos; pytest en verde.

### Decisiones

- Los negados no pasan por recuperación ni segunda llamada al LLM (menos tiempo; no se sugieren de todos modos).
- No se registran tokens: los proveedores solo devuelven texto; se registran llamadas y segundos.
- `usar_negacion=False` queda disponible para la ablación de la etapa 6.

### Verificación

- `uv run pytest -q`: en verde.
- Prueba manual con una nota ficticia (sin guardar resultados en `results/`): con `LLM_PROVIDER=ollama` (qwen2.5:7b-instruct) y con `LLM_PROVIDER=gemini` cambiando solo la variable, ambos devuelven `cefalea occipital` → R51 con evidencia literal y `dolor torácico` como negado. Gemini además codificó `DM2` → E11.9; Ollama no extrajo HTA ni DM2 como menciones.

### Pendientes

- Revisión manual de salidas (¿la evidencia justifica el código?) y métricas: etapa 6.
- La extracción de menciones del modelo local omite diagnósticos; evaluar si requiere mejorar el prompt.
