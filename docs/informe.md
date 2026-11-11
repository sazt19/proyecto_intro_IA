# Agente de calidad del aire del Valle de Aburrá

**SI3003 – Inteligencia Artificial · Universidad EAFIT · 2026-1**

## 1. Problema

Muchas personas no entienden qué significa el PM2.5 del día ni qué hacer al respecto. Construimos un
dashboard que muestra la calidad del aire por estación, predice la categoría del día siguiente y ofrece
un agente conversacional que responde con datos medidos, la predicción del modelo y la normativa.

## 2. Datos

- **SIATA:** PM2.5 horario de 16 estaciones, julio 2023 – julio 2024
  (`https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_pm25.json`).
- **Open-Meteo (ERA5):** lluvia, viento máximo y humedad diarios por estación.
- **Limpieza:** se descartan valores inválidos (-9999, negativos o > 500) y estaciones fuera de operación; el
  promedio diario se calcula solo con ≥ 18 horas válidas (75 %). Resultado: 5.728 días-estación.
- **Plan B:** los CSV procesados están en el repositorio.

Hallazgos: PM2.5 diario promedio de 18,7 µg/m³. Marzo de 2024 fue el peor mes (31,3 µg/m³ de promedio y
98 días-estación por encima de 37). La estación más contaminada es CEN-TRAF (26,4 µg/m³).

## 3. Parte 1: regresión logística multinomial (implementación propia)

- **Tarea:** predecir la clase del día siguiente: buena (≤ 12), aceptable (12–37) o dañina (> 37).
- **Modelo:** softmax escrito en numpy, entrenado con descenso de gradiente, regularización L2 y pesos por clase.
- **Variables:** PM2.5 de hoy y de ayer, media de 3 días, tendencia, lluvia, viento, humedad, fin de semana y
  mes (seno/coseno).
- **Validación temporal:** entrenamiento hasta marzo de 2024; prueba de abril a julio de 2024 (1.816 días-estación).

| Modelo | Accuracy | F1-macro | F1 dañina |
|---|---|---|---|
| Regresión logística (propia) | **0.769** | 0.674 | 0.52 |
| Persistencia (mañana = hoy) | 0.764 | **0.689** | **0.62** |
| Clase mayoritaria | 0.692 | 0.273 | 0.00 |

El modelo supera a la clase mayoritaria y empata con la persistencia en accuracy, pero no la supera en
F1-macro. Su ventaja es que entrega probabilidades, que el agente usa para comunicar la incertidumbre.

## 4. Parte 2: agente LLM con RAG

- **Loop agéntico manual** (sin frameworks) sobre la API de NVIDIA NIM compatible con OpenAI, con el modelo
  `nvidia/nemotron-3.5-lightning-30b-a3b`. Máximo T_max = 5 pasos.
- **Manejo de errores:** si una herramienta falla, el error vuelve al modelo como observación y la app no se cae.
- **Herramientas:** `list_stations`, `get_pm25`, `station_summary`, `predict_tomorrow` (usa el modelo de la Parte 1) y `search_docs`.
- **RAG:** `search_docs` busca en 19 fragmentos (Resolución 2254 de 2017, OMS 2021, AMVA/POECA y
  estadísticas del proyecto) con TF-IDF y similitud coseno, filtrando palabras vacías del español. Se eligió
  TF-IDF porque el corpus es pequeño y no requiere GPU ni descargar modelos.
- **Respuestas solo con contexto:** el prompt obliga a responder únicamente con lo que devuelven las
  herramientas y a citar la fuente. Si no se consultó ninguna herramienta, la respuesta se reemplaza por
  "No puedo responder esa pregunta…".

**Integración Parte 1 + Parte 2:** el agente llama al modelo propio a través de `predict_tomorrow`, explica sus
probabilidades y las complementa con la normativa recuperada por el RAG.

## 5. Dashboard

Streamlit con tres pestañas:

- **Dashboard:** mapa de estaciones coloreado por ICA, serie de tiempo con franjas ICA y la norma de 37 µg/m³, y la predicción de mañana.
- **Agente:** chat con el agente; muestra solo la respuesta final, con sus fuentes citadas.
- **Modelo:** métricas y matrices de confusión.

El usuario elige la estación, la fecha y el rango de la serie.

## 6. Evaluación del agente

13 preguntas fijas (`python -m src.evaluate`, detalle en `docs/evaluacion_agente.md`):

- **Herramienta correcta en 13/13** y **fuente citada en 12/12**. La pregunta fuera de tema no se cuenta en las citas.
- Incluye casos de borde: una fecha fuera de rango (el agente explica el rango disponible) y una pregunta
  fuera de tema (el agente se niega a responder).
- **Errores observados:** en una respuesta el agente dijo la fecha predicha equivocada (12 de abril en lugar de 11), y
  en otra añadió de más la frase de rechazo después de responder bien.

## 7. Limitaciones

- **Los datos son históricos (2023–2024):** "hoy" en la app es el 30 de julio de 2024.
- **Hubo muy pocos días críticos** (2 por encima de 55 µg/m³), por eso se agruparon las categorías altas.
- **La persistencia es una línea base fuerte;** el modelo no la supera en F1-macro.
- **La meteorología es un reanálisis,** no medición en la estación.
- **El RAG es léxico y el corpus es pequeño:** son resúmenes redactados por el equipo a partir de las fuentes citadas.
- **El LLM puede equivocarse en detalles,** y la disponibilidad de modelos en NVIDIA cambia.
- **Es una herramienta académica:** no es consejo médico ni reemplaza las alertas oficiales de SIATA/AMVA.
