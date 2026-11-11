# Agente de calidad del aire del Valle de Aburrá

Proyecto final de **SI3003 – Inteligencia Artificial** (Universidad EAFIT, 2026-1), idea 3.1.

Muchas personas no entienden qué significa el PM2.5 del día ni qué hacer al respecto. Esta aplicación
combina datos reales de SIATA, un modelo clásico implementado por el equipo y un agente LLM con
herramientas, todo dentro de un dashboard interactivo.

> ⚠️ Herramienta académica. No es consejo médico ni reemplaza las alertas oficiales de SIATA / AMVA.

## Cómo correrlo

Requisitos: Python 3.10 o superior (probado con 3.12) y Git.

```powershell
# 1. Clonar (recomendado: fuera de carpetas sincronizadas como OneDrive)
git clone https://github.com/sazt19/proyecto_intro_IA.git
cd proyecto_intro_IA

# 2. Entorno virtual
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Dependencias
pip install -r requirements.txt

# 4. Clave de NVIDIA (gratis en https://build.nvidia.com)
copy .env.example .env          # Windows  (cp .env.example .env en macOS/Linux)
# Abre .env y pega tu clave en NVIDIA_API_KEY=

# 5. Dashboard
streamlit run app.py
```

Se abre en `http://localhost:8501`.

- **Sin clave:** el dashboard (mapa, serie de tiempo, predicción y métricas) funciona igual. Solo el chat necesita la clave.
- **Datos y modelo incluidos:** ya vienen en `data/processed/`, así que no hay que descargar ni entrenar nada para correr la app.
- **La clave nunca se sube al repositorio:** `.env` está en `.gitignore`.

### Comandos opcionales

```powershell
python -m src.data --download   # vuelve a descargar los datos de SIATA y Open-Meteo
python -m src.model             # reentrena el modelo e imprime métricas vs. líneas base
python -m src.agent "¿Cómo estará el aire mañana en Bello?"   # agente desde la terminal
python -m src.rag "límite diario de PM2.5"                     # prueba solo el buscador RAG (sin clave)
python -m src.evaluate          # evalúa el agente con 13 preguntas fijas -> docs/evaluacion_agente.md
```

## Datos

| Fuente | Qué se usa | Acceso |
|---|---|---|
| [SIATA](https://siata.gov.co) – Sistema de Alerta Temprana de Medellín y el Valle de Aburrá | PM2.5 horario de 16 estaciones activas, jul 2023 – jul 2024 | `https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_pm25.json` |
| [Open-Meteo Historical Weather API](https://open-meteo.com) (reanálisis ERA5) | Lluvia, viento máximo y humedad diarios por estación | API pública sin clave |

Procesamiento (`src/data.py`):
- Se descartan los valores inválidos (-9999, negativos o mayores a 500 µg/m³) y las estaciones marcadas como fuera de operación.
- Se calcula el promedio diario solo cuando hay al menos 18 horas válidas, es decir, 75 % de representatividad.
- El resultado son 5.728 días-estación válidos.
- **Plan B:** los CSV procesados están en el repositorio por si las APIs fallan.

Las categorías siguen el ICA de PM2.5 de la Resolución 2254 de 2017, en µg/m³ promedio de 24 h. Para el modelo se agrupan en 3 clases:

| Clase | Rango | Días en los datos |
|---|---|---|
| buena | 0 – 12 | 1.021 |
| aceptable | 12 – 37 | 4.520 |
| dañina | > 37 (dañina para grupos sensibles o peor) | 187 |

## Parte 1: modelo clásico (implementación propia)

`src/model.py` contiene una **regresión logística multinomial (softmax) escrita en numpy**. Se entrena con descenso de gradiente, regularización L2 y pesos por clase parciales. Predice la categoría ICA del **día siguiente**.

- **Variables:** PM2.5 de hoy y de ayer, media móvil de 3 días, tendencia, lluvia, viento, humedad, fin de semana y mes (codificación cíclica).
- **Validación temporal:** se entrena hasta marzo de 2024 y se prueba de abril a julio de 2024, con 1.816 días-estación.
- **Métricas:** accuracy y F1-macro, implementadas a mano.

| Modelo | Accuracy | F1-macro | F1 "dañina" |
|---|---|---|---|
| **Regresión logística (propia)** | **0.769** | 0.674 | 0.52 |
| Persistencia (mañana = hoy) | 0.764 | **0.689** | **0.62** |
| Clase mayoritaria | 0.692 | 0.273 | 0.00 |

El modelo supera ligeramente a la persistencia en accuracy, pero no en F1-macro. Además entrega
probabilidades por clase, que el agente usa para comunicar la incertidumbre. El análisis completo de
limitaciones está en la sección final.

## Parte 2: agente LLM

`src/agent.py` implementa el **loop agéntico de forma manual**, sin frameworks, sobre la API compatible con OpenAI de **NVIDIA NIM**.

- **Modelo:** `nvidia/nemotron-3.5-lightning-30b-a3b`. Se puede cambiar con `NVIDIA_MODEL` en `.env`.
- **Respaldo:** `mistralai/mistral-large-4-1t-a52b`. Funciona, pero es más lento.
- **Límite de pasos:** T_max = 5 iteraciones.
- **Errores:** si una herramienta falla, el error se le devuelve al modelo como observación y la app no se cae.
- **Solo responde con contexto:** si una pregunta no se puede contestar con las herramientas, responde *"No puedo responder esa pregunta porque no tengo la información necesaria para hacerlo."* Los saludos y agradecimientos sí se responden.

Herramientas (`src/tools.py`):

| Herramienta | Qué hace |
|---|---|
| `list_stations()` | Lista las estaciones disponibles |
| `get_pm25(estacion, fecha)` | PM2.5 diario medido y su categoría ICA |
| `station_summary(estacion)` | Resumen histórico: peor y mejor día, promedio y días sobre la norma |
| `predict_tomorrow(estacion, fecha)` | Predicción del día siguiente con el modelo propio |
| `search_docs(query)` | Búsqueda en normativa y guías (RAG) |

### RAG

`src/rag.py` busca en `docs/fuentes.json` (19 fragmentos: Resolución 2254 de 2017, directrices de la
OMS 2021, AMVA/POECA y estadísticas de los propios datos) con **TF‑IDF y similitud coseno**. Se eligió
TF‑IDF en lugar de embeddings porque el corpus es pequeño, no requiere GPU ni descargar modelos y la app
arranca rápido. Cada fragmento tiene título, sección y URL, que el agente usa para citar la fuente.

### Evaluación del agente

`python -m src.evaluate` hace 13 preguntas fijas (datos, predicción, normativa, combinadas, fecha fuera
de rango y fuera de tema) y revisa si el agente llamó a la herramienta correcta y si citó la fuente.
Resultado: **13/13 herramienta correcta, 12/12 citó fuente**. Detalle en
[docs/evaluacion_agente.md](docs/evaluacion_agente.md).

## Dashboard

`app.py` (Streamlit) tiene tres pestañas:

- **Dashboard:**
  - Mapa de estaciones coloreado por categoría ICA.
  - Serie de tiempo con las franjas ICA y la norma diaria de 37 µg/m³.
  - Predicción de mañana.
- **Agente:** chat con el agente; muestra solo la respuesta final, con sus fuentes citadas. Las herramientas que usa se pueden ver con `python -m src.evaluate`.
- **Modelo:** métricas y matrices de confusión del modelo propio frente a la persistencia.

En la barra lateral se eligen la estación, la fecha y el rango de la serie.

## Estructura

```
app.py                  Dashboard Streamlit
src/config.py           Rutas, credenciales (.env / st.secrets) y categorías ICA
src/data.py             Descarga y limpieza (SIATA + Open-Meteo)
src/model.py            Regresión logística propia, líneas base y métricas
src/tools.py            Herramientas del agente
src/agent.py            Loop agéntico con NVIDIA NIM
src/rag.py              Búsqueda documental (TF-IDF)
src/evaluate.py         Evaluación del agente con preguntas fijas
docs/fuentes.json       Corpus del RAG
docs/                   Evaluación, informe y declaración de uso de IA
data/processed/         Datos diarios y modelo entrenado (plan B)
```

## Estado del proyecto

- [x] Datos reales descargados y procesados
- [x] Modelo de la Parte 1 con métrica frente a línea base
- [x] Agente con herramientas y dashboard funcional
- [x] RAG sobre normativa (Res. 2254/2017, POECA AMVA, guías OMS)
- [x] Evaluación del agente (13 preguntas)
- [x] Informe ([docs/informe.md](docs/informe.md)) y declaración de uso de IA ([docs/declaracion_ia.md](docs/declaracion_ia.md))

## Limitaciones

- **Datos históricos:** el JSON público de SIATA cubre julio de 2023 a julio de 2024, así que "hoy" en la app se refiere al último día disponible.
- **Pocos episodios críticos:** en ese año solo hubo 2 días por encima de 55 µg/m³. Por eso las categorías "dañina para grupos sensibles" y superiores se agruparon en una sola clase.
- **La persistencia es una línea base fuerte** para el PM2.5 diario. El modelo propio no la supera en F1-macro.
- **La meteorología es un reanálisis** (ERA5 vía Open-Meteo), no medición en la estación.
- **Disponibilidad de modelos en NVIDIA:** el catálogo cambia. Algunos modelos que aparecen en la lista no responden o dan error 404.
- **RAG léxico y corpus pequeño:** TF‑IDF busca por palabras, no por significado; los fragmentos son resúmenes escritos por el equipo a partir de las fuentes citadas, no los documentos completos.
- **El LLM puede cometer errores menores:** en la evaluación, por ejemplo, el agente confundió una vez la fecha predicha (dijo 12 de abril en lugar de 11) y en otra respuesta correcta añadió de más la frase de "no puedo responder".
