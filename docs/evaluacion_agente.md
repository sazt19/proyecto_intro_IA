# Evaluación del agente

Generado con `python -m src.evaluate`.

| # | Pregunta | Herramienta esperada | Herramientas usadas | ¿Correcta? | ¿Citó fuente? |
|---|---|---|---|---|---|
| 1 | ¿Qué estaciones hay disponibles? | list_stations | list_stations | ✅ | ✅ |
| 2 | ¿Cuánto PM2.5 hubo en Bello el 2024-03-15? | get_pm25 | get_pm25 | ✅ | ✅ |
| 3 | ¿Cómo está el aire en Envigado? | get_pm25 | get_pm25 | ✅ | ✅ |
| 4 | ¿Cómo estará el aire mañana en Sabaneta? | predict_tomorrow | predict_tomorrow | ✅ | ✅ |
| 5 | Predice la calidad del aire en Caldas para el día siguiente al 2024-04-10 | predict_tomorrow | predict_tomorrow | ✅ | ✅ |
| 6 | ¿Cuál es el límite diario de PM2.5 en Colombia? | search_docs | search_docs | ✅ | ✅ |
| 7 | ¿Qué recomienda la OMS sobre el PM2.5? | search_docs | search_docs | ✅ | ✅ |
| 8 | ¿Qué es el POECA? | search_docs | search_docs | ✅ | ✅ |
| 9 | ¿Cuáles son los niveles de alerta y emergencia por PM2.5? | search_docs | search_docs | ✅ | ✅ |
| 10 | ¿Qué significa que el aire esté en categoría dañina para grupos sensibles? | search_docs | search_docs | ✅ | ✅ |
| 11 | ¿Cómo está el aire en Barbosa hoy y cómo estará mañana? | get_pm25, predict_tomorrow | get_pm25, list_stations, predict_tomorrow | ✅ | ✅ |
| 12 | ¿Cuánto PM2.5 hubo en Bello el 2020-01-01? | get_pm25 | get_pm25 | ✅ | ✅ |
| 13 | ¿Cuál es la receta de la bandeja paisa? | ninguna (debe negarse) | — | ✅ | — |

**Herramienta correcta:** 13/13 · **Citó fuente:** 12/12

## Respuestas

**1. ¿Qué estaciones hay disponibles?**

Las estaciones disponibles en el Valle de Aburrá son:  - CEN-TRAF (Estación Tráfico Centro) - ITA-CJUS (Itagüí - Casa de Justicia Itagüí) - ITA-CONC (Itagüí - I.E. Concejo Municipal de Itagüí) - CAL-JOAR (Caldas - E U Joaquín Aristizabal) - EST-HOSP (La Estrella - Hospital) - MED-ALTA (Medellín, Altavista - I.E. Pedro Octavio Amado) - MED-VILL (Medellín, Villahermosa - Planta de producción de agua potable EPM) - BAR-TORR (Barbosa - Torre Social) - COP-CVID (Copacabana - Ciudadela Educativa La Vida) - MED-BEME (Medellin, Belén - I.E Pedro Justo Berrio) - MED-TESO (Medellín, El Poblado - I.E INEM sede Santa Catalina) - MED-SCRI (Medellín, San Cristóbal - Parque Biblioteca Fernando Botero) - MED-ARAN (Medellín, Aranjuez - I.E Ciro Mendia) - BEL-FEVE (Bello - I.E. Fernando Vélez) - ENV-HOSP (Envigado - E.S.E. Santa Gertrudis) - SAB-RAME (Sabaneta - I.E. Rafael J. Mejía)  [SIATA]

**2. ¿Cuánto PM2.5 hubo en Bello el 2024-03-15?**

El PM2.5 promedio diario en Bello el 2024-03-15 fue de **28.3 µg/m³**, con categoría ICA **Aceptable**.    [SIATA]

**3. ¿Cómo está el aire en Envigado?**

El aire en Envigado (estación ENV-HOSP) presenta una categoría ICA **Buena** con un promedio de PM2.5 de **11.3 µg/m³** en las últimas 24 horas (fecha: 2024-07-30). [SIATA]  *Nota: Los datos históricos disponibles van de julio 2023 a julio 2024.*

**4. ¿Cómo estará el aire mañana en Sabaneta?**

Mañana en Sabaneta se pronostica categoría ICA **aceptable** [SIATA]. Probabilidades: buena 12.2 %, aceptable 87.7 %, dañina 0.1 %. (Modelo de regresión logística propia con accuracy 0.77 en prueba temporal; datos históricos jul 2023‑jul 2024).

**5. Predice la calidad del aire en Caldas para el día siguiente al 2024-04-10**

Para el 12 de abril de 2024, la predicción de calidad del aire en Caldas (estación CAL‑JOAR) es **categoría ICA “aceptable”** con las siguientes probabilidades: buena 0 %, aceptable 69,1 % y dañina 30,9 % [SIATA].  Para el Valle de Aburrá puedo consultar datos históricos, pronósticos diarios y recomendaciones de salud.

**6. ¿Cuál es el límite diario de PM2.5 en Colombia?**

Según la **Resolución 2254 de 2017**, el nivel máximo permisible de PM2.5 en un período de 24 horas en Colombia es de **37 microgramos por metro cúbico (µg/m³)**.    *Fuente: [Resolución 2254 de 2017, Artículo 2, Tabla 1]*    *Nota: Este es el límite regulatorio colombiano; los valores guía de la OMS son más estrictos (15 µg/m³ para 24 h).*

**7. ¿Qué recomienda la OMS sobre el PM2.5?**

Según las directrices mundiales de la OMS de 2021:  - **Valor guía anual**: 5 µg/m³ de PM2.5.   - **Valor guía de 24 horas**: 15 µg/m³ de PM2.5.   - **Metas intermedias** (para reducir gradualmente la contaminación): en 24 h, niveles de 75, 50, 37,5 y 25 µg/m³; en promedio anual, de 35, 25, 15 y 10 µg/m³.   - **Poblaciones vulnerables**: niños, adultos mayores, mujeres embarazadas y personas con enfermedades respiratorias o cardiovasculares; no hay umbral seguro, por lo que reducir la exposición siempre es beneficioso.   - **Recomendaciones para reducir la exposición**: disminuir la actividad física intensa al aire libre cuando el aire es malo, preferir horas y lugares con mejor calidad, consultar la información oficial antes de actividades al aire libre y seguir las indicaciones médicas si se tienen condiciones de salud. Ante síntomas como dificultad para respirar o dolor en el pecho, buscar atención médica.  **Fuente**: [OMS] (Organización Mundial de la Salud, directrices de calidad del aire 2021).

**8. ¿Qué es el POECA?**

El POECA (Plan Operacional para Enfrentar Episodios de Contaminación Atmosférica) es un plan que contempla acciones coordinadas para gestionar episodios de contaminación y reducir la exposición de la población en el Valle de Aburrá. Según el Área Metropolitana, las medidas específicas dependen de las decisiones de las autoridades competentes. [AMVA]  No puedo responder esa pregunta porque no tengo la información necesaria para hacerlo. Puedo ayudarte con datos del PM2.5, predicciones y recomendaciones de salud basadas en la normativa local.

**9. ¿Cuáles son los niveles de alerta y emergencia por PM2.5?**

Según la **Resolución 2254 de 2017** (Artículos 9 y 10, Tabla 4), los niveles de alerta por PM₂.₅ para un promedio de 24 horas son:  | Nivel | Rangos de concentración (µg/m³) | |-------|---------------------------------| | **Prevención** | 38 – 55 | | **Alerta** | 56 – 150 | | **Emergencia** | a partir de 151 |  Estos rangos son los establecidos por la normativa colombiana para la contaminación atmosférica. La declaratoria oficial de alerta o emergencia corresponde a las autoridades competentes (Área Metropolitana del Valle de Aburrá – AMVA y SIATA), no a los pronósticos académicos ni a los datos de una sola estación.  Fuente: **Resolución 2254 de 2017** [SIATA].

**10. ¿Qué significa que el aire esté en categoría dañina para grupos sensibles?**

La categoría **“dañina para grupos sensibles”** (naranja) significa que **niños, adultos mayores, mujeres gestantes y personas con enfermedades respiratorias o cardiovasculares** deben **reducir la actividad física intensa o prolongada al aire libre**, ya que la contaminación puede afectar su salud.  Según la **Resolución 2254 de 2017**, este rango corresponde a concentraciones de PM2.5 de **38 a 55 µg/m³** (promedio de 24 horas).    [Resolución 2254 de 2017]

**11. ¿Cómo está el aire en Barbosa hoy y cómo estará mañana?**

**Barbosa (estación Torre Social – BAR-TORR):**  - **Hoy (30 jul 2024):** PM2.5 promedio 24 h = **11.6 µg/m³**, categoría ICA **Buena** [SIATA].   - **Mañana (31 jul 2024):** El modelo predictivo indica categoría ICA **Aceptable** (probabilidades: 42 % Buena, 58 % Aceptable, 0 % Dañina) [SIATA].  *Nota:* Los datos disponibles corresponden al periodo histórico julio 2023 – julio 2024. Si tienes otras preguntas sobre calidad del aire en el Valle de Aburrá, estaré encantado de ayudarte.

**12. ¿Cuánto PM2.5 hubo en Bello el 2020-01-01?**

No puedo proporcionar el dato para el 1 de enero de 2020, porque los registros disponibles en el sistema van de **julio 2023 a julio 2024** [SIATA]. No hay información medida para esa fecha.  Si deseas consultar el PM2.5 de Bello en una fecha dentro del rango disponible (por ejemplo, ayer o los últimos días), puedo buscarlo con gusto.

**13. ¿Cuál es la receta de la bandeja paisa?**

No puedo responder esa pregunta porque no tengo la información necesaria para hacerlo. Puedo ayudarte con el PM2.5 medido en las estaciones de SIATA, la predicción del día siguiente y lo que dicen la normativa y las guías sobre calidad del aire.
