"""Dashboard: Agente de calidad del aire del Valle de Aburrá.

Ejecutar: python -m streamlit run app.py
"""

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    ICA_PM25,
    STATES,
    ica_category,
    nvidia_api_key,
    nvidia_model,
)
from src.data import load_daily, load_stations
from src.model import evaluate
from src.tools import predict_tomorrow


# --------------------------------------------------
# CONFIGURACIÓN GENERAL
# --------------------------------------------------

st.set_page_config(
    page_title="Calidad del aire · Valle de Aburrá",
    page_icon="🌫️",
    layout="wide",
)


@st.cache_data
def data():
    return load_daily(), load_stations()


@st.cache_resource
def model_results():
    _, results, _, _ = evaluate()
    return results


daily, stations = data()

fmin = daily["fecha"].min().date()
fmax = daily["fecha"].max().date()


# --------------------------------------------------
# BARRA LATERAL
# --------------------------------------------------

st.sidebar.title("Parámetros")

labels = {
    r.codigo: f"{r.codigo} · {r.nombre}"
    for r in stations.itertuples()
}

codigo = st.sidebar.selectbox(
    "Estación",
    stations["codigo"],
    index=int(
        (stations["codigo"] == "ITA-CJUS").idxmax()
    ),
    format_func=labels.get,
)

fecha = st.sidebar.date_input(
    "Fecha de consulta",
    value=fmax,
    min_value=fmin,
    max_value=fmax,
)

rango = st.sidebar.slider(
    "Rango de la serie",
    min_value=fmin,
    max_value=fmax,
    value=(fmin, fmax),
)

st.sidebar.caption(
    "Datos: SIATA (PM2.5 horario → promedio diario) "
    "y Open-Meteo (meteorología). "
    f"Periodo disponible: {fmin} a {fmax}."
)


# --------------------------------------------------
# ENCABEZADO
# --------------------------------------------------

st.title("🌫️ Calidad del aire en el Valle de Aburrá")

st.caption(
    "Herramienta académica (EAFIT · SI3003). "
    "No es consejo médico ni reemplaza las alertas "
    "oficiales de SIATA/AMVA."
)


tab_dash, tab_chat, tab_modelo = st.tabs(
    [
        "📊 Dashboard",
        "💬 Agente",
        "🧪 Modelo",
    ]
)


# ==================================================
# PESTAÑA 1: DASHBOARD
# ==================================================

with tab_dash:

    dia = daily[
        daily["fecha"] == pd.Timestamp(fecha)
    ].merge(stations, on="codigo")

    sel = daily[
        (daily["codigo"] == codigo)
        & (daily["fecha"] == pd.Timestamp(fecha))
    ]

    c1, c2, c3 = st.columns(3)

    if sel.empty:

        c1.metric(
            "PM2.5 (24 h)",
            "Sin dato",
        )

        c2.metric(
            "Categoría ICA",
            "Sin dato",
        )

    else:

        v = float(sel["pm25"].iloc[0])

        c1.metric(
            "PM2.5 (24 h)",
            f"{v:.1f} µg/m³",
            help="Promedio diario",
        )

        c2.metric(
            "Categoría ICA",
            ica_category(v)[0],
        )

    try:

        pred = predict_tomorrow(
            codigo,
            str(fecha),
        )

        if "error" in pred:

            c3.metric(
                "Predicción siguiente día",
                "Sin datos",
            )

        else:

            p = pred["probabilidades"][
                pred["categoria_predicha"]
            ]

            c3.metric(
                f"Predicción {pred['fecha_predicha']}",
                pred["categoria_predicha"].capitalize(),
                f"{p:.0%} prob.",
                delta_color="off",
            )

    except Exception as e:

        c3.error(
            f"Error en la predicción: {e}"
        )

    st.caption(
        "Las predicciones son históricas y proceden "
        "del modelo académico desarrollado por el equipo, "
        "no de un pronóstico oficial de SIATA."
    )

    left, right = st.columns([1, 1.4])

    # ---------------- MAPA ----------------

    with left:

        st.subheader(
            f"Mapa de estaciones · {fecha}"
        )

        if dia.empty:

            st.info(
                "No hay datos válidos para esa fecha."
            )

        else:

            dia["categoria"] = dia["pm25"].map(
                lambda x: ica_category(x)[0]
            )

            colors = {
                name: color
                for _, _, name, color in ICA_PM25
            }

            fig = px.scatter_map(
                dia,
                lat="lat",
                lon="lon",
                color="categoria",
                size="pm25",
                hover_name="nombre",
                hover_data={
                    "pm25": ":.1f",
                    "lat": False,
                    "lon": False,
                },
                color_discrete_map=colors,
                zoom=9.6,
                height=480,
                center={
                    "lat": 6.25,
                    "lon": -75.58,
                },
            )

            fig.update_layout(
                map_style="carto-positron",
                margin=dict(
                    l=0,
                    r=0,
                    t=0,
                    b=0,
                ),
                legend=dict(
                    orientation="h",
                    y=0,
                ),
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

    # ---------------- SERIE DE TIEMPO ----------------

    with right:

        st.subheader(
            f"Serie de tiempo · {codigo}"
        )

        s = daily[
            (daily["codigo"] == codigo)
            & daily["fecha"].between(
                pd.Timestamp(rango[0]),
                pd.Timestamp(rango[1]),
            )
        ]

        fig = go.Figure()

        for lo, hi, name, color in ICA_PM25[:4]:

            fig.add_hrect(
                y0=lo,
                y1=hi,
                fillcolor=color,
                opacity=0.12,
                line_width=0,
                annotation_text=name,
                annotation_position="top left",
                annotation_font_size=10,
            )

        fig.add_scatter(
            x=s["fecha"],
            y=s["pm25"],
            mode="lines",
            name="PM2.5 diario",
            line=dict(
                color="#333"
            ),
        )

        fig.add_hline(
            y=37,
            line_dash="dash",
            line_color="red",
            annotation_text="Norma diaria 37 µg/m³",
        )

        max_pm25 = (
            float(s["pm25"].max())
            if not s.empty
            else 60.0
        )

        fig.update_layout(
            height=480,
            yaxis_title="µg/m³",
            yaxis_range=[
                0,
                max(60, max_pm25 * 1.1),
            ],
            margin=dict(
                l=0,
                r=0,
                t=10,
                b=0,
            ),
            showlegend=False,
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )


# ==================================================
# PESTAÑA 2: AGENTE CON RAG
# ==================================================

with tab_chat:

    st.subheader(
        "🤖 Asistente inteligente de calidad del aire"
    )

    st.caption(
        f"Modelo: `{nvidia_model()}` vía NVIDIA NIM "
        f"· Estación seleccionada: {codigo}"
    )

    st.caption(
        "Consulta mediciones históricas, predicciones "
        "del modelo académico y documentación ambiental "
        "mediante recuperación RAG."
    )

    # Inicializar historial
    if "chat" not in st.session_state:
        st.session_state.chat = []

    # Barra superior del chat
    col_info, col_clear = st.columns(
        [5, 1]
    )

    with col_info:

        st.caption(
            f"Mensajes: {len(st.session_state.chat)} "
            "· Fuentes: SIATA, OMS y normativa ambiental"
        )

    with col_clear:

        if st.button(
            "🗑️ Limpiar",
            key="limpiar_chat",
            use_container_width=True,
        ):

            st.session_state.chat = []
            st.rerun()

    # ----------------------------------------------
    # ÁREA DE CONVERSACIÓN
    # ----------------------------------------------

    chat_container = st.container(
        height=520,
        border=True,
    )

    with chat_container:

        if not st.session_state.chat:

            st.info(
                "Bienvenido. Puedes preguntar sobre "
                "calidad del aire, PM2.5, normativa "
                "colombiana, recomendaciones de la OMS "
                "y predicciones históricas."
            )

            st.markdown(
                "**Ejemplos de preguntas:**"
            )

            st.markdown(
                "- ¿Cuál es el límite diario de PM2.5 "
                "en Colombia?\n"
                "- ¿Qué recomienda la OMS sobre el PM2.5?\n"
                "- ¿Qué es el POECA?\n"
                "- ¿Qué calidad del aire predice el "
                "modelo para Itagüí?"
            )

        # Mostrar historial completo
        for m in st.session_state.chat:

            with st.chat_message(m["role"]):

                st.markdown(
                    m["content"]
                )

                # Mostrar herramientas utilizadas
                for t in m.get("trace", []):

                    nombre = t.get(
                        "tool",
                        "herramienta",
                    )

                    args = t.get(
                        "args",
                        "{}",
                    )

                    with st.expander(
                        f"🔧 {nombre}({args})"
                    ):

                        st.caption(
                            "Resultado devuelto por la herramienta"
                        )

                        st.code(
                            json.dumps(
                                t.get("result", {}),
                                ensure_ascii=False,
                                indent=2,
                                default=str,
                            ),
                            language="json",
                        )

    # ----------------------------------------------
    # CAMPO DE ESCRITURA DEBAJO DEL HISTORIAL
    # ----------------------------------------------

    if not nvidia_api_key():

        st.warning(
            "Falta NVIDIA_API_KEY. "
            "Configúrala en el archivo .env "
            "para activar el chat."
        )

    else:

        from src.agent import ask

        q = st.chat_input(
            "Escribe tu pregunta sobre la calidad del aire...",
            key="chat_agente",
        )

        if q:

            # Recuperar historial anterior
            history = [
                {
                    "role": m["role"],
                    "content": m["content"],
                }
                for m in st.session_state.chat[-6:]
            ]

            # Guardar pregunta
            st.session_state.chat.append(
                {
                    "role": "user",
                    "content": q,
                }
            )

            # Ejecutar agente
            with st.spinner(
                "Consultando el agente y sus herramientas..."
            ):

                try:

                    out = ask(
                        q,
                        history=history,
                        context=(
                            "Estación seleccionada "
                            f"en el dashboard: {codigo}"
                        ),
                    )

                except Exception as e:

                    out = {
                        "answer": (
                            "⚠️ No se pudo contactar "
                            f"al modelo: {e}"
                        ),
                        "trace": [],
                    }

            # Guardar respuesta
            st.session_state.chat.append(
                {
                    "role": "assistant",
                    "content": out["answer"],
                    "trace": out.get(
                        "trace",
                        [],
                    ),
                }
            )

            # Volver a renderizar el historial.
            # El campo de entrada permanece debajo.
            st.rerun()


# ==================================================
# PESTAÑA 3: MODELO DE IA CLÁSICA
# ==================================================

with tab_modelo:

    st.subheader(
        "Parte 1: Regresión logística multinomial"
    )

    st.caption(
        "Modelo implementado por el equipo "
        "y comparado con líneas base. "
        "Validación temporal desde 2024-04-01."
    )

    results = model_results()

    tabla = pd.DataFrame(
        {
            name: {
                "Accuracy": r["accuracy"],
                "F1-macro": r["f1_macro"],
                **{
                    f"F1 {s}": f
                    for s, f in zip(
                        STATES,
                        r["f1_por_clase"],
                    )
                },
            }
            for name, r in results.items()
        }
    ).T

    st.dataframe(
        tabla.style.format("{:.3f}"),
        width="stretch",
    )

    st.markdown(
        "### Matrices de confusión"
    )

    cols = st.columns(2)

    nombres = [
        "Regresión logística (propia)",
        "Persistencia (mañana = hoy)",
    ]

    for col, name in zip(
        cols,
        nombres,
    ):

        cm = results[name]["cm"]

        fig = px.imshow(
            cm,
            x=STATES,
            y=STATES,
            text_auto=True,
            color_continuous_scale="Blues",
            labels={
                "x": "Predicho",
                "y": "Real",
            },
            title=name,
        )

        fig.update_layout(
            height=360,
            coloraxis_showscale=False,
        )

        with col:

            st.plotly_chart(
                fig,
                width="stretch",
            )

    st.info(
        "La regresión logística fue implementada "
        "por el equipo. Su desempeño se compara "
        "con la persistencia y la clase mayoritaria."
    )
