import os
import streamlit as st
from utils.env_setup import co, client
from utils.embeddings import retrieve, embeddings_store
from utils.query_processing import improve_query
from utils.reranker import rerank_with_cohere
from utils.generator import generate_response
from utils.helpers import score_response_async
import re

# -------------------- Environment Setup -------------------- #

# Get the absolute path of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths for assets and data
IMAGE_PATH = os.path.join(BASE_DIR, "assets", "mastercard-logo.png")

# -------------------- Streamlit UI Setup -------------------- #

st.set_page_config(
    page_title="ChatCIS",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    if os.path.exists(IMAGE_PATH):
        st.image(IMAGE_PATH, width=120)
    else:
        st.warning(f"⚠️ Logo not found: {IMAGE_PATH}")

    st.markdown("### Customer Interface Specification")
    st.markdown("---")

    st.markdown("⚠️ **Nota:** Este chat responde una consulta a la vez, ya que está en fase de prueba. "
                "El objetivo es centrarse en la precisión de cada respuesta. "
                "**Importante: Hasta no enviar la nueva consulta, no permite realizar otra**")

    st.markdown("---")

    st.markdown("🆔 **ID Consulta:** Cada interacción (consulta-respuesta) tiene un identificador único. "
                "Si aparte de esta pantalla se quieren hacer comentarios adicionales, "
                "favor de hacer referencia a este indicador.")

# Main UI
st.title("🔍 Customer Interface Specification")

# Session state defaults
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

if "generated_response" not in st.session_state:
    st.session_state.generated_response = None

if "trace_id_code" not in st.session_state:
    st.session_state.trace_id_code = None

if "final_context" not in st.session_state:
    st.session_state.final_context = ""

# Input field (bound to session state)
st.session_state.query_input = st.text_input(
    "Escribe tu consulta:",
    placeholder="Ej: 'Cual es la tabla de atributos del DE 1?'",
    value=st.session_state.query_input
)

CONFIDENCE_THRESHOLD = 0.60

# Query processing
if st.session_state.query_input and st.session_state.generated_response is None:
    with st.spinner("🔄 Procesando tu consulta... Por favor esperar."):

        improved_query = improve_query(st.session_state.query_input)

        context = []
        for item in improved_query:

            reference = item['reference']
            match = re.search(r'->\s*(.*)', reference)
            if match:
                reference_clean = match.group(1)
            else:
                reference_clean = reference

            initial_results = retrieve(reference_clean, top_k=20)

            reranked_results = rerank_with_cohere(reference_clean, initial_results)

            for chunk_id, confidence, header, data in reranked_results:
                if confidence >= CONFIDENCE_THRESHOLD:
                    context.append(data.get("content", ""))
        


        st.session_state.final_context = "\n\n".join(context)
        response, trace_id_code = generate_response(st.session_state.final_context, st.session_state.query_input)

        st.session_state.generated_response = response
        st.session_state.trace_id_code = trace_id_code

# Display response
if st.session_state.generated_response is not None:
    st.write("### Respuesta generada:")

    if st.session_state.final_context:
        st.write(f"ID Consulta: {st.session_state.trace_id_code}")
        st.markdown(st.session_state.generated_response)
    else:
        st.write(f"ID Consulta: {st.session_state.trace_id_code}")
        st.warning("⚠️ No se encontró información relevante para responder a la consulta.")

    # Feedback section
    st.markdown("**Dar Feedback**")

    if "feedback_score" not in st.session_state:
        st.session_state.feedback_score = "🙂 3"

    st.session_state.feedback_score = st.radio(
        "¿Qué tan acertada fue esta respuesta?",
        ["😞 1", "😐 2", "🙂 3", "😀 4", "🚀 5"],
        index=2,
        horizontal=True
    )

    if st.button("Enviar Feedback"):
        score_response_async(
            trace_id=st.session_state.trace_id_code,
            feedback=st.session_state.feedback_score
        )

        # Clear state and rerun app
        keys_to_clear = [
            "query_input",
            "generated_response",
            "trace_id_code",
            "final_context",
            "feedback_score"
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.success("✅ Feedback enviado. ¡Gracias!")
        st.rerun()


