import os
import streamlit as st
from utils.env_setup import co, client
from utils.embeddings import retrieve, embeddings_store
from utils.query_processing import improve_query
from utils.reranker import rerank_with_cohere
from utils.generator import generate_response
from utils.helpers import score_response_async

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
                "El objetivo es centrarse en la precisión de cada respuesta y en el feedback recibido. "
                "**Importante: Hasta no enviar el feedback, no permite realizar una nueva consulta**")

    st.markdown("---")

    st.markdown("🆔 **ID Consulta:** Cada interacción (consulta-respuesta) tiene un identificador único. "
                "Si aparte de enviar el feedback a través de esta pantalla se quieren hacer comentarios adicionales, "
                "favor de hacer referencia a este indicador.")

# Main UI
st.title("🔍 Customer Interface Specification")
st.markdown("📄💬 **Enter your query below to get information from the document:**")

# User Query Input
query_input = st.text_input("Escribe tu consulta:", placeholder="Ej: 'Cual es la tabla de atributos del DE 1?'")

CONFIDENCE_THRESHOLD = 0.65  # Adjust based on rerank score distribution

# Store response in session state to avoid unnecessary reprocessing
if "generated_response" not in st.session_state:
    st.session_state.generated_response = None
if "trace_id_code" not in st.session_state:
    st.session_state.trace_id_code = None

if query_input and st.session_state.generated_response is None:
    with st.spinner("🔄 Procesando tu consulta... Por favor esperar."):

        # Improve query
        improved_query = improve_query(query_input)

        # Retrieve relevant chunks
        initial_results = retrieve(improved_query)

        # Rerank results with Cohere
        reranked_results = rerank_with_cohere(improved_query, initial_results)

        # Filter based on confidence threshold
        filtered_results = [
            (chunk_id, conf) for chunk_id, conf, _ in reranked_results if conf >= CONFIDENCE_THRESHOLD
        ]

        # ✅ Corrected: Retrieve content from already loaded embeddings_store
        context = "\n\n".join([embeddings_store[chunk_id]["content"] for chunk_id, _ in filtered_results]) if filtered_results else ""

        # Generate response or inform no relevant info found
        if context:
            response, trace_id_code = generate_response(context, improved_query)
        else:
            response = "⚠️ **No se ha encontrado información relevante.** Por favor reformula tu pregunta."
            trace_id_code = None
            st.session_state.generated_response = None  

        # Store response in session state
        if response != "⚠️ **No se ha encontrado información relevante.** Por favor reformula tu pregunta.":
            st.session_state.generated_response = response
            st.session_state.trace_id_code = trace_id_code

# Display Response
if st.session_state.trace_id_code is not None:
    st.write(f"**ID consulta:** {st.session_state.trace_id_code}")
    st.markdown(st.session_state.generated_response)

    # Feedback Section (Only if an answer was generated)
    if st.session_state.trace_id_code is not None:
        st.markdown("**Dar Feedback**")
        feedback = st.radio("Qué tan acertada fue esta respuesta?", 
                            ["😞 1", "😐 2", "🙂 3", "😀 4", "🚀 5"], index=2, horizontal=True)

        if st.button("Enviar Feedback"):
            score_response_async(st.session_state.trace_id_code, feedback)  # Run in background
            st.success("Gracias por el feedback!")

            # Clear response and trace ID after feedback is submitted
            st.session_state.generated_response = None
            st.session_state.trace_id_code = None
