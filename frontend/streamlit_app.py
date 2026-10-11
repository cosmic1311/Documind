"""DocuMind Streamlit frontend. Talks to the FastAPI backend over HTTP only."""

import os

import requests
import streamlit as st

API_URL = os.getenv("DOCUMIND_API_URL", "http://127.0.0.1:8000").rstrip("/")
UPLOAD_TIMEOUT_SECONDS = 300  # embedding a large PDF can take minutes on CPU
ASK_TIMEOUT_SECONDS = 120

st.set_page_config(page_title="DocuMind", page_icon="📄", layout="wide")


def api_error_message(response: requests.Response) -> str:
    """Pull the readable message out of a FastAPI error response."""
    try:
        return response.json().get("detail", response.text)
    except ValueError:
        return response.text or f"HTTP {response.status_code}"


def backend_is_up() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=3).status_code == 200
    except requests.RequestException:
        return False


def upload_pdf(uploaded_file) -> dict:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
    response = requests.post(f"{API_URL}/upload", files=files, timeout=UPLOAD_TIMEOUT_SECONDS)
    if response.status_code != 201:
        raise RuntimeError(api_error_message(response))
    return response.json()


def ask(question: str) -> dict:
    response = requests.post(
        f"{API_URL}/ask", json={"question": question}, timeout=ASK_TIMEOUT_SECONDS
    )
    if response.status_code != 200:
        raise RuntimeError(api_error_message(response))
    return response.json()


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for number, source in enumerate(sources, start=1):
            st.markdown(
                f"**[{number}] {source['filename']}, page {source['page_number']}** "
                f"· similarity {source['score']:.2f}"
            )
            st.caption(source["text"])


# Session state lives in this browser session only. The backend keeps the index in memory.
if "documents" not in st.session_state:
    st.session_state.documents = []
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("📄 DocuMind")
st.caption("Upload PDFs and ask questions. Answers come only from your documents, with page citations.")

if not backend_is_up():
    st.error(
        f"Cannot reach the backend at {API_URL}. "
        "Start it with: python -m uvicorn app.main:app --reload"
    )
    st.stop()

with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader("Add PDF files", type=["pdf"], accept_multiple_files=True)
    if st.button("Index selected files", disabled=not uploaded_files):
        indexed_names = {doc["filename"] for doc in st.session_state.documents}
        for uploaded_file in uploaded_files:
            if uploaded_file.name in indexed_names:
                st.info(f"{uploaded_file.name} is already indexed.")
                continue
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                try:
                    result = upload_pdf(uploaded_file)
                    st.session_state.documents.append(result["document"])
                    st.success(result["message"])
                except requests.RequestException as exc:
                    st.error(f"Network error while uploading {uploaded_file.name}: {exc}")
                except RuntimeError as exc:
                    st.error(f"{uploaded_file.name}: {exc}")

    st.divider()
    st.subheader("Indexed this session")
    if not st.session_state.documents:
        st.caption("No documents yet.")
    for doc in st.session_state.documents:
        st.write(f"**{doc['filename']}** — {doc['page_count']} pages with text, {doc['chunk_count']} chunks")

if not st.session_state.documents:
    st.info("Upload and index a PDF in the sidebar to start asking questions.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))

question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            try:
                result = ask(question)
            except requests.RequestException as exc:
                st.error(f"Network error: {exc}")
            except RuntimeError as exc:
                st.error(f"The backend could not answer: {exc}")
            else:
                st.markdown(result["answer"])
                render_sources(result["sources"])
                st.session_state.messages.append(
                    {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
                )