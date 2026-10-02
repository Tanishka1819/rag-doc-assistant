import os
import time
from pathlib import Path

import numpy as np
import streamlit as st
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# If you get a "model not found" error, copy the current free model name from Google AI Studio
MODEL_NAME = "gemini-3.8-flash"

st.set_page_config(page_title="Chat with my documents", page_icon="📄")
st.title("📄 Chat with my documents")


def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY")


def chunk_text(text, size=800, overlap=150):
    # cut a page into small overlapping pieces so no sentence gets lost at an edge
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


@st.cache_resource
def load_embedder():
    # turns text into numbers that capture its meaning
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def build_index():
    chunks = []
    for pdf in sorted(Path("docs").glob("*.pdf")):
        reader = PdfReader(pdf)
        for page_no, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for c in chunk_text(text):
                if c.strip():
                    chunks.append({"source": pdf.name, "page": page_no, "text": c})
    vectors = load_embedder().encode([c["text"] for c in chunks], normalize_embeddings=True)
    return chunks, vectors


def search(question, chunks, vectors, k=4):
    # find the k chunks whose meaning is closest to the question
    q = load_embedder().encode([question], normalize_embeddings=True)[0]
    scores = vectors @ q
    top = np.argsort(-scores)[:k]
    return [{**chunks[i], "score": float(scores[i])} for i in top]


def answer(question, passages):
    context = "\n\n".join(f"[{p['source']} p.{p['page']}]\n{p['text']}" for p in passages)
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say \"I couldn't find that in the documents.\" "
        "Mention the source file and page number you used.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    )
    client = genai.Client(api_key=get_api_key())
    last_error = None
    for attempt in range(4):
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            return response.text
        except Exception as e:
            last_error = e
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                time.sleep(3 * (attempt + 1))
            else:
                raise
    raise last_error
    
pdfs = list(Path("docs").glob("*.pdf"))
if not pdfs:
    st.error("Put at least one PDF in the docs folder.")
    st.stop()
if not get_api_key():
    st.error("No API key found. Add GEMINI_API_KEY to .streamlit/secrets.toml")
    st.stop()

with st.spinner("Reading the documents (first time takes a minute)..."):
    chunks, vectors = build_index()
st.caption(f"Loaded {len(chunks)} text chunks from {len(pdfs)} PDF(s).")

question = st.text_input("Ask a question about the documents:")
if question:
    with st.spinner("Thinking..."):
        passages = search(question, chunks, vectors)
        try:
            reply = answer(question, passages)
        except Exception as e:
            st.error(f"The AI call failed: {e}")
            st.stop()
    st.subheader("Answer")
    st.write(reply)
    with st.expander("Sources used"):
        for p in passages:
            st.markdown(f"**{p['source']}, page {p['page']}** (match score {p['score']:.2f})")
            st.write(p["text"][:400] + "...")