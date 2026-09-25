# %%writefile app.py
import os
import io
import requests
import numpy as np
import faiss
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
TOP_K = 3

# ---------- Step 1: PDF Extraction ----------
def extract_pdf_text(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages

# ---------- Step 2: Chunking ----------
def chunk_text(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    for p in pages:
        text = p["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append({"page": p["page"], "text": chunk})
            start += chunk_size - overlap
    return chunks

# ---------- Step 3: Embeddings ----------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def create_embeddings(chunks, model):
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(embeddings)
    return embeddings

# ---------- Step 4: FAISS ----------
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def retrieve_relevant_chunks(question, model, index, chunks, top_k=TOP_K):
    q_emb = model.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "page": chunks[idx]["page"],
            "text": chunks[idx]["text"],
            "score": float(score)
        })
    return results

# ---------- Step 5: OpenRouter LLM ----------
def generate_answer(question, context_chunks):
    if not OPENROUTER_API_KEY:
        return "⚠️ Configuration error: OPENROUTER_API_KEY is missing. Please set it in your .env file."

    context = "\n\n".join(
        [f"[Page {c['page']}]: {c['text']}" for c in context_chunks]
    )

    prompt = f"""You are an AI document assistant.

Answer the user's question using ONLY the provided document context.

Do not invent facts.

If the answer cannot be found in the provided context, say:
"I couldn't find that information in the uploaded document."

Keep the answer concise and clear.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}
"""

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        return f"⚠️ Error contacting OpenRouter: {e}"
    except (KeyError, IndexError):
        return "⚠️ Unexpected response format from OpenRouter."

# ---------- Streamlit UI ----------
st.set_page_config(page_title="AI Document Q&A", page_icon="📄", layout="wide")

st.title("📄 AI Document Q&A")
st.caption("Ask questions about your document using Retrieval-Augmented Generation")

if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "index" not in st.session_state:
    st.session_state.index = None
if "filename" not in st.session_state:
    st.session_state.filename = None
if "num_pages" not in st.session_state:
    st.session_state.num_pages = 0

model = load_embedding_model()

with st.sidebar:
    st.header("📁 Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        if st.session_state.filename != uploaded_file.name:
            with st.spinner("Processing document..."):
                try:
                    file_bytes = uploaded_file.read()
                    pages = extract_pdf_text(file_bytes)

                    if not pages:
                        st.error("❌ No readable text found. This may be a scanned/image-only PDF.")
                    else:
                        chunks = chunk_text(pages)
                        embeddings = create_embeddings(chunks, model)
                        index = build_faiss_index(embeddings)

                        st.session_state.chunks = chunks
                        st.session_state.index = index
                        st.session_state.filename = uploaded_file.name
                        st.session_state.num_pages = len(pages)
                        st.success("✅ Document processed successfully")
                except Exception as e:
                    st.error(f"❌ Failed to process PDF: {e}")

    if st.session_state.filename:
        st.markdown("---")
        st.write(f"**File:** {st.session_state.filename}")
        st.write(f"**Pages:** {st.session_state.num_pages}")
        st.write(f"**Chunks:** {len(st.session_state.chunks)}")

st.subheader("Ask a question about your document")
question = st.text_input("Your question", placeholder="e.g. What is the main conclusion of this document?")
ask_clicked = st.button("Ask AI")

if ask_clicked:
    if not st.session_state.chunks:
        st.warning("⚠️ Please upload a PDF first.")
    elif not question.strip():
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("Searching document and generating answer..."):
            try:
                results = retrieve_relevant_chunks(
                    question, model, st.session_state.index, st.session_state.chunks
                )
                if not results:
                    st.warning("No relevant content found in the document.")
                else:
                    answer = generate_answer(question, results)

                    st.markdown("### 🤖 Answer")
                    st.write(answer)

                    st.markdown("### 📚 Sources")
                    for r in results:
                        with st.expander(f"Page {r['page']} — similarity score {r['score']:.3f}"):
                            st.write(r["text"])
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
