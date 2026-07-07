import streamlit as st
import os

from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

load_dotenv()
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

MODEL_NAME = "gemini-2.5-flash-lite"
TEMPERATURE = 1.2

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=TEMPERATURE
)

template = """
Kamu adalah Wonosobo Travel Assistant.

Tugasmu:

- Menjadi pemandu wisata Wonosobo.
- Memberikan rekomendasi wisata.
- Menjelaskan harga tiket jika tersedia.
- Menjelaskan kuliner khas.
- Menjelaskan rute perjalanan.
- Menjawab dengan bahasa santai dan ramah.

Jika informasi tidak ditemukan pada context,
katakan bahwa informasi tidak tersedia.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={
        "prompt": prompt
    }
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.title("🏔️ Wonosobo Assistant")

    if st.button("➕ Percakapan Baru"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.subheader("📜 Riwayat Pertanyaan")

    user_questions = [
        msg["content"]
        for msg in st.session_state.messages
        if msg["role"] == "user"
    ]

    if len(user_questions) == 0:
        st.caption("Belum ada percakapan")
    else:
        for q in reversed(user_questions[-10:]):
            st.caption(f"• {q}")

    st.divider()

    st.subheader("📊 Statistik")

    st.caption(f"Total Pesan: {len(st.session_state.messages)}")

    st.divider()

    st.subheader("📡 Status Sistem")

    st.success("Gemini API Aktif")
    st.success("ChromaDB Terhubung")
    st.success("RAG Siap Digunakan")

    st.divider()

    st.subheader("⚙️ Konfigurasi Model")

    st.caption(f"🤖 Model : {MODEL_NAME}")
    st.caption(f"🌡️ Temperature : {TEMPERATURE}")

# =========================
# MAIN CHAT
# =========================

st.title("🏔️ Wonosobo Travel Assistant")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

query = st.chat_input(
    "Tanyakan wisata Wonosobo..."
)

if query:

    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):
        st.write(query)

    try:

        with st.spinner(
            f"🔍 Sedang mencari jawaban dari '{query}' ..."
        ):

            response = qa_chain.invoke({
                "query": query
            })

            answer = response["result"]

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        with st.chat_message("assistant"):
            st.write(answer)

    except Exception as e:

        error_msg = str(e)

        if "429" in error_msg:
            st.warning("""
⚠️ Kuota Gemini API telah mencapai batas penggunaan.

Silakan:
- Tunggu beberapa saat
- Gunakan API Key lain
- Periksa kuota Gemini API Anda
""")

        elif "quota" in error_msg.lower():
            st.warning("""
⚠️ Batas penggunaan Gemini API telah tercapai.
Silakan coba kembali nanti.
""")

        elif "rate limit" in error_msg.lower():
            st.warning("""
⚠️ Terlalu banyak permintaan ke Gemini API.
Silakan tunggu beberapa saat lalu coba lagi.
""")

        else:
            st.error(f"Terjadi error: {error_msg}")