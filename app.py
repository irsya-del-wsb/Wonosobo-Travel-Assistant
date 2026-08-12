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

st.set_page_config(
    page_title="Chatbot Wisata WSB",
    page_icon="data/WSB_LOGO.png",
    layout="wide"
)

load_dotenv()

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY tidak ditemukan.")
    st.stop()

MODEL_NAME = "gemini-3.5-flash-lite"
TEMPERATURE = 1.2

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
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
    temperature=TEMPERATURE,
    google_api_key=GOOGLE_API_KEY
)

template = """
Kamu adalah Wonosobo Travel Assistant, asisten wisata yang ramah
dan membantu pengguna menjelajahi Wonosobo dan Dieng.

Tugasmu:
- Memberikan rekomendasi wisata.
- Menjelaskan tempat wisata.
- Menjelaskan harga tiket jika tersedia.
- Menjelaskan kuliner khas.
- Menjelaskan rute perjalanan.
- Menjawab dengan bahasa Indonesia yang santai, ramah, dan informatif.

ATURAN PENTING:

1. Gunakan informasi dari CONTEXT sebagai sumber utama jika
   pertanyaan pengguna berkaitan dengan informasi yang ada di dalamnya.

2. Jika informasi yang ditanyakan TIDAK ADA di CONTEXT, kamu
   BOLEH memberikan informasi umum berdasarkan pengetahuanmu.

3. Jika memberikan informasi berdasarkan pengetahuan umum dan bukan
   dari CONTEXT, jangan mengklaim bahwa informasi tersebut berasal
   dari dokumen.

4. Untuk informasi yang sangat spesifik seperti:
   - harga tiket
   - nomor telepon
   - alamat
   - jadwal
   - jam buka
   - harga hotel
   - harga makanan
   - rute/detail transportasi

   jangan mengarang informasi jika tidak tersedia di CONTEXT.
   Katakan bahwa informasi tersebut tidak tersedia di dokumen.

5. Jika pengguna meminta rekomendasi, kamu boleh memberikan beberapa
   pilihan berdasarkan pengetahuan umum meskipun daftar tersebut
   tidak terdapat di CONTEXT.

6. Jangan mengatakan "informasi tidak tersedia" hanya karena
   pertanyaan pengguna tidak memiliki jawaban persis di CONTEXT.
   Gunakan CONTEXT untuk informasi yang tersedia dan pengetahuan
   umum untuk melengkapi jawaban jika diperlukan.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
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
if "example_query" not in st.session_state:
    st.session_state.example_query = None
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

if "processing" not in st.session_state:
    st.session_state.processing = False
# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.title("Wonosobo Assistant")
    st.caption("made with sshdq")

    if st.button("Percakapan Baru"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.subheader("Riwayat Pertanyaan")

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

    st.subheader("Statistik")

    st.caption(f"Total Pesan: {len(st.session_state.messages)}")

    st.divider()

    st.subheader("Status Sistem")

    st.success("Gemini API Aktif")
    st.success("ChromaDB Terhubung")
    st.success("RAG Siap Digunakan")

    st.divider()

    st.subheader("Konfigurasi Model")

    st.caption(f"Model : {MODEL_NAME}")
    st.caption(f"Temperature : {TEMPERATURE}")

# =========================
# MAIN CHAT
# =========================
col1, col2 = st.columns([1,6])
with col1:
    st.image("data/WSB_LOGO.png")
with col2 :
    st.title(" Wonosobo Travel Assistant")
    st.caption(
        "Asisten digitalmu buat kulonuwun ke Wonosobo! Nanya apa aja bebas—dari "
        "tempat makan hits, hotel estetik, sampai hidden gems wisata ada di sini."
    )

if len(st.session_state.messages) == 0:

    st.markdown("""
Hai, ada yang bisa saya bantu??

Saya siap membantu Anda menjelajahi Wonosobo.

Contoh pertanyaan:
""")

    col1, col2 = st.columns(2)

    with col1:

        if st.button("Wisata terbaik di Dieng"):
            st.session_state.example_query = "Wisata terbaik di Dieng"
            st.rerun()

        if st.button("Kuliner khas Wonosobo"):
            st.session_state.example_query = "Kuliner khas Wonosobo"
            st.rerun()

    with col2:

        if st.button("Hotel dekat Dieng"):
            st.session_state.example_query = "Hotel di Dieng"
            st.rerun()

        if st.button("Cara ke Dieng"):
            st.session_state.example_query = "Rute ke Dieng"
            st.rerun()

# =========================
# TAMPILKAN RIWAYAT CHAT DULU
# =========================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# =========================
# INPUT CHAT
# =========================

query = st.chat_input(
    "Tanyakan wisata Wonosobo...",
    disabled=st.session_state.processing
)
# Jika user klik contoh pertanyaan
if st.session_state.example_query:
    query = st.session_state.example_query
    st.session_state.example_query = None

# =========================
# PROSES PERTANYAAN
# =========================
if st.session_state.processing and st.session_state.pending_query:

    query = st.session_state.pending_query
    st.session_state.pending_query = None

if query and not st.session_state.processing:

    st.session_state.pending_query = query
    st.session_state.processing = True
    st.rerun()

if st.session_state.processing and query:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
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

        st.session_state.processing = False
        st.rerun()

    except Exception as e:

        st.session_state.processing = False

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

            st.error(
                f"Terjadi error: {error_msg}"
            )