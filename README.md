# Wonosobo Travel Assistant

Chatbot wisata Wonosobo menggunakan:
- Streamlit
- Gemini 2.5 Flash Lite
- LangChain
- ChromaDB
- RAG

## Instalasi

```bash
pip install -r requirements.txt

## Buat API KEY
GOOGLE_API_KEY=YOUR_API_KEY

# Buat Vector Database
python build_rag.py

# Jalankan
streamlit run app.py
