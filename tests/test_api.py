from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

os.environ["GOOGLE_API_KEY"] = "AQ.Ab8RN6JaWXSG1L6BlWQnwf2vYq8cvjKW3ixJVjBm8G1maUSAwg"

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

result = embeddings.embed_query("halo")

print(len(result))