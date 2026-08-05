from google import genai

client = genai.Client(
    api_key="AQ.Ab8RN6Kd4Ej3AjfldOxARRfC74GvBvDGPitZwiKKFUhPHgaVAw"
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Halo"
)

print(response.text)