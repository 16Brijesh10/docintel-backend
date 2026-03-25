import os
from google.genai import Client

# 1) Set your API key in environment (preferred)
os.environ["GOOGLE_API_KEY"] = "AIzaSyB9NAhvek99o2fCw_Kgo8owfVilZJJcPGg"

# 2) Initialize the client
client = Client()

# 3) Choose a model (from your list, e.g., Gemini 2.5 Flash)
model_name = "models/gemini-2.5-flash"

# 4) Generate text
response = client.models.generate_content(
    model=model_name,
    contents="Explain what RAG (Retrieval‑Augmented Generation) is in simple terms."
)

# 5) Print the result
print("AI response:\n", response.text)