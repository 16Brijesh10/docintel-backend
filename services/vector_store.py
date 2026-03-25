#Vector Store (Chroma)
import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.Client()
collection = client.get_or_create_collection(name="docs")

model = SentenceTransformer("all-MiniLM-L6-v2")

"""def store_chunks(chunks, filename):

    embeddings = model.encode(chunks).tolist()

    ids = [f"{filename}_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"source": filename}] * len(chunks)
    )"""
    
def store_chunks(text, filename):
    # 1. Split text into semantic chunks
    chunks = chunk_text(text, chunk_size=500)

    if not chunks:
        print(f"No readable chunks for {filename}")
        return

    # 2. Encode
    embeddings = model.encode(chunks).tolist()
    ids = [f"{filename}_{i}" for i in range(len(chunks))]

    # 3. Add to collection
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"source": filename}] * len(chunks)
    )

    print(f"Stored {len(chunks)} chunks for {filename}")