from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder # Added CrossEncoder
from pypdf import PdfReader
from docx import Document
from typing import List
from io import BytesIO
from google.genai import Client

# Initialize Gemini Client
clients = Client(api_key="Enter your key") 
model_name = "models/gemini-flash-latest"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://doc-intel-frontend-nine.vercel.app",
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ChromaDB Client
db_client = chromadb.Client()

# 1. BI-ENCODER: For fast initial retrieval (Stage 1)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. CROSS-ENCODER: For high-accuracy re-ranking (Stage 2)
# This model is small and works well on CPU
re_ranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    collection = db_client.get_or_create_collection(name="docs")
    
    try:
        all_docs = collection.get()
        if all_docs['ids']:
            collection.delete(ids=all_docs['ids'])
    except:
        pass 

    all_chunks = []
    all_embeddings = []
    all_ids = []
    all_metadata = []

    for file in files:
        content = await file.read()
        text = ""

        if file.filename.endswith(".pdf"):
            reader = PdfReader(BytesIO(content))
            for page in reader.pages:
                text += page.extract_text() or ""
        elif file.filename.endswith(".docx"):
            doc = Document(BytesIO(content))
            for para in doc.paragraphs:
                text += para.text + "\n"
        else:
            try:
                text = content.decode("utf-8")
            except:
                text = ""

        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        if not chunks:
            continue

        embeddings = embedding_model.encode(chunks).tolist()
        ids = [f"{file.filename}_{i}" for i in range(len(chunks))]

        all_chunks.extend(chunks)
        all_embeddings.extend(embeddings)
        all_ids.extend(ids)
        all_metadata.extend([{"source": file.filename}] * len(chunks))

    if not all_chunks:
        return {"message": "No readable text found"}

    collection.add(
        documents=all_chunks,
        embeddings=all_embeddings,
        ids=all_ids,
        metadatas=all_metadata
    )

    return {"message": f"{len(files)} files uploaded successfully"}

@app.post("/query")
def query(q: str):
    collection = db_client.get_or_create_collection(name="docs")
    
    # STAGE 1: Broad Retrieval
    # We fetch 10 results instead of 3 to give the re-ranker options
    query_embedding = embedding_model.encode([q]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=10)

    retrieved_chunks = results["documents"][0] if results["documents"] else []
    retrieved_metadatas = results["metadatas"][0] if results["metadatas"] else []

    if not retrieved_chunks:
        return {"answer": "No relevant info found in docs.", "sources": []}

    # STAGE 2: Re-ranking with Cross-Encoder
    # We pair the query with every retrieved chunk for comparison
    sentence_combinations = [[q, chunk] for chunk in retrieved_chunks]
    
    # Get relevancy scores (higher is better)
    scores = re_ranker.predict(sentence_combinations)

    # Sort chunks based on the Cross-Encoder scores
    # We combine (score, chunk, metadata) and sort by score descending
    ranked_results = sorted(
        zip(scores, retrieved_chunks, retrieved_metadatas), 
        key=lambda x: x[0], 
        reverse=True
    )

    # Pick the top 3 after re-ranking
    top_ranked = ranked_results[:3]
    final_chunks = [item[1] for item in top_ranked]
    final_metadatas = [item[2] for item in top_ranked]

    context = "\n\n".join(final_chunks)
    
    prompt = f"""
    You are a strict filtering assistant. 
    The user is asking ONLY about: "{q}".
    Using ONLY the context below, provide a structured summary. 
    If context does not contain the answer, respond ONLY with: "I'm sorry, I couldn't find any information regarding that in the uploaded documents."
    Do NOT mention "Based on the provided context".
    Context:
    {context}
    Question: {q}
    """
    
    response = clients.models.generate_content(model=model_name, contents=prompt)
    unique_sources = list({meta.get('source', 'Unknown') for meta in final_metadatas})

    return {
        "answer": response.text.strip() if response.text else "No answer found",
        "sources": unique_sources
    }

@app.post("/clear")
async def clear_data():
    try:
        db_client.delete_collection(name="docs") 
        db_client.get_or_create_collection(name="docs")
        return {"message": "Database and chat history cleared successfully"}
    except Exception as e:
        db_client.get_or_create_collection(name="docs")
        return {"message": "Database was already empty or has been reset"}
    
@app.get("/files")
async def get_files():
    try:
        collection = db_client.get_or_create_collection(name="docs")
        results = collection.get(include=['metadatas'])
        metadatas = results.get('metadatas', [])
        unique_files = list(set([m.get('source') for m in metadatas if m.get('source')]))
        return {"files": unique_files}
    except Exception as e:
        return {"files": [], "error": str(e)}
