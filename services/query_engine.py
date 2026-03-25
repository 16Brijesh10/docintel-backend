#Query Engine
from app.services.vector_store import collection, model

def get_answer(query):
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    docs = results["documents"][0]
    sources = results["metadatas"][0]

    context = " ".join(docs)

    # Simple response (replace with LLM later)
    return {
        "answer": context[:300],
        "sources": sources
    }