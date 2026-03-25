#Document Processing
from app.utils.loader import extract_text
from app.utils.chunking import chunk_text
from app.services.vector_store import store_chunks

def process_document(filename, content):
    text = extract_text(filename, content)
    chunks = chunk_text(text)
    store_chunks(chunks, filename)