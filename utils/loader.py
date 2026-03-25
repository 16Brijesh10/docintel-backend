#Text Extraction
from pypdf import PdfReader
from docx import Document
import io

def extract_text(filename, content):
    if filename.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content))
        return " ".join([page.extract_text() or "" for page in reader.pages])

    elif filename.endswith(".docx"):
        doc = Document(io.BytesIO(content))
        return " ".join([p.text for p in doc.paragraphs])

    elif filename.endswith(".txt"):
        return content.decode("utf-8")

    return ""