#File Upload API
from fastapi import APIRouter, UploadFile, File
from app.services.document_processor import process_document

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    process_document(file.filename, content)
    return {"message": f"{file.filename} uploaded successfully"}