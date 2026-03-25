#Query Api
from fastapi import APIRouter
from app.services.query_engine import get_answer

router = APIRouter()

@router.post("/query")
def query(q: str):
    return get_answer(q)