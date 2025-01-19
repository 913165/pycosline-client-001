# app/routers/health.py
from fastapi import APIRouter
from app.services.rag_processor import rag_processor
from app.config.settings import MISTRAL_API_KEY

router = APIRouter()

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "vector_store_initialized": rag_processor.vector_store is not None,
        "embedding_model": "mistral-embed",
        "api_key_configured": bool(MISTRAL_API_KEY)
    }