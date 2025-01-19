# app/routers/__init__.py
from app.routers.document import router as document_router
from app.routers.embeddings import router as embeddings_router
from app.routers.health import router as health_router
from app.routers.similarity import router as similarity_router

__all__ = [
    'document_router',
    'embeddings_router',
    'health_router',
    'similarity_router'
]