# app/models/request_models.py
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class QueryRequest(BaseModel):
    question: str
    context_filter: Optional[dict] = None

class ProcessingConfig(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200
    model_name: str = "mistral-embed"

class EmbeddingRequest(BaseModel):
    text: str

class BatchEmbeddingRequest(BaseModel):
    texts: List[str]
    batch_size: Optional[int] = 32