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


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[ChatMessage]] = []
    similarity_threshold: float = 0.7
    top_k: int = 5