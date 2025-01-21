from uuid import UUID
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Union, Any

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[str]] = None

class EmbeddingResponse(BaseModel):
    id: UUID
    text: str
    embedding: List[float]
    metadata: Optional[Dict[str, Any]] = None

class BatchEmbeddingResponse(BaseModel):
    embeddings: List[Dict[str, Union[str, List[float]]]]

class Point(BaseModel):
    id: UUID
    metadata: Dict[str, Any]
    content: str
    media: List[str]
    embedding: List[float]

    class Config:
        from_attributes = True

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[str]] = None
    similarity_scores: Optional[List[float]] = None
    processing_time: float