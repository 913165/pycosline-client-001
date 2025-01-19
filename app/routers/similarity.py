# app/routers/similarity.py
import numpy as np
import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException
import aiohttp
from pydantic import BaseModel
from typing import Dict, Any, List

from app.config.settings import VECTOR_DB_URL
from app.services.rag_processor import rag_processor

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()


class FilterExpression(BaseModel):
    field: str
    operator: str
    value: Any


class SearchRequest(BaseModel):
    query: str
    topK: int = 10
    similarityThreshold: float = 0.7
    filterExpression: Optional[FilterExpression] = None


class Document(BaseModel):
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    documents: List[Document]
    total: int
    scores: List[float]


class Similarity:
    def __init__(self, key: str, score: float):
        self.key = key
        self.score = score


def cosine_similarity(vector1: List[float], vector2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    v1 = np.array(vector1)
    v2 = np.array(vector2)

    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0


def convert_filter_to_query_params(filter_expression: Optional[FilterExpression]) -> str:
    """Convert filter expression to query parameters."""
    if not filter_expression:
        return ""

    operator_map = {
        "equals": "=",
        "gt": ">",
        "lt": "<",
        "gte": ">=",
        "lte": "<=",
        "contains": "~"
    }

    op = operator_map.get(filter_expression.operator, "=")
    return f"{filter_expression.field}{op}{filter_expression.value}"


async def get_documents_from_api(collection_name: str, filter_expression: Optional[FilterExpression]) -> List[Document]:
    """Fetch documents from the vector database API."""
    query_params = convert_filter_to_query_params(filter_expression)
    base_url = f"{VECTOR_DB_URL}/api/v1/collections/{collection_name}/payload"

    url = f"{base_url}?{query_params}" if query_params else base_url
    logger.info(f"Fetching documents from URL: {url}")

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Error fetching documents: {await response.text()}"
                )

            documents = await response.json()
            return [Document(**doc) for doc in documents]


@router.post("/similarity-search", response_model=SearchResponse)
async def similarity_search(request: SearchRequest, collection_name: str = "vector_store"):
    """
    Perform similarity search on documents using cosine similarity.
    """
    try:
        # Get query embeddings
        query_embedding = await rag_processor.get_embeddings(request.query)
        logger.info(f"Generated query embedding for: {request.query}")

        # Fetch documents from API
        documents = await get_documents_from_api(collection_name, request.filterExpression)
        logger.info(f"Retrieved {len(documents)} documents from API")

        # Create document map for quick lookup
        document_map = {doc.id: doc for doc in documents}

        # Calculate similarities and filter results
        similarities = []
        for doc in documents:
            score = cosine_similarity(query_embedding, doc.embedding)
            logger.info(f"Document ID: {doc.id}, Score: {score}")

            if score >= request.similarityThreshold:
                similarities.append(Similarity(doc.id, score))

        # Sort by similarity score and limit results
        sorted_similarities = sorted(
            similarities,
            key=lambda x: x.score,
            reverse=True
        )[:request.topK]

        # Get final results
        result_documents = []
        result_scores = []
        for sim in sorted_similarities:
            doc = document_map[sim.key]
            result_documents.append(doc)
            result_scores.append(sim.score)

        logger.info(f"Returning {len(result_documents)} documents from similarity search")

        return SearchResponse(
            documents=result_documents,
            total=len(result_documents),
            scores=result_scores
        )

    except Exception as e:
        logger.error(f"Error in similarity search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add to __init__.py:
# from app.routers.similarity import router as similarity_router
# __all__.append('similarity_router')