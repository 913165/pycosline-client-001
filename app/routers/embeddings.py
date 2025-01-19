# app/routers/embeddings.py
from uuid import uuid4
from fastapi import APIRouter, HTTPException

from app.models.request_models import EmbeddingRequest, BatchEmbeddingRequest
from app.models.response_models import EmbeddingResponse, BatchEmbeddingResponse
from app.services.rag_processor import rag_processor

router = APIRouter()

@router.post("/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    try:
        embedding = await rag_processor.get_embeddings(request.text)
        point_id = uuid4()
        return EmbeddingResponse(
            text=request.text,
            embedding=embedding,
            id=point_id,
            metadata={
                "source": "single_request",
                "id": str(point_id)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-embeddings", response_model=BatchEmbeddingResponse)
async def get_batch_embeddings(request: BatchEmbeddingRequest):
    try:
        embeddings = []
        for i in range(0, len(request.texts), request.batch_size or 32):
            batch = request.texts[i:i + request.batch_size or 32]
            batch_embeddings = await rag_processor.get_batch_embeddings(batch)

            for text, embedding in zip(batch, batch_embeddings):
                point_id = uuid4()
                embeddings.append({
                    "id": point_id,
                    "text": text,
                    "embedding": embedding,
                    "metadata": {
                        "source": "batch_request",
                        "id": str(point_id),
                        "batch_index": len(embeddings)
                    }
                })

        return BatchEmbeddingResponse(embeddings=embeddings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))