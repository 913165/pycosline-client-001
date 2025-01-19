# app/routers/document.py
import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional
import aiohttp

from app.models.request_models import QueryRequest, ProcessingConfig
from app.models.response_models import QueryResponse
from app.services.rag_processor import rag_processor
from app.config.settings import VECTOR_DB_URL

router = APIRouter()

@router.post("/upload")
async def upload_document(
        file: UploadFile = File(...),
        config: Optional[ProcessingConfig] = None
):
    if not file.filename.endswith('.txt'):
        raise HTTPException(
            status_code=400,
            detail="Only .txt files are currently supported"
        )

    try:
        file_path = f"temp/{file.filename}"
        os.makedirs("temp", exist_ok=True)

        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        result = await rag_processor.process_document(file_path)
        os.remove(file_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    try:
        answer = await rag_processor.query(request.question)
        return QueryResponse(
            answer=answer,
            sources=None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process-file")
async def process_file_to_vectors(
        file: UploadFile = File(...),
        collection_name: str = "vector_store"
):
    if not file.filename.endswith(('.txt', '.csv')):
        raise HTTPException(
            status_code=400,
            detail="Only .txt and .csv files are currently supported"
        )

    try:
        content = await file.read()
        points = await rag_processor.process_file_for_embeddings(content)
        vector_db_url = f"{VECTOR_DB_URL}/api/v1/collections/{collection_name}/payload"

        formatted_points = [
            {
                "id": str(point.id),
                "metadata": point.metadata,
                "content": point.content,
                "media": point.media,
                "embedding": point.embedding
            }
            for point in points
        ]

        async with aiohttp.ClientSession() as session:
            responses = []
            for point in formatted_points:
                async with session.post(
                        vector_db_url,
                        json=point
                ) as response:
                    if response.status not in [200, 201]:
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"Error from vector database: {await response.text()}"
                        )
                    result = await response.json()
                    responses.append(result)

        return {
            "status": "success",
            "processed_points": len(points),
            "vector_db_responses": responses
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )