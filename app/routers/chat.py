# app/routers/chat.py
from fastapi import APIRouter, HTTPException
from typing import List, Optional

from app.models.request_models import ChatRequest
from app.models.response_models import ChatResponse
from app.services.rag_processor import rag_processor
import logging

# print infor to console

logging.basicConfig(level=logging.INFO)
logging = logging.getLogger(__name__)


router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
# FastAPI route usage example
@router.post("/chat", response_model=ChatResponse)
async def chat_with_documents(request: ChatRequest):
    try:
        answer, sources, scores = await rag_processor.process_chat(
            question=request.question,
            chat_history=request.chat_history,
            similarity_threshold=request.similarity_threshold,
            top_k=request.top_k
        )

        return ChatResponse(
            answer=answer,
            sources=sources,
            similarity_scores=scores,
            processing_time=0.0  # You can add actual processing time if needed
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))