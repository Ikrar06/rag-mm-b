"""Chat API router."""

import logging

from fastapi import APIRouter, HTTPException

from backend.models.schemas import ChatRequest, ChatResponse, SourceDocument
from backend.services.rag_pipeline import query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main RAG chat endpoint."""
    try:
        result = query(request.query)
        sources = [
            SourceDocument(**s) for s in result["sources"]
        ]
        return ChatResponse(answer=result["answer"], sources=sources)
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
