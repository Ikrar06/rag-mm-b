"""FastAPI application entry point."""

import logging

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient

from backend.config import OLLAMA_BASE_URL, QDRANT_URL, LLM_MODEL
from backend.models.schemas import HealthResponse, IndexRequest, IndexResponse
from backend.routers.chat import router as chat_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Chatbot UNHAS", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Check if Ollama and Qdrant are reachable."""
    ollama_ok = False
    qdrant_ok = False

    # Check Ollama
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            ollama_ok = resp.status_code == 200
    except Exception:
        pass

    # Check Qdrant
    try:
        qc = QdrantClient(url=QDRANT_URL, timeout=5)
        qc.get_collections()
        qdrant_ok = True
    except Exception:
        pass

    status = "healthy" if (ollama_ok and qdrant_ok) else "degraded"
    return HealthResponse(status=status, ollama=ollama_ok, qdrant=qdrant_ok)


@app.post("/api/index", response_model=IndexResponse)
async def index_documents(request: IndexRequest):
    """Trigger document indexing to Qdrant."""
    from backend.services.indexing import index_documents as do_index

    try:
        count = do_index(request.directory)
        return IndexResponse(status="ok", documents_indexed=count)
    except Exception as e:
        logger.error(f"Indexing error: {e}")
        return IndexResponse(status=f"error: {e}", documents_indexed=0)


@app.on_event("startup")
async def startup_event():
    logger.info(f"RAG Chatbot UNHAS starting...")
    logger.info(f"LLM model: {LLM_MODEL}")
    logger.info(f"Ollama: {OLLAMA_BASE_URL}")
    logger.info(f"Qdrant: {QDRANT_URL}")
