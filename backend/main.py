"""FastAPI application entry point."""

import logging
import os

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from qdrant_client import QdrantClient

from backend.config import LLM_BASE_URL, LLM_PROVIDER, QDRANT_URL, LLM_MODEL
from backend.models.schemas import HealthResponse, IndexRequest, IndexResponse
from backend.routers.chat import router as chat_router
from backend.routers.auth import router as auth_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Chatbot UNHAS", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(chat_router)
app.include_router(auth_router)

# Serve frontend static files
_frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
app.mount("/static", StaticFiles(directory=_frontend_dir), name="static")


@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(_frontend_dir, "index.html"))


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    ollama_ok = False
    qdrant_ok = False
    postgres_ok = False
    redis_ok = False

    # Check LLM endpoint (Ollama atau vLLM)
    try:
        check_url = (
            f"{LLM_BASE_URL}/api/tags" if LLM_PROVIDER == "ollama"
            else f"{LLM_BASE_URL}/v1/models"
        )
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(check_url)
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

    # Check PostgreSQL
    try:
        from backend.db.database import check_db_connection
        postgres_ok = check_db_connection()
    except Exception:
        pass

    # Check Redis
    try:
        from backend.services.cache import check_redis_connection
        redis_ok = check_redis_connection()
    except Exception:
        pass

    status = "healthy" if (ollama_ok and qdrant_ok) else "degraded"
    return HealthResponse(
        status=status,
        ollama=ollama_ok,
        qdrant=qdrant_ok,
        postgres=postgres_ok,
        redis=redis_ok,
    )


@app.post("/api/index", response_model=IndexResponse)
async def index_documents(request: IndexRequest):
    from backend.services.indexing import index_documents as do_index
    try:
        count = do_index(request.directory, force=request.force)
        return IndexResponse(status="ok", documents_indexed=count)
    except Exception as e:
        logger.error(f"Indexing error: {e}")
        return IndexResponse(status=f"error: {e}", documents_indexed=0)


@app.on_event("startup")
async def startup_event():
    logger.info("RAG Chatbot UNHAS v2 starting...")
    logger.info(f"LLM: {LLM_PROVIDER} / {LLM_MODEL}")
    logger.info(f"Qdrant: {QDRANT_URL}")

    # Init database tables
    try:
        from backend.db.database import init_db
        init_db()
    except Exception as e:
        logger.error(f"DB init failed: {e}")
        logger.warning("Chatbot will run without session persistence.")
