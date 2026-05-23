"""FastAPI application entry point."""

import logging
import os
import time
import uuid

import httpx
import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from qdrant_client import QdrantClient
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from backend.config import (
    LLM_BASE_URL, LLM_PROVIDER, QDRANT_URL, LLM_MODEL,
    ALLOWED_ORIGINS, LOG_LEVEL,
)
from backend.logging_config import configure_logging
from backend.limiter import limiter
from backend.models.schemas import HealthResponse, IndexRequest, IndexResponse
from backend.routers.chat import router as chat_router
from backend.routers.auth import router as auth_router
from backend.routers.query import router as query_router

configure_logging(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Chatbot UNHAS", version="2.0.0")

# ── Prometheus metrics — expose /metrics untuk scraping ───────────────────────
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
    logger.info("prometheus_metrics_enabled endpoint=/metrics")
except ImportError:
    logger.info("prometheus_instrumentator_not_installed — /metrics endpoint disabled")

# ── Rate limiting ──────────────────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# ── Request timing + correlation ID middleware ────────────────────────────────
@app.middleware("http")
async def log_request_timing(request: Request, call_next):
    # Gunakan X-Request-ID dari client jika ada, atau generate baru
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)

    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)

    logger.info(
        f"http_request method={request.method} path={request.url.path} "
        f"status={response.status_code} duration_ms={duration_ms} request_id={request_id}"
    )
    response.headers["X-Request-ID"] = request_id
    return response

app.include_router(chat_router)
app.include_router(auth_router)
app.include_router(query_router)

# ── Static files ──────────────────────────────────────────────────────────────
_frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
app.mount("/static", StaticFiles(directory=_frontend_dir), name="static")


@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(_frontend_dir, "index.html"))


@app.get("/api/files/{key:path}", tags=["infrastructure"], include_in_schema=False)
async def serve_storage_file(key: str):
    """Serve image yang di-upload user — hanya untuk STORAGE_BACKEND=filesystem (dev).

    Production (MinIO): user dapat presigned URL langsung ke MinIO, endpoint
    ini tidak terpakai.
    """
    from fastapi import HTTPException
    from fastapi.responses import Response

    if os.getenv("STORAGE_BACKEND", "filesystem") != "filesystem":
        raise HTTPException(404, "Endpoint hanya aktif untuk filesystem storage")

    if ".." in key or key.startswith("/"):
        raise HTTPException(400, "Invalid key")

    from backend.services.storage import get_storage
    try:
        data = get_storage().get_sync(key)
    except FileNotFoundError:
        raise HTTPException(404, "File tidak ditemukan")
    except Exception:
        raise HTTPException(500, "Gagal baca file")

    media = "image/jpeg" if key.endswith((".jpg", ".jpeg")) else "image/png"
    return Response(content=data, media_type=media)


@app.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Status semua service",
    tags=["infrastructure"],
)
async def health_check():
    """
    Cek status semua dependency service.

    - `ollama` / LLM: endpoint `/api/tags` atau `/v1/models`
    - `qdrant`: Qdrant vector store
    - `postgres`: PostgreSQL (session persistence)
    - `redis`: Redis cache
    - `intent_model`: file model IndoBERT di `INTENT_MODEL_PATH`

    Status `healthy` = ollama + qdrant + intent_model semua OK.
    Status `degraded` = salah satu service tidak tersedia.
    """
    ollama_ok = qdrant_ok = postgres_ok = redis_ok = intent_ok = False

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

    try:
        qc = QdrantClient(url=QDRANT_URL, timeout=5)
        qc.get_collections()
        qdrant_ok = True
    except Exception:
        pass

    try:
        from backend.db.database import check_db_connection
        postgres_ok = check_db_connection()
    except Exception:
        pass

    try:
        from backend.services.cache import check_redis_connection
        redis_ok = check_redis_connection()
    except Exception:
        pass

    try:
        from pathlib import Path
        from backend.config import INTENT_MODEL_PATH
        intent_ok = (Path(INTENT_MODEL_PATH) / "config.json").exists()
    except Exception:
        pass

    mod_circuit = "unknown"
    try:
        from backend.services.moderation import get_circuit_state
        mod_circuit = get_circuit_state()["state"]
    except Exception:
        pass

    vision_ok = False
    try:
        from backend.services.vision import vision_supported
        vision_ok = vision_supported()
    except Exception:
        pass

    status = "healthy" if (ollama_ok and qdrant_ok and intent_ok) else "degraded"
    return HealthResponse(
        status=status,
        ollama=ollama_ok,
        qdrant=qdrant_ok,
        postgres=postgres_ok,
        redis=redis_ok,
        intent_model=intent_ok,
        moderation_circuit=mod_circuit,
        vision_enabled=vision_ok,
    )


@app.post(
    "/api/index",
    response_model=IndexResponse,
    summary="Index dokumen PDF ke Qdrant",
    tags=["infrastructure"],
    responses={500: {"description": "Error saat indexing"}},
)
async def index_documents(request: IndexRequest):
    """
    Trigger indexing dokumen PDF ke Qdrant vector store.

    - Scan folder `directory` (default: `data/pdfs`) untuk file PDF baru.
    - Jika `force=true`, hapus semua chunk lama lalu index ulang.
    - Jika `force=false` (default), hanya index file yang belum ada di Qdrant.

    **Catatan:** Endpoint ini tidak memerlukan auth saat ini.
    Untuk indexing narasi JSON (data akademik UNHAS), gunakan script:
    `python backend/services/index_narratives.py`
    """
    from backend.services.indexing import index_documents as do_index
    try:
        count = do_index(request.directory, force=request.force)
        return IndexResponse(status="ok", documents_indexed=count)
    except Exception as e:
        logger.error(f"indexing_error error={e}")
        return IndexResponse(status=f"error: {e}", documents_indexed=0)


@app.on_event("startup")
async def startup_event():
    logger.info(f"RAG Chatbot UNHAS v2 starting llm={LLM_PROVIDER}/{LLM_MODEL} qdrant={QDRANT_URL}")

    try:
        from backend.db.database import init_db
        init_db()
    except Exception as e:
        logger.error(f"db_init_failed error={e}")
        logger.warning("Chatbot will run without session persistence.")

    try:
        from backend.services.auth import seed_users_from_json
        seed_users_from_json()
    except Exception as e:
        logger.warning(f"user_seed_skipped reason={e}")

    # Warm up L1 keyword filter — fail fast kalau YAML config rusak.
    try:
        from backend.services import keyword_filter
        keyword_filter.warm_up()
    except Exception as e:
        logger.error(f"keyword_filter_warmup_failed error={e}")
        raise

    # Warm up L3 intent classifier — fail soft (boleh degraded mode kalau model error).
    try:
        from backend.services.intent_classifier import warm_up as intent_warm_up
        intent_warm_up()
    except Exception as e:
        logger.warning(f"intent_classifier_warmup_failed error={e}")
