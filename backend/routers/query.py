"""Clean query API — untuk integrasi BE eksternal tanpa session management."""

import json
import logging

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from backend.limiter import limiter
from backend.config import RATE_LIMIT_TEXT_PER_MINUTE
from backend.models.schemas import QueryRequest, QueryResponse, SourceDocument, DebugInfo
from backend.services.rag_pipeline import query, query_stream
from backend.services.output_filter import filter_output

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["query"])


@router.post("/query", response_model=QueryResponse)
@limiter.limit(f"{RATE_LIMIT_TEXT_PER_MINUTE}/minute")
async def query_endpoint(request: Request, body: QueryRequest):
    """
    RAG query tanpa session management — untuk dipanggil BE eksternal.

    BE bertanggung jawab atas:
    - Autentikasi user (kirim role yang sesuai)
    - Penyimpanan history (kirim history di setiap request)
    - Session tracking

    Input:
    - question: pertanyaan user
    - history: riwayat percakapan [{"role": "user"|"assistant", "content": "..."}]
    - role: "public" | "mahasiswa" | "admin"
    """
    history = [m.model_dump() for m in body.history]

    result = query(
        question=body.question,
        history=history,
        role=body.role,
    )

    result["answer"] = filter_output(result["answer"])

    sources = [SourceDocument(**s) for s in result.get("sources", [])]
    debug = DebugInfo(**result["debug"]) if result.get("debug") else None

    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        condensed_question=result.get("condensed_question"),
        debug=debug,
    )


@router.post("/query/stream")
@limiter.limit(f"{RATE_LIMIT_TEXT_PER_MINUTE}/minute")
async def query_stream_endpoint(request: Request, body: QueryRequest):
    """
    Streaming variant dari /api/query — server-sent events.

    Event types:
    - {"type": "token", "delta": "..."}   — token per token
    - {"type": "meta", "answer": "...", "sources": [...], "debug": {...}}
    - {"type": "error", "message": "..."}
    """
    history = [m.model_dump() for m in body.history]

    def event_generator():
        answer_parts: list[str] = []
        meta_event: dict = {}

        try:
            for event in query_stream(body.question, history, body.role):
                if event["type"] == "token":
                    answer_parts.append(event.get("delta", ""))
                elif event["type"] == "meta":
                    event["answer"] = filter_output(event.get("answer", ""))
                    meta_event = event
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.error(f"query_stream_error error={e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
