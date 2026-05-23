"""Clean query API — untuk integrasi BE eksternal tanpa session management."""

import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from backend.limiter import limiter
from backend.config import RATE_LIMIT_TEXT_PER_MINUTE
from backend.models.schemas import QueryRequest, QueryResponse, SourceDocument, DebugInfo
from backend.services.rag_pipeline import query, query_stream
from backend.services.output_filter import filter_output
from backend.services.vision import validate_and_process, VisionError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="RAG query untuk integrasi BE eksternal (Tim 1)",
    responses={
        429: {"description": "Rate limit terlampaui (20 req/menit per IP)"},
        422: {"description": "Validasi input gagal"},
    },
)
@limiter.limit(f"{RATE_LIMIT_TEXT_PER_MINUTE}/minute")
async def query_endpoint(request: Request, body: QueryRequest):
    """
    Endpoint RAG tanpa session management — dirancang untuk **Tim 1 (BE)**.

    **Tanggung jawab BE pemanggil:**
    - Autentikasi user sendiri → kirim `role` yang sesuai
    - Simpan history di DB sendiri → kirim kembali di setiap request
    - Tidak ada JWT/cookie yang dibutuhkan di endpoint ini

    **Alur integrasi:**
    ```
    FE → BE Tim 1 → POST /api/query → RAG Core → jawaban
    ```

    **Field `role`:**
    - `public` — tamu / calon mahasiswa (default)
    - `mahasiswa` — mahasiswa terdaftar, dapat akses `get_info_private`
    - `admin` — staf akademik

    **Field `images`** (opsional):
    - List of `{"mime_type": "image/jpeg|png|webp", "data": "<base64>"}`
    - Max 2 gambar per request, max 10 MB per gambar.
    - Auto-resize ke 1280px supaya hemat token VL.
    - Hanya aktif kalau backend dikonfigurasi `LLM_SUPPORTS_VISION=true`.

    **Field `debug.mode`** menjelaskan jalur jawaban:
    - `rag` — dari dokumen RAG
    - `vision_rag` — RAG + analisis gambar via Qwen3-VL
    - `vision_error` — gambar invalid atau LLM vision gagal
    - `chitchat` — sapaan / basa-basi
    - `out_of_scope` — di luar topik akademik UNHAS
    - `get_info_private` — dari API UNHAS (aktif jika `UNHAS_API_BASE_URL` dikonfigurasi)
    - `cache_hit` — dari cache Redis (jawaban identik sebelumnya)
    - `blocked` — diblokir keyword berbahaya
    - `rag_low_relevance` — tidak ada dokumen relevan ditemukan
    """
    history = [m.model_dump() for m in body.history]

    try:
        processed_images = validate_and_process(
            [img.model_dump() for img in body.images]
        )
    except VisionError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = query(
        question=body.question,
        history=history,
        role=body.role,
        images=processed_images or None,
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


@router.post(
    "/query/stream",
    summary="RAG query streaming untuk BE eksternal (SSE)",
    responses={429: {"description": "Rate limit terlampaui"}},
)
@limiter.limit(f"{RATE_LIMIT_TEXT_PER_MINUTE}/minute")
async def query_stream_endpoint(request: Request, body: QueryRequest):
    """
    Versi streaming dari `/api/query` menggunakan **Server-Sent Events (SSE)**.

    - Request body sama persis dengan `/api/query`.
    - Tidak memerlukan auth/session.

    **Format event stream:**
    ```
    data: {"type": "token", "delta": "Untuk "}
    data: {"type": "token", "delta": "mengajukan "}
    ...
    data: {"type": "meta", "answer": "...", "sources": [...], "debug": {...}, "condensed_question": "..."}
    ```
    Error: `data: {"type": "error", "message": "..."}`

    **Catatan implementasi:** consume stream sampai event `meta` diterima,
    lalu gunakan `meta.answer` sebagai jawaban final (sudah difilter output_filter).
    Token individual belum difilter — gunakan hanya untuk tampilan live streaming.
    """
    history = [m.model_dump() for m in body.history]

    try:
        processed_images = validate_and_process(
            [img.model_dump() for img in body.images]
        )
    except VisionError as e:
        raise HTTPException(status_code=400, detail=str(e))

    def event_generator():
        answer_parts: list[str] = []
        meta_event: dict = {}

        try:
            for event in query_stream(
                body.question, history, body.role,
                images=processed_images or None,
            ):
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
