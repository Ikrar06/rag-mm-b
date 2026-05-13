"""Chat API router — session, auth, multi-turn history, rate limiting."""

import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session as DBSession

from backend.db.database import get_db, SessionLocal
from backend.limiter import limiter
from backend.config import RATE_LIMIT_TEXT_PER_MINUTE
from backend.models.schemas import (
    ChatRequest, ChatResponse, SourceDocument, DebugInfo,
    SessionListResponse, SessionInfo,
)
from backend.services.rag_pipeline import query, query_stream
from backend.services.auth import decode_token
from backend.services.output_filter import filter_output
from backend.services.session import (
    get_or_create_session, get_history, save_user_message, save_assistant_message,
    list_sessions,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])

_COOKIE_NAME = "access_token"


def _get_current_user(request: Request) -> dict:
    """Extract user dari JWT cookie/header. Default: anonymous public."""
    token = request.cookies.get(_COOKIE_NAME)
    if not token:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]

    if token:
        payload = decode_token(token)
        if payload:
            return {
                "username": payload.get("sub", "anonymous"),
                "role": payload.get("role", "public"),
                "name": payload.get("name", "Anonymous"),
                "token": token,
            }

    return {"username": "anonymous", "role": "public", "name": "Anonymous", "token": ""}


@router.post("/chat", response_model=ChatResponse)
@limiter.limit(f"{RATE_LIMIT_TEXT_PER_MINUTE}/minute")
async def chat(request: Request, request_body: ChatRequest, db: DBSession = Depends(get_db)):
    """Main RAG chat endpoint dengan session + multi-turn history."""
    user = _get_current_user(request)

    try:
        session = get_or_create_session(
            db,
            session_id=request_body.session_id,
            username=user["username"],
            role=user["role"],
        )
        session_id_str = str(session.id)
        history = get_history(db, session_id_str)
        save_user_message(db, session_id_str, request_body.query)

        result = query(
            question=request_body.query,
            history=history,
            role=user["role"],
        )

        # Layer 6 — output filter (backstop, RAG pipeline sudah filter tapi dobel aman)
        result["answer"] = filter_output(result["answer"], session_id_str)

        save_assistant_message(
            db,
            session_id=session_id_str,
            content=result["answer"],
            sources=result.get("sources"),
            debug=result.get("debug"),
            mode=result["debug"].get("mode") if result.get("debug") else None,
            condensed_question=result.get("condensed_question"),
        )

        sources = [SourceDocument(**s) for s in result.get("sources", [])]
        debug = DebugInfo(**result["debug"]) if result.get("debug") else None

        return ChatResponse(
            answer=result["answer"],
            session_id=session_id_str,
            sources=sources,
            condensed_question=result.get("condensed_question"),
            debug=debug,
        )

    except Exception as e:
        logger.error(f"chat_error error={e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
@limiter.limit(f"{RATE_LIMIT_TEXT_PER_MINUTE}/minute")
async def chat_stream(request: Request, request_body: ChatRequest):
    """Streaming chat endpoint — server-sent events."""
    user = _get_current_user(request)

    setup_db = SessionLocal()
    try:
        session = get_or_create_session(
            setup_db,
            session_id=request_body.session_id,
            username=user["username"],
            role=user["role"],
        )
        session_id_str = str(session.id)
        history = get_history(setup_db, session_id_str)
        save_user_message(setup_db, session_id_str, request_body.query)
    except Exception as e:
        setup_db.close()
        logger.error(f"stream_setup_error error={e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        setup_db.close()

    def event_generator():
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id_str})}\n\n"

        answer_parts: list[str] = []
        meta_event: dict = {}

        try:
            for event in query_stream(request_body.query, history, user["role"]):
                if event["type"] == "token":
                    answer_parts.append(event.get("delta", ""))
                elif event["type"] == "meta":
                    # Filter final answer in meta event
                    event["answer"] = filter_output(event.get("answer", ""), session_id_str)
                    meta_event = event
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.error(f"stream_error error={e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            full_answer = meta_event.get("answer") or "".join(answer_parts)
            if full_answer:
                save_db = SessionLocal()
                try:
                    save_assistant_message(
                        save_db,
                        session_id=session_id_str,
                        content=full_answer,
                        sources=meta_event.get("sources", []),
                        debug=meta_event.get("debug"),
                        mode=(meta_event.get("debug") or {}).get("mode"),
                        condensed_question=meta_event.get("condensed_question"),
                    )
                except Exception as e:
                    logger.error(f"stream_save_error error={e}")
                finally:
                    save_db.close()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/sessions", response_model=SessionListResponse)
async def get_sessions(request: Request, db: DBSession = Depends(get_db)):
    user = _get_current_user(request)
    if user["username"] == "anonymous":
        return SessionListResponse(sessions=[])

    sessions = list_sessions(db, user["username"])
    return SessionListResponse(sessions=[
        SessionInfo(
            id=str(s.id),
            title=s.title,
            message_count=s.message_count or 0,
            last_message_at=s.last_message_at.isoformat() if s.last_message_at else None,
        )
        for s in sessions
    ])
