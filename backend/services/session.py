"""Session service — CRUD untuk sessions + messages di PostgreSQL."""

import uuid
import logging
from typing import Optional
from sqlalchemy.orm import Session as DBSession

from backend.db.models import Session, Message

logger = logging.getLogger(__name__)


def _to_json_safe(obj):
    """Recursively convert numpy/non-standard types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return obj
    # numpy float32/float64, int32/int64, etc.
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def create_session(db: DBSession, username: str, role: str, title: Optional[str] = None) -> Session:
    session = Session(
        user_username=username,
        user_role=role,
        title=title,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_session(db: DBSession, session_id: str, username: str) -> Optional[Session]:
    """Load session — pastikan milik user yang request (ownership check)."""
    try:
        sid = uuid.UUID(session_id)
    except ValueError:
        return None
    return (
        db.query(Session)
        .filter(Session.id == sid, Session.user_username == username)
        .first()
    )


def get_or_create_session(
    db: DBSession,
    session_id: Optional[str],
    username: str,
    role: str,
) -> Session:
    """Ambil session yang ada, atau buat baru kalau session_id None/tidak ditemukan."""
    if session_id:
        session = get_session(db, session_id, username)
        if session:
            return session
    return create_session(db, username, role)


def list_sessions(db: DBSession, username: str, limit: int = 20) -> list[Session]:
    return (
        db.query(Session)
        .filter(Session.user_username == username, Session.is_archived == False)  # noqa: E712
        .order_by(Session.last_message_at.desc())
        .limit(limit)
        .all()
    )


def get_history(db: DBSession, session_id: str, limit_turns: int = 5) -> list[dict]:
    """Ambil N turn terakhir (1 turn = 1 user message + 1 assistant message)."""
    try:
        sid = uuid.UUID(session_id)
    except ValueError:
        return []

    messages = (
        db.query(Message)
        .filter(Message.session_id == sid)
        .order_by(Message.created_at.desc())
        .limit(limit_turns * 2)  # 2 messages per turn
        .all()
    )
    messages.reverse()  # chronological order
    return [{"role": m.role, "content": m.content} for m in messages]


def save_user_message(
    db: DBSession,
    session_id: str,
    content: str,
    images: Optional[list[dict]] = None,
) -> Message:
    """Simpan user message. `images` adalah list of dict
    {"url": str, "mime_type": str, "size_bytes": int} dari hasil storage upload.
    """
    msg = Message(
        session_id=uuid.UUID(session_id),
        role="user",
        content=content,
        images=_to_json_safe(images) if images else None,
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


def save_assistant_message(
    db: DBSession,
    session_id: str,
    content: str,
    sources: Optional[list] = None,
    debug: Optional[dict] = None,
    mode: Optional[str] = None,
    condensed_question: Optional[str] = None,
) -> Message:
    msg = Message(
        session_id=uuid.UUID(session_id),
        role="assistant",
        content=content,
        sources=_to_json_safe(sources),
        debug=_to_json_safe(debug),
        mode=mode,
        condensed_question=condensed_question,
    )
    db.add(msg)

    # Update session metadata
    session = db.query(Session).filter(Session.id == uuid.UUID(session_id)).first()
    if session:
        from datetime import datetime, timezone
        session.last_message_at = datetime.now(timezone.utc)
        session.message_count = (session.message_count or 0) + 1

    db.commit()
    db.refresh(msg)
    return msg
