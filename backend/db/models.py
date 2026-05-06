"""SQLAlchemy ORM models — sessions, messages."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, String, Text, Boolean, Integer, DateTime,
    ForeignKey, JSON,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from backend.db.database import Base


def _now():
    return datetime.now(timezone.utc)


class Session(Base):
    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_username = Column(String(100), nullable=False, index=True)
    user_role = Column(String(30), nullable=False, default="public")
    title = Column(String(200), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)
    last_message_at = Column(DateTime(timezone=True), default=_now)
    is_archived = Column(Boolean, default=False)
    message_count = Column(Integer, default=0)

    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan",
                            order_by="Message.created_at")


class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"),
                        nullable=False, index=True)
    role = Column(String(20), nullable=False)         # "user" | "assistant"
    content = Column(Text, nullable=False)
    sources = Column(JSON, nullable=True)             # list of source dicts
    debug = Column(JSON, nullable=True)               # debug info dict
    mode = Column(String(30), nullable=True)          # rag|chitchat|blocked|cache_hit|...
    condensed_question = Column(Text, nullable=True)  # hasil query condensation
    created_at = Column(DateTime(timezone=True), default=_now)

    session = relationship("Session", back_populates="messages")
