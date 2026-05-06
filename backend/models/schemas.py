"""Pydantic models for request/response schemas."""

from typing import Optional
from pydantic import BaseModel, Field


# ─── Chat ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = Field(None, description="UUID session — None untuk session baru")


class SourceDocument(BaseModel):
    file_name: str
    page: Optional[int] = None
    chunk_index: Optional[int] = None
    element_type: Optional[str] = None
    score: float
    text_preview: str


class DebugInfo(BaseModel):
    mode: str
    total_time_s: float
    top_score: Optional[float] = None
    similarity_top_k: Optional[int] = None
    reranker_top_n: Optional[int] = None
    sources_returned: Optional[int] = None
    model: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[SourceDocument] = []
    condensed_question: Optional[str] = None
    debug: Optional[DebugInfo] = None


# ─── Session ──────────────────────────────────────────────────────────────────

class SessionInfo(BaseModel):
    id: str
    title: Optional[str] = None
    message_count: int = 0
    last_message_at: Optional[str] = None


class SessionListResponse(BaseModel):
    sessions: list[SessionInfo]


# ─── Infrastructure ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    ollama: bool
    qdrant: bool
    postgres: bool = False
    redis: bool = False


class IndexRequest(BaseModel):
    directory: str = Field(default="data/pdfs")
    force: bool = Field(default=False)


class IndexResponse(BaseModel):
    status: str
    documents_indexed: int
