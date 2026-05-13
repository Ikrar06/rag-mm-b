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
    mode: str = Field(
        ...,
        description=(
            "Mode jawaban: rag | chitchat | out_of_scope | blocked | "
            "blocked_moderation | clarification_needed | get_info_private | "
            "cache_hit | rag_low_relevance"
        ),
    )
    total_time_s: float = Field(..., description="Total waktu proses dalam detik")
    top_score: Optional[float] = Field(None, description="Skor relevansi tertinggi dari reranker (0–1)")
    similarity_top_k: Optional[int] = Field(None, description="Jumlah dokumen yang diambil dari vector store")
    reranker_top_n: Optional[int] = Field(None, description="Jumlah dokumen setelah reranking")
    sources_returned: Optional[int] = Field(None, description="Jumlah sumber yang dikembalikan")
    model: Optional[str] = Field(None, description="Nama model LLM yang digunakan")
    intent: Optional[str] = Field(
        None,
        description="Hasil L3 intent classifier: chitchat | out_of_scope | get_info_public | get_info_private",
    )
    intent_confidence: Optional[float] = Field(
        None, description="Skor kepercayaan intent classifier (0–1)"
    )


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


# ─── Clean Query API (untuk integrasi BE eksternal, tanpa session) ────────────

class HistoryMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$", description="Pengirim: 'user' atau 'assistant'")
    content: str = Field(..., min_length=1, max_length=4000, description="Isi pesan")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="Pertanyaan dari user")
    history: list[HistoryMessage] = Field(
        default_factory=list,
        max_length=20,
        description=(
            "Riwayat percakapan sebelumnya dalam format [{role, content}]. "
            "BE bertanggung jawab menyimpan dan mengirim ini di setiap request."
        ),
    )
    role: str = Field(
        default="public",
        pattern="^(public|mahasiswa|admin)$",
        description=(
            "Role user terverifikasi dari BE. "
            "public: tamu/calon mahasiswa. "
            "mahasiswa: mahasiswa terdaftar (dapat akses get_info_private). "
            "admin: staf akademik."
        ),
    )


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument] = []
    condensed_question: Optional[str] = None
    debug: Optional[DebugInfo] = None


# ─── Infrastructure ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    ollama: bool
    qdrant: bool
    postgres: bool = False
    redis: bool = False
    intent_model: bool = False


class IndexRequest(BaseModel):
    directory: str = Field(default="data/pdfs")
    force: bool = Field(default=False)


class IndexResponse(BaseModel):
    status: str
    documents_indexed: int
