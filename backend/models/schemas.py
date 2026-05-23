"""Pydantic models for request/response schemas."""

from typing import Optional
from pydantic import BaseModel, Field


# ─── Chat ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Pesan / pertanyaan dari user")
    session_id: Optional[str] = Field(
        None,
        description="UUID session percakapan. Kosongkan (null) untuk memulai session baru. "
                    "UUID dikembalikan di field session_id pada response.",
    )


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
    answer: str = Field(..., description="Jawaban dari RAG chatbot")
    session_id: str = Field(..., description="UUID session yang digunakan — simpan untuk request berikutnya")
    sources: list[SourceDocument] = Field(default=[], description="Dokumen sumber yang digunakan untuk menjawab")
    condensed_question: Optional[str] = Field(None, description="Pertanyaan setelah dikondensasi dari history (multi-turn)")
    debug: Optional[DebugInfo] = Field(None, description="Info debug — mode, waktu, skor relevansi, intent")


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
    # Circuit breaker state: "closed" (normal), "open" (failing), "half_open" (probing)
    moderation_circuit: str = "closed"


class IndexRequest(BaseModel):
    directory: str = Field(default="data/pdfs", description="Path folder berisi file PDF yang akan di-index")
    force: bool = Field(default=False, description="Jika true, hapus index lama dan index ulang semua file")


class IndexResponse(BaseModel):
    status: str = Field(..., description="'ok' jika berhasil, atau pesan error")
    documents_indexed: int = Field(..., description="Jumlah dokumen yang berhasil di-index")
