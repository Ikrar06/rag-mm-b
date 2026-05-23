"""Pydantic models for request/response schemas."""

from typing import Optional
from pydantic import BaseModel, Field


# ─── Image input ──────────────────────────────────────────────────────────────

class ImageInput(BaseModel):
    """Image attachment di chat request — base64-encoded.

    mime_type harus salah satu dari ALLOWED_IMAGE_TYPES di config.
    data adalah base64 string (TANPA prefix "data:image/...;base64,").
    """
    mime_type: str = Field(
        ...,
        pattern=r"^image/(jpeg|jpg|png|webp)$",
        description="MIME type gambar. Hanya jpeg/png/webp yang diterima.",
    )
    data: str = Field(
        ...,
        min_length=20,
        description=(
            "Base64-encoded image data tanpa prefix 'data:image/...;base64,'. "
            "Max size MAX_IMAGE_SIZE_MB MB setelah decode."
        ),
    )


# ─── Chat ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Pesan / pertanyaan dari user")
    session_id: Optional[str] = Field(
        None,
        description="UUID session percakapan. Kosongkan (null) untuk memulai session baru. "
                    "UUID dikembalikan di field session_id pada response.",
    )
    images: list[ImageInput] = Field(
        default_factory=list,
        description=(
            "Lampiran gambar (KRS, KTM, formulir, dll). Max MAX_IMAGES_PER_MESSAGE "
            "gambar per request. Hanya aktif kalau LLM_SUPPORTS_VISION=true."
        ),
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
            "Mode jawaban: rag | vision_rag | vision_error | chitchat | "
            "out_of_scope | blocked | blocked_moderation | clarification_needed | "
            "get_info_private | cache_hit | rag_low_relevance"
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
    confidence_band: Optional[str] = Field(
        None, description="high | marginal | low (lihat LOW_CONFIDENCE_BUFFER)"
    )
    has_images: Optional[bool] = Field(None, description="True kalau request punya image attachment")
    image_count: Optional[int] = Field(None, description="Jumlah image yang diproses di vision mode")


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
    images: list[ImageInput] = Field(
        default_factory=list,
        description=(
            "Lampiran gambar (KRS, KTM, formulir, dll). Max MAX_IMAGES_PER_MESSAGE "
            "gambar per request. Hanya aktif kalau LLM_SUPPORTS_VISION=true."
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
    # True kalau backend support vision (LLM_SUPPORTS_VISION=true + LLM_PROVIDER=vllm)
    vision_enabled: bool = False


class IndexRequest(BaseModel):
    directory: str = Field(default="data/pdfs", description="Path folder berisi file PDF yang akan di-index")
    force: bool = Field(default=False, description="Jika true, hapus index lama dan index ulang semua file")


class IndexResponse(BaseModel):
    status: str = Field(..., description="'ok' jika berhasil, atau pesan error")
    documents_indexed: int = Field(..., description="Jumlah dokumen yang berhasil di-index")
