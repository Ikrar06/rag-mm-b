"""Pydantic models for request/response schemas."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Pertanyaan user")


class SourceDocument(BaseModel):
    file_name: str
    page: int | None = None
    chunk_index: int | None = None
    element_type: str | None = None
    score: float
    text_preview: str = Field(description="Potongan teks sumber (max 300 char)")


class DebugInfo(BaseModel):
    mode: str  # "rag" or "chitchat"
    total_time_s: float
    similarity_top_k: int | None = None
    reranker_top_n: int | None = None
    sources_returned: int | None = None
    model: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceDocument] = []
    debug: DebugInfo | None = None


class HealthResponse(BaseModel):
    status: str
    ollama: bool
    qdrant: bool


class IndexRequest(BaseModel):
    directory: str = Field(default="data/pdfs", description="Path ke folder PDF")
    force: bool = Field(default=False, description="Hapus collection lama sebelum re-index")


class IndexResponse(BaseModel):
    status: str
    documents_indexed: int
