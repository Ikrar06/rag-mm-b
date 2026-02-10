"""Pydantic models for request/response schemas."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Pertanyaan user")


class SourceDocument(BaseModel):
    file_name: str
    page: int | None = None
    score: float
    text_preview: str = Field(description="Potongan teks sumber (max 200 char)")


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceDocument] = []


class HealthResponse(BaseModel):
    status: str
    ollama: bool
    qdrant: bool


class IndexRequest(BaseModel):
    directory: str = Field(default="data/pdfs", description="Path ke folder PDF")


class IndexResponse(BaseModel):
    status: str
    documents_indexed: int
