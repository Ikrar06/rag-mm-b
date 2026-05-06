"""Configuration for RAG Chatbot UNHAS — v2 POC-ready."""

import os
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

# === Paths ===
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "pdfs")
IMAGES_DIR = os.path.join(_PROJECT_ROOT, "data", "images")

# =============================================================================
# LLM — Switchable provider: "ollama" (dev) atau "vllm" (POC)
# =============================================================================
# Dev (RTX 3060): LLM_PROVIDER=ollama, LLM_MODEL=qwen2.5:7b via Ollama
# POC (L40S):     LLM_PROVIDER=vllm,   LLM_MODEL=Qwen/Qwen3-VL-8B-Instruct via vLLM
# =============================================================================

LLM_PROVIDER: Literal["ollama", "vllm", "openai"] = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
LLM_API_KEY = os.getenv("LLM_API_KEY", "not-needed")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_REQUEST_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", "120"))
LLM_SUPPORTS_VISION = os.getenv("LLM_SUPPORTS_VISION", "false").lower() == "true"

# Backward-compat alias
OLLAMA_BASE_URL = LLM_BASE_URL

# =============================================================================
# Embedding — Switchable provider: "huggingface" (dev, in-process) atau "tei" (POC)
# =============================================================================
# Dev:  in-process HuggingFace model
# POC:  Text Embeddings Inference (TEI) HTTP service — dikelola IOH
# =============================================================================

EMBED_PROVIDER: Literal["huggingface", "tei"] = os.getenv("EMBED_PROVIDER", "huggingface")
EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "")   # hanya dipakai kalau EMBED_PROVIDER=tei
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cuda")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
EMBED_DIMENSION = 1024

# Backward-compat aliases
EMBED_MODEL_NAME = EMBED_MODEL

# =============================================================================
# Re-ranker — Switchable provider: "sentence_transformers" (dev) atau "tei" (POC)
# =============================================================================

RERANKER_PROVIDER: Literal["sentence_transformers", "tei"] = os.getenv("RERANKER_PROVIDER", "sentence_transformers")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", os.path.join(_PROJECT_ROOT, "models", "bge-reranker-v2-m3"))
RERANKER_BASE_URL = os.getenv("RERANKER_BASE_URL", "")   # hanya dipakai kalau RERANKER_PROVIDER=tei
RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", "4"))
RERANKER_USE_FP16 = True

# Backward-compat alias
RERANKER_MODEL_NAME = RERANKER_MODEL

# =============================================================================
# Qdrant
# =============================================================================

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", os.getenv("QDRANT_COLLECTION_NAME", "unhas_docs"))
QDRANT_COLLECTION_NAME = QDRANT_COLLECTION  # backward-compat

# =============================================================================
# PaddleOCR
# =============================================================================

OCR_LANG = "id"
OCR_USE_GPU = True

# =============================================================================
# Chunking
# =============================================================================

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# =============================================================================
# RAG Pipeline
# =============================================================================

SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "8"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.3"))

# =============================================================================
# Conversation history (multi-turn)
# =============================================================================

HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "5"))
HISTORY_MAX_TOKENS = int(os.getenv("HISTORY_MAX_TOKENS", "1500"))

# =============================================================================
# Vision — aktif saat LLM_SUPPORTS_VISION=true (POC dengan Qwen3-VL)
# =============================================================================

MAX_IMAGES_PER_MESSAGE = int(os.getenv("MAX_IMAGES_PER_MESSAGE", "2"))
MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
IMAGE_RESIZE_MAX_DIM = int(os.getenv("IMAGE_RESIZE_MAX_DIM", "1280"))

# =============================================================================
# PostgreSQL — sessions, messages, audit
# Dev: postgresql://ragchat:dev@localhost:5432/ragchat  (lihat docker-compose.dev.yml)
# POC: postgresql+asyncpg://... (IOH provision)
# =============================================================================

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://ragchat:dev@localhost:5432/ragchat")

# =============================================================================
# Redis — cache + rate limit + circuit breaker
# Dev: redis://localhost:6379/0  (lihat docker-compose.dev.yml)
# POC: redis://redis:6379/0
# =============================================================================

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))

# =============================================================================
# Auth (JWT)
# =============================================================================

JWT_SECRET = os.getenv("JWT_SECRET", "unhas-rag-demo-secret-2026")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "24"))

# =============================================================================
# POC-only — Uncomment saat deploy ke L40S / production
# =============================================================================

# --- Storage (MinIO untuk POC, filesystem untuk dev) ---
# STORAGE_BACKEND: Literal["filesystem", "minio"] = os.getenv("STORAGE_BACKEND", "filesystem")
# STORAGE_LOCAL_PATH = os.getenv("STORAGE_LOCAL_PATH", "./storage")
# MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
# MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "")
# MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "")
# MINIO_BUCKET = os.getenv("MINIO_BUCKET", "ragchat-images")

# --- Rate limiting (production scale) ---
# RATE_LIMIT_TEXT_PER_MINUTE = int(os.getenv("RATE_LIMIT_TEXT_PER_MINUTE", "8"))
# RATE_LIMIT_VISION_PER_MINUTE = int(os.getenv("RATE_LIMIT_VISION_PER_MINUTE", "2"))
# RATE_LIMIT_BURST_PER_SECOND = int(os.getenv("RATE_LIMIT_BURST_PER_SECOND", "3"))

# --- Circuit breaker ---
# CIRCUIT_OPEN_THRESHOLD = int(os.getenv("CIRCUIT_OPEN_THRESHOLD", "60"))
# CIRCUIT_RECOVERY_THRESHOLD = int(os.getenv("CIRCUIT_RECOVERY_THRESHOLD", "20"))
