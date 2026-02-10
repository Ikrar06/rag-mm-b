"""Configuration for RAG Chatbot UNHAS."""

import os
from dotenv import load_dotenv

load_dotenv()

# === LLM (Multimodal VLM via Ollama) ===
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5vl:7b")  # Multimodal VLM
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))
LLM_REQUEST_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", "120"))

# === Embedding Model ===
EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBED_DEVICE = "cuda"
EMBED_BATCH_SIZE = 32
EMBED_DIMENSION = 1024

# === Re-ranker ===
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANKER_TOP_N = 3
RERANKER_USE_FP16 = True

# === Qdrant ===
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "unhas_docs")

# === RAG Pipeline ===
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
SIMILARITY_TOP_K = 10  # Ambil 10 dari Qdrant, re-rank jadi top 3

# === Paths ===
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "pdfs")
