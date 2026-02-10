"""RAG pipeline service — orchestrate retrieval, rerank, and generation."""

import logging
import time

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore

from backend.config import (
    OLLAMA_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_REQUEST_TIMEOUT,
    EMBED_MODEL_NAME,
    EMBED_DEVICE,
    EMBED_BATCH_SIZE,
    RERANKER_MODEL_NAME,
    RERANKER_TOP_N,
    QDRANT_COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SIMILARITY_TOP_K,
)
from backend.prompts.templates import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT, CHITCHAT_SYSTEM_PROMPT
from backend.services.indexing import get_qdrant_client

logger = logging.getLogger(__name__)

# Singleton: models loaded once at startup
_query_engine = None

# Kata kunci chitchat sederhana — skip RAG, langsung ke LLM
_CHITCHAT_KEYWORDS = {
    "halo", "hai", "hello", "hi", "hey",
    "selamat pagi", "selamat siang", "selamat sore", "selamat malam",
    "terima kasih", "makasih", "thanks", "thank you",
    "apa kabar", "siapa kamu", "siapa anda",
    "bye", "dadah", "sampai jumpa",
}


def _configure_settings():
    """Configure LlamaIndex global settings."""
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        device=EMBED_DEVICE,
        trust_remote_code=True,
        embed_batch_size=EMBED_BATCH_SIZE,
    )
    Settings.llm = Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=LLM_TEMPERATURE,
        request_timeout=LLM_REQUEST_TIMEOUT,
        system_prompt=RAG_SYSTEM_PROMPT,
    )
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP


def get_query_engine():
    """Get or create the query engine (lazy init, singleton)."""
    global _query_engine
    if _query_engine is not None:
        return _query_engine

    logger.info("Initializing RAG pipeline...")
    _configure_settings()

    qdrant_client = get_qdrant_client()
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=QDRANT_COLLECTION_NAME,
    )

    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    reranker = SentenceTransformerRerank(
        model=RERANKER_MODEL_NAME,
        top_n=RERANKER_TOP_N,
    )

    _query_engine = index.as_query_engine(
        similarity_top_k=SIMILARITY_TOP_K,
        node_postprocessors=[reranker],
        text_qa_template=None,  # uses default with system_prompt from Settings.llm
    )

    logger.info("RAG pipeline ready.")
    return _query_engine


def _is_chitchat(question: str) -> bool:
    """Deteksi apakah pertanyaan adalah sapaan/chitchat sederhana."""
    q = question.strip().lower().rstrip("?!.,")
    return q in _CHITCHAT_KEYWORDS


def _chitchat_response(question: str) -> str:
    """Jawab chitchat langsung via LLM tanpa RAG."""
    llm = Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=LLM_TEMPERATURE,
        request_timeout=LLM_REQUEST_TIMEOUT,
    )
    response = llm.complete(
        f"System: {CHITCHAT_SYSTEM_PROMPT}\n\nPertanyaan: {question}\n\nJawaban:"
    )
    return str(response)


def query(question: str) -> dict:
    """Run RAG query and return answer + sources.

    Chitchat (sapaan, terima kasih, dll) langsung ke LLM tanpa RAG pipeline.

    Returns:
        dict with keys: answer (str), sources (list of dicts)
    """
    t_start = time.time()

    # Chitchat shortcut — skip embedding, search, re-rank
    if _is_chitchat(question):
        logger.info(f"Chitchat detected: '{question}' — skipping RAG")
        answer = _chitchat_response(question)
        t_total = round(time.time() - t_start, 2)
        logger.info(f"Chitchat response in {t_total}s")
        return {
            "answer": answer,
            "sources": [],
            "debug": {
                "mode": "chitchat",
                "total_time_s": t_total,
            },
        }

    engine = get_query_engine()
    response = engine.query(question)
    t_total = round(time.time() - t_start, 2)

    sources = []
    for node in response.source_nodes:
        meta = node.metadata
        sources.append({
            "file_name": meta.get("file_name", "unknown"),
            "page": meta.get("page"),
            "chunk_index": meta.get("chunk_index"),
            "element_type": meta.get("element_type"),
            "score": round(node.score, 4) if node.score else 0.0,
            "text_preview": node.text[:300],
        })

    debug_info = {
        "mode": "rag",
        "total_time_s": t_total,
        "similarity_top_k": SIMILARITY_TOP_K,
        "reranker_top_n": RERANKER_TOP_N,
        "sources_returned": len(sources),
        "model": LLM_MODEL,
    }
    logger.info(f"RAG query completed in {t_total}s — {len(sources)} sources")

    return {
        "answer": str(response),
        "sources": sources,
        "debug": debug_info,
    }
