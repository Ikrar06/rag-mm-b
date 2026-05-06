"""Semantic cache — Redis exact-match cache dengan graceful fallback.

Tier 1: exact match (SHA-256 hash of role:question)
Tier 2: semantic similarity — TODO untuk POC (butuh RedisSearch vector index)
TTL: 24 jam (info akademik bisa berubah)
Cache key include role: prevent cached answer untuk staf masuk ke mahasiswa

Fallback: kalau Redis tidak bisa diakses → NoOpCache (no caching, normal flow)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Optional

from backend.config import REDIS_URL, CACHE_ENABLED, CACHE_TTL_SECONDS

logger = logging.getLogger(__name__)


# ─── Redis client (lazy init) ─────────────────────────────────────────────────

_redis_client = None


def _get_redis():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        import redis
        client = redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)
        client.ping()
        _redis_client = client
        logger.info(f"Redis cache connected: {REDIS_URL}")
    except Exception as e:
        logger.warning(f"Redis not available ({e}) — cache disabled")
        _redis_client = None
    return _redis_client


# ─── Cache implementation ─────────────────────────────────────────────────────

def _cache_key(question: str, role: str) -> str:
    h = hashlib.sha256(f"{role}:{question}".encode()).hexdigest()[:16]
    return f"ragchat:cache:{h}"


def cache_get(question: str, role: str = "public") -> Optional[dict]:
    """Return cached result dict or None."""
    if not CACHE_ENABLED:
        return None
    r = _get_redis()
    if r is None:
        return None
    try:
        key = _cache_key(question, role)
        raw = r.get(key)
        if raw:
            logger.info(f"Cache hit: {question[:50]!r} (role={role})")
            return json.loads(raw)
    except Exception as e:
        logger.warning(f"Cache get error: {e}")
    return None


def cache_set(question: str, answer: str, sources: list, role: str = "public", debug: dict = None) -> None:
    """Store result in Redis cache."""
    if not CACHE_ENABLED:
        return
    r = _get_redis()
    if r is None:
        return
    try:
        key = _cache_key(question, role)
        data = {
            "answer": answer,
            "sources": sources,
            "debug": debug or {},
            "cached_at": time.time(),
        }
        r.setex(key, CACHE_TTL_SECONDS, json.dumps(data))
        logger.info(f"Cached: {question[:50]!r} (role={role}, TTL={CACHE_TTL_SECONDS}s)")
    except Exception as e:
        logger.warning(f"Cache set error: {e}")


def cache_invalidate(question: str, role: str = "public") -> None:
    """Hapus entry cache tertentu."""
    r = _get_redis()
    if r is None:
        return
    try:
        r.delete(_cache_key(question, role))
    except Exception as e:
        logger.warning(f"Cache invalidate error: {e}")


def check_redis_connection() -> bool:
    """Cek apakah Redis bisa diakses."""
    try:
        import redis
        r = redis.from_url(REDIS_URL, socket_connect_timeout=2)
        r.ping()
        return True
    except Exception:
        return False
