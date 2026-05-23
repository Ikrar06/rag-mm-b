"""Layer 1 — Keyword filter dengan hard-block + soft-flag + safe context.

Replace _is_harmful sederhana sebelumnya. Pipeline:
    raw_input
       → normalize (NFKC, lowercase, homoglyph map, leet, strip zero-width)
       → hard_block check          → kalau match: BLOCK
       → soft_flag check
           → kalau ada safe_context: PASS (downgrade)
           → kalau tidak: FLAG (lanjut ke L2 dengan info flag)
       → kalau tidak ada flag/block: PASS

Keyword list di-load dari YAML config, di-cache 5 menit untuk hot-reload
tanpa restart service.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

logger = logging.getLogger(__name__)

_KEYWORDS_PATH = Path(__file__).resolve().parent / "blocked_keywords.yaml"
_RELOAD_INTERVAL_SEC = 300  # 5 menit

# Mapping leet → normal. Konservatif: hanya substitusi yang aman antar-konteks
# (mis. "0" → "o" tidak rusak query "level 0" karena context window beda).
_LEET_MAP = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a",
    "5": "s", "7": "t", "@": "a", "$": "s",
})

# Homoglyph: char Cyrillic/Greek yang look-alike Latin.
_HOMOGLYPH_MAP = {
    "а": "a",  # Cyrillic а
    "ο": "o",  # Greek ο
    "е": "e",  # Cyrillic е
    "р": "p",  # Cyrillic р
    "с": "c",  # Cyrillic с
    "х": "x",  # Cyrillic х
    "о": "o",  # Cyrillic о
}

# Zero-width & invisible chars yang sering dipakai bypass.
_ZERO_WIDTH_RE = re.compile(r"[​‌‍﻿⁠]")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class KeywordFilterResult:
    """Hasil cek keyword filter.

    mode:
        - "block": hard-block matched, langsung tolak
        - "flag":  soft-flag matched tanpa safe_context, lanjut ke L2 dengan flag
        - "pass":  bersih ATAU soft-flag dengan safe_context (downgrade)
    """
    mode: Literal["block", "flag", "pass"]
    matched_phrases: list[str] = field(default_factory=list)
    safe_context_phrases: list[str] = field(default_factory=list)
    normalized_text: str = ""

    @property
    def is_blocked(self) -> bool:
        return self.mode == "block"

    @property
    def is_flagged(self) -> bool:
        return self.mode == "flag"


@dataclass
class _KeywordBundle:
    version: int
    updated_at: str
    hard_block: tuple[str, ...]
    soft_flag: tuple[str, ...]
    safe_context: tuple[str, ...]
    loaded_at: float


_bundle_cache: _KeywordBundle | None = None


def _load_yaml_bundle() -> _KeywordBundle:
    """Load + validate YAML keyword config. Raise kalau file rusak."""
    if not _KEYWORDS_PATH.exists():
        raise FileNotFoundError(f"Keyword config tidak ada: {_KEYWORDS_PATH}")

    with _KEYWORDS_PATH.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("YAML root harus dict")

    for key in ("hard_block", "soft_flag", "safe_context"):
        if not isinstance(raw.get(key), list):
            raise ValueError(f"YAML field '{key}' harus list")

    # Pre-normalize semua keyword saat load — hemat di hot path.
    hard = tuple(sorted({_normalize_text(p) for p in raw["hard_block"] if p}))
    soft = tuple(sorted({_normalize_text(p) for p in raw["soft_flag"] if p}))
    safe = tuple(sorted({_normalize_text(p) for p in raw["safe_context"] if p}))

    return _KeywordBundle(
        version=int(raw.get("version", 0)),
        updated_at=str(raw.get("updated_at", "")),
        hard_block=hard,
        soft_flag=soft,
        safe_context=safe,
        loaded_at=time.time(),
    )


def _get_bundle() -> _KeywordBundle:
    """Lazy + cached load dengan auto-reload tiap _RELOAD_INTERVAL_SEC."""
    global _bundle_cache
    now = time.time()
    if _bundle_cache is None or (now - _bundle_cache.loaded_at) > _RELOAD_INTERVAL_SEC:
        try:
            _bundle_cache = _load_yaml_bundle()
            logger.info(
                "keyword_filter_loaded version=%s updated_at=%s hard=%d soft=%d safe=%d",
                _bundle_cache.version,
                _bundle_cache.updated_at,
                len(_bundle_cache.hard_block),
                len(_bundle_cache.soft_flag),
                len(_bundle_cache.safe_context),
            )
        except Exception as e:
            if _bundle_cache is None:
                # First load gagal — fail loud, jangan run tanpa filter.
                raise
            logger.error("keyword_filter_reload_failed using_old_cache error=%s", e)
            _bundle_cache.loaded_at = now  # delay retry
    return _bundle_cache


def _normalize_text(text: str) -> str:
    """Normalisasi defensive untuk anti-bypass:
    1. NFKC unicode decomposition + recomposition canonical
    2. Lowercase
    3. Homoglyph Cyrillic/Greek → Latin
    4. Strip zero-width invisible chars
    5. Collapse whitespace
    6. Leet → normal (0→o, 1→i, dll)
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    for src, dst in _HOMOGLYPH_MAP.items():
        if src in text:
            text = text.replace(src, dst)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    text = text.translate(_LEET_MAP)
    return text


def _hash_query(text: str) -> str:
    """SHA256 hash 12 char untuk logging — privacy preserving."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def check(raw_text: str) -> KeywordFilterResult:
    """Cek keyword filter terhadap raw text.

    Returns:
        KeywordFilterResult dengan mode (block/flag/pass) dan matched phrases.
    """
    if not raw_text or not raw_text.strip():
        return KeywordFilterResult(mode="pass")

    normalized = _normalize_text(raw_text)
    if not normalized:
        return KeywordFilterResult(mode="pass", normalized_text=normalized)

    bundle = _get_bundle()
    query_hash = _hash_query(raw_text)

    # 1. Hard block — niat eksplisit, no context override.
    hard_matches = [p for p in bundle.hard_block if p in normalized]
    if hard_matches:
        logger.warning(
            "keyword_filter_block query_hash=%s matched=%s",
            query_hash,
            hard_matches,
        )
        return KeywordFilterResult(
            mode="block",
            matched_phrases=hard_matches,
            normalized_text=normalized,
        )

    # 2. Soft flag — keyword ambigu.
    soft_matches = [p for p in bundle.soft_flag if p in normalized]
    if not soft_matches:
        return KeywordFilterResult(mode="pass", normalized_text=normalized)

    # 3. Safe context check — kalau soft_flag tapi konteks akademik/policy → pass.
    safe_matches = [p for p in bundle.safe_context if p in normalized]
    if safe_matches:
        logger.info(
            "keyword_filter_downgrade query_hash=%s soft=%s safe=%s",
            query_hash,
            soft_matches,
            safe_matches,
        )
        return KeywordFilterResult(
            mode="pass",
            matched_phrases=soft_matches,
            safe_context_phrases=safe_matches,
            normalized_text=normalized,
        )

    # 4. Soft flag tanpa safe context → flag (lanjut ke L2 dengan info).
    logger.warning(
        "keyword_filter_flag query_hash=%s matched=%s",
        query_hash,
        soft_matches,
    )
    return KeywordFilterResult(
        mode="flag",
        matched_phrases=soft_matches,
        normalized_text=normalized,
    )


def warm_up() -> None:
    """Trigger first YAML load saat startup — fail fast kalau config rusak."""
    _get_bundle()
