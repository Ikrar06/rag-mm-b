"""Describe gambar PDF menggunakan vision LLM (Qwen3-VL) untuk indexing.

Dipanggil saat indexing PDF yang punya informative image (diagram, flowchart, dll).
Output deskripsi text disimpan sebagai chunk RAG, bukan gambar mentah.

Graceful degradation:
- Kalau LLM_SUPPORTS_VISION=false (dev): skip, return None
- Kalau vLLM tidak reachable: log warning, return None
"""

import base64
import hashlib
import logging
from pathlib import Path
from typing import Optional

from backend.config import (
    LLM_BASE_URL, LLM_MODEL, LLM_PROVIDER, LLM_SUPPORTS_VISION,
    PDF_MAX_IMAGE_DIM, PDF_MIN_IMAGE_SIZE_KB,
)

logger = logging.getLogger(__name__)

_DESCRIPTION_PROMPT = """\
Deskripsikan gambar ini dalam Bahasa Indonesia secara faktual dan ringkas (maks 150 kata).

Fokus pada:
- Jika diagram/flowchart: sebutkan semua langkah, label, dan arah panah
- Jika tabel: sebutkan judul kolom, baris penting, dan angka kunci
- Jika screenshot UI aplikasi: sebutkan tombol, menu, field yang terlihat
- Jika foto/dekorasi (logo, sampul, tanda tangan): cukup tulis "DEKORATIF"

JANGAN tafsirkan atau menambah informasi yang tidak terlihat. Jika gambar tidak jelas, tulis "TIDAK JELAS".
"""

# Parameter generasi deskripsi gambar. Diberi nama supaya run_manifest.json
# dapat membacanya langsung dari sumbernya, bukan menyalin angka yang bisa
# melenceng diam-diam. Hanya jalur vLLM yang mengirimkannya; jalur Ollama
# memakai default model (lihat _describe_via_ollama).
DESCRIPTION_MAX_TOKENS = 300
DESCRIPTION_TEMPERATURE = 0.1

# Cache hasil deskripsi by image hash supaya tidak panggil LLM 2× untuk gambar yang sama
_description_cache: dict[str, str] = {}


def _hash_image(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()[:16]


def _resize_image_if_needed(image_bytes: bytes, max_dim: int = PDF_MAX_IMAGE_DIM) -> bytes:
    """Resize image jika dimensi melebihi max_dim untuk hemat token VL."""
    try:
        from io import BytesIO
        from PIL import Image

        img = Image.open(BytesIO(image_bytes))
        w, h = img.size
        if max(w, h) <= max_dim:
            return image_bytes

        ratio = max_dim / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

        buf = BytesIO()
        fmt = img.format or "PNG"
        if fmt == "JPEG":
            img.save(buf, format="JPEG", quality=85)
        else:
            img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()
    except Exception as e:
        logger.debug(f"image_resize_skip reason={e}")
        return image_bytes


def is_likely_informative(image_path: str | Path) -> bool:
    """Heuristic: gambar informatif (worth describing) vs dekoratif (skip)."""
    try:
        p = Path(image_path)
        size_kb = p.stat().st_size / 1024
        if size_kb < PDF_MIN_IMAGE_SIZE_KB:
            return False  # terlalu kecil, kemungkinan icon/logo

        from PIL import Image
        with Image.open(p) as img:
            w, h = img.size

        # Aspect ratio ekstrem (mis. line separator) → skip
        if max(w, h) / min(w, h) > 15:
            return False
        # Dimensi sangat kecil → skip
        if w < 100 or h < 100:
            return False

        return True
    except Exception:
        return True  # fail-open: kalau ragu, deskripsikan saja


def describe_image(image_bytes: bytes) -> Optional[str]:
    """Deskripsikan gambar via vision LLM.

    Returns:
        - String deskripsi jika sukses
        - None jika skip (vision off, error, atau image dekoratif)
    """
    if not LLM_SUPPORTS_VISION:
        return None

    img_hash = _hash_image(image_bytes)
    if img_hash in _description_cache:
        return _description_cache[img_hash]

    image_bytes = _resize_image_if_needed(image_bytes)

    try:
        if LLM_PROVIDER == "vllm":
            description = _describe_via_vllm(image_bytes)
        elif LLM_PROVIDER == "ollama":
            description = _describe_via_ollama(image_bytes)
        else:
            logger.debug(f"image_describer_unsupported provider={LLM_PROVIDER}")
            return None
    except Exception as e:
        logger.warning(f"image_describer_error error={e}")
        return None

    if not description:
        return None

    # Filter hasil — kalau model bilang dekoratif, treat as skip
    cleaned = description.strip()
    if cleaned.upper().startswith("DEKORATIF") or cleaned.upper().startswith("TIDAK JELAS"):
        _description_cache[img_hash] = ""
        return None

    _description_cache[img_hash] = cleaned
    return cleaned


def _describe_via_vllm(image_bytes: bytes) -> Optional[str]:
    """Vision via vLLM OpenAI-compatible API."""
    import httpx

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "model": LLM_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": _DESCRIPTION_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
        "max_tokens": DESCRIPTION_MAX_TOKENS,
        "temperature": DESCRIPTION_TEMPERATURE,
    }

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(f"{LLM_BASE_URL}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

    return data["choices"][0]["message"]["content"]


def _describe_via_ollama(image_bytes: bytes) -> Optional[str]:
    """Vision via Ollama generate API (untuk model multimodal Ollama)."""
    import httpx

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "model": LLM_MODEL,
        "prompt": _DESCRIPTION_PROMPT,
        "images": [b64],
        "stream": False,
    }

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(f"{LLM_BASE_URL}/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()

    return data.get("response")
