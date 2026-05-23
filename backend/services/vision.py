"""Vision RAG — handle image attachment di chat query.

Pipeline:
  raw images (base64)
    → validate (mime, size, count)
    → decode + resize ke IMAGE_RESIZE_MAX_DIM (hemat token vLLM)
    → re-encode ke base64
    → kirim ke vLLM /v1/chat/completions dengan content multimodal
       (text prompt + image_url x N)

Catatan:
- Llama Guard 3 1B TIDAK lihat image content (text-only model). Validasi
  image-content abuse rely on Qwen3-VL refusal behavior + L1 keyword filter
  pada caption/question text.
- Cache (Redis) di-skip kalau ada image — setiap image unik, cache by-text
  tidak applicable (CACHE_SKIP_IF_HAS_IMAGE=true di config).
- Indexing image describer pakai endpoint sama tapi prompt berbeda (deskripsi
  bukan jawab pertanyaan). Lihat backend/services/image_describer.py.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from io import BytesIO
from typing import Iterable

import httpx
from PIL import Image

from backend.config import (
    ALLOWED_IMAGE_TYPES,
    IMAGE_RESIZE_MAX_DIM,
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_REQUEST_TIMEOUT,
    LLM_SUPPORTS_VISION,
    LLM_TEMPERATURE,
    MAX_IMAGES_PER_MESSAGE,
    MAX_IMAGE_SIZE_MB,
)

logger = logging.getLogger(__name__)

_MAX_IMAGE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024


class VisionError(Exception):
    """Raised saat validasi/decode/resize image gagal — caught di router level
    dan dikembalikan sebagai HTTP 400 dengan pesan natural.
    """


@dataclass(frozen=True)
class ProcessedImage:
    """Image siap dikirim ke vLLM."""
    mime_type: str
    data_b64: str   # base64 (TANPA prefix data: URL)
    size_bytes: int
    width: int
    height: int


@dataclass(frozen=True)
class StoredImage:
    """Image yang sudah di-upload ke storage — URL untuk persist & QA Sheet."""
    url: str         # presigned MinIO URL (POC) atau /api/files/... (dev)
    mime_type: str
    size_bytes: int


def vision_supported() -> bool:
    """Check apakah backend dikonfigurasi support vision."""
    return LLM_SUPPORTS_VISION and LLM_PROVIDER == "vllm"


def validate_and_process(images: Iterable[dict]) -> list[ProcessedImage]:
    """Validate, decode, resize, re-encode image dari request.

    Args:
        images: Iterable of dict dengan key "mime_type" dan "data" (base64).

    Returns:
        list[ProcessedImage]

    Raises:
        VisionError dengan pesan user-friendly.
    """
    image_list = list(images)
    if not image_list:
        return []

    if not vision_supported():
        raise VisionError("Fitur kirim gambar belum aktif di server.")

    if len(image_list) > MAX_IMAGES_PER_MESSAGE:
        raise VisionError(
            f"Maksimum {MAX_IMAGES_PER_MESSAGE} gambar per pesan. "
            f"Diterima {len(image_list)}."
        )

    processed: list[ProcessedImage] = []
    for idx, raw in enumerate(image_list, start=1):
        try:
            processed.append(_process_one(raw, idx))
        except VisionError:
            raise
        except Exception as e:
            logger.error("image_process_unexpected idx=%d error=%s", idx, e)
            raise VisionError(f"Gambar #{idx} tidak bisa diproses.") from e

    return processed


def _process_one(raw: dict, idx: int) -> ProcessedImage:
    mime_type = (raw.get("mime_type") or "").lower().strip()
    if mime_type not in ALLOWED_IMAGE_TYPES:
        raise VisionError(
            f"Gambar #{idx}: format {mime_type or 'tidak diketahui'} tidak didukung. "
            f"Pakai JPEG, PNG, atau WebP."
        )

    b64 = raw.get("data") or ""
    # Strip data URL prefix kalau ada (defensive — client kadang kirim full data URL).
    if b64.startswith("data:"):
        comma = b64.find(",")
        if comma > 0:
            b64 = b64[comma + 1:]

    try:
        decoded = base64.b64decode(b64, validate=True)
    except Exception as e:
        raise VisionError(f"Gambar #{idx}: base64 invalid.") from e

    if len(decoded) > _MAX_IMAGE_BYTES:
        raise VisionError(
            f"Gambar #{idx} terlalu besar ({len(decoded) // 1024 // 1024} MB). "
            f"Maksimum {MAX_IMAGE_SIZE_MB} MB."
        )

    try:
        img = Image.open(BytesIO(decoded))
        img.verify()  # paranoid check: format valid?
    except Exception as e:
        raise VisionError(f"Gambar #{idx}: file rusak atau bukan gambar valid.") from e

    # Re-open setelah verify (verify destroys image object).
    img = Image.open(BytesIO(decoded))
    img = _resize_if_needed(img)
    resized_bytes, out_mime = _encode_image(img, mime_type)

    return ProcessedImage(
        mime_type=out_mime,
        data_b64=base64.b64encode(resized_bytes).decode("ascii"),
        size_bytes=len(resized_bytes),
        width=img.width,
        height=img.height,
    )


def _resize_if_needed(img: Image.Image) -> Image.Image:
    """Resize image kalau dimensi melebihi IMAGE_RESIZE_MAX_DIM (hemat token VL)."""
    w, h = img.size
    if max(w, h) <= IMAGE_RESIZE_MAX_DIM:
        return img
    ratio = IMAGE_RESIZE_MAX_DIM / max(w, h)
    new_size = (int(w * ratio), int(h * ratio))
    logger.info("vision_resize from=%dx%d to=%dx%d", w, h, *new_size)
    return img.resize(new_size, Image.Resampling.LANCZOS)


def _encode_image(img: Image.Image, original_mime: str) -> tuple[bytes, str]:
    """Re-encode image. JPEG → JPEG 85% quality, lainnya → PNG optimized.
    Convert RGBA ke RGB kalau output JPEG (JPEG tidak support alpha).
    """
    buf = BytesIO()
    if original_mime in ("image/jpeg", "image/jpg"):
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=85, optimize=True)
        return buf.getvalue(), "image/jpeg"
    # PNG / WebP → save sebagai PNG (universally supported di OpenAI-compat clients)
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGBA")
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), "image/png"


def upload_to_storage(images: list[ProcessedImage], session_id: str) -> list[StoredImage]:
    """Upload ProcessedImage ke storage, return list of StoredImage dengan URL.

    Key format: chat-uploads/{session_id}/{uuid4}.{ext}
    URL: MinIO presigned (POC, 7-day) atau /api/files/... (dev).

    Fail-safe: kalau upload gagal, log warning + skip (vision query masih bisa
    jalan tanpa persist URL; cuma QA Sheet yang miss).
    """
    if not images:
        return []

    from uuid import uuid4
    from backend.services.storage import get_storage

    try:
        storage = get_storage()
    except Exception as e:
        logger.error("storage_init_failed error=%s — images tidak akan tersimpan", e)
        return []

    stored: list[StoredImage] = []
    for img in images:
        ext = "jpg" if img.mime_type == "image/jpeg" else "png"
        key = f"chat-uploads/{session_id}/{uuid4().hex}.{ext}"
        try:
            data = base64.b64decode(img.data_b64)
            url = storage.put_sync(key, data, img.mime_type)
            stored.append(StoredImage(
                url=url,
                mime_type=img.mime_type,
                size_bytes=img.size_bytes,
            ))
            logger.info("image_uploaded key=%s size_kb=%d", key, len(data) // 1024)
        except Exception as e:
            logger.error("image_upload_failed key=%s error=%s", key, e)
            # Continue dengan image berikutnya — vision query masih bisa jalan.
    return stored


def generate_with_vision(
    prompt: str,
    images: list[ProcessedImage],
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """Call vLLM /v1/chat/completions dengan content multimodal (text + images).

    Args:
        prompt: text instruction (sudah diformat lengkap dengan context_str).
        images: list of ProcessedImage hasil validate_and_process().
        max_tokens: override LLM_MAX_TOKENS untuk request ini (opsional).
        temperature: override LLM_TEMPERATURE (opsional).

    Returns:
        Raw text response dari LLM (belum filter_output).

    Raises:
        VisionError kalau LLM call gagal.
    """
    if not vision_supported():
        raise VisionError("Vision mode tidak aktif.")

    content: list[dict] = [{"type": "text", "text": prompt}]
    for img in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{img.mime_type};base64,{img.data_b64}"},
        })

    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens or LLM_MAX_TOKENS,
        "temperature": temperature if temperature is not None else LLM_TEMPERATURE,
    }

    try:
        with httpx.Client(timeout=float(LLM_REQUEST_TIMEOUT)) as client:
            resp = client.post(f"{LLM_BASE_URL}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.TimeoutException as e:
        logger.error("vision_llm_timeout error=%s", e)
        raise VisionError("Server LLM lambat — silakan coba lagi.") from e
    except httpx.HTTPError as e:
        logger.error("vision_llm_http_error error=%s", e)
        raise VisionError("Server LLM error — silakan coba lagi.") from e
    except Exception as e:
        logger.error("vision_llm_unexpected error=%s", e)
        raise VisionError("Gagal memproses gambar. Coba lagi nanti.") from e

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        logger.error("vision_llm_bad_response data=%s", data)
        raise VisionError("Respons LLM tidak valid.") from e
