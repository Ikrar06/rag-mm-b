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
    RESEARCH_VISION_SEED, VISION_MAX_TOKENS, VISION_MODEL,
    VISION_NUM_CTX, VISION_TEMPERATURE,
)
from backend.services import vision_cache

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

# Parameter generasi deskripsi gambar, dibaca dari config supaya run_manifest
# dan images.jsonl mencatat nilai yang benar-benar dipakai.
#
# KEDUA jalur (vLLM dan Ollama) kini mengirim nilai yang SAMA. Sebelumnya jalur
# Ollama tidak mengirim temperature maupun batas token sama sekali, sehingga
# memakai default model — dua provider menghasilkan distribusi berbeda dari
# prompt yang sama.
DESCRIPTION_MAX_TOKENS = VISION_MAX_TOKENS
DESCRIPTION_TEMPERATURE = VISION_TEMPERATURE
DESCRIPTION_SEED = RESEARCH_VISION_SEED
DESCRIPTION_NUM_CTX = VISION_NUM_CTX

# Peta format PIL -> (ekstensi, mime). SATU sumber kebenaran: preprocessing
# mengimpornya untuk menamai berkas di disk, dan modul ini memakainya untuk
# melabeli payload yang dikirim ke model. Ekstensi di disk dan MIME di payload
# karenanya tidak bisa berselisih.
PIL_FORMAT_MAP: dict[str, tuple[str, str]] = {
    "PNG": ("png", "image/png"),
    "JPEG": ("jpg", "image/jpeg"),
    "WEBP": ("webp", "image/webp"),
    "GIF": ("gif", "image/gif"),
    "TIFF": ("tiff", "image/tiff"),
    "BMP": ("bmp", "image/bmp"),
}
_FALLBACK_FORMAT = ("png", "image/png")

# Cache hasil deskripsi by image hash supaya tidak panggil LLM 2× untuk gambar yang sama
_description_cache: dict[str, str] = {}


def _hash_image(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()[:16]


def probe_format(image_bytes: bytes) -> tuple[str, str, int | None, int | None]:
    """(ekstensi, mime, width, height) dari isi bytes — bukan asumsi.

    Sebelumnya jalur vLLM meng-hardcode `image/png` untuk semua gambar, sehingga
    JPEG yang tidak di-resize dikirim dengan label yang salah.
    """
    try:
        from io import BytesIO
        from PIL import Image
        with Image.open(BytesIO(image_bytes)) as img:
            fmt = (img.format or "").upper()
            w, h = img.size
        ext, mime = PIL_FORMAT_MAP.get(fmt, _FALLBACK_FORMAT)
        return ext, mime, w, h
    except Exception as e:
        logger.debug("image_probe_failed error=%s", e)
        return (*_FALLBACK_FORMAT, None, None)


def _detect_mime(image_bytes: bytes) -> str:
    return probe_format(image_bytes)[1]


def vision_provenance() -> dict:
    """Parameter yang menentukan isi deskripsi, untuk dicatat per gambar.

    `model_digest` diambil dari /api/tags — digest sha256 PENUH, bukan ID pendek
    yang tampil di `ollama list`. Tag bergerak (`qwen3-vl:8b` bisa menunjuk bobot
    berbeda setelah `ollama pull`), digest tidak.
    """
    return {
        "vision_provider": LLM_PROVIDER,
        "vision_model": VISION_MODEL,
        "vision_model_digest": _ollama_model_digest(VISION_MODEL),
        "vision_temperature": DESCRIPTION_TEMPERATURE,
        "vision_max_tokens": DESCRIPTION_MAX_TOKENS,
        "vision_seed": DESCRIPTION_SEED if DESCRIPTION_SEED >= 0 else None,
        "vision_num_ctx": DESCRIPTION_NUM_CTX,
        "prompt_sha256": hashlib.sha256(_DESCRIPTION_PROMPT.encode("utf-8")).hexdigest(),
    }


_digest_cache: dict[str, str | None] = {}


def _ollama_model_digest(model: str) -> str | None:
    """Digest sha256 penuh model dari GET /api/tags, atau None.

    None BUKAN diam: kegagalan dicatat sebagai warning, dan pemanggil menandainya
    di manifest. Tanpa digest, bobot yang menghasilkan deskripsi tidak dapat
    dibuktikan setelahnya.
    """
    if LLM_PROVIDER != "ollama":
        return None
    if model in _digest_cache:
        return _digest_cache[model]

    digest = None
    try:
        import httpx
        r = httpx.get(f"{LLM_BASE_URL.rstrip('/')}/api/tags", timeout=10)
        r.raise_for_status()
        for m in r.json().get("models", []):
            if m.get("name") == model or m.get("model") == model:
                digest = m.get("digest")
                break
        if digest is None:
            logger.warning(
                "vision_model_digest_absent model=%r — tidak ada di /api/tags; "
                "bobot yang menghasilkan deskripsi tidak dapat dibuktikan", model,
            )
    except Exception as e:
        logger.warning(
            "vision_model_digest_unreachable model=%r base_url=%s error=%s: %s "
            "— provenance model vision tidak lengkap di run ini",
            model, LLM_BASE_URL, type(e).__name__, e,
        )

    _digest_cache[model] = digest
    return digest


def _resize_image_if_needed(image_bytes: bytes, max_dim: int = PDF_MAX_IMAGE_DIM) -> bytes:
    """Resize image jika dimensi melebihi max_dim untuk hemat token VL."""
    try:
        from io import BytesIO
        from PIL import Image

        img = Image.open(BytesIO(image_bytes))
        w, h = img.size
        # Format HARUS dibaca sebelum .resize(): objek hasil resize tidak
        # membawa .format, sehingga `img.format or "PNG"` dulu selalu jatuh ke
        # PNG dan cabang JPEG di bawah tidak pernah terjangkau.
        fmt = (img.format or "PNG").upper()
        if max(w, h) <= max_dim:
            return image_bytes

        ratio = max_dim / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

        buf = BytesIO()
        if fmt == "JPEG":
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")   # JPEG tidak mendukung alpha
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


def _classify(raw: str) -> tuple[str, Optional[str]]:
    """Petakan balasan model ke (verdict, teks).

    Memisahkan KEPUTUSAN MODEL dari KEGAGALAN. Keduanya dulu sama-sama menulis
    string kosong ke cache proses-lokal — atau, untuk respons kosong, tidak
    menulis sama sekali. Ambiguitas itu tidak boleh diwarisi cache persisten:
    keputusan model layak disimpan, kegagalan transient tidak.
    """
    cleaned = (raw or "").strip()
    if not cleaned:
        return vision_cache.VERDICT_DESCRIBED, None   # dipakai pemanggil sbg gagal
    upper = cleaned.upper()
    if upper.startswith("DEKORATIF"):
        return vision_cache.VERDICT_DECORATIVE, None
    if upper.startswith("TIDAK JELAS"):
        return vision_cache.VERDICT_UNCLEAR, None
    return vision_cache.VERDICT_DESCRIBED, cleaned


def describe_image(image_bytes: bytes) -> Optional[str]:
    """Deskripsikan gambar via vision LLM.

    Returns:
        - String deskripsi jika sukses
        - None jika skip (vision off, error, atau image dekoratif)

    Cache persisten dikonsultasi lebih dulu bila aktif. Yang disimpan hanya
    keputusan model (described / decorative / unclear) — kegagalan transient
    tidak pernah masuk, karena satu timeout yang ter-cache membuat gambar itu
    hilang dari seluruh eksperimen.
    """
    if not LLM_SUPPORTS_VISION:
        return None

    img_hash = _hash_image(image_bytes)
    if img_hash in _description_cache:
        return _description_cache[img_hash]

    # ── Cache persisten. Kunci memakai sha256 PENUH bytes asli, sebelum resize.
    cache_key = None
    if vision_cache.enabled():
        prov = vision_provenance()
        full_sha = hashlib.sha256(image_bytes).hexdigest()
        cache_key = vision_cache.make_key(
            full_sha, vision_cache.VARIANT_NARRATIVE,
            prov["prompt_sha256"], prov["vision_model_digest"],
        )
        hit = vision_cache.get(cache_key)
        if hit is not None:
            text = hit.usable_text
            _description_cache[img_hash] = text or ""
            return text

    image_bytes_original = image_bytes
    image_bytes = _resize_image_if_needed(image_bytes)

    if LLM_PROVIDER not in ("vllm", "ollama"):
        # Dulu logger.debug lalu return None — SELURUH deskripsi gambar hilang
        # dan hanya terlihat bila level log dinaikkan. Untuk riset, kehilangan
        # sebesar itu tidak boleh senyap.
        raise ValueError(
            f"LLM_PROVIDER={LLM_PROVIDER!r} tidak mendukung deskripsi gambar. "
            f"Hanya 'vllm' dan 'ollama' yang punya jalur multimodal. "
            f"Dengan provider ini SETIAP gambar akan gagal dideskripsikan."
        )

    try:
        if LLM_PROVIDER == "vllm":
            description = _describe_via_vllm(image_bytes)
        else:
            description = _describe_via_ollama(image_bytes)
    except Exception as e:
        logger.warning(
            "image_describer_error provider=%s model=%s sha=%s error=%s: %s",
            LLM_PROVIDER, VISION_MODEL, img_hash, type(e).__name__, e,
        )
        return None

    if not description or not description.strip():
        # KEGAGALAN TRANSIENT, bukan keputusan model. Di-cache proses-lokal
        # supaya tidak dipanggil berulang dalam satu proses, tapi TIDAK PERNAH
        # masuk cache persisten — satu timeout yang ter-cache membuat gambar itu
        # hilang dari seluruh eksperimen.
        logger.warning(
            "image_describer_empty provider=%s model=%s sha=%s — model "
            "mengembalikan respons kosong; TIDAK di-cache persisten",
            LLM_PROVIDER, VISION_MODEL, img_hash,
        )
        _description_cache[img_hash] = ""
        return None

    verdict, text = _classify(description)

    if cache_key is not None:
        prov = vision_provenance()
        vision_cache.put(
            cache_key,
            image_sha256=hashlib.sha256(image_bytes_original).hexdigest(),
            variant=vision_cache.VARIANT_NARRATIVE,
            prompt_sha256=prov["prompt_sha256"],
            vision_model=prov["vision_model"],
            vision_model_digest=prov["vision_model_digest"],
            verdict=verdict,
            description=text,
        )

    if verdict != vision_cache.VERDICT_DESCRIBED:
        logger.info(
            "image_describer_verdict verdict=%s sha=%s — gambar tetap tersimpan "
            "di disk, hanya tanpa narrative_summary", verdict, img_hash,
        )

    _description_cache[img_hash] = text or ""
    return text


def _describe_via_vllm(image_bytes: bytes) -> Optional[str]:
    """Vision via vLLM OpenAI-compatible API."""
    import httpx

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    mime = _detect_mime(image_bytes)   # bukan asumsi image/png
    payload = {
        "model": VISION_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": _DESCRIPTION_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }],
        "max_tokens": DESCRIPTION_MAX_TOKENS,
        "temperature": DESCRIPTION_TEMPERATURE,
    }
    if DESCRIPTION_SEED >= 0:
        payload["seed"] = DESCRIPTION_SEED

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(f"{LLM_BASE_URL}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        logger.error(
            "image_describer_bad_response provider=vllm model=%s error=%s "
            "keys=%s — bentuk respons tak terduga",
            VISION_MODEL, type(e).__name__,
            list(data)[:6] if isinstance(data, dict) else type(data).__name__,
        )
        return None


def _describe_via_ollama(image_bytes: bytes) -> Optional[str]:
    """Vision via Ollama generate API (untuk model multimodal Ollama).

    Parameter generasi dikirim EKSPLISIT lewat `options`, disamakan dengan jalur
    vLLM. Sebelumnya jalur ini tidak mengirim temperature maupun batas token
    sama sekali, sehingga memakai default model.

    `num_ctx` wajib dikirim: tanpa itu Ollama memotong konteks ke 4096 token
    secara senyap, dan satu gambar saja bisa menghabiskannya.
    """
    import httpx

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    options: dict = {
        "temperature": DESCRIPTION_TEMPERATURE,
        "num_predict": DESCRIPTION_MAX_TOKENS,   # padanan max_tokens di Ollama
        "num_ctx": DESCRIPTION_NUM_CTX,
    }
    if DESCRIPTION_SEED >= 0:
        options["seed"] = DESCRIPTION_SEED

    payload = {
        "model": VISION_MODEL,
        "prompt": _DESCRIPTION_PROMPT,
        "images": [b64],
        "stream": False,
        "options": options,
    }

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(f"{LLM_BASE_URL}/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()

    # Dulu `data.get("response")` — bentuk respons tak terduga mengembalikan
    # None DIAM-DIAM tanpa satu baris log pun, sehingga jumlah gambar yang gagal
    # dideskripsikan tidak dapat direkonstruksi dari log. Kini setara jalur vLLM.
    if not isinstance(data, dict) or "response" not in data:
        logger.error(
            "image_describer_bad_response provider=ollama model=%s keys=%s "
            "— kunci 'response' tidak ada di respons",
            VISION_MODEL,
            list(data)[:6] if isinstance(data, dict) else type(data).__name__,
        )
        return None
    return data["response"]
