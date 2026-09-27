"""I/O bersama alat ukur vision: buka area PDF dan panggil model Ollama.

Dipakai scripts/ukur_transkripsi.py dan scripts/klasifikasi_gambar.py. Tidak
menulis ke koleksi, cache, atau dump.
"""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass
from pathlib import Path

from lib.transkripsi_ukur import jenis_halaman

NUM_CTX = 16384
SEED = 1337
TIMEOUT_DETIK = 1800.0


@dataclass(frozen=True)
class Area:
    png: bytes
    # Kata UTUH yang beririsan dengan area. Klip karakter memotong angka di
    # tepi (ukt p5: "8,000,000" jadi "8,000"), dan potongan itu terhitung
    # "angka hilang" padahal artefak rujukan.
    teks_area: str
    teks_halaman: str
    jenis_halaman: str
    # Kotak kata ternormalisasi seluruh halaman, untuk perluas_bbox.
    kata: tuple[tuple[float, float, float, float], ...]


def _klip(page, bbox):
    import fitz
    r = page.rect
    return fitz.Rect(r.x0 + bbox[0] * r.width, r.y0 + bbox[1] * r.height,
                     r.x0 + bbox[2] * r.width, r.y0 + bbox[3] * r.height)


def buka_area(pdf: Path, halaman: int, bbox, dpi: int | None) -> Area:
    """Render area bbox (dpi None: tanpa render) dan lapisan teksnya."""
    import fitz
    with fitz.open(str(pdf)) as doc:
        page = doc[halaman - 1]
        r = page.rect
        klip = _klip(page, bbox)
        kata = page.get_text("words")
        luas = r.width * r.height or 1.0
        terbesar = max((fitz.Rect(i["bbox"]).get_area() / luas
                        for i in page.get_image_info()), default=0.0)
        return Area(
            png=page.get_pixmap(dpi=dpi, clip=klip).tobytes("png") if dpi else b"",
            teks_area=" ".join(w[4] for w in kata if fitz.Rect(w[:4]).intersects(klip)),
            teks_halaman=page.get_text("text"),
            jenis_halaman=jenis_halaman(bool(kata), terbesar),
            kata=tuple((w[0] / r.width, w[1] / r.height, w[2] / r.width, w[3] / r.height)
                       for w in kata),
        )


def perkecil(png: bytes, sisi: int) -> bytes:
    """Perkecil ke sisi terpanjang `sisi` px; gambar yang lebih kecil dibiarkan."""
    from PIL import Image
    img = Image.open(io.BytesIO(png)).convert("RGB")
    skala = sisi / max(img.size)
    if skala < 1:
        img = img.resize((max(1, round(img.width * skala)), max(1, round(img.height * skala))))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def png_pemanasan() -> bytes:
    """Gambar kecil untuk memuat model ke memori; waktunya tidak dihitung."""
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(buf, format="PNG")
    return buf.getvalue()


def panggil(png: bytes, prompt: str, num_predict: int) -> dict:
    import httpx
    from backend.config import LLM_BASE_URL, LLM_PROVIDER, VISION_MODEL
    if LLM_PROVIDER != "ollama":
        raise SystemExit(f"alat ukur hanya untuk ollama, LLM_PROVIDER={LLM_PROVIDER}")
    payload = {"model": VISION_MODEL, "prompt": prompt, "stream": False,
               "images": [base64.b64encode(png).decode()],
               "options": {"temperature": 0, "seed": SEED, "num_ctx": NUM_CTX,
                           "num_predict": num_predict}}
    t0 = time.monotonic()
    with httpx.Client(timeout=TIMEOUT_DETIK) as c:
        data = c.post(f"{LLM_BASE_URL}/api/generate", json=payload).raise_for_status().json()
    return {"detik": round(time.monotonic() - t0, 1), "response": data.get("response", ""),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count": data.get("eval_count"), "done_reason": data.get("done_reason")}
