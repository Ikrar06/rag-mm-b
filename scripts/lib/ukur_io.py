"""I/O bersama alat ukur vision: buka area PDF dan panggil model Ollama.

Dipakai scripts/ukur_transkripsi.py dan scripts/klasifikasi_gambar.py. Tidak
menulis ke koleksi, cache, atau dump.

Setiap gambar yang dikirim ke model lewat `panggil` diamankan ukurannya
(lihat `batas_model` dan `rencana_ukuran`), dan kegagalan 5xx/timeout dicoba
ulang dengan jeda bertahap sebelum dilaporkan sebagai `GagalVision`.
"""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from lib.transkripsi_ukur import jenis_halaman, rencana_ukuran

NUM_CTX = 16384
SEED = 1337
TIMEOUT_DETIK = 1800.0
# Tiga kali coba ulang setelah percobaan pertama, jeda bertahap (detik).
JEDA_COBA_ULANG = (5.0, 15.0, 45.0)
# Batas rasio SmartResize Ollama: panic bila max(sisi) // min(sisi) > 200.
RASIO_MAKS = 200


class GagalVision(Exception):
    """Model tidak menjawab setelah semua percobaan. Pesannya alasan kegagalan."""


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


@lru_cache(maxsize=1)
def batas_model() -> dict:
    """Sisi minimum gambar untuk model vision, dibaca dari Ollama, bukan ditebak.

    Image processor qwen3vl Ollama (model/models/qwen3vl/imageprocessor.go):
    factor = vision.patch_size * vision.spatial_merge_size, dan SmartResize
    panic bila tinggi atau lebar < factor. Kedua nilai diambil dari
    model_info /api/show model yang sedang dipakai. Bila tidak ada, run
    berhenti: menebak di sini berarti memilih antara 28 dan 32.
    """
    import httpx
    from backend.config import LLM_BASE_URL, VISION_MODEL
    with httpx.Client(timeout=60.0) as c:
        info = c.post(f"{LLM_BASE_URL}/api/show", json={"model": VISION_MODEL}
                      ).raise_for_status().json().get("model_info") or {}
        versi = c.get(f"{LLM_BASE_URL}/api/version").json().get("version")
    patch = next((v for k, v in info.items() if k.endswith(".vision.patch_size")), None)
    gabung = next((v for k, v in info.items() if k.endswith(".vision.spatial_merge_size")), None)
    if not patch or not gabung:
        raise SystemExit(f"model_info {VISION_MODEL} tidak memuat vision.patch_size / "
                         "vision.spatial_merge_size; sisi minimum tidak dapat dipastikan")
    return {"model": VISION_MODEL, "ollama": versi, "patch_size": int(patch),
            "spatial_merge_size": int(gabung), "sisi_min": int(patch) * int(gabung),
            "rasio_maks": RASIO_MAKS}


def siapkan_png(png: bytes, sisi_maks: int | None = None) -> tuple[bytes, bool]:
    """Perkecil (opsional) lalu tambal putih sampai aman bagi model.

    Mengembalikan (png, dipadding). Gambar yang sudah aman dan tidak perlu
    diperkecil dikirim apa adanya, byte demi byte.
    """
    from PIL import Image
    b = batas_model()
    img = Image.open(io.BytesIO(png))
    rencana = rencana_ukuran(img.width, img.height, sisi_maks, b["sisi_min"], b["rasio_maks"])
    if (rencana.kanvas_lebar, rencana.kanvas_tinggi) == img.size:
        return png, False
    img = img.convert("RGB")
    if (rencana.lebar, rencana.tinggi) != img.size:
        img = img.resize((rencana.lebar, rencana.tinggi))
    if rencana.dipadding:
        kanvas = Image.new("RGB", (rencana.kanvas_lebar, rencana.kanvas_tinggi), "white")
        kanvas.paste(img, ((rencana.kanvas_lebar - rencana.lebar) // 2,
                           (rencana.kanvas_tinggi - rencana.tinggi) // 2))
        img = kanvas
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), rencana.dipadding


def png_pemanasan() -> bytes:
    """Gambar untuk memuat model ke memori; waktunya tidak dihitung."""
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(buf, format="PNG")
    return buf.getvalue()


def _boleh_diulang(e: Exception) -> bool:
    import httpx
    if isinstance(e, httpx.HTTPStatusError):
        return e.response.status_code >= 500
    return isinstance(e, (httpx.TimeoutException, httpx.TransportError))


def _kirim(payload: dict) -> dict:
    import httpx
    from backend.config import LLM_BASE_URL
    with httpx.Client(timeout=TIMEOUT_DETIK) as c:
        return c.post(f"{LLM_BASE_URL}/api/generate", json=payload).raise_for_status().json()


def panggil(png: bytes, prompt: str, num_predict: int) -> dict:
    """Satu panggilan model dengan coba ulang. GagalVision bila tetap gagal."""
    from backend.config import LLM_PROVIDER, VISION_MODEL
    if LLM_PROVIDER != "ollama":
        raise SystemExit(f"alat ukur hanya untuk ollama, LLM_PROVIDER={LLM_PROVIDER}")
    aman, dipadding = siapkan_png(png)
    payload = {"model": VISION_MODEL, "prompt": prompt, "stream": False,
               "images": [base64.b64encode(aman).decode()],
               "options": {"temperature": 0, "seed": SEED, "num_ctx": NUM_CTX,
                           "num_predict": num_predict}}
    galat: list[str] = []
    for percobaan, jeda in enumerate((*JEDA_COBA_ULANG, None), 1):
        t0 = time.monotonic()
        try:
            data = _kirim(payload)
        except Exception as e:
            galat.append(f"{type(e).__name__}: {str(e)[:120]}")
            if not _boleh_diulang(e) or jeda is None:
                raise GagalVision(f"{percobaan} percobaan; terakhir {galat[-1]}") from e
            time.sleep(jeda)
            continue
        return {"detik": round(time.monotonic() - t0, 1), "response": data.get("response", ""),
                "prompt_eval_count": data.get("prompt_eval_count"),
                "eval_count": data.get("eval_count"), "done_reason": data.get("done_reason"),
                "percobaan": percobaan, "dipadding": dipadding}
    raise AssertionError("tidak tercapai")
