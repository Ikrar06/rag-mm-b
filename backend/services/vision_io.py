"""Panggilan model vision yang aman ukuran dan tahan galat sementara.

Dua masalah terukur yang ditangani di satu tempat:

1. Image processor qwen3vl Ollama (model/models/qwen3vl/imageprocessor.go)
   panic — HTTP 500 — bila sisi gambar < patch_size x spatial_merge_size atau
   max(sisi) // min(sisi) > 200. Klasifikasi pernah berhenti di gambar
   543/1.441 karena manual_p23_c03 (2087x118) diperkecil ke 512x29. Batasnya
   dibaca dari model_info /api/show, bukan ditebak; gambar yang kurang
   ditambal putih, tidak pernah diregangkan.
2. Galat 5xx dan timeout dicoba ulang tiga kali dengan jeda bertahap. Yang
   tetap gagal dilempar sebagai GagalVision; pemanggil memakai fallback
   (teks OCR untuk tabel, narasi untuk gambar) dan TIDAK menyimpannya di
   cache, supaya run berikutnya mencoba lagi.

Gambar yang sudah aman dikirim byte demi byte, jadi pengaman ini tidak
mengubah apa pun yang sebelumnya berhasil — termasuk isi cache v4.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from functools import lru_cache

from backend.config import LLM_BASE_URL, LLM_PROVIDER, RESEARCH_VISION_SEED, VISION_MODEL
from backend.services.transkripsi_murni import rencana_ukuran

logger = logging.getLogger(__name__)

# Tiga kali coba ulang setelah percobaan pertama, jeda bertahap (detik).
JEDA_COBA_ULANG = (5.0, 15.0, 45.0)
# Batas rasio SmartResize Ollama: panic bila max(sisi) // min(sisi) > 200.
RASIO_MAKS = 200
TIMEOUT_DETIK = 1800.0


class GagalVision(Exception):
    """Model tidak menjawab setelah semua percobaan. Pesannya alasan kegagalan."""


@lru_cache(maxsize=1)
def batas_model() -> dict:
    """Sisi minimum gambar untuk VISION_MODEL, dari /api/show.

    Raise RuntimeError bila tidak dapat dipastikan — gerbang indexing
    memanggilnya sebelum PDF pertama, jadi run gagal di awal, bukan di jam ke-10.
    """
    import httpx
    if LLM_PROVIDER != "ollama":
        raise RuntimeError(f"transkripsi tabel hanya untuk ollama, LLM_PROVIDER={LLM_PROVIDER}")
    with httpx.Client(timeout=60.0) as c:
        info = c.post(f"{LLM_BASE_URL}/api/show", json={"model": VISION_MODEL}
                      ).raise_for_status().json().get("model_info") or {}
        versi = c.get(f"{LLM_BASE_URL}/api/version").json().get("version")
    patch = next((v for k, v in info.items() if k.endswith(".vision.patch_size")), None)
    gabung = next((v for k, v in info.items() if k.endswith(".vision.spatial_merge_size")), None)
    if not patch or not gabung:
        raise RuntimeError(f"model_info {VISION_MODEL} tidak memuat vision.patch_size / "
                           "vision.spatial_merge_size; sisi minimum tidak dapat dipastikan")
    return {"model": VISION_MODEL, "ollama": versi, "patch_size": int(patch),
            "spatial_merge_size": int(gabung), "sisi_min": int(patch) * int(gabung),
            "rasio_maks": RASIO_MAKS}


def siapkan_png(png: bytes, sisi_maks: int | None = None,
                sisi_min: int | None = None) -> tuple[bytes, bool]:
    """Perkecil (opsional) lalu tambal putih sampai aman. -> (bytes, dipadding).

    `sisi_min` None: dibaca dari batas_model(). Gambar yang sudah aman dan tak
    perlu diperkecil dikembalikan apa adanya.
    """
    from PIL import Image
    minimum = sisi_min if sisi_min is not None else batas_model()["sisi_min"]
    img = Image.open(io.BytesIO(png))
    r = rencana_ukuran(img.width, img.height, sisi_maks, minimum, RASIO_MAKS)
    if (r.kanvas_lebar, r.kanvas_tinggi) == img.size:
        return png, False
    img = img.convert("RGB")
    if (r.lebar, r.tinggi) != img.size:
        img = img.resize((r.lebar, r.tinggi))
    if r.dipadding:
        kanvas = Image.new("RGB", (r.kanvas_lebar, r.kanvas_tinggi), "white")
        kanvas.paste(img, ((r.kanvas_lebar - r.lebar) // 2, (r.kanvas_tinggi - r.tinggi) // 2))
        img = kanvas
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), r.dipadding


def _boleh_diulang(e: Exception) -> bool:
    import httpx
    if isinstance(e, httpx.HTTPStatusError):
        return e.response.status_code >= 500
    return isinstance(e, (httpx.TimeoutException, httpx.TransportError))


def _kirim(payload: dict) -> dict:
    import httpx
    with httpx.Client(timeout=TIMEOUT_DETIK) as c:
        return c.post(f"{LLM_BASE_URL}/api/generate", json=payload).raise_for_status().json()


def panggil(png: bytes, prompt: str, *, num_predict: int, num_ctx: int,
            temperature: float = 0.0) -> dict:
    """Satu panggilan model dengan pengaman ukuran dan coba ulang.

    -> {"response", "done_reason", "eval_count", "prompt_eval_count", "detik",
        "percobaan", "dipadding"}. GagalVision bila tetap gagal.
    """
    aman, dipadding = siapkan_png(png)
    options = {"temperature": temperature, "num_ctx": num_ctx, "num_predict": num_predict}
    if RESEARCH_VISION_SEED >= 0:
        options["seed"] = RESEARCH_VISION_SEED
    payload = {"model": VISION_MODEL, "prompt": prompt, "stream": False,
               "images": [base64.b64encode(aman).decode()], "options": options}
    galat: list[str] = []
    for percobaan, jeda in enumerate((*JEDA_COBA_ULANG, None), 1):
        t0 = time.monotonic()
        try:
            data = _kirim(payload)
        except Exception as e:
            galat.append(f"{type(e).__name__}: {str(e)[:120]}")
            if not _boleh_diulang(e) or jeda is None:
                raise GagalVision(f"{percobaan} percobaan; terakhir {galat[-1]}") from e
            logger.warning("vision_coba_ulang percobaan=%d jeda=%.0fs galat=%s",
                           percobaan, jeda, galat[-1])
            time.sleep(jeda)
            continue
        if not isinstance(data, dict) or "response" not in data:
            raise GagalVision(f"respons tanpa 'response': {list(data)[:6] if isinstance(data, dict) else type(data).__name__}")
        return {"response": data["response"], "done_reason": data.get("done_reason"),
                "eval_count": data.get("eval_count"),
                "prompt_eval_count": data.get("prompt_eval_count"),
                "detik": round(time.monotonic() - t0, 1),
                "percobaan": percobaan, "dipadding": dipadding}
    raise AssertionError("tidak tercapai")


def amankan(gambar: bytes) -> bytes:
    """Pengaman ukuran untuk jalur lama (deskripsi, adjudikasi).

    Gambar yang sudah aman dikembalikan byte demi byte, jadi isi cache v4 tidak
    berubah: di korpus ini tidak ada gambar deskripsi di bawah batas (ukuran
    asli, maks 1.280 px). Bila batas model tidak dapat dibaca, gambar dikirim
    apa adanya seperti sebelumnya — jalur lama tidak boleh gagal karena
    pengaman baru.
    """
    global _batas_tak_terbaca
    if _batas_tak_terbaca:
        return gambar
    try:
        return siapkan_png(gambar)[0]
    except Exception as e:
        # Dicatat sekali per proses: batas_model tidak di-cache saat gagal, dan
        # mengulang /api/show untuk setiap gambar hanya menambah beban dan log.
        _batas_tak_terbaca = f"{type(e).__name__}: {e}"
        logger.warning("pengaman_ukuran_dilewati error=%s — gambar dikirim apa adanya "
                       "sepanjang proses ini", _batas_tak_terbaca)
        return gambar


_batas_tak_terbaca: str | None = None
