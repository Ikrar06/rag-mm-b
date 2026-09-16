"""Adjudikasi pasangan tabel ambigu oleh model vision.

Dipakai HANYA untuk pasangan yang tidak dapat diputuskan sinyal struktural.
Sinyal struktural terkuat — batas kolom — hanya tersedia pada PDF berlapis teks
(68,4% halaman bertabel di korpus ini). Pasangan pada halaman pindai tidak punya
sinyal itu sama sekali, dan membiarkannya "mungkin" berarti membuangnya dari
pertimbangan. Model vision melihat apa yang dilihat manusia: gambar kedua
potongan.

BUKAN pengganti tinjauan manusia. Putusannya mengisi kolom SARAN di
`table_continuation.json`; manusia yang menyetujui. Gunanya agar peninjau
membaca putusan yang sudah beralasan, bukan menebak dari baris pertama teks.

Determinisme: memakai `vision_cache` yang sama dengan deskripsi gambar, dengan
`variant="table_continuation"`. Kuncinya sudah memuat digest model dan hash
prompt, jadi entri dari konfigurasi berbeda tidak saling tertukar. Parameter
generasi diambil dari config yang sama (temperature 0, seed, num_ctx) dan
dikirim lewat pola pemanggilan `image_describer` — tidak ada klien baru.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import re
from dataclasses import dataclass

from backend.config import (
    LLM_BASE_URL, LLM_PROVIDER, RESEARCH_VISION_SEED, VISION_MAX_TOKENS,
    VISION_MODEL, VISION_NUM_CTX, VISION_TEMPERATURE,
)
from backend.services import vision_cache

logger = logging.getLogger(__name__)

VARIANT = "table_continuation"

# DPI render potongan tabel. 72 dipilih dari pengukuran di server: pada
# halaman penuh ia menjawab BENAR sama seperti 110 dan 150 dpi, tetapi 5,6x
# lebih cepat (110,6 s vs 618,1 s). Menaikkan DPI tidak memperbaiki putusan,
# hanya memperbesar jumlah token gambar.
#
# Dicatat di run_manifest.json lewat provenance(): mengubahnya mengubah biaya
# DAN berpotensi mengubah putusan, jadi ia bagian konfigurasi eksperimen.
RENDER_DPI = 72

# Batas waktu satu panggilan model, detik. 90 detik menjamin kegagalan total:
# terukur di server, bahkan 72 dpi pada halaman penuh butuh 110,6 detik.
TIMEOUT_DETIK = 600.0

# Tinggi pita pemisah antara dua potongan, dalam piksel.
TINGGI_PEMISAH = 24

PROMPT = """Kamu melihat satu gambar berisi DUA potongan tabel dari dokumen PDF.
Potongan ATAS diambil dari halaman N. Potongan BAWAH dari halaman N+1.
Keduanya dipisahkan garis horizontal.

Pertanyaan: apakah potongan BAWAH adalah LANJUTAN dari tabel yang sama dengan
potongan ATAS, atau tabel yang BERBEDA?

Perhatikan: jumlah dan lebar kolom, apakah potongan bawah punya baris header
sendiri, dan apakah isinya melanjutkan urutan potongan atas.

Jawab HANYA dengan JSON satu baris, tanpa penjelasan tambahan:
{"putusan": "LANJUTAN" | "BERBEDA" | "TIDAK_JELAS", "keyakinan": "tinggi" | "sedang" | "rendah", "alasan": "<satu kalimat>"}"""

PROMPT_SHA = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()

_PUTUSAN = {
    "LANJUTAN": vision_cache.VERDICT_LANJUTAN,
    "BERBEDA": vision_cache.VERDICT_BUKAN_LANJUTAN,
    "TIDAK_JELAS": vision_cache.VERDICT_TIDAK_JELAS,
}


_HALAMAN_RE = None      # dikompilasi saat pertama dipakai


def halaman_dikarang(alasan: str, halaman_asli) -> tuple[int, ...]:
    """Nomor halaman yang disebut model tapi TIDAK termasuk yang dikirim.

    Terukur di server: pada ketiga DPI, model menulis "Tabel di gambar kedua
    (halaman 34)..." padahal yang dikirim halaman 38 dan 39. Putusannya benar,
    alasannya mengarang.

    Dideteksi, bukan disensor. Menghapus angkanya akan menyembunyikan bahwa
    alasan model tidak dapat dipercaya; menandainya membuat peninjau tahu
    persis kalimat mana yang harus diabaikan.
    """
    global _HALAMAN_RE
    if _HALAMAN_RE is None:
        _HALAMAN_RE = re.compile(r"(?:halaman|hal\.?|page|pg\.?)\s*(\d{1,4})",
                                 re.IGNORECASE)
    sah = {int(h) for h in halaman_asli if isinstance(h, int)}
    return tuple(sorted({
        int(m) for m in _HALAMAN_RE.findall(alasan or "") if int(m) not in sah
    }))


@dataclass(frozen=True)
class Putusan:
    """Hasil adjudikasi. Immutable.

    `verdict` None berarti model TIDAK MENJAWAB — kegagalan transient, bukan
    putusan. Itu tidak pernah di-cache dan tidak boleh dibaca sebagai "berbeda".
    """

    verdict: str | None = None
    keyakinan: str = ""
    alasan: str = ""
    dari_cache: bool = False
    error: str = ""
    # Nomor halaman yang disebut model tapi tidak termasuk yang dikirim.
    # Non-kosong berarti alasannya memuat karangan.
    halaman_dikarang: tuple[int, ...] = ()
    # Ukuran gambar yang benar-benar dikirim, untuk menakar biaya.
    piksel_dikirim: int = 0

    @property
    def lanjutan(self) -> bool:
        return self.verdict == vision_cache.VERDICT_LANJUTAN


def _render(pdf_path, halaman: int, bbox) -> bytes | None:
    """Render area bbox sebuah halaman jadi PNG. None bila gagal."""
    import fitz

    if not bbox or len(bbox) != 4:
        return None
    try:
        with fitz.open(str(pdf_path)) as doc:
            if not 1 <= halaman <= doc.page_count:
                return None
            page = doc[halaman - 1]
            r = page.rect
            klip = fitz.Rect(
                r.x0 + bbox[0] * r.width, r.y0 + bbox[1] * r.height,
                r.x0 + bbox[2] * r.width, r.y0 + bbox[3] * r.height,
            )
            return page.get_pixmap(dpi=RENDER_DPI, clip=klip).tobytes("png")
    except Exception as e:
        logger.warning("render_tabel_gagal halaman=%s error=%s", halaman, e)
        return None


def susun(png_atas: bytes, png_bawah: bytes) -> bytes | None:
    """Tumpuk dua potongan jadi satu gambar dengan garis pemisah.

    Satu gambar, bukan dua: model diminta membandingkan, dan perbandingan lebih
    andal bila keduanya berada dalam satu bidang pandang dengan pemisah yang
    jelas. Lebarnya disamakan ke yang terlebar supaya lebar kolom tetap dapat
    dibandingkan secara visual.
    """
    from PIL import Image

    try:
        a = Image.open(io.BytesIO(png_atas)).convert("RGB")
        b = Image.open(io.BytesIO(png_bawah)).convert("RGB")
        lebar = max(a.width, b.width)
        kanvas = Image.new("RGB", (lebar, a.height + TINGGI_PEMISAH + b.height), "white")
        kanvas.paste(a, (0, 0))
        garis = Image.new("RGB", (lebar, 4), "black")
        kanvas.paste(garis, (0, a.height + (TINGGI_PEMISAH - 4) // 2))
        kanvas.paste(b, (0, a.height + TINGGI_PEMISAH))
        buf = io.BytesIO()
        kanvas.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        logger.warning("susun_gambar_gagal error=%s", e)
        return None


def parse_jawaban(raw: str) -> Putusan:
    """Urai jawaban model. Fungsi murni.

    Model kerap membungkus JSON dengan teks atau pagar kode, jadi objek JSON
    pertama diekstrak alih-alih menuntut seluruh keluaran berupa JSON.
    """
    if not raw or not raw.strip():
        return Putusan(error="jawaban kosong")

    cocok = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not cocok:
        return Putusan(error=f"tidak ada JSON di jawaban: {raw.strip()[:80]!r}")
    try:
        d = json.loads(cocok.group(0))
    except json.JSONDecodeError as e:
        return Putusan(error=f"JSON tidak valid: {e}")
    if not isinstance(d, dict):
        return Putusan(error="JSON bukan object")

    putusan = str(d.get("putusan", "")).strip().upper()
    verdict = _PUTUSAN.get(putusan)
    if verdict is None:
        return Putusan(error=f"putusan tidak dikenal: {putusan!r}")
    return Putusan(
        verdict=verdict,
        keyakinan=str(d.get("keyakinan", "")).strip().lower(),
        alasan=" ".join(str(d.get("alasan", "")).split())[:200],
    )


def _tanya_model(png: bytes) -> str | None:
    """Kirim gambar ke model vision. Pola dan parameter sama dengan image_describer."""
    import httpx

    b64 = base64.b64encode(png).decode("utf-8")
    if LLM_PROVIDER == "vllm":
        payload = {
            "model": VISION_MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}],
            "temperature": VISION_TEMPERATURE,
            "max_tokens": VISION_MAX_TOKENS,
        }
        if RESEARCH_VISION_SEED >= 0:
            payload["seed"] = RESEARCH_VISION_SEED
        url = f"{LLM_BASE_URL}/v1/chat/completions"
        with httpx.Client(timeout=TIMEOUT_DETIK) as c:
            data = c.post(url, json=payload).raise_for_status().json()
        pilihan = (data or {}).get("choices") or []
        if not pilihan:
            logger.error("adjudikasi_bad_response provider=vllm keys=%s", list(data or {})[:6])
            return None
        return (pilihan[0].get("message") or {}).get("content")

    options = {
        "temperature": VISION_TEMPERATURE,
        "num_predict": VISION_MAX_TOKENS,
        "num_ctx": VISION_NUM_CTX,
    }
    if RESEARCH_VISION_SEED >= 0:
        options["seed"] = RESEARCH_VISION_SEED
    payload = {"model": VISION_MODEL, "prompt": PROMPT, "images": [b64],
               "stream": False, "options": options}
    with httpx.Client(timeout=TIMEOUT_DETIK) as c:
        data = c.post(f"{LLM_BASE_URL}/api/generate", json=payload).raise_for_status().json()
    if not isinstance(data, dict) or "response" not in data:
        logger.error("adjudikasi_bad_response provider=ollama keys=%s",
                     list(data)[:6] if isinstance(data, dict) else type(data).__name__)
        return None
    return data["response"]


def adjudikasi(pdf_path, hal_a: int, bbox_a, hal_b: int, bbox_b) -> Putusan:
    """Putuskan apakah potongan B kelanjutan tabel A. Memakai cache bila aktif.

    Kegagalan apa pun mengembalikan `verdict=None` dengan `error` terisi, dan
    TIDAK di-cache. Kegagalan transient tidak boleh membeku jadi putusan
    permanen — aturan yang sama sudah dipakai cache deskripsi gambar.
    """
    png_a, png_b = _render(pdf_path, hal_a, bbox_a), _render(pdf_path, hal_b, bbox_b)
    if png_a is None or png_b is None:
        return Putusan(error="render potongan gagal")
    png = susun(png_a, png_b)
    if png is None:
        return Putusan(error="penyusunan gambar gagal")

    piksel = _piksel(png)
    sha = hashlib.sha256(png).hexdigest()
    prov = _provenance()
    kunci = vision_cache.make_key(sha, VARIANT, PROMPT_SHA, prov["digest"])

    if vision_cache.enabled():
        tersimpan = vision_cache.get(kunci)
        if tersimpan is not None:
            alasan = tersimpan.description or ""
            return Putusan(
                verdict=tersimpan.verdict, alasan=alasan, dari_cache=True,
                halaman_dikarang=halaman_dikarang(alasan, (hal_a, hal_b)),
                piksel_dikirim=piksel,
            )

    try:
        raw = _tanya_model(png)
    except Exception as e:
        logger.error("adjudikasi_gagal hal=%s->%s error=%s", hal_a, hal_b, e)
        return Putusan(error=f"{type(e).__name__}: {e}")
    if raw is None:
        return Putusan(error="model tidak menjawab")

    hasil = parse_jawaban(raw)
    karangan = halaman_dikarang(hasil.alasan, (hal_a, hal_b))
    if karangan:
        logger.warning(
            "adjudikasi_halaman_dikarang dikirim=%s disebut_model=%s — alasan "
            "model tidak dapat dipercaya sebagai fakta", (hal_a, hal_b), karangan,
        )
    hasil = Putusan(
        verdict=hasil.verdict, keyakinan=hasil.keyakinan, alasan=hasil.alasan,
        error=hasil.error, halaman_dikarang=karangan, piksel_dikirim=piksel,
    )
    if hasil.verdict is not None and vision_cache.enabled():
        vision_cache.put(
            kunci, image_sha256=sha, variant=VARIANT, prompt_sha256=PROMPT_SHA,
            vision_model=VISION_MODEL, vision_model_digest=prov["digest"],
            verdict=hasil.verdict, description=hasil.alasan or None,
        )
    return hasil


def _piksel(png: bytes) -> int:
    """Jumlah piksel gambar yang dikirim. Penakar biaya: waktu model tumbuh
    seiring jumlah token gambar, dan token gambar tumbuh seiring piksel."""
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(png))
        return im.width * im.height
    except Exception:
        return 0


def _provenance() -> dict:
    """Digest model vision, lewat helper image_describer yang sudah ada."""
    try:
        from backend.services.image_describer import vision_provenance
        return {"digest": vision_provenance().get("vision_model_digest")}
    except Exception:
        return {"digest": None}


def provenance() -> dict:
    """Konfigurasi adjudikasi untuk dicatat di run_manifest.json.

    RENDER_DPI ikut karena ia mengubah biaya DAN berpotensi mengubah putusan;
    dua run dengan DPI berbeda tidak dapat dibandingkan begitu saja.
    """
    return {
        "vision_model": VISION_MODEL,
        "vision_model_digest": _provenance()["digest"],
        "prompt_sha256": PROMPT_SHA,
        "render_dpi": RENDER_DPI,
        "timeout_detik": TIMEOUT_DETIK,
        "variant": VARIANT,
    }
