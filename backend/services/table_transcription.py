"""Transkripsi tabel dan klasifikasi gambar oleh vision, di dalam jalur indexing.

Dipanggil preprocessing.extract_from_pdf SETELAH gambar disimpan ke disk dan
SEBELUM deskripsi gambar serta chunking, sehingga index dapat dibangun ulang
dari satu jalan pipeline dan dump-nya identik untuk strategi mana pun.

Dua flag, sendiri-sendiri:
- INDEX_TABLE_TRANSCRIPTION: setiap element Table ditranskripsi dari area
  render (bbox ditarik ke kata utuh yang terpotong tepinya). Teks OCR pindah
  ke `teks_ocr`; `raw_html` TIDAK disentuh karena sidiknya menjaga berkas
  keputusan tabel lanjutan dan jangkar migrasi gold.
- INDEX_IMAGE_TABLE_TRANSCRIPTION: setiap element Image diklasifikasi (prompt
  v3: jenis + memuat_tabel_data) lalu diperlakukan menurut
  klasifikasi_gambar.perlakuan_gambar: buang (cap di atas tabel), transkripsi
  (gambar-tabel jadi element Table), narasi+tabel, atau narasi seperti v4.

Kegagalan model tidak pernah menghentikan run dan tidak pernah di-cache:
tabel jatuh ke teks OCR, gambar jatuh ke narasi, alasannya dicatat di
`transkripsi_peringatan` dan dihitung di manifest.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from pathlib import Path
from typing import Callable

from backend import config
from backend.services import vision_cache
from backend.services.klasifikasi_gambar import (
    PROMPT_KLASIFIKASI, Klasifikasi, perlakuan_gambar, urai_klasifikasi,
)
from backend.services.transkripsi_murni import (
    PROMPT_KOREKSI_KOLOM, PROMPT_TRANSKRIPSI, PROMPT_TRANSKRIPSI_SEMUA, AreaTidakSah,
    baris_meleset, klip_aman,
    bersihkan, bersihkan_semua, jenis_halaman, kolom_tidak_konsisten, perluas_bbox,
    peringatan, prompt_koreksi_kolom, rasio_tumpang, urai_markdown,
)
from backend.services.vision_io import GagalVision, panggil, siapkan_png

logger = logging.getLogger(__name__)

VARIANT_TABEL = "table_transcription"
VARIANT_TABEL_SEMUA = "table_transcription_all"
VARIANT_KLASIFIKASI = "image_class_v3"
VARIANT_KOREKSI = "table_transcription_fix"
PROMPT_PER_VARIANT = {
    VARIANT_TABEL: PROMPT_TRANSKRIPSI,
    VARIANT_TABEL_SEMUA: PROMPT_TRANSKRIPSI_SEMUA,
    VARIANT_KLASIFIKASI: PROMPT_KLASIFIKASI,
    # Templat; sha-nya yang dicatat. Lihat transkripsi_murni.PROMPT_KOREKSI_KOLOM.
    VARIANT_KOREKSI: PROMPT_KOREKSI_KOLOM,
}
PROMPT_SHA = {v: hashlib.sha256(p.encode("utf-8")).hexdigest() for v, p in PROMPT_PER_VARIANT.items()}
# JSON klasifikasi dua kunci ~20 token.
NUM_PREDICT_KLASIFIKASI = 40

_stats: Counter = Counter()


def reset_stats() -> None:
    _stats.clear()


def ringkasan_run() -> str:
    """Satu baris hitungan kunci untuk akhir run."""
    kunci = ("tabel_ditranskripsi", "tabel_fallback_ocr", "galat_elemen", "kolom_tidak_konsisten_awal",
             "koreksi_kolom_berhasil", "kolom_tidak_konsisten_akhir", "gagal_vision",
             "terpotong", "gambar_buang", "gambar_transkripsi", "gambar_narasi+tabel")
    return "  ".join(f"{k}={_stats.get(k, 0)}" for k in kunci)


def aktif() -> bool:
    return config.INDEX_TABLE_TRANSCRIPTION or config.INDEX_IMAGE_TABLE_TRANSCRIPTION


def provenance() -> dict | None:
    """Konfigurasi dan hitungan run untuk run_manifest.json, atau None bila mati."""
    if not aktif():
        return None
    from backend.services.image_describer import vision_provenance
    from backend.services.vision_io import batas_model
    try:
        batas = batas_model()
    except Exception as e:      # manifest tidak boleh menjatuhkan dump
        batas = {"error": str(e)}
    return {
        "INDEX_TABLE_TRANSCRIPTION": config.INDEX_TABLE_TRANSCRIPTION,
        "INDEX_IMAGE_TABLE_TRANSCRIPTION": config.INDEX_IMAGE_TABLE_TRANSCRIPTION,
        "render_dpi": config.TABLE_TRANSCRIPTION_DPI,
        "num_predict": config.TABLE_TRANSCRIPTION_NUM_PREDICT,
        "num_ctx": config.TABLE_TRANSCRIPTION_NUM_CTX,
        "render_scan_margin": config.TABLE_RENDER_SCAN_MARGIN,
        "classification_dpi": config.IMAGE_CLASSIFICATION_DPI,
        "classification_side": config.IMAGE_CLASSIFICATION_SIDE,
        "cap_min_overlap": config.IMAGE_CAP_MIN_OVERLAP,
        "prompt_sha256": dict(PROMPT_SHA),
        "vision_model": config.VISION_MODEL,
        "vision_model_digest": vision_provenance()["vision_model_digest"],
        "batas_gambar": batas,
        "hitungan": dict(sorted(_stats.items())),
    }


# ─── Model dengan cache ───────────────────────────────────────────────────────

def _tanya(png: bytes, variant: str, num_predict: int,
           sah: Callable[[str], bool], verdict: str,
           prompt: str | None = None, kunci_tambahan: str = "") -> tuple[str | None, str]:
    """(jawaban mentah yang sah, alasan gagal). Hanya jawaban sah yang di-cache.

    Gagal = galat jaringan setelah coba ulang, terpotong (done_reason=length;
    tabel yang barisnya hilang tanpa jejak lebih buruk daripada teks OCR), atau
    tidak lolos `sah`. Tak satu pun masuk cache, supaya run berikutnya mencoba lagi.

    `prompt`: teks prompt terisi untuk varian bertemplat (koreksi kolom). Kunci
    cache memakai sha TEMPLAT; `kunci_tambahan` (nilai yang mengisi templat)
    ikut ke komponen gambar kunci supaya isian berbeda tidak bertukar jawaban.
    """
    from backend.services.image_describer import vision_provenance
    img_sha = hashlib.sha256(png).hexdigest()
    digest = vision_provenance()["vision_model_digest"]
    key = (vision_cache.make_key(img_sha + kunci_tambahan, variant, PROMPT_SHA[variant], digest)
           if vision_cache.enabled() else None)
    if key is not None:
        hit = vision_cache.get(key)
        if hit is not None and hit.verdict == verdict:
            _stats["cache_hit"] += 1
            return hit.description, ""
    try:
        h = panggil(png, prompt or PROMPT_PER_VARIANT[variant], num_predict=num_predict,
                    num_ctx=config.TABLE_TRANSCRIPTION_NUM_CTX)
    except GagalVision as e:
        _stats["gagal_vision"] += 1
        return None, f"gagal_vision:{e}"
    _stats["dipadding"] += int(h["dipadding"])
    _stats["diulang"] += int(h["percobaan"] > 1)
    if h["done_reason"] == "length":
        _stats["terpotong"] += 1
        return None, "terpotong"
    if not sah(h["response"]):
        # Jawaban lengkap tapi bukan tabel/label yang sah. Tidak di-cache, jadi
        # cuplikannya dicatat di sini — satu-satunya jejak untuk diagnosis
        # (sop12 p10_c00: isian formulir tanpa garis yang dideteksi OCR sebagai Table).
        logger.warning("jawaban_tak_sah variant=%s sha=%s cuplikan=%r",
                       variant, img_sha[:12], h["response"][:400])
        return None, "tak_terurai"
    if key is not None:
        vision_cache.put(key, image_sha256=img_sha, variant=variant,
                         prompt_sha256=PROMPT_SHA[variant], vision_model=config.VISION_MODEL,
                         vision_model_digest=digest, verdict=verdict,
                         description=h["response"])
    return h["response"], ""


def transkripsi_satu_tabel(png: bytes) -> tuple[str | None, str]:
    """(markdown, alasan gagal) satu tabel, dengan SATU koreksi kolom.

    Bila jumlah sel header tidak sama dengan baris data, model diminta ulang
    dengan prompt koreksi yang menyebut angkanya. Yang disimpan hasil dengan
    baris meleset paling sedikit (seri: yang pertama). Tetap tidak konsisten:
    TIDAK jatuh ke OCR — di UKT p5 dan standar-biaya OCR-nya lebih buruk;
    penanda kolom_tidak_konsisten dipasang oleh peringatan().
    """
    raw, alasan = _tanya(png, VARIANT_TABEL, config.TABLE_TRANSCRIPTION_NUM_PREDICT,
                         _tabel_sah, vision_cache.VERDICT_TRANSCRIBED)
    if raw is None:
        return None, alasan
    md = bersihkan(raw)
    baris = urai_markdown(md)
    kolom = kolom_tidak_konsisten(baris)
    if kolom is None:
        return md, ""
    _stats["kolom_tidak_konsisten_awal"] += 1
    raw2, alasan2 = _tanya(png, VARIANT_KOREKSI, config.TABLE_TRANSCRIPTION_NUM_PREDICT,
                           _tabel_sah, vision_cache.VERDICT_TRANSCRIBED,
                           prompt=prompt_koreksi_kolom(*kolom),
                           kunci_tambahan=f":{kolom[0]}:{','.join(map(str, kolom[1]))}")
    if raw2 is not None:
        md2 = bersihkan(raw2)
        baris2 = urai_markdown(md2)
        if baris_meleset(baris2) < baris_meleset(baris):
            md, baris = md2, baris2
    if kolom_tidak_konsisten(baris) is None:
        _stats["koreksi_kolom_berhasil"] += 1
    else:
        _stats["kolom_tidak_konsisten_akhir"] += 1
        logger.warning("kolom_tidak_konsisten_setelah_koreksi awal=%s koreksi=%s",
                       kolom, alasan2 or kolom_tidak_konsisten(urai_markdown(bersihkan(raw2 or ""))))
    return md, ""


def _tabel_sah(raw: str) -> bool:
    return urai_markdown(bersihkan(raw)) is not None


def _tabel_semua_sah(raw: str) -> bool:
    return bool(bersihkan_semua(raw))


def _klasifikasi_sah(raw: str) -> bool:
    return urai_klasifikasi(raw).jenis is not None


# ─── Halaman PDF ──────────────────────────────────────────────────────────────

class _Halaman:
    """Kata ternormalisasi, jenis lapisan teks, dan teks satu halaman. Dibaca sekali."""

    def __init__(self, page):
        r = page.rect
        luas = (r.width * r.height) or 1.0
        kata = page.get_text("words")
        terbesar = max((abs(i["bbox"][2] - i["bbox"][0]) * abs(i["bbox"][3] - i["bbox"][1]) / luas
                        for i in page.get_image_info()), default=0.0)
        self.page = page
        self.jenis = jenis_halaman(bool(kata), terbesar)
        self.teks = page.get_text("text")
        self.kata = tuple((w[0] / r.width, w[1] / r.height, w[2] / r.width, w[3] / r.height)
                          for w in kata)

    def render(self, bbox, dpi: int) -> bytes:
        """PNG area bbox. AreaTidakSah bila area atau pixmap-nya nol — dipakai
        SEMUA jalur (klasifikasi, transkripsi tabel/gambar, koreksi)."""
        import fitz
        r = self.page.rect
        x0, y0, x1, y1 = klip_aman(bbox, r.width, r.height)
        klip = fitz.Rect(r.x0 + x0, r.y0 + y0, r.x0 + x1, r.y0 + y1)
        pm = self.page.get_pixmap(dpi=dpi, clip=klip)
        if pm.width < 1 or pm.height < 1:
            raise AreaTidakSah(f"pixmap {pm.width}x{pm.height} untuk bbox {bbox!r}")
        return pm.tobytes("png")


def _rujukan(hal: _Halaman, teks_ocr: str) -> tuple[str, str]:
    """(label, teks) rujukan angka. Lapisan OCR pindaian bukan kebenaran —
    labelnya dibawa supaya penandanya dibaca sebagai petunjuk lemah."""
    if hal.jenis == "digital_asli":
        return "lapisan_teks", hal.teks
    if hal.jenis == "pindai_lapisan_ocr":
        return "lapisan_ocr", hal.teks
    return "teks_ocr", teks_ocr


# ─── Tabel ────────────────────────────────────────────────────────────────────

def area_render(bbox, hal: _Halaman, milik_lain) -> list[float]:
    """bbox Table yang ditarik ke kata utuh beririsan (margin tetap bila tanpa lapisan).

    `milik_lain`: bbox Table dan gambar lain di halaman yang sama. Kata milik
    chunk teks TETAP menarik — memotongnya menghilangkan kolom harga
    standar-biaya (349 tabel) demi 1 item gold.
    """
    if hal.jenis == "pindai_tanpa_lapisan":
        return perluas_bbox(bbox, (), config.TABLE_RENDER_SCAN_MARGIN)
    return perluas_bbox(bbox, hal.kata, milik_lain=milik_lain)


def transkripsi_tabel(el: dict, hal: _Halaman, milik_lain) -> dict:
    """Element Table baru berisi transkripsi, atau element lama + penanda fallback."""
    meta = el.get("metadata") or {}
    bbox = meta.get("bbox")
    if not bbox:
        _stats["tabel_tanpa_bbox"] += 1
        return {**el, "metadata": {**meta, "table_source": "ocr_fallback",
                                    "transkripsi_peringatan": ["gagal:tanpa_bbox"]}}
    area = area_render(bbox, hal, milik_lain)
    png = hal.render(area, config.TABLE_TRANSCRIPTION_DPI)
    md, alasan = transkripsi_satu_tabel(png)
    if md is None:
        _stats["tabel_fallback_ocr"] += 1
        logger.warning("tabel_fallback_ocr halaman=%s bbox=%s alasan=%s",
                       el.get("page"), bbox, alasan)
        return {**el, "metadata": {**meta, "table_source": "ocr_fallback", "render_bbox": area,
                                    "transkripsi_peringatan": [f"gagal:{alasan}"]}}
    baris = urai_markdown(md)
    label, teks_rujukan = _rujukan(hal, el.get("text") or "")
    tanda = list(peringatan(baris, teks_rujukan))
    _stats["tabel_ditranskripsi"] += 1
    _stats["tabel_berpenanda"] += bool(tanda)
    return {**el, "text": md, "metadata": {
        **meta, "teks_ocr": el.get("text") or "", "table_source": "vision_transcription",
        "table_format": "markdown", "render_bbox": area, "transkripsi_baris": baris,
        "transkripsi_peringatan": [f"rujukan={label}", *tanda] if tanda else [],
    }}


# ─── Gambar ───────────────────────────────────────────────────────────────────

def klasifikasi(el: dict, hal: _Halaman, tabel_sehalaman) -> tuple[Klasifikasi, float, str]:
    """(klasifikasi, rasio tumpang terbesar dengan Table sehalaman, perlakuan)."""
    bbox = (el.get("metadata") or {}).get("bbox")
    if not bbox:
        _stats["gambar_tanpa_bbox"] += 1
        return Klasifikasi(None, None), 0.0, "narasi"
    rasio = max((rasio_tumpang(bbox, t) for t in tabel_sehalaman), default=0.0)
    png, _ = siapkan_png(hal.render(bbox, config.IMAGE_CLASSIFICATION_DPI),
                         config.IMAGE_CLASSIFICATION_SIDE)
    raw, alasan = _tanya(png, VARIANT_KLASIFIKASI, NUM_PREDICT_KLASIFIKASI,
                         _klasifikasi_sah, vision_cache.VERDICT_CLASSIFIED)
    k = urai_klasifikasi(raw)
    if k.jenis is None:
        # Jawaban tak dikenali atau gagal: narasi seperti v4, tidak ada isi hilang.
        _stats["klasifikasi_gagal"] += 1
        logger.warning("klasifikasi_gagal image_id=%s alasan=%s",
                       (el.get("metadata") or {}).get("image_id"), alasan)
    perlakuan = perlakuan_gambar(k, rasio, config.IMAGE_CAP_MIN_OVERLAP)
    _stats[f"gambar_{perlakuan}"] += 1
    return k, rasio, perlakuan


def _transkripsi_gambar(el: dict, hal: _Halaman, semua: bool) -> tuple[str | None, str]:
    """(markdown, alasan gagal). `semua`: prompt setiap tabel untuk narasi+tabel."""
    png = hal.render(el["metadata"]["bbox"], config.TABLE_TRANSCRIPTION_DPI)
    if not semua:
        return transkripsi_satu_tabel(png)
    raw, alasan = _tanya(png, VARIANT_TABEL_SEMUA, config.TABLE_TRANSCRIPTION_NUM_PREDICT,
                         _tabel_semua_sah, vision_cache.VERDICT_TRANSCRIBED)
    if raw is None:
        return None, alasan
    return "\n\n".join(bersihkan_semua(raw)), ""


def _tanda_kolom(md: str) -> list[str]:
    """Penanda kolom_tidak_konsisten per tabel, untuk transkripsi gambar.

    Gambar tidak punya rujukan angka (tak ada lapisan teks di dalam gambar),
    jadi hanya konsistensi kolom yang dapat diperiksa.
    """
    tanda = []
    for tabel in bersihkan_semua(md) or [md]:
        k = kolom_tidak_konsisten(urai_markdown(tabel))
        if k:
            tanda.append(f"kolom_tidak_konsisten:{k[0]}/{','.join(map(str, k[1]))}")
    return tanda


def perlakukan_gambar(el: dict, hal: _Halaman, tabel_sehalaman) -> tuple[dict | None, dict]:
    """(element baru atau None bila dibuang, info untuk images.jsonl)."""
    k, rasio, perlakuan = klasifikasi(el, hal, tabel_sehalaman)
    meta = el.get("metadata") or {}
    info = {"klasifikasi_jenis": k.jenis, "memuat_tabel_data": k.memuat_tabel_data,
            "rasio_tumpang_tabel": round(rasio, 4), "perlakuan": perlakuan}
    if perlakuan == "buang":
        return None, {**info, "image_content": None}
    if perlakuan in ("transkripsi", "narasi+tabel"):
        md, alasan = _transkripsi_gambar(el, hal, semua=perlakuan == "narasi+tabel")
        if md is None:
            _stats[f"gambar_{perlakuan}_gagal"] += 1
            return ({**el, "metadata": {**meta, "image_content": "narasi",
                                        "transkripsi_peringatan": [f"gagal:{alasan}"]}},
                    {**info, "image_content": "narasi"})
        if perlakuan == "transkripsi":
            baris = urai_markdown(md)
            return ({"text": md, "category": "Table", "page": el.get("page"), "metadata": {
                        "raw_html": "", "table_format": "markdown", "bbox": meta.get("bbox"),
                        "image_id": meta.get("image_id"), "table_origin": "image",
                        "table_source": "vision_transcription", "image_content": "tabel",
                        "transkripsi_baris": baris, "transkripsi_peringatan": _tanda_kolom(md)}},
                    {**info, "image_content": "tabel"})
        return ({**el, "metadata": {**meta, "image_content": "narasi+tabel",
                                    "transkripsi_tabel": md,
                                    "transkripsi_peringatan": _tanda_kolom(md)}},
                {**info, "image_content": "narasi+tabel"})
    return {**el, "metadata": {**meta, "image_content": "narasi"}}, {**info, "image_content": "narasi"}


# ─── Satu dokumen ─────────────────────────────────────────────────────────────

def _lolos_deskripsi(el: dict) -> bool:
    """Gambar yang juga lolos ke deskripsi v4 — satu-satunya yang diklasifikasi.

    v4 tidak punya saringan ukuran (is_likely_informative adalah fungsi mati);
    yang menyaring adalah MODEL: gambar berputusan DEKORATIF/TIDAK JELAS, atau
    yang deskripsinya gagal, tidak menjadi chunk. Himpunan yang sama dipakai
    klasifikasi_gambar.py (chunk ImageDescription dump v4). Deskripsi di sini
    diambil dari cache; _describe_image_elements memanggilnya lagi dari cache
    proses-lokal, jadi tidak ada panggilan model tambahan. Gambar yang tidak
    lolos diteruskan apa adanya dan dibuang deskripsi persis seperti v4 —
    termasuk garis/serpihan yang bbox-nya merosot jadi nol piksel.
    """
    import base64
    from backend.services.image_describer import describe_image
    akan = (config.PDF_DESCRIBE_IMAGES == "true"
            or (config.PDF_DESCRIBE_IMAGES == "auto" and config.LLM_SUPPORTS_VISION))
    b64 = (el.get("metadata") or {}).get("image_base64")
    if not akan or not b64:
        _stats["gambar_tidak_dideskripsi"] += 1
        return False
    try:
        lolos = bool(describe_image(base64.b64decode(b64)))
    except Exception as e:
        logger.warning("saringan_deskripsi_gagal image_id=%s error=%s: %s",
                       (el.get("metadata") or {}).get("image_id"), type(e).__name__, e)
        lolos = False
    if not lolos:
        _stats["gambar_tidak_dideskripsi"] += 1
    return lolos


def _aman(fungsi, fallback, el: dict, *args):
    """Satu element: galat APA PUN jatuh ke fallback element itu, bukan ke dokumen.

    Reindex v5 pertama kehilangan UKT dan Pedoman Tesis seluruhnya karena satu
    render gambar gagal. Galat dicatat (hitungan `galat_elemen`, log dengan
    traceback) supaya tidak tersembunyi.
    """
    try:
        return fungsi(el, *args)
    except Exception as e:
        _stats["galat_elemen"] += 1
        _stats[f"galat_{type(e).__name__}"] += 1
        meta = el.get("metadata") or {}
        logger.error("transkripsi_galat_elemen kategori=%s halaman=%s bbox=%s image_id=%s "
                     "error=%s: %s", el.get("category"), el.get("page"), meta.get("bbox"),
                     meta.get("image_id"), type(e).__name__, e, exc_info=True)
        return fallback(el, f"galat:{type(e).__name__}:{str(e)[:120]}")


def _fallback_tabel(el: dict, alasan: str) -> dict:
    _stats["tabel_fallback_ocr"] += 1
    meta = el.get("metadata") or {}
    return {**el, "metadata": {**meta, "table_source": "ocr_fallback",
                                "transkripsi_peringatan": [alasan]}}


def _fallback_gambar(el: dict, alasan: str) -> tuple[dict, dict]:
    meta = el.get("metadata") or {}
    return ({**el, "metadata": {**meta, "image_content": "narasi",
                                "transkripsi_peringatan": [alasan]}},
            {"klasifikasi_jenis": None, "memuat_tabel_data": None, "rasio_tumpang_tabel": None,
             "perlakuan": "narasi", "image_content": "narasi"})

def proses(elements: list[dict], pdf_path: Path,
           halaman_terpilih: set[int] | None = None) -> tuple[list[dict], dict[str, dict]]:
    """(element baru, info per image_id). Urutan element dipertahankan.

    `halaman_terpilih`: hanya element di halaman ini yang diproses; sisanya
    persis seperti flag mati. Untuk uji sampel lewat jalur pipeline tanpa
    mentranskripsi seluruh dokumen — indexing tidak pernah mengisinya.
    """
    if not aktif():
        return elements, {}
    import fitz

    tabel_per_hal: dict = {}
    gambar_per_hal: dict = {}
    for el in elements:
        b = (el.get("metadata") or {}).get("bbox")
        if b and el.get("category") == "Table":
            tabel_per_hal.setdefault(el.get("page"), []).append(b)
        elif b and el.get("category") == "Image":
            gambar_per_hal.setdefault(el.get("page"), []).append(b)

    halaman: dict[int, _Halaman] = {}
    keluar, info = [], {}
    with fitz.open(str(pdf_path)) as doc:
        def hal(no) -> _Halaman | None:
            if not isinstance(no, int) or not 1 <= no <= doc.page_count:
                return None
            if no not in halaman:
                halaman[no] = _Halaman(doc[no - 1])
            return halaman[no]

        for el in elements:
            kat, h = el.get("category"), hal(el.get("page"))
            if halaman_terpilih is not None and el.get("page") not in halaman_terpilih:
                keluar.append(el)
                continue
            if kat == "Table" and config.INDEX_TABLE_TRANSCRIPTION and h is not None:
                b = (el.get("metadata") or {}).get("bbox")
                lain = [x for x in tabel_per_hal.get(el.get("page"), []) if x is not b]
                lain += gambar_per_hal.get(el.get("page"), [])
                keluar.append(_aman(transkripsi_tabel, _fallback_tabel, el, h, lain))
            elif (kat == "Image" and config.INDEX_IMAGE_TABLE_TRANSCRIPTION and h is not None
                  and _lolos_deskripsi(el)):
                baru, i = _aman(perlakukan_gambar, _fallback_gambar, el, h,
                                tabel_per_hal.get(el.get("page"), []))
                image_id = (el.get("metadata") or {}).get("image_id")
                if image_id:
                    info[image_id] = i
                if baru is not None:
                    keluar.append(baru)
            else:
                keluar.append(el)
    logger.info("transkripsi_tabel file=%s hitungan=%s", pdf_path.name, dict(_stats))
    return keluar, info
