"""PDF preprocessing — production-grade extraction dengan layout awareness.

Strategi:
- "fast": PyMuPDF text-only + PaddleOCR fallback (cepat, untuk SOP/dokumen text-heavy)
- "hi_res": Unstructured.io partition_pdf dengan layout detection (lambat tapi extract tabel + gambar)
- "auto": deteksi otomatis per file berdasarkan jumlah gambar/tabel yang terdeteksi

Element types yang dihasilkan:
- text: paragraf, daftar, judul
- table: tabel terstruktur (markdown)
- image_description: deskripsi gambar dari vision LLM
"""

import hashlib
import logging
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import fitz  # pymupdf

from backend.config import (
    OCR_LANG, OCR_USE_GPU,
    CHUNK_SIZE, CHUNK_OVERLAP,
    IMAGES_DIR,
    INDEX_MAX_CHUNK_TOKENS,
    INDEX_MIN_CHUNK_TOKENS,
    INDEX_PERSIST_IMAGES,
    INDEX_STRUCTURAL_METADATA,
    INDEX_TABLES_AS_OWN_CHUNKS,
    PDF_EXTRACTION_STRATEGY,
    PDF_EXTRACT_IMAGES, PDF_DESCRIBE_IMAGES,
    PDF_EXTRACT_TABLES, PDF_TABLE_MAX_CHARS,
    LLM_SUPPORTS_VISION,
)

logger = logging.getLogger(__name__)

# ─── Lazy initialized engines ─────────────────────────────────────────────────

_ocr_engine = None
_tokenizer = None


def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR
        logger.info("ocr_engine_init lang=%s gpu=%s", OCR_LANG, OCR_USE_GPU)
        # PaddleOCR 3.x: use_angle_cls → use_textline_orientation (renamed).
        # device terima 'gpu', 'cpu', atau 'gpu:0' untuk pilih device spesifik.
        _ocr_engine = PaddleOCR(
            use_textline_orientation=True,
            lang=OCR_LANG,
            device="gpu" if OCR_USE_GPU else "cpu",
        )
    return _ocr_engine


def _extract_texts_from_ocr_result(result) -> list[str]:
    """Robust text extractor untuk PaddleOCR 3.x predict() result.

    result = list[OCRResult]. Setiap OCRResult adalah dict-like dengan key 'rec_texts'.
    Fallback ke .json property kalau dict-access tidak tersedia (defensif untuk
    perubahan minor antar versi 3.x).
    """
    texts: list[str] = []
    for ocr_res in result:
        rec_texts: list[str] = []

        # Path 1: dict-like access (paddleocr 3.0.x canonical)
        try:
            rec_texts = list(ocr_res["rec_texts"])
        except (KeyError, TypeError, AttributeError):
            pass

        # Path 2: .json property (fallback)
        if not rec_texts:
            try:
                data = getattr(ocr_res, "json", None)
                if isinstance(data, dict):
                    rec_texts = list(
                        data.get("rec_texts")
                        or data.get("res", {}).get("rec_texts", [])
                    )
            except Exception:
                pass

        texts.extend(t for t in rec_texts if t)

    return texts


# ─── File hashing untuk incremental indexing ──────────────────────────────────

def file_sha256(path: str | Path) -> str:
    """SHA256 hash isi file (untuk deteksi perubahan saat re-index)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ─── Hash teks chunk (jaring pengaman pemetaan ulang anotasi gold) ────────────

_WHITESPACE_RUN_RE = re.compile(r"\s+")


def normalize_for_hash(text: str) -> str:
    """Normalisasi teks sebelum di-hash.

    NFKC menyatukan bentuk unicode yang setara; deretan whitespace dirapatkan
    karena ekstraksi PDF dan OCR menghasilkan spasi/newline yang tidak stabil
    antar-run. Kapitalisasi TIDAK diturunkan: casing stabil antar re-index, dan
    menurunkannya hanya menambah peluang tabrakan.
    """
    return _WHITESPACE_RUN_RE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def text_sha(text: str) -> str:
    """Hash pendek isi chunk, untuk memetakan ulang anotasi gold saat
    penomoran bergeser. 16 hex = 64 bit, cukup untuk skala korpus akademik.

    Dihitung atas konten milik chunk itu SENDIRI — sebelum prefix overlap
    ditempel — sehingga tidak ikut berubah saat chunk tetangga berubah.
    """
    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()[:16]


# ─── Laporan degradasi per dokumen ────────────────────────────────────────────

@dataclass
class ExtractionReport:
    """Catatan apa yang TIDAK berjalan sebagaimana mestinya saat ekstraksi.

    Dibutuhkan riset: dokumen yang jatuh dari hi_res ke fast tidak punya tabel
    terstruktur maupun koordinat, dan halaman yang gagal OCR tidak punya teks
    sama sekali. Keduanya sebelumnya hanya muncul sebagai baris log yang mudah
    terlewat, padahal menentukan apakah sebuah dokumen layak masuk analisis.
    """
    strategy_requested: str = ""
    strategy_used: str = ""
    hi_res_fallback_reason: str | None = None
    ocr_failed_pages: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "strategy_requested": self.strategy_requested,
            "strategy_used": self.strategy_used,
            "hi_res_fallback_reason": self.hi_res_fallback_reason,
            "ocr_failed_pages": list(self.ocr_failed_pages),
        }

    @property
    def degraded(self) -> bool:
        return bool(self.hi_res_fallback_reason or self.ocr_failed_pages)


# ─── Strategy detection ───────────────────────────────────────────────────────

# Rata-rata gambar per halaman di atas ambang ini -> hi_res.
STRATEGY_IMAGE_THRESHOLD = 1.0
# Jumlah halaman pertama yang disampel detect_strategy.
STRATEGY_SAMPLE_PAGES = 5


def detect_strategy(pdf_path: Path) -> str:
    """Pilih strategy otomatis berdasarkan karakteristik PDF.

    - Banyak gambar/tabel → hi_res
    - Pure text → fast
    """
    try:
        doc = fitz.open(str(pdf_path))
        total_images = 0
        total_pages = len(doc)
        sample_pages = min(STRATEGY_SAMPLE_PAGES, total_pages)

        for i in range(sample_pages):
            page = doc[i]
            total_images += len(page.get_images())

        doc.close()

        avg_images_per_page = total_images / max(sample_pages, 1)
        # Heuristic: > 1 image per page → kemungkinan rich content
        return "hi_res" if avg_images_per_page > STRATEGY_IMAGE_THRESHOLD else "fast"
    except Exception as e:
        logger.debug(f"detect_strategy_failed file={pdf_path.name} error={e}")
        return "fast"


# ─── FAST PATH — PyMuPDF + OCR fallback ──────────────────────────────────────

# Ambang teks di bawahnya halaman dianggap hasil scan dan dilempar ke OCR.
OCR_TEXT_THRESHOLD_CHARS = 50
# Resolusi render halaman sebelum dikirim ke OCR.
OCR_RENDER_DPI = 300


def _extract_text_from_page_fast(page: fitz.Page) -> tuple[str, str | None]:
    """Extract text via PyMuPDF, fallback ke PaddleOCR jika scan.

    Returns (text, ocr_error). `ocr_error` berisi ringkasan kegagalan bila OCR
    tidak dapat dijalankan; teks yang dikembalikan lalu apa adanya dari PyMuPDF
    (bisa kosong).

    Sebelumnya blok try di sini hanya punya `finally`, sehingga ImportError
    paddleocr atau kegagalan predict merambat naik lewat _extract_fast dan
    extract_from_pdf sampai ditangkap indexing.py — yang lalu MELEWATI SELURUH
    PDF. Satu halaman scan di halaman 40 membuang 99 halaman lain. Sekarang
    degradasinya per halaman.
    """
    text = page.get_text().strip()
    if len(text) > OCR_TEXT_THRESHOLD_CHARS:
        return text, None

    logger.info(f"page_ocr_fallback page={page.number + 1}")
    pix = page.get_pixmap(dpi=OCR_RENDER_DPI)
    img_bytes = pix.tobytes("png")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(img_bytes)
        tmp_path = tmp.name

    try:
        ocr = _get_ocr_engine()
        # PaddleOCR 3.x: .predict() ganti .ocr(). Return list[OCRResult].
        result = ocr.predict(tmp_path)
        if not result:
            return text, None
        texts = _extract_texts_from_ocr_result(result)
        return ("\n".join(texts) if texts else text), None
    except Exception as e:
        logger.error(
            "page_ocr_failed page=%d error=%s: %s — halaman didegradasi ke teks "
            "PyMuPDF apa adanya (%d karakter)",
            page.number + 1, type(e).__name__, e, len(text),
        )
        return text, f"{type(e).__name__}: {e}"
    finally:
        os.unlink(tmp_path)


def _extract_fast(pdf_path: Path, report: "ExtractionReport | None" = None) -> list[dict]:
    """Fast path: pure text extraction. Return list of elements per page."""
    doc = fitz.open(str(pdf_path))
    elements = []

    for page in doc:
        text, ocr_error = _extract_text_from_page_fast(page)
        if ocr_error and report is not None:
            report.ocr_failed_pages.append({
                "page": page.number + 1,
                "error": ocr_error,
                "fallback_chars": len(text.strip()),
            })
        if not text.strip():
            continue
        elements.append({
            "text": text,
            "category": "NarrativeText",
            "page": page.number + 1,
            "metadata": {},
        })

    doc.close()
    return elements


# ─── HI_RES PATH — Unstructured.io layout-aware ───────────────────────────────

def _html_table_to_markdown(html: str) -> tuple[str, str]:
    """Convert tabel HTML dari Unstructured ke Markdown.

    Returns (text, format) dengan format "markdown" atau "html". Saat markdownify
    gagal, fungsi ini mengembalikan HTML mentah — tanpa penanda, chunk
    element_type="Table" bisa berisi Markdown atau HTML tanpa cara membedakannya.
    Penanda inilah yang diteruskan ke payload sebagai `table_format`.
    """
    try:
        from markdownify import markdownify
        return markdownify(html, heading_style="ATX").strip(), "markdown"
    except Exception as e:
        logger.warning("html_table_md_fail error=%s — chunk berisi HTML mentah", e)
        return html, "html"


def _element_bbox(el) -> list[float] | None:
    """Bounding box element sebagai [x0, y0, x1, y1] ternormalisasi ke [0, 1].

    `el.metadata.coordinates` (unstructured 0.16.11, elements.py:164) berisi
    `.points` — tuple pasangan (x, y) di ruang piksel layout — dan `.system`
    dengan `.width`/`.height`. Piksel layout tidak bermakna di luar konteks
    ekstraksi itu, jadi koordinat dinormalisasi terhadap dimensi layout supaya
    bisa dipakai lintas alat tanpa perlu menyimpan dimensinya.

    Mengembalikan None bila koordinat tidak tersedia — `bbox` di skema riset
    memang bertipe "array[float] atau null, bila tersedia".
    """
    coords = getattr(getattr(el, "metadata", None), "coordinates", None)
    if coords is None:
        return None

    points = getattr(coords, "points", None)
    system = getattr(coords, "system", None)
    if not points or system is None:
        return None

    width = getattr(system, "width", None)
    height = getattr(system, "height", None)
    if not width or not height:
        return None

    try:
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
    except (TypeError, IndexError, ValueError):
        return None

    if not xs or not ys:
        return None

    return [
        round(min(xs) / width, 4),
        round(min(ys) / height, 4),
        round(max(xs) / width, 4),
        round(max(ys) / height, 4),
    ]


def _extract_hi_res(
    pdf_path: Path, report: "ExtractionReport | None" = None
) -> list[dict]:
    """Hi-res path: layout-aware extraction via Unstructured.io.

    Mengeluarkan element terstruktur: Title, NarrativeText, Table, Image, dll.

    Bila partition_pdf gagal, fungsi ini jatuh ke jalur fast — dan itu DICATAT
    di `report`. Dokumen yang jatuh tidak punya tabel terstruktur, tidak punya
    raw_html, dan tidak punya koordinat; tanpa catatan, kegagalan itu hanya
    tampak sebagai satu baris log yang mudah terlewat.
    """
    from unstructured.partition.pdf import partition_pdf

    extract_image_block_types = []
    if PDF_EXTRACT_IMAGES:
        extract_image_block_types = ["Image", "Figure"]

    try:
        raw_elements = partition_pdf(
            filename=str(pdf_path),
            strategy="hi_res",
            infer_table_structure=PDF_EXTRACT_TABLES,
            extract_image_block_types=extract_image_block_types,
            extract_image_block_to_payload=True,  # base64 di metadata
            languages=["ind", "eng"],
        )
    except Exception as e:
        logger.warning(f"hi_res_failed_fallback_fast file={pdf_path.name} error={e}")
        if report is not None:
            report.hi_res_fallback_reason = f"{type(e).__name__}: {e}"
            report.strategy_used = "fast"
        return _extract_fast(pdf_path, report=report)

    elements = []
    for el in raw_elements:
        category = el.category
        page = getattr(el.metadata, "page_number", None) or 0

        # Skip noise
        if category in ("Header", "Footer", "PageNumber", "PageBreak"):
            continue

        bbox = _element_bbox(el)

        if category == "Table":
            html = getattr(el.metadata, "text_as_html", None)
            if html:
                text, table_format = _html_table_to_markdown(html)
            else:
                text, table_format = el.text, "text"
            elements.append({
                "text": text,
                "category": "Table",
                "page": page,
                "metadata": {
                    "raw_html": html or "",
                    "table_format": table_format,
                    "bbox": bbox,
                },
            })

        elif category in ("Image", "Figure"):
            # base64 image di metadata, akan di-describe nanti
            image_b64 = getattr(el.metadata, "image_base64", None)
            if not image_b64:
                continue
            elements.append({
                "text": "",  # diisi nanti dengan deskripsi
                "category": "Image",
                "page": page,
                "metadata": {"image_base64": image_b64, "bbox": bbox},
            })

        else:
            # Title, NarrativeText, ListItem, dll → text biasa
            text = el.text.strip() if el.text else ""
            if not text:
                continue
            elements.append({
                "text": text,
                "category": category,
                "page": page,
                "metadata": {"bbox": bbox},
            })

    return elements


# ─── Penyimpanan gambar ke disk ───────────────────────────────────────────────

# Format PIL -> ekstensi berkas. MIME/ekstensi TIDAK diasumsikan PNG: gambar
# yang tidak di-resize keluar dalam format aslinya.
_PIL_FORMAT_EXT = {
    "PNG": ("png", "image/png"),
    "JPEG": ("jpg", "image/jpeg"),
    "WEBP": ("webp", "image/webp"),
    "GIF": ("gif", "image/gif"),
    "TIFF": ("tiff", "image/tiff"),
    "BMP": ("bmp", "image/bmp"),
}


def _probe_image(raw: bytes) -> tuple[str, str, int | None, int | None]:
    """(ekstensi, mime, width, height) dari bytes gambar apa adanya.

    Fallback ke png/image/png hanya bila PIL tidak dapat mengenali formatnya.
    """
    try:
        from io import BytesIO
        from PIL import Image
        with Image.open(BytesIO(raw)) as img:
            fmt = (img.format or "").upper()
            w, h = img.size
        ext, mime = _PIL_FORMAT_EXT.get(fmt, ("png", "image/png"))
        return ext, mime, w, h
    except Exception as e:
        logger.debug("image_probe_failed error=%s", e)
        return "png", "image/png", None, None


def _persist_image_elements(
    elements: list[dict], document_id: str | None, pdf_name: str
) -> tuple[list[dict], list[dict]]:
    """Tulis setiap element Image ke disk dan beri identitas.

    Dipanggil SEBELUM _describe_image_elements dengan sengaja: gambar yang nanti
    dinilai DEKORATIF atau gagal dideskripsikan tetap harus tersimpan, karena
    strategi pembanding memvektorkan gambar aslinya dan riset ini harus bisa
    menilai ulang tanpa mengulang ekstraksi PDF.

    Penamaan mengikuti skema images.jsonl:
        image_id  = {document_id}_p{page}_img{NN}
        file_path = images/{document_id}/p{page}_img{NN}.{ext}   (relatif ke data/)

    `sha256` dihitung atas bytes ASLI, sebelum resize apa pun — resize adalah
    re-encode lossy yang keluarannya bergantung versi Pillow, jadi hash setelah
    resize tidak stabil antar lingkungan.

    Returns (elements, image_records). Tanpa document_id atau saat flag mati,
    elements dikembalikan apa adanya dan records kosong.
    """
    if not INDEX_PERSIST_IMAGES:
        return elements, []
    if not document_id:
        logger.error(
            "image_persist_skipped file=%s — INDEX_PERSIST_IMAGES aktif tapi "
            "document_id tidak ada; gambar tidak disimpan", pdf_name,
        )
        return elements, []

    import base64

    target_dir = Path(IMAGES_DIR) / document_id
    target_dir.mkdir(parents=True, exist_ok=True)

    counters: dict[str, int] = {}
    records: list[dict] = []
    out: list[dict] = []

    for el in elements:
        if el.get("category") != "Image":
            out.append(el)
            continue

        b64 = (el.get("metadata") or {}).get("image_base64")
        if not b64:
            out.append(el)
            continue

        try:
            raw = base64.b64decode(b64)
        except Exception as e:
            logger.error("image_decode_failed file=%s error=%s", pdf_name, e)
            out.append(el)
            continue

        segment = _page_segment(el.get("page"))
        ordinal = counters.get(segment, 0)
        counters[segment] = ordinal + 1

        ext, mime, width, height = _probe_image(raw)
        image_id = f"{document_id}_{segment}_img{ordinal:02d}"
        rel_path = f"images/{document_id}/{segment}_img{ordinal:02d}.{ext}"

        try:
            (target_dir / f"{segment}_img{ordinal:02d}.{ext}").write_bytes(raw)
        except OSError as e:
            logger.error("image_write_failed id=%s error=%s", image_id, e)
            out.append(el)
            continue

        records.append({
            "image_id": image_id,
            "document_id": document_id,
            "page_number": el.get("page") if _page_segment(el.get("page")) != "pNA" else None,
            "file_path": rel_path,
            "visual_type": None,        # anotasi manusia, menyusul
            "structured_summary": None,  # varian (c), menyusul
            "narrative_summary": None,   # diisi setelah _describe_image_elements
            # ── kolom tambahan di luar skema minimum ──
            "sha256": hashlib.sha256(raw).hexdigest(),
            "mime_type": mime,
            "size_bytes": len(raw),
            "width": width,
            "height": height,
            "source_file": pdf_name,
        })

        out.append({**el, "metadata": {**el["metadata"], "image_id": image_id}})

    if records:
        logger.info(
            "images_persisted file=%s count=%d dir=%s", pdf_name, len(records), target_dir
        )
    return out, records


# ─── Image description (panggil vision LLM) ───────────────────────────────────

def _describe_image_elements(elements: list[dict]) -> list[dict]:
    """Untuk setiap element Image, panggil vision LLM untuk deskripsi.

    Element yang gagal dideskripsi (vision off, dekoratif, error) dihapus dari list.
    """
    should_describe = (
        PDF_DESCRIBE_IMAGES == "true"
        or (PDF_DESCRIBE_IMAGES == "auto" and LLM_SUPPORTS_VISION)
    )

    if not should_describe:
        return [e for e in elements if e["category"] != "Image"]

    from backend.services.image_describer import describe_image
    import base64

    output = []
    for el in elements:
        if el["category"] != "Image":
            output.append(el)
            continue

        b64 = el["metadata"].get("image_base64")
        if not b64:
            continue

        try:
            image_bytes = base64.b64decode(b64)
        except Exception:
            continue

        description = describe_image(image_bytes)
        if not description:
            continue  # skip dekoratif atau error

        output.append({
            "text": description,
            "category": "ImageDescription",
            "page": el["page"],
            # bbox diteruskan; image_base64 tetap dibuang di sini (gambar belum
            # disimpan ke disk — lihat INSPECTION_REPORT_2.md butir #3).
            "metadata": {
                "bbox": el["metadata"].get("bbox"),
                "image_id": el["metadata"].get("image_id"),
            },
        })

    return output


# ─── Pembatas ukuran chunk (opt-in via INDEX_MAX_CHUNK_TOKENS) ───────────────

# Urutan pemisah dari yang paling menjaga struktur ke yang paling merusak.
# Baris ("\n") didahulukan sebelum kalimat karena page.get_text() di jalur fast
# memisahkan baris dengan newline tunggal, dan daftar bernomor SOP lebih baik
# pecah per baris daripada di tengah kalimat.
_SPLIT_SEPARATORS = ("\n\n", "\n", ". ", " ")


def _get_tokenizer():
    """Tokenizer yang SAMA dengan yang dipakai node parser LlamaIndex.

    Diimpor lazy supaya modul ini tetap bisa dipakai untuk ekstraksi murni
    tanpa llama-index terpasang.
    """
    global _tokenizer
    if _tokenizer is None:
        from llama_index.core.utils import get_tokenizer
        _tokenizer = get_tokenizer()
    return _tokenizer


def _ntok(text: str) -> int:
    return len(_get_tokenizer()(text))


def _pack(units: list[str], sep: str, budget: int) -> list[str]:
    """Gabungkan unit berurutan sampai mendekati budget token."""
    out: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}{sep}{unit}" if current else unit
        if current and _ntok(candidate) > budget:
            out.append(current)
            current = unit
        else:
            current = candidate
    if current:
        out.append(current)
    return out


def _hard_cut(text: str, budget: int) -> list[str]:
    """Potong paksa blok yang tidak punya pemisah apa pun.

    Memakai pencarian biner atas panjang karakter — tokenizer tidak menyediakan
    pemetaan balik token->karakter yang stabil.
    """
    pieces: list[str] = []
    sisa = text
    while sisa and _ntok(sisa) > budget:
        lo, hi, cut = 1, len(sisa), 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if _ntok(sisa[:mid]) <= budget:
                cut, lo = mid, mid + 1
            else:
                hi = mid - 1
        pieces.append(sisa[:cut])
        sisa = sisa[cut:]
    if sisa:
        pieces.append(sisa)
    return pieces


def _merge_fitting(pieces: list[str], budget: int) -> list[str]:
    """Gabungkan kembali potongan bertetangga yang muat dalam satu budget.

    Pemecahan berjenjang bisa meninggalkan serpihan kecil (mis. ekor paragraf
    atau baris judul). Penggabungan ini tidak pernah melewati budget, jadi aman
    dijalankan setelah pemecahan.
    """
    if len(pieces) < 2:
        return pieces

    out = [pieces[0]]
    for piece in pieces[1:]:
        candidate = f"{out[-1]}\n\n{piece}"
        if _ntok(candidate) <= budget:
            out[-1] = candidate
        else:
            out.append(piece)
    return out


def _apply_overlap(pieces: list[str], budget: int) -> list[str]:
    """Awali tiap potongan (selain pertama) dengan ekor potongan sebelumnya.

    Konsisten dengan overlap antar-chunk yang sudah ada di _chunk_elements.
    Prefix dipangkas bila membuat potongan balik melewati budget.
    """
    if len(pieces) < 2 or CHUNK_OVERLAP <= 0:
        return pieces

    out = [pieces[0]]
    for prev, piece in zip(pieces, pieces[1:]):
        prefix = prev[-CHUNK_OVERLAP:]
        while prefix and _ntok(f"{prefix}\n{piece}") > budget:
            prefix = prefix[len(prefix) // 2:] if len(prefix) > 8 else ""
        out.append(f"{prefix}\n{piece}" if prefix else piece)
    return out


def _split_for_budget(text: str, splittable: bool = True) -> list[str]:
    """Pecah `text` agar tiap potongan muat dalam INDEX_MAX_CHUNK_TOKENS.

    Mengembalikan [text] apa adanya bila fitur mati (budget <= 0), bila teks
    sudah muat, atau bila `splittable` False (tabel & deskripsi gambar).
    """
    if INDEX_MAX_CHUNK_TOKENS <= 0 or not text:
        return [text]

    budget = INDEX_MAX_CHUNK_TOKENS
    if _ntok(text) <= budget:
        return [text]

    if not splittable:
        logger.warning(
            "chunk_over_budget_kept_whole tokens=%d budget=%d chars=%d "
            "— sengaja tidak dipecah (relasi struktural)",
            _ntok(text), budget, len(text),
        )
        return [text]

    pieces = [text]
    for sep in _SPLIT_SEPARATORS:
        if all(_ntok(p) <= budget for p in pieces):
            break
        expanded: list[str] = []
        for p in pieces:
            expanded.extend(_pack(p.split(sep), sep, budget) if _ntok(p) > budget else [p])
        pieces = expanded

    final: list[str] = []
    for p in pieces:
        final.extend(_hard_cut(p, budget) if _ntok(p) > budget else [p])

    final = _merge_fitting([p for p in final if p.strip()], budget)
    logger.info(
        "chunk_split_oversized tokens=%d budget=%d pieces=%d",
        _ntok(text), budget, len(final),
    )
    # Overlap TIDAK diterapkan di sini. Pemanggil (`emit`) menempelkannya setelah
    # menghitung text_sha, supaya hash mencerminkan konten milik chunk itu
    # sendiri dan tidak ikut berubah saat chunk tetangga berubah.
    return final


# ─── Chunking — preserve section context ──────────────────────────────────────

def _page_segment(page) -> str:
    """Segmen halaman untuk chunk_id/image_id.

    `_extract_hi_res` memakai 0 sebagai penanda "halaman tidak diketahui"
    (`page_number` absen). Nilai itu ditangani eksplisit sebagai "pNA" — memakai
    "p0" akan terbaca sebagai halaman nol dan mencampur dua hal berbeda.
    """
    try:
        n = int(page)
    except (TypeError, ValueError):
        return "pNA"
    return f"p{n}" if n > 0 else "pNA"


def _chunk_elements(
    elements: list[dict], pdf_name: str, document_id: str | None = None
) -> list[dict]:
    """Smart chunking: group element di bawah Title, jangan split Table.

    Strategy:
    - Title memulai section baru
    - Text/ListItem di-merge sampai mendekati CHUNK_SIZE
    - Table & ImageDescription jadi chunk tersendiri (tidak di-merge)
    - Overlap CHUNK_OVERLAP karakter antar chunk text

    `document_id` hanya dipakai saat INDEX_STRUCTURAL_METADATA aktif, untuk
    membangun chunk_id berformat {document_id}_p{page}_c{NN}.
    """
    chunks: list[dict] = []
    current_section: str = ""
    current_buffer: list[str] = []
    current_page: int = 1
    current_categories: set[str] = set()
    current_bboxes: list[list[float]] = []
    # Ekor chunk sebelumnya yang disemai ke buffer ini sebagai overlap. Dikeluarkan
    # dari dasar text_sha supaya hash tidak bergantung pada isi chunk tetangga.
    current_overlap_seed: str = ""

    # Pencacah ordinal DALAM halaman, terpisah per segmen halaman. Ember "pNA"
    # menampung element yang halamannya tidak diketahui.
    page_counters: dict[str, int] = {}
    unknown_page_chunks = 0

    def _is_title_only(text: str) -> bool:
        """True bila chunk hanya berisi baris judul section.

        Chunk semacam itu tidak membawa konten unik — `section` sudah ada di
        metadata setiap chunk — tapi tetap divektorkan dan mencemari presisi.
        """
        if not current_section:
            return False
        return normalize_for_hash(text) == normalize_for_hash(f"# {current_section}")

    def emit(
        text: str,
        page: int,
        element_type: str,
        splittable: bool = True,
        extra: dict | None = None,
        inherited_prefix: str = "",
    ):
        """Tambahkan chunk, dipecah dulu bila melewati INDEX_MAX_CHUNK_TOKENS.

        Urutan operasi penting: pecah -> saring -> hash -> tempel overlap.

        text_sha dihitung atas konten milik chunk itu SENDIRI, dengan dua sumber
        overlap dikeluarkan dari perhitungan:
        1. Overlap antar-potongan hasil pemecahan — ditempel setelah hash.
        2. Overlap antar-chunk dari buffer (lihat cabang teks di bawah, yang
           menyemai buffer baru dengan ekor element terakhir) — dipotong lewat
           `inherited_prefix`.

        Tanpa keduanya, hash sebuah chunk ikut berubah saat chunk tetangganya
        berubah, dan pemetaan ulang anotasi gold gagal justru pada skenario
        kaskade yang paling sering terjadi.
        """
        nonlocal unknown_page_chunks

        pieces = _split_for_budget(text, splittable=splittable)
        budget = INDEX_MAX_CHUNK_TOKENS if INDEX_MAX_CHUNK_TOKENS > 0 else 0
        final_texts = _apply_overlap(pieces, budget) if budget else pieces

        for position, (core, final_text) in enumerate(zip(pieces, final_texts)):
            if not final_text.strip():
                continue

            # Potongan pertama bisa diawali ekor chunk sebelumnya; buang dari
            # dasar hash supaya isi milik chunk ini saja yang terhitung.
            hash_basis = core
            if position == 0 and inherited_prefix and core.startswith(inherited_prefix):
                trimmed = core[len(inherited_prefix):].strip()
                if trimmed:
                    hash_basis = trimmed

            if INDEX_MIN_CHUNK_TOKENS > 0:
                if _is_title_only(core):
                    logger.info(
                        "chunk_dropped reason=title_only section=%r", current_section
                    )
                    continue
                if _ntok(final_text) < INDEX_MIN_CHUNK_TOKENS:
                    logger.info(
                        "chunk_dropped reason=below_min_tokens tokens=%d min=%d text=%r",
                        _ntok(final_text), INDEX_MIN_CHUNK_TOKENS, final_text[:60],
                    )
                    continue

            chunk = {
                "text": final_text,
                "page": page,
                "file_name": pdf_name,
                "chunk_index": len(chunks),
                "element_type": element_type,
                "section": current_section,
            }

            if INDEX_STRUCTURAL_METADATA:
                segment = _page_segment(page)
                if segment == "pNA":
                    unknown_page_chunks += 1
                ordinal = page_counters.get(segment, 0)
                page_counters[segment] = ordinal + 1
                if document_id:
                    chunk["document_id"] = document_id
                    chunk["chunk_id"] = f"{document_id}_{segment}_c{ordinal:02d}"
                chunk["text_sha"] = text_sha(hash_basis)
                for key, value in (extra or {}).items():
                    if value not in (None, ""):
                        chunk[key] = value

            chunks.append(chunk)

    def _union_bbox() -> list[float] | None:
        """Gabungan bbox element yang ada di buffer saat ini.

        Chunk teks bisa merangkum beberapa element, jadi bbox-nya adalah kotak
        yang melingkupi semuanya. Untuk chunk hasil pemecahan, bbox tetap
        merujuk element sumber — bukan potongan — karena pemecahan terjadi di
        ruang teks, bukan ruang halaman.
        """
        if not current_bboxes:
            return None
        return [
            round(min(b[0] for b in current_bboxes), 4),
            round(min(b[1] for b in current_bboxes), 4),
            round(max(b[2] for b in current_bboxes), 4),
            round(max(b[3] for b in current_bboxes), 4),
        ]

    def flush():
        if not current_buffer:
            return
        text = "\n\n".join(current_buffer).strip()
        if not text:
            return
        emit(
            text,
            current_page,
            "+".join(sorted(current_categories)) or "text",
            splittable=True,
            extra={"bbox": _union_bbox()},
            inherited_prefix=current_overlap_seed,
        )

    for el in elements:
        cat = el["category"]
        text = el["text"]
        page = el["page"]

        el_bbox = (el.get("metadata") or {}).get("bbox")

        if cat == "Title":
            # Flush sebelumnya, mulai section baru
            flush()
            current_buffer = []
            current_categories = set()
            current_bboxes = []
            current_overlap_seed = ""
            current_section = text
            current_page = page
            # Title juga jadi bagian buffer (header context)
            current_buffer.append(f"# {text}")
            current_categories.add("Title")
            if el_bbox:
                current_bboxes.append(el_bbox)
            continue

        if cat == "Table":
            # Tabel jadi chunk independen kalau besar, atau kalau
            # INDEX_TABLES_AS_OWN_CHUNKS aktif — yang terakhir supaya setiap
            # tabel punya hubungan 1:1 dengan satu chunk beserta raw_html-nya.
            mandiri = len(text) > PDF_TABLE_MAX_CHARS or INDEX_TABLES_AS_OWN_CHUNKS
            if mandiri:
                flush()
                current_buffer = []
                current_categories = set()
                current_bboxes = []
                current_overlap_seed = ""
                # splittable=False: memecah Markdown tabel memisahkan baris
                # header dari baris data, dan relasi baris-kolom itu justru
                # yang diukur RCAA di lapis 3.
                emit(
                    (f"## {current_section}\n\n" if current_section else "") + text,
                    page,
                    "Table",
                    splittable=False,
                    extra={
                        "raw_html": (el.get("metadata") or {}).get("raw_html"),
                        "table_format": (el.get("metadata") or {}).get("table_format"),
                        "bbox": el_bbox,
                    },
                )
            else:
                # Tabel kecil → append ke buffer dengan separator.
                # raw_html tabel ini TIDAK terwakili di payload mana pun: chunk
                # hasilnya bercampur prosa sehingga tidak ada hubungan 1:1.
                # Aktifkan INDEX_TABLES_AS_OWN_CHUNKS untuk mengubahnya.
                current_buffer.append(f"**Tabel:**\n{text}")
                current_categories.add("Table")
                if el_bbox:
                    current_bboxes.append(el_bbox)
                if not current_page:
                    current_page = page

        elif cat == "ImageDescription":
            # Image description jadi chunk independen
            flush()
            current_buffer = []
            current_categories = set()
            current_bboxes = []
            current_overlap_seed = ""
            # splittable=False: satu deskripsi = satu gambar. Memecahnya merusak
            # relasi narrative_summary <-> image_id. Praktisnya tidak pernah
            # terpicu karena image_describer.py:150 membatasi max_tokens=300.
            emit(
                (f"## {current_section}\n\n" if current_section else "") + f"[Deskripsi Gambar] {text}",
                page,
                "ImageDescription",
                splittable=False,
                extra={
                    "bbox": el_bbox,
                    "image_id": (el.get("metadata") or {}).get("image_id"),
                },
            )

        else:
            # Text biasa → tambah ke buffer
            buffer_size = sum(len(s) for s in current_buffer)
            if buffer_size + len(text) > CHUNK_SIZE:
                flush()
                # Overlap: keep last paragraph for context
                tail = current_buffer[-1] if current_buffer else ""
                overlap_text = tail[-CHUNK_OVERLAP:] if len(tail) > CHUNK_OVERLAP else tail
                current_buffer = [overlap_text] if overlap_text else []
                current_categories = set()
                current_bboxes = []
                current_overlap_seed = overlap_text

            current_buffer.append(text)
            current_categories.add(cat)
            if el_bbox:
                current_bboxes.append(el_bbox)
            if not current_page:
                current_page = page

    flush()

    if unknown_page_chunks:
        logger.warning(
            "chunk_id_page_unknown file=%s chunks=%d — memakai segmen 'pNA'. "
            "Sumbernya page_number absen di metadata Unstructured "
            "(preprocessing.py:254).",
            pdf_name, unknown_page_chunks,
        )

    return chunks


# ─── Public API ───────────────────────────────────────────────────────────────

def extract_from_pdf(pdf_path: str | Path, document_id: str | None = None) -> dict:
    """Extract elements terstruktur dari PDF.

    Returns:
        {
            "elements": [...],   # raw elements (text/table/image)
            "file_name": "...",
            "file_hash": "...",
            "strategy": "fast" | "hi_res",   # strategi yang DIMINTA
            "report": ExtractionReport,      # termasuk strategi yang TERPAKAI
            "images": [...],                 # record images.jsonl, bisa kosong
        }

    Catatan: `strategy` adalah strategi yang diminta. Bila hi_res gagal dan
    jatuh ke fast, `report.strategy_used` yang mencerminkan kenyataannya.
    """
    pdf_path = Path(pdf_path)
    file_hash = file_sha256(pdf_path)

    strategy = PDF_EXTRACTION_STRATEGY
    if strategy == "auto":
        strategy = detect_strategy(pdf_path)

    report = ExtractionReport(strategy_requested=strategy, strategy_used=strategy)

    logger.info(f"pdf_extract file={pdf_path.name} strategy={strategy} hash={file_hash[:8]}")

    image_records: list[dict] = []
    if strategy == "hi_res":
        elements = _extract_hi_res(pdf_path, report=report)
        # Penulisan gambar SEBELUM keputusan deskripsi — gambar yang nanti
        # dinilai DEKORATIF atau gagal dideskripsikan tetap tersimpan.
        elements, image_records = _persist_image_elements(
            elements, document_id, pdf_path.name
        )
        elements = _describe_image_elements(elements)
        # Deskripsi yang berhasil ditautkan balik ke record gambarnya.
        by_id = {r["image_id"]: r for r in image_records}
        for el in elements:
            if el.get("category") != "ImageDescription":
                continue
            rec = by_id.get((el.get("metadata") or {}).get("image_id"))
            if rec is not None:
                rec["narrative_summary"] = el["text"]
    else:
        elements = _extract_fast(pdf_path, report=report)

    if report.degraded:
        logger.warning(
            "pdf_degraded file=%s hi_res_fallback=%s ocr_failed_pages=%d",
            pdf_path.name,
            bool(report.hi_res_fallback_reason),
            len(report.ocr_failed_pages),
        )

    return {
        "elements": elements,
        "file_name": pdf_path.name,
        "file_hash": file_hash,
        "strategy": strategy,
        "report": report,
        "images": image_records,
    }


def chunk_documents(
    elements: list[dict] | dict,
    file_name: str | None = None,
    document_id: str | None = None,
) -> list[dict]:
    """Chunk elements jadi documents siap di-embed.

    Backward compatible: terima output dari extract_from_pdf (dict) atau list elements.

    `document_id` hanya dipakai saat INDEX_STRUCTURAL_METADATA aktif; pemanggil
    (indexing.py) yang meresolusinya lewat backend.services.document_registry.
    """
    if isinstance(elements, dict):
        file_name = elements.get("file_name", file_name or "unknown.pdf")
        elements = elements.get("elements", [])
    elif file_name is None:
        file_name = "unknown.pdf"

    if not elements:
        return []

    chunks = _chunk_elements(elements, file_name, document_id=document_id)
    logger.info(f"pdf_chunked file={file_name} chunks={len(chunks)}")
    return chunks
