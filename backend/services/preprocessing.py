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
import tempfile
from pathlib import Path
from typing import Iterable

import fitz  # pymupdf

from backend.config import (
    OCR_LANG, OCR_USE_GPU,
    CHUNK_SIZE, CHUNK_OVERLAP,
    IMAGES_DIR,
    INDEX_MAX_CHUNK_TOKENS,
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


# ─── Strategy detection ───────────────────────────────────────────────────────

def detect_strategy(pdf_path: Path) -> str:
    """Pilih strategy otomatis berdasarkan karakteristik PDF.

    - Banyak gambar/tabel → hi_res
    - Pure text → fast
    """
    try:
        doc = fitz.open(str(pdf_path))
        total_images = 0
        total_pages = len(doc)
        sample_pages = min(5, total_pages)

        for i in range(sample_pages):
            page = doc[i]
            total_images += len(page.get_images())

        doc.close()

        avg_images_per_page = total_images / max(sample_pages, 1)
        # Heuristic: > 1 image per page → kemungkinan rich content
        return "hi_res" if avg_images_per_page > 1.0 else "fast"
    except Exception as e:
        logger.debug(f"detect_strategy_failed file={pdf_path.name} error={e}")
        return "fast"


# ─── FAST PATH — PyMuPDF + OCR fallback ──────────────────────────────────────

def _extract_text_from_page_fast(page: fitz.Page) -> str:
    """Extract text via PyMuPDF, fallback ke PaddleOCR jika scan."""
    text = page.get_text().strip()
    if len(text) > 50:
        return text

    logger.info(f"page_ocr_fallback page={page.number + 1}")
    pix = page.get_pixmap(dpi=300)
    img_bytes = pix.tobytes("png")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(img_bytes)
        tmp_path = tmp.name

    try:
        ocr = _get_ocr_engine()
        # PaddleOCR 3.x: .predict() ganti .ocr(). Return list[OCRResult].
        result = ocr.predict(tmp_path)
        if not result:
            return text
        texts = _extract_texts_from_ocr_result(result)
        return "\n".join(texts) if texts else text
    finally:
        os.unlink(tmp_path)


def _extract_fast(pdf_path: Path) -> list[dict]:
    """Fast path: pure text extraction. Return list of elements per page."""
    doc = fitz.open(str(pdf_path))
    elements = []

    for page in doc:
        text = _extract_text_from_page_fast(page)
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

def _html_table_to_markdown(html: str) -> str:
    """Convert tabel HTML dari Unstructured ke Markdown."""
    try:
        from markdownify import markdownify
        md = markdownify(html, heading_style="ATX").strip()
        return md
    except Exception as e:
        logger.debug(f"html_table_md_fail error={e}")
        return html


def _extract_hi_res(pdf_path: Path) -> list[dict]:
    """Hi-res path: layout-aware extraction via Unstructured.io.

    Mengeluarkan element terstruktur: Title, NarrativeText, Table, Image, dll.
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
        return _extract_fast(pdf_path)

    elements = []
    for el in raw_elements:
        category = el.category
        page = getattr(el.metadata, "page_number", None) or 0

        # Skip noise
        if category in ("Header", "Footer", "PageNumber", "PageBreak"):
            continue

        if category == "Table":
            html = getattr(el.metadata, "text_as_html", None)
            md = _html_table_to_markdown(html) if html else el.text
            elements.append({
                "text": md,
                "category": "Table",
                "page": page,
                "metadata": {"raw_html": html or ""},
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
                "metadata": {"image_base64": image_b64},
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
                "metadata": {},
            })

    return elements


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
            "metadata": {},
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
    final = _apply_overlap(final, budget)
    logger.info(
        "chunk_split_oversized tokens=%d budget=%d pieces=%d",
        _ntok(text), budget, len(final),
    )
    return final


# ─── Chunking — preserve section context ──────────────────────────────────────

def _chunk_elements(elements: list[dict], pdf_name: str) -> list[dict]:
    """Smart chunking: group element di bawah Title, jangan split Table.

    Strategy:
    - Title memulai section baru
    - Text/ListItem di-merge sampai mendekati CHUNK_SIZE
    - Table & ImageDescription jadi chunk tersendiri (tidak di-merge)
    - Overlap CHUNK_OVERLAP karakter antar chunk text
    """
    chunks: list[dict] = []
    current_section: str = ""
    current_buffer: list[str] = []
    current_page: int = 1
    current_categories: set[str] = set()

    def emit(text: str, page: int, element_type: str, splittable: bool = True):
        """Tambahkan chunk, dipecah dulu bila melewati INDEX_MAX_CHUNK_TOKENS.

        chunk_index diberikan berurutan per potongan — satu nomor per chunk
        keluaran, supaya tidak ada dua chunk yang berbagi nomor.
        """
        for piece in _split_for_budget(text, splittable=splittable):
            if not piece.strip():
                continue
            chunks.append({
                "text": piece,
                "page": page,
                "file_name": pdf_name,
                "chunk_index": len(chunks),
                "element_type": element_type,
                "section": current_section,
            })

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
        )

    for el in elements:
        cat = el["category"]
        text = el["text"]
        page = el["page"]

        if cat == "Title":
            # Flush sebelumnya, mulai section baru
            flush()
            current_buffer = []
            current_categories = set()
            current_section = text
            current_page = page
            # Title juga jadi bagian buffer (header context)
            current_buffer.append(f"# {text}")
            current_categories.add("Title")
            continue

        if cat == "Table":
            # Table jadi chunk independen kalau besar, atau di-append jika kecil
            if len(text) > PDF_TABLE_MAX_CHARS:
                flush()
                current_buffer = []
                current_categories = set()
                # splittable=False: memecah Markdown tabel memisahkan baris
                # header dari baris data, dan relasi baris-kolom itu justru
                # yang diukur RCAA di lapis 3.
                emit(
                    (f"## {current_section}\n\n" if current_section else "") + text,
                    page,
                    "Table",
                    splittable=False,
                )
            else:
                # Tabel kecil → append ke buffer dengan separator
                current_buffer.append(f"**Tabel:**\n{text}")
                current_categories.add("Table")
                if not current_page:
                    current_page = page

        elif cat == "ImageDescription":
            # Image description jadi chunk independen
            flush()
            current_buffer = []
            current_categories = set()
            # splittable=False: satu deskripsi = satu gambar. Memecahnya merusak
            # relasi narrative_summary <-> image_id. Praktisnya tidak pernah
            # terpicu karena image_describer.py:150 membatasi max_tokens=300.
            emit(
                (f"## {current_section}\n\n" if current_section else "") + f"[Deskripsi Gambar] {text}",
                page,
                "ImageDescription",
                splittable=False,
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

            current_buffer.append(text)
            current_categories.add(cat)
            if not current_page:
                current_page = page

    flush()
    return chunks


# ─── Public API ───────────────────────────────────────────────────────────────

def extract_from_pdf(pdf_path: str | Path) -> dict:
    """Extract elements terstruktur dari PDF.

    Returns:
        {
            "elements": [...],   # raw elements (text/table/image)
            "file_name": "...",
            "file_hash": "...",
            "strategy": "fast" | "hi_res",
        }
    """
    pdf_path = Path(pdf_path)
    file_hash = file_sha256(pdf_path)

    strategy = PDF_EXTRACTION_STRATEGY
    if strategy == "auto":
        strategy = detect_strategy(pdf_path)

    logger.info(f"pdf_extract file={pdf_path.name} strategy={strategy} hash={file_hash[:8]}")

    if strategy == "hi_res":
        elements = _extract_hi_res(pdf_path)
        elements = _describe_image_elements(elements)
    else:
        elements = _extract_fast(pdf_path)

    return {
        "elements": elements,
        "file_name": pdf_path.name,
        "file_hash": file_hash,
        "strategy": strategy,
    }


def chunk_documents(elements: list[dict] | dict, file_name: str | None = None) -> list[dict]:
    """Chunk elements jadi documents siap di-embed.

    Backward compatible: terima output dari extract_from_pdf (dict) atau list elements.
    """
    if isinstance(elements, dict):
        file_name = elements.get("file_name", file_name or "unknown.pdf")
        elements = elements.get("elements", [])
    elif file_name is None:
        file_name = "unknown.pdf"

    if not elements:
        return []

    chunks = _chunk_elements(elements, file_name)
    logger.info(f"pdf_chunked file={file_name} chunks={len(chunks)}")
    return chunks
