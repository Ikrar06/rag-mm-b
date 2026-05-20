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
    PDF_EXTRACTION_STRATEGY,
    PDF_EXTRACT_IMAGES, PDF_DESCRIBE_IMAGES,
    PDF_EXTRACT_TABLES, PDF_TABLE_MAX_CHARS,
    LLM_SUPPORTS_VISION,
)

logger = logging.getLogger(__name__)

# ─── Lazy initialized engines ─────────────────────────────────────────────────

_ocr_engine = None


def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR
        logger.info("ocr_engine_init lang=%s gpu=%s", OCR_LANG, OCR_USE_GPU)
        _ocr_engine = PaddleOCR(
            use_angle_cls=True, lang=OCR_LANG,
            device="gpu" if OCR_USE_GPU else "cpu",
        )
    return _ocr_engine


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
        result = ocr.ocr(tmp_path)
        if not result or not result[0]:
            return text
        return "\n".join(line[1][0] for line in result[0])
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

    def flush():
        if not current_buffer:
            return
        text = "\n\n".join(current_buffer).strip()
        if not text:
            return
        chunks.append({
            "text": text,
            "page": current_page,
            "file_name": pdf_name,
            "chunk_index": len(chunks),
            "element_type": "+".join(sorted(current_categories)) or "text",
            "section": current_section,
        })

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
                chunks.append({
                    "text": (f"## {current_section}\n\n" if current_section else "") + text,
                    "page": page,
                    "file_name": pdf_name,
                    "chunk_index": len(chunks),
                    "element_type": "Table",
                    "section": current_section,
                })
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
            chunks.append({
                "text": (f"## {current_section}\n\n" if current_section else "") + f"[Deskripsi Gambar] {text}",
                "page": page,
                "file_name": pdf_name,
                "chunk_index": len(chunks),
                "element_type": "ImageDescription",
                "section": current_section,
            })

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
