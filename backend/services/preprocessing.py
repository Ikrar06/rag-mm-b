"""PDF preprocessing service — PaddleOCR for scanned pages, Unstructured.io for structure-aware chunking."""

import logging
import os
import tempfile
from pathlib import Path

import fitz  # pymupdf
from paddleocr import PaddleOCR
from unstructured.partition.text import partition_text
from unstructured.chunking.title import chunk_by_title

from backend.config import (
    OCR_LANG,
    OCR_USE_GPU,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    IMAGES_DIR,
)

logger = logging.getLogger(__name__)

_ocr_engine: PaddleOCR | None = None


def _get_ocr_engine() -> PaddleOCR:
    global _ocr_engine
    if _ocr_engine is None:
        logger.info("Initializing PaddleOCR...")
        _ocr_engine = PaddleOCR(use_angle_cls=True, lang=OCR_LANG, use_gpu=OCR_USE_GPU)
    return _ocr_engine


def _extract_text_from_page(page: fitz.Page) -> str:
    """Extract text dari satu halaman PDF.

    Strategi:
    1. Coba extract teks digital dulu (PyMuPDF) — cepat & akurat
    2. Jika hasilnya kosong/sedikit (PDF scan), fallback ke PaddleOCR
    """
    text = page.get_text().strip()

    if len(text) > 50:
        return text

    logger.info(f"Page {page.number + 1}: teks digital minim, fallback ke OCR")
    pix = page.get_pixmap(dpi=300)
    img_bytes = pix.tobytes("png")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(img_bytes)
        tmp_path = tmp.name

    try:
        ocr = _get_ocr_engine()
        result = ocr.ocr(tmp_path, cls=True)

        if not result or not result[0]:
            return text

        lines = [line[1][0] for line in result[0]]
        return "\n".join(lines)
    finally:
        os.unlink(tmp_path)


def _extract_images_from_page(page: fitz.Page, pdf_name: str) -> list[dict]:
    """Extract gambar dari halaman PDF untuk multimodal.

    Returns:
        List of dict: [{"image_path": "...", "page": 1, "file_name": "doc.pdf"}, ...]
    """
    images = []
    os.makedirs(IMAGES_DIR, exist_ok=True)

    for img_index, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        base_image = page.parent.extract_image(xref)

        if base_image and base_image["image"]:
            ext = base_image.get("ext", "png")
            img_filename = f"{Path(pdf_name).stem}_p{page.number + 1}_img{img_index + 1}.{ext}"
            img_path = os.path.join(IMAGES_DIR, img_filename)

            with open(img_path, "wb") as f:
                f.write(base_image["image"])

            images.append({
                "image_path": img_path,
                "page": page.number + 1,
                "file_name": pdf_name,
            })

    return images


def extract_from_pdf(pdf_path: str | Path) -> dict:
    """Extract teks + gambar dari semua halaman PDF.

    Returns:
        dict: {"pages": [...], "images": [...]}
    """
    pdf_path = Path(pdf_path)
    logger.info(f"Processing: {pdf_path.name}")

    doc = fitz.open(str(pdf_path))
    pages = []
    all_images = []

    for page in doc:
        text = _extract_text_from_page(page)
        if text.strip():
            pages.append({
                "text": text,
                "page": page.number + 1,
                "file_name": pdf_path.name,
            })

        images = _extract_images_from_page(page, pdf_path.name)
        all_images.extend(images)

    doc.close()
    logger.info(f"  Extracted {len(pages)} pages, {len(all_images)} images from {pdf_path.name}")
    return {"pages": pages, "images": all_images}


def chunk_documents(pages: list[dict]) -> list[dict]:
    """Chunk pages menggunakan Unstructured.io chunk_by_title.

    Pipeline: teks per halaman → partition_text (deteksi struktur) → chunk_by_title (chunking cerdas)

    Returns:
        List of dict: [{"text": "...", "page": 1, "file_name": "doc.pdf", "chunk_index": 0}, ...]
    """
    chunks = []

    for page_data in pages:
        # Unstructured partition_text: deteksi struktur (judul, paragraf, list, dll)
        elements = partition_text(text=page_data["text"])

        # chunk_by_title: gabung elemen di bawah satu heading, potong jika kebesaran
        chunked_elements = chunk_by_title(
            elements,
            max_characters=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )

        for element in chunked_elements:
            text = str(element).strip()
            if text:
                chunks.append({
                    "text": text,
                    "page": page_data["page"],
                    "file_name": page_data["file_name"],
                    "chunk_index": len(chunks),
                    "element_type": element.category,
                })

    logger.info(f"  Chunked {len(pages)} pages into {len(chunks)} chunks (structure-aware)")
    return chunks


def process_pdf_directory(data_dir: str | Path) -> dict:
    """Process semua PDF dalam directory: OCR + extract images + chunk.

    Returns:
        dict: {"chunks": [...], "images": [...]}
    """
    data_dir = Path(data_dir)
    all_chunks = []
    all_images = []

    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {data_dir}")
        return {"chunks": [], "images": []}

    logger.info(f"Found {len(pdf_files)} PDF files")

    for pdf_path in pdf_files:
        result = extract_from_pdf(pdf_path)
        chunks = chunk_documents(result["pages"])
        all_chunks.extend(chunks)
        all_images.extend(result["images"])

    logger.info(f"Total: {len(all_chunks)} chunks, {len(all_images)} images from {len(pdf_files)} files")
    return {"chunks": all_chunks, "images": all_images}
