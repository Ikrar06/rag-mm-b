"""Document indexing service — preprocess PDFs, embed, store to Qdrant."""

import logging
import time
from pathlib import Path

from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from qdrant_client.models import Distance, VectorParams

from backend.config import (
    QDRANT_URL,
    QDRANT_COLLECTION_NAME,
    EMBED_DIMENSION,
    DATA_DIR,
)
from backend.services.preprocessing import extract_from_pdf, chunk_documents

logger = logging.getLogger(__name__)


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def clear_collection() -> bool:
    """Hapus semua points di collection (keep collection structure).

    Tidak menghapus collection itu sendiri karena Qdrant Docker
    bisa race-condition antara delete storage dan create ulang.
    """
    try:
        client = get_qdrant_client()
        collections = [c.name for c in client.get_collections().collections]
        if QDRANT_COLLECTION_NAME not in collections:
            logger.info(f"Collection '{QDRANT_COLLECTION_NAME}' not found, nothing to clear.")
            return True
        client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=Filter(),  # empty filter = match all points
        )
        logger.info(f"All points in '{QDRANT_COLLECTION_NAME}' cleared.")
        return True
    except Exception as e:
        logger.warning(f"Could not clear collection: {e}")
        return False


def get_collection_count() -> int:
    """Cek jumlah dokumen di collection."""
    try:
        client = get_qdrant_client()
        info = client.get_collection(QDRANT_COLLECTION_NAME)
        return info.points_count
    except Exception:
        return 0


def get_indexed_files() -> set[str]:
    """Ambil daftar file_name yang sudah di-index di Qdrant."""
    try:
        client = get_qdrant_client()
        # Scroll semua points, ambil unique file_name dari metadata
        indexed = set()
        offset = None
        while True:
            results, offset = client.scroll(
                collection_name=QDRANT_COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in results:
                file_name = point.payload.get("file_name")
                if file_name:
                    indexed.add(file_name)
            if offset is None:
                break
        return indexed
    except Exception:
        return set()


def delete_file_chunks(file_name: str) -> int:
    """Hapus semua chunks dari file tertentu di Qdrant."""
    try:
        client = get_qdrant_client()
        result = client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(key="file_name", match=MatchValue(value=file_name))]
            ),
        )
        logger.info(f"Deleted chunks for '{file_name}'")
        return 1
    except Exception as e:
        logger.warning(f"Could not delete chunks for '{file_name}': {e}")
        return 0


def _ensure_collection(client: QdrantClient):
    """Pastikan collection ada di Qdrant, buat jika belum.

    Includes retry logic untuk handle orphaned storage dari
    previous failed delete (Qdrant Docker issue).
    """
    collections = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION_NAME in collections:
        return

    logger.info(f"Creating collection '{QDRANT_COLLECTION_NAME}' (dim={EMBED_DIMENSION})...")
    max_retries = 5
    for attempt in range(max_retries):
        try:
            client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBED_DIMENSION, distance=Distance.COSINE),
            )
            return
        except Exception as e:
            if "already exists" in str(e).lower() and attempt < max_retries - 1:
                wait = 2 * (attempt + 1)
                logger.warning(
                    f"Orphaned storage detected, retry in {wait}s ({attempt + 1}/{max_retries})"
                )
                try:
                    client.delete_collection(QDRANT_COLLECTION_NAME)
                except Exception:
                    pass
                time.sleep(wait)
            else:
                raise


def _embed_and_store(documents: list[Document]):
    """Embed documents dan simpan ke Qdrant."""
    qdrant_client = get_qdrant_client()

    # Buat collection dulu jika belum ada (hindari race condition di LlamaIndex)
    _ensure_collection(qdrant_client)

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=QDRANT_COLLECTION_NAME,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )


def index_documents(data_dir: str | None = None, force: bool = False) -> int:
    """Index PDF dari data_dir ke Qdrant (incremental by default).

    Pipeline: PDF → PaddleOCR → Unstructured.io (chunk_by_title) → Embed → Qdrant

    Args:
        data_dir: Path ke folder PDF (default: DATA_DIR)
        force: Jika True, hapus collection lama sebelum re-index semua

    Returns jumlah chunks yang berhasil di-index.
    """
    target_dir = Path(data_dir or DATA_DIR)

    if force:
        existing = get_collection_count()
        logger.info(f"Force mode: clearing {existing} existing chunks...")
        clear_collection()

    # Cari semua PDF di directory
    pdf_files = sorted(target_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {target_dir}")
        return 0

    # Cek file mana yang sudah di-index (skip jika bukan force)
    indexed_files = set() if force else get_indexed_files()
    new_files = [f for f in pdf_files if f.name not in indexed_files]
    skipped = len(pdf_files) - len(new_files)

    if skipped > 0:
        logger.info(f"Skipping {skipped} already-indexed files")
    if not new_files:
        logger.info("No new PDF files to index.")
        return 0

    logger.info(f"Indexing {len(new_files)} new PDF files:")
    for f in new_files:
        logger.info(f"  - {f.name}")

    # Step 1: Preprocessing semua file baru
    all_documents = []
    total_images = 0

    for pdf_path in new_files:
        result = extract_from_pdf(pdf_path)
        chunks = chunk_documents(result["pages"])
        total_images += len(result["images"])

        if not chunks:
            logger.warning(f"  No chunks from {pdf_path.name}, skipping")
            continue

        for chunk in chunks:
            doc = Document(
                text=chunk["text"],
                metadata={
                    "file_name": chunk["file_name"],
                    "page": chunk["page"],
                    "chunk_index": chunk["chunk_index"],
                    "element_type": chunk.get("element_type", "unknown"),
                },
            )
            all_documents.append(doc)

        logger.info(f"  {pdf_path.name}: {len(chunks)} chunks prepared")

    if not all_documents:
        logger.warning("No chunks produced from any file.")
        return 0

    # Step 2: Embed + store semua sekaligus ke Qdrant
    logger.info(f"Embedding & storing {len(all_documents)} chunks to Qdrant...")
    _embed_and_store(all_documents)

    logger.info(f"Total: {len(all_documents)} chunks, {total_images} images from {len(new_files)} files")
    return len(all_documents)
