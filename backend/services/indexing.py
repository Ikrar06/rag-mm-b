"""Document indexing service — preprocess PDFs, embed, store to Qdrant."""

import logging

from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from backend.config import (
    QDRANT_URL,
    QDRANT_COLLECTION_NAME,
    DATA_DIR,
)
from backend.services.preprocessing import process_pdf_directory

logger = logging.getLogger(__name__)


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def index_documents(data_dir: str | None = None) -> int:
    """Index semua PDF dari data_dir ke Qdrant.

    Pipeline: PDF → PaddleOCR → Unstructured.io (chunk_by_title) → Embed → Qdrant

    Returns jumlah chunks yang berhasil di-index.
    """
    target_dir = data_dir or DATA_DIR
    logger.info(f"Loading documents from {target_dir}")

    # Step 1: Preprocessing (OCR + image extraction + structure-aware chunking)
    result = process_pdf_directory(target_dir)
    chunks = result["chunks"]
    images = result["images"]

    if not chunks:
        logger.warning("No chunks produced.")
        return 0

    logger.info(f"Extracted {len(images)} images for multimodal")

    # Step 2: Convert ke LlamaIndex Document objects
    documents = []
    for chunk in chunks:
        doc = Document(
            text=chunk["text"],
            metadata={
                "file_name": chunk["file_name"],
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
            },
        )
        documents.append(doc)

    logger.info(f"Prepared {len(documents)} chunks for indexing")

    # Step 3: Embed + store ke Qdrant
    qdrant_client = get_qdrant_client()
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

    logger.info(f"Indexed {len(documents)} chunks to collection '{QDRANT_COLLECTION_NAME}'")
    return len(documents)
