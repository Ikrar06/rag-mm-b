"""Document indexing service — load PDFs, embed, store to Qdrant."""

import logging

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from backend.config import (
    QDRANT_URL,
    QDRANT_COLLECTION_NAME,
    DATA_DIR,
)

logger = logging.getLogger(__name__)


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def index_documents(data_dir: str | None = None) -> int:
    """Index semua PDF dari data_dir ke Qdrant.

    Returns jumlah dokumen yang berhasil di-index.
    """
    target_dir = data_dir or DATA_DIR
    logger.info(f"Loading documents from {target_dir}")

    documents = SimpleDirectoryReader(target_dir).load_data()
    if not documents:
        logger.warning("No documents found.")
        return 0

    logger.info(f"Loaded {len(documents)} document chunks")

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
