"""Script untuk indexing PDF ke Qdrant.

Usage:
    python -m scripts.index_documents              # index (append)
    python -m scripts.index_documents --force       # hapus collection lama, index ulang
    python -m scripts.index_documents --force path  # custom directory

Catatan: indexing pakai GPU (PaddleOCR + Embedding).
Jika VRAM penuh, matikan Ollama dulu saat indexing:
    ollama stop qwen2.5:7b
"""

import argparse
import sys
import logging
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.services.rag_pipeline import _configure_settings
from backend.services.indexing import index_documents, get_collection_count
from backend.config import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Index PDF documents ke Qdrant")
    parser.add_argument("directory", nargs="?", default=DATA_DIR, help="Path ke folder PDF")
    parser.add_argument("--force", action="store_true", help="Hapus collection lama sebelum re-index")
    args = parser.parse_args()

    data_dir = args.directory

    # Check PDF files exist
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    if not pdf_files:
        logger.error(f"No PDF files found in {data_dir}")
        sys.exit(1)

    logger.info(f"Found {len(pdf_files)} PDF files in {data_dir}:")
    for f in pdf_files:
        logger.info(f"  - {f}")

    existing = get_collection_count()
    if existing > 0:
        logger.info(f"Existing chunks in Qdrant: {existing}")
        if not args.force:
            logger.warning("Collection sudah ada. Gunakan --force untuk re-index (hapus + index ulang)")

    # Configure embedding model (needed for indexing)
    logger.info("Loading embedding model...")
    _configure_settings()

    # Index
    logger.info(f"Starting indexing {'(force re-index)' if args.force else '(append)'}...")
    count = index_documents(data_dir, force=args.force)
    logger.info(f"Done! Indexed {count} chunks.")


if __name__ == "__main__":
    main()
