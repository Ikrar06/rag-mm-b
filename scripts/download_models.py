"""
download_models.py
==================
Download model yang diperlukan untuk RAG UNHAS.

  - intent_classifier : fine-tuned IndoBERT (HuggingFace)
  - bge-reranker-v2-m3: pre-trained reranker (HuggingFace)

Usage:
    python scripts/download_models.py                  # semua model
    python scripts/download_models.py --model intent   # hanya intent classifier
    python scripts/download_models.py --model reranker # hanya reranker
"""

import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("[ERROR] huggingface_hub belum terinstall.")
    print("        Jalankan: pip install huggingface_hub")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "intent": {
        "repo_id": "ikrarrr/rag-unhas-intent-classifier",
        "local_dir": ROOT / "models" / "intent_classifier",
        "description": "IndoBERT fine-tuned 4-class intent classifier",
    },
    "reranker": {
        "repo_id": "BAAI/bge-reranker-v2-m3",
        "local_dir": ROOT / "models" / "bge-reranker-v2-m3",
        "description": "BGE Reranker v2 M3 (cross-encoder)",
    },
}


def download(key: str):
    m = MODELS[key]
    print(f"\nDownloading: {m['description']}")
    print(f"  Source : {m['repo_id']}")
    print(f"  Target : {m['local_dir']}")
    snapshot_download(repo_id=m["repo_id"], local_dir=str(m["local_dir"]))
    print(f"  Done.")


def main():
    parser = argparse.ArgumentParser(description="Download model untuk RAG UNHAS")
    parser.add_argument(
        "--model",
        choices=["intent", "reranker", "all"],
        default="all",
        help="Model yang akan didownload (default: all)",
    )
    args = parser.parse_args()

    targets = list(MODELS.keys()) if args.model == "all" else [args.model]

    print("=" * 55)
    print("  RAG UNHAS — Model Downloader")
    print("=" * 55)

    for key in targets:
        download(key)

    print("\n" + "=" * 55)
    print("  Selesai! Semua model sudah tersedia di ./models/")
    print("=" * 55)


if __name__ == "__main__":
    main()
