"""verify_chunk_ids.py — hitung chunk_id duplikat langsung dari Qdrant.

Read-only: tidak mengindeks, tidak menulis, tidak memuat model. Pemeriksaan yang
sama ini berjalan otomatis di akhir `index_documents`; skrip ini untuk memeriksa
koleksi yang SUDAH terlanjur ada tanpa harus mengindeks ulang lebih dulu.

Keluar dengan kode 1 bila ada duplikat, supaya bisa dipakai sebagai gerbang di
skrip lain.

Usage:
    python scripts/verify_chunk_ids.py
    python scripts/verify_chunk_ids.py --collection rag_mm_b_varian_c
    python scripts/verify_chunk_ids.py --all          # semua contoh, tanpa dipotong
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from backend.config import QDRANT_COLLECTION_NAME, QDRANT_URL
    from backend.services import index_verify
    from backend.services.indexing import get_qdrant_client

    ap = argparse.ArgumentParser(description="Hitung chunk_id duplikat di Qdrant")
    ap.add_argument("--collection", default=QDRANT_COLLECTION_NAME)
    ap.add_argument("--all", action="store_true",
                    help="Cetak semua chunk_id bertabrakan, bukan hanya 10 teratas")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    if args.all:
        index_verify._MAX_CONTOH = 10**9

    print(f"Qdrant   : {QDRANT_URL}")
    print(f"Koleksi  : {args.collection}")

    try:
        h = index_verify.scan_chunk_ids(get_qdrant_client(), args.collection)
    except Exception as e:
        print(f"\nGAGAL membaca koleksi: {e}")
        return 2

    print(f"\n  {'titik total':<24}{h['total_titik']:>9}")
    print(f"  {'punya chunk_id':<24}{h['dengan_chunk_id']:>9}")
    print(f"  {'tanpa chunk_id':<24}{h['tanpa_chunk_id']:>9}"
          f"   {'<-- tidak dapat dirujuk gold_chunk_ids' if h['tanpa_chunk_id'] else ''}")
    print(f"  {'chunk_id unik':<24}{h['unik']:>9}")
    print(f"  {'titik berlebih':<24}{h['duplikat']:>9}")
    print(f"  {'chunk_id bertabrakan':<24}{h['chunk_id_bertabrakan']:>9}")
    print(f"  {'dokumen terdampak':<24}{h['dokumen_terdampak']:>9}")

    if not h["duplikat"]:
        print("\nLOLOS — seluruh chunk_id unik.")
        return 0

    print(f"\nchunk_id bertabrakan"
          f"{'' if args.all else ' (10 teratas; --all untuk semua)'}:")
    for c in h["contoh"]:
        berkas = ", ".join(c["file_name"]) if c["file_name"] else "<file_name tidak ada>"
        print(f"  {c['titik']:>4} titik   {c['chunk_id']:<52}{berkas}")

    print("\nGAGAL — satu gold_chunk_id akan menunjuk ke lebih dari satu titik")
    print("dengan teks berbeda. Lihat backend/services/node_passthrough.py.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
