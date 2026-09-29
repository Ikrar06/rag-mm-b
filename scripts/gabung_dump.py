"""gabung_dump.py — satukan dump run utama dan run susulan menjadi satu dataset.

Dipakai setelah indexing inkremental (tanpa --force) memasukkan dokumen yang
gagal di run utama ke collection yang sama. Menolak bila konfigurasi penentu
isi chunk berbeda antar run, atau dokumen/chunk_id/image_id bertabrakan.
Direktori masukan tidak diubah; hasil ditulis ke --out.

Usage:
    python scripts/gabung_dump.py --utama <dump v5 utama> --susulan <dump susulan> \\
        --out <dump v5 gabungan>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.gabung_dump import beda_konfigurasi, gabung, manifest_gabungan  # noqa: E402


def baca_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def tulis_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def tulis_csv(path: Path, kolom, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=kolom, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--utama", type=Path, required=True)
    ap.add_argument("--susulan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        print(f"GAGAL: {a.out} sudah berisi berkas; pakai direktori baru")
        return 2

    from backend.services import chunk_dump
    m_u = json.loads((a.utama / "run_manifest.json").read_text(encoding="utf-8"))
    m_s = json.loads((a.susulan / "run_manifest.json").read_text(encoding="utf-8"))
    beda = beda_konfigurasi(m_u, m_s)
    if beda:
        print("GAGAL: konfigurasi penentu isi chunk berbeda antar run:")
        for b in beda[:20]:
            print(f"  {b}")
        return 1

    try:
        chunks = gabung(baca_jsonl(a.utama / "chunks.jsonl"), baca_jsonl(a.susulan / "chunks.jsonl"),
                        "chunk_id")
        images = gabung(baca_jsonl(a.utama / "images.jsonl"), baca_jsonl(a.susulan / "images.jsonl"),
                        "image_id")
    except ValueError as e:
        print(f"GAGAL: {e}")
        return 1

    dok_s = sorted({r.get("document_id") for r in baca_jsonl(a.susulan / "chunks.jsonl")})
    a.out.mkdir(parents=True, exist_ok=True)
    tulis_jsonl(a.out / "chunks.jsonl", chunks)
    tulis_csv(a.out / "chunks_review.csv", chunk_dump._CSV_COLUMNS, map(chunk_dump._row, chunks))
    tulis_jsonl(a.out / "images.jsonl", images)
    tulis_csv(a.out / "images_review.csv", chunk_dump._IMAGE_CSV_COLUMNS,
              map(chunk_dump._image_row, images))
    m = manifest_gabungan(m_u, m_s, len(chunks), len(images),
                          sum(1 for i in images if i.get("narrative_summary")), dok_s)
    (a.out / "run_manifest.json").write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n",
                                             encoding="utf-8")
    n_dok = len({r.get("document_id") for r in chunks})
    print(f"gabungan: {len(chunks)} chunk, {n_dok} dokumen, {len(images)} gambar -> {a.out}")
    print(f"susulan: {dok_s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
