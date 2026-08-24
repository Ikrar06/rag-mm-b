"""scaffold_document_registry.py — buat/perbarui kerangka document_registry.json.

Memindai folder PDF, menghitung sha256 tiap berkas, lalu menulis registry dengan
`document_id` KOSONG untuk berkas baru — siap diisi manual.

Aman dijalankan berulang:
- `document_id` yang sudah diisi TIDAK PERNAH ditimpa.
- Field tambahan yang sudah kamu tulis (title, source_unit, dst) dipertahankan.
- `sha256` diperbarui bila isi berkas berubah, dan perubahannya dilaporkan.
- Entri untuk berkas yang sudah tidak ada di folder dipertahankan dan ditandai,
  bukan dihapus — anotasi gold bisa jadi masih menunjuk ke sana.

Folder kosong menghasilkan registry kosong yang valid ({}), bukan error.

Usage:
    python scripts/scaffold_document_registry.py
    python scripts/scaffold_document_registry.py --pdf-dir data/pdfs
    python scripts/scaffold_document_registry.py --out data/document_registry.json
    python scripts/scaffold_document_registry.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import DATA_DIR, DOCUMENT_REGISTRY_PATH
from backend.services.preprocessing import file_sha256


def _load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"[ERROR] {path} bukan JSON valid — {e}")
        print("        Perbaiki atau pindahkan dulu; scaffolder menolak menimpanya.")
        sys.exit(1)
    if not isinstance(raw, dict):
        print(f"[ERROR] {path}: akar JSON harus object.")
        sys.exit(1)
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Buat/perbarui kerangka document_registry.json"
    )
    parser.add_argument("--pdf-dir", default=DATA_DIR, help="Folder PDF (default: DATA_DIR)")
    parser.add_argument("--out", default=DOCUMENT_REGISTRY_PATH, help="Path registry")
    parser.add_argument("--dry-run", action="store_true", help="Tampilkan saja, jangan tulis")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out_path = Path(args.out)

    existing = _load_existing(out_path)
    registry = {k: dict(v) if isinstance(v, dict) else {} for k, v in existing.items()}

    if not pdf_dir.exists():
        print(f"[WARN ] Folder PDF tidak ada: {pdf_dir}")
        print("        Registry tetap ditulis dari entri yang sudah ada.")
        pdf_files: list[Path] = []
    else:
        pdf_files = sorted(pdf_dir.glob("*.pdf"))

    baru, berubah, tetap = [], [], []

    for pdf in pdf_files:
        sha = file_sha256(pdf)
        entry = registry.get(pdf.name)

        if entry is None:
            registry[pdf.name] = {"document_id": "", "sha256": sha}
            baru.append(pdf.name)
            continue

        entry.setdefault("document_id", "")
        lama = (entry.get("sha256") or "").strip()
        if lama and lama != sha:
            berubah.append((pdf.name, lama, sha))
        entry["sha256"] = sha
        tetap.append(pdf.name)

    hilang = [n for n in registry if n not in {p.name for p in pdf_files}]

    # ── Laporan ──────────────────────────────────────────────────────────────
    print(f"Folder PDF : {pdf_dir}")
    print(f"Registry   : {out_path}")
    print(f"PDF terbaca: {len(pdf_files)}")
    print()
    print(f"  baru (document_id kosong, perlu diisi) : {len(baru)}")
    for n in baru:
        print(f"      + {n}")
    print(f"  sudah terdaftar                        : {len(tetap)}")
    if berubah:
        print(f"  ISI BERUBAH (sha256 berbeda)           : {len(berubah)}")
        for n, lama, sha in berubah:
            print(f"      ! {n}")
            print(f"        {lama[:16]} -> {sha[:16]}")
        print("        Periksa apakah ini revisi dokumen. Bila ya, dokumen revisi")
        print("        biasanya butuh document_id BARU plus superseded_by di entri lama.")
    if hilang:
        print(f"  entri tanpa berkas (dipertahankan)     : {len(hilang)}")
        for n in hilang:
            print(f"      ? {n}")

    belum_diisi = [n for n, e in registry.items() if not (e.get("document_id") or "").strip()]
    print()
    print(f"  BELUM punya document_id                : {len(belum_diisi)}")
    if belum_diisi:
        print("      Berkas ini akan DILEWATI saat indexing selama")
        print("      INDEX_STRUCTURAL_METADATA=true dan document_id masih kosong.")

    if args.dry_run:
        print("\n[dry-run] tidak ada yang ditulis.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nDitulis: {out_path}  ({len(registry)} entri)")


if __name__ == "__main__":
    main()
