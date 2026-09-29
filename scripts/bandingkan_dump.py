"""bandingkan_dump.py — jumlah chunk per dokumen dan per halaman, dump lama vs baru.

Untuk menjelaskan selisih jumlah chunk SEBELUM migrasi gold: dokumen mana yang
berubah, di halaman mana, jenis element apa yang bertambah/berkurang, dan
chunk_id mana yang bergeser. Read-only.

Sebab yang dilabeli (dari data, bukan tebakan):
- `tabel_kecil_lolos`: chunk Table baru di v5 yang tak berpadanan di v4, dan
  teks OCR-nya (teks_ocr) pendek — di v4 kemungkinan dibuang
  INDEX_MIN_CHUNK_TOKENS, di v5 transkripsinya cukup panjang.
- `gambar_baru` / `gambar_hilang`: ImageDescription (atau Table asal gambar)
  yang image_id-nya hanya ada di satu sisi.
- `teks`: jenis lain.

Usage:
    python scripts/bandingkan_dump.py --lama <dump v4>/chunks.jsonl \\
        --baru <dump v5>/chunks.jsonl [--kecuali ukt-tahun-2025,...] --out ~/banding_v4_v5.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.banding_dump import bandingkan  # noqa: E402


def baca(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lama", type=Path, required=True)
    ap.add_argument("--baru", type=Path, required=True)
    ap.add_argument("--kecuali", default="", help="document_id dipisah koma (mis. yang gagal)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    kecuali = {x.strip() for x in a.kecuali.split(",") if x.strip()}
    lama = [r for r in baca(a.lama) if r.get("document_id") not in kecuali]
    baru = [r for r in baca(a.baru) if r.get("document_id") not in kecuali]
    h = bandingkan(lama, baru)

    print(f"chunk lama {len(lama)}  baru {len(baru)}  selisih {len(baru) - len(lama):+d}")
    print(f"dokumen hanya di lama: {h['hanya_lama']}\ndokumen hanya di baru: {h['hanya_baru']}")
    print(f"\ndokumen yang jumlah chunk-nya berubah: {len(h['dokumen'])}")
    for d in h["dokumen"]:
        print(f"  {d['selisih']:+4d}  {d['document_id']}  ({d['lama']} -> {d['baru']})  "
              f"per jenis {d['per_jenis']}")
        for hal in d["halaman"]:
            print(f"        hal {hal['halaman']:>4}: {hal['selisih']:+d}  sebab={hal['sebab']}  "
                  f"id bergeser={len(hal['bergeser'])}")
            for x in hal["tambahan"][:3]:
                print(f"            + {x}")
            for x in hal["hilang"][:3]:
                print(f"            - {x}")
    print(f"\nringkasan sebab (halaman): {dict(h['sebab'])}")
    print(f"halaman dengan jumlah sama tapi urutan jenis berubah: {len(h['urutan_berubah'])}")
    for x in h["urutan_berubah"][:10]:
        print(f"    {x}")
    print(f"total chunk_id bergeser: {h['n_bergeser']}")
    print(f"\nCHUNK DARI GAMBAR YANG DI DUMP LAMA BUKAN CHUNK: {len(h['gambar_tanpa_padanan'])}")
    for x in h["gambar_tanpa_padanan"][:30]:
        print(f"    {x}")
    if h["gambar_tanpa_padanan"]:
        print("  Tidak nol: saringan deskripsi (perbaikan a) mengubah dokumen ini juga —")
        print("  run susulan saja TIDAK konsisten dengan run utama; ulangi run penuh.")
    if a.out:
        a.out.write_text(json.dumps(h, ensure_ascii=False, indent=1, default=list), encoding="utf-8")
        print(f"\nrincian: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
