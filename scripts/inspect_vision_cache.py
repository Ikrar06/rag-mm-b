"""inspect_vision_cache.py — periksa isi cache deskripsi gambar.

Read-only: tidak menjalankan indexing dan tidak memanggil model. Dibuat supaya
tim evaluasi dapat melihat isi cache tanpa menjalankan apa pun yang mengubahnya.

Usage:
    python scripts/inspect_vision_cache.py
    python scripts/inspect_vision_cache.py --path ~/rag_mm_b_shared/vision_cache.db
    python scripts/inspect_vision_cache.py --sample 5          # contoh deskripsi
    python scripts/inspect_vision_cache.py --image-sha <sha>   # telusuri satu gambar
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _content_sha256(conn: sqlite3.Connection) -> str:
    """Sama persis dengan vision_cache.content_sha256 — hash BARIS terurut,
    bukan byte berkas, supaya bebas dari halaman bebas SQLite dan jurnal WAL."""
    h = hashlib.sha256()
    for row in conn.execute(
        "SELECT cache_key, verdict, COALESCE(description,'') "
        "FROM descriptions ORDER BY cache_key"
    ):
        h.update("\x1f".join(row).encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()


def main() -> int:
    from backend.config import VISION_CACHE_PATH

    ap = argparse.ArgumentParser(description="Periksa cache deskripsi gambar")
    ap.add_argument("--path", default=None, help="Path cache (default: VISION_CACHE_PATH)")
    ap.add_argument("--sample", type=int, default=0, help="Tampilkan N contoh deskripsi")
    ap.add_argument("--image-sha", default=None, help="Telusuri entri satu gambar")
    args = ap.parse_args()

    path = Path(os.path.expanduser(args.path or VISION_CACHE_PATH))
    print(f"Cache : {path}")

    if not path.exists():
        print("  BELUM ADA. Dibuat saat indexing pertama dengan "
              "VISION_CACHE_ENABLED=true.")
        return 0

    print(f"Ukuran: {path.stat().st_size / 1024 / 1024:.1f} MB"
          f"      Dapat ditulis: {'ya' if os.access(path, os.W_OK) else 'TIDAK (read-only)'}")

    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=15)
        total = conn.execute("SELECT COUNT(*) FROM descriptions").fetchone()[0]
    except sqlite3.Error as e:
        print(f"  GAGAL dibuka: {e}")
        return 1

    print(f"Entri : {total}")
    if total == 0:
        print("\n(kosong)")
        return 0

    # ── Telusuri satu gambar ────────────────────────────────────────────────
    if args.image_sha:
        rows = conn.execute(
            "SELECT variant, vision_model, vision_model_digest, verdict, "
            "       created_at, COALESCE(description,'') "
            "FROM descriptions WHERE image_sha256 LIKE ? ORDER BY created_at",
            (args.image_sha + "%",),
        ).fetchall()
        print(f"\nEntri untuk image_sha256 {args.image_sha!r}: {len(rows)}")
        for r in rows:
            print(f"  varian={r[0]}  verdict={r[3]}  model={r[1]}")
            print(f"    digest  : {r[2]}")
            print(f"    ditulis : {r[4]}")
            if r[5]:
                print(f"    teks    : {' '.join(r[5].split())[:110]}")
        return 0

    # ── Verdict ─────────────────────────────────────────────────────────────
    CATATAN = {
        "described":  "dipakai sebagai narrative_summary",
        "decorative": "model menjawab DEKORATIF — gambar tetap di disk, tanpa narasi",
        "unclear":    "model menjawab TIDAK JELAS — gambar tetap di disk, tanpa narasi",
    }
    print("\nVERDICT")
    for verdict, n in conn.execute(
        "SELECT verdict, COUNT(*) FROM descriptions GROUP BY verdict ORDER BY COUNT(*) DESC"
    ):
        print(f"  {verdict:<12}{n:>7}   {100 * n / total:>5.1f}%   {CATATAN.get(verdict, '')}")

    # ── Konfigurasi vision ──────────────────────────────────────────────────
    konfig = conn.execute(
        "SELECT vision_model, vision_model_digest, prompt_sha256, variant, "
        "       COUNT(*), MIN(created_at), MAX(created_at) "
        "FROM descriptions "
        "GROUP BY vision_model, vision_model_digest, prompt_sha256, variant "
        "ORDER BY COUNT(*) DESC"
    ).fetchall()

    print(f"\nKONFIGURASI VISION  ({len(konfig)} kombinasi)")
    for k in konfig:
        print(f"  {k[4]:>7} entri  varian={k[3]}  model={k[0]}")
        print(f"          digest : {k[1] or '<tidak ada>'}")
        print(f"          prompt : {k[2][:32]}")
        print(f"          ditulis: {k[5]}  ..  {k[6]}")

    if len({k[1] for k in konfig}) > 1 or len({k[2] for k in konfig}) > 1:
        print("\n  PERINGATAN: cache memuat lebih dari satu konfigurasi vision.")
        print("  Sebagian chunk berasal dari model atau prompt berbeda, jadi")
        print("  perbandingan varian tidak lagi mengukur strategi indexing saja.")
        print("  RESEARCH_MODE=true akan MENOLAK run dalam kondisi ini.")

    # ── Hash isi ────────────────────────────────────────────────────────────
    print("\nHASH ISI  (cocokkan dengan run_manifest.json -> vision_cache.content_sha256)")
    print(f"  {_content_sha256(conn)}")
    print("  Dihitung atas BARIS terurut, bukan byte berkas — bebas dari halaman")
    print("  bebas SQLite dan jurnal WAL, jadi dua cache dengan hash sama memuat")
    print("  deskripsi yang sama persis.")

    # ── Contoh ──────────────────────────────────────────────────────────────
    if args.sample:
        print(f"\nCONTOH ({args.sample} entri 'described')")
        for r in conn.execute(
            "SELECT image_sha256, description FROM descriptions "
            "WHERE verdict='described' LIMIT ?", (args.sample,),
        ):
            print(f"  {r[0][:16]}  {' '.join((r[1] or '').split())[:96]}")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
