"""migrasi_keputusan.py — petakan kunci table_continuation.json ke penomoran baru.

Berkas keputusan dibuat dari koleksi v2, SEBELUM perbaikan current_page. Di v3
chunk_id tabel bergeser, sehingga kunci v2 menunjuk chunk lain — bahkan chunk
teks. Skrip ini menulis berkas BARU berkunci penomoran dump baru, membawa
seluruh kolom tinjauan manusia (keputusan, catatan_peninjau) apa adanya, dan
menambahkan sidik html kedua sisi supaya indexing dapat MEMVERIFIKASI bahwa
kunci yang cocok memang menunjuk tabel yang sama.

Berkas asli TIDAK ditimpa. Entri yang tidak dapat dipetakan dengan yakin
dipindah ke bagian `tidak_terpetakan` — di luar `pasangan`, jadi indexing tidak
pernah membacanya.

Usage:
    python scripts/migrasi_keputusan.py \\
        --keputusan ~/rag_mm_b_shared/table_continuation.json \\
        --chunks-lama .../20260826T100956Z-6dd92574/chunks.jsonl \\
        --chunks-baru .../20260921T095218Z-589ddf85/chunks.jsonl \\
        --keluar ~/rag_mm_b_shared/table_continuation.v3.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.services.table_continuation import html_sha  # noqa: E402
from lib.keputusan_migrasi import petakan_id, posisi_tabel  # noqa: E402


def baca_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Petakan ulang kunci berkas keputusan")
    ap.add_argument("--keputusan", required=True)
    ap.add_argument("--chunks-lama", required=True, help="dump yang dipakai membuat berkas keputusan")
    ap.add_argument("--chunks-baru", required=True, help="dump berpenomoran tujuan")
    ap.add_argument("--keluar", required=True)
    args = ap.parse_args()

    src = Path(args.keputusan).expanduser()
    keluar = Path(args.keluar).expanduser()
    if keluar.resolve() == src.resolve():
        print("GAGAL: --keluar sama dengan --keputusan. Berkas asli tidak ditimpa.")
        return 2
    try:
        raw = json.loads(src.read_text(encoding="utf-8"))
        lama = baca_jsonl(Path(args.chunks_lama).expanduser())
        baru = baca_jsonl(Path(args.chunks_baru).expanduser())
    except (OSError, ValueError) as e:
        print(f"GAGAL membaca masukan: {e}")
        return 2

    pos_lama, _ = posisi_tabel(lama)
    _, dari_pos_baru = posisi_tabel(baru)
    sidik_lama = {r["chunk_id"]: html_sha(r.get("text_as_html")) for r in lama if r.get("chunk_id")}
    sidik_baru = {r["chunk_id"]: html_sha(r.get("text_as_html")) for r in baru if r.get("chunk_id")}
    jenis_baru = {r["chunk_id"]: r.get("element_type") for r in baru if r.get("chunk_id")}

    pasangan_baru, tak_terpetakan = {}, {}
    status_pasangan = Counter()
    kunci_lama_bukan_tabel = 0
    for kunci, entri in (raw.get("pasangan") or {}).items():
        a, _, b = kunci.partition("__")
        if jenis_baru.get(a) != "Table" or jenis_baru.get(b) != "Table":
            kunci_lama_bukan_tabel += 1
        ha = petakan_id(a, pos_lama, dari_pos_baru, sidik_lama, sidik_baru)
        hb = petakan_id(b, pos_lama, dari_pos_baru, sidik_lama, sidik_baru)
        migrasi = {"chunk_id_lama_a": a, "chunk_id_lama_b": b,
                   "status_a": ha.status, "status_b": hb.status}
        if ha.baru and hb.baru:
            st = "id_sama" if ha.status == hb.status == "id_sama" else "dipetakan"
            status_pasangan[st] += 1
            pasangan_baru[f"{ha.baru}__{hb.baru}"] = {
                **entri,
                "chunk_id_a": ha.baru, "chunk_id_b": hb.baru,
                "html_sha_a": sidik_baru.get(ha.baru, ""),
                "html_sha_b": sidik_baru.get(hb.baru, ""),
                "_migrasi": migrasi,
            }
        else:
            status_pasangan["tidak_terpetakan"] += 1
            tak_terpetakan[kunci] = {**entri, "_migrasi": {
                **migrasi, "alasan": "; ".join(x.alasan for x in (ha, hb) if not x.baru)}}

    hasil = {
        "_meta": {**(raw.get("_meta") or {}),
                  "migrasi": {"dari": str(src), "chunks_lama": args.chunks_lama,
                              "chunks_baru": args.chunks_baru,
                              "waktu": datetime.now(timezone.utc).isoformat(),
                              "tool": "scripts/migrasi_keputusan.py"}},
        "pasangan": pasangan_baru,
        "tidak_terpetakan": tak_terpetakan,
    }
    keluar.write_text(json.dumps(hasil, indent=2, ensure_ascii=False), encoding="utf-8")

    def terima(d):
        return sum(1 for v in d.values() if str(v.get("keputusan", "")).strip().lower() == "terima")

    n = len(raw.get("pasangan") or {})
    print(f"Berkas asli   : {src}  ({n} pasangan, {terima(raw.get('pasangan') or {})} diterima)")
    print(f"Kunci lama yang di dump BARU tidak menunjuk dua tabel: {kunci_lama_bukan_tabel}")
    print("  (inilah yang dibaca v3 dan validasi sebelumnya — sebagian menunjuk chunk teks)\n")
    for st in ("id_sama", "dipetakan", "tidak_terpetakan"):
        print(f"  {st:<18}{status_pasangan[st]:>5}")
    print(f"\n  diterima setelah migrasi : {terima(pasangan_baru)}")
    print(f"  diterima tak terpetakan  : {terima(tak_terpetakan)}   <-- perlu ditinjau ulang")
    for st, jml in Counter(v["_migrasi"]["alasan"] for v in tak_terpetakan.values()).most_common(5):
        print(f"    {jml:>4}  {st}")
    print(f"\nDitulis: {keluar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
