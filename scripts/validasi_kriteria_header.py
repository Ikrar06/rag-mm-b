"""validasi_kriteria_header.py — uji kriteria header terhadap DATA NYATA.

READ-ONLY. Tidak mengindeks, tidak menulis ke Qdrant, tidak mengubah pipeline.

Latar
-----
Kriteria header pernah diuji terhadap dua belas contoh yang baris TUBUHNYA
dikarang, padahal aturan (d) — kontras kolom — bergantung penuh pada tubuh
tabel sungguhan. "Nol salah terima" karenanya belum terbukti. Skrip ini
menjalankan kedua varian atas seluruh pasangan yang DISETUJUI di berkas
keputusan, memakai baris nyata dari dump.

Kenapa membaca `text_as_html`, bukan `text_content`
---------------------------------------------------
Teks chunk di v3 SUDAH tergabung: potongan yang merupakan lanjutan sudah
diawali header (yang mungkin salah pilih). Menilai kriteria di atas teks itu
berarti menilai hasil keputusan lama, bukan tabel aslinya. `text_as_html` tidak
pernah disentuh penggabungan — ia catatan ekstraksi yang setia — jadi ia yang
dipakai. Ia juga satu-satunya sumber penanda `<th>`/`<thead>`.

Diagnosis butir 6
-----------------
Untuk pasangan yang teks B-nya TIDAK berubah, dicek kondisi mana yang berlaku:
`baris_header_markdown` mengembalikan kosong (A tidak berbentuk tabel Markdown),
atau B memang sudah diawali header itu. Perbandingan v2<->v3 memakai `text_sha`,
bukan `chunk_id`: penomoran halaman diperbaiki di antara kedua run sehingga id
bisa bergeser, sedangkan sha ikut isinya.

Usage:
    python scripts/validasi_kriteria_header.py \
        --keputusan ~/rag_mm_b_shared/table_continuation.json \
        --chunks-v3 .../20260921T095218Z-589ddf85/chunks.jsonl \
        --chunks-v2 .../20260826T100956Z-6dd92574/chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.header_tabel import deteksi_header  # noqa: E402
from lib.tabel_html import urai  # noqa: E402

PEMISAH_MD = re.compile(r"\|[\s:|-]+\|")


def baca_jsonl(path: Path) -> list[dict]:
    out = []
    for i, ln in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not ln.strip():
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} baris {i}: {e}") from e
    return out


def baris_header_markdown(teks: str) -> str:
    """Salinan persis logika produksi, untuk mendiagnosis butir 6."""
    baris = (teks or "").splitlines()
    mulai = next((i for i, b in enumerate(baris) if b.lstrip().startswith("|")), None)
    if mulai is None or mulai + 1 >= len(baris):
        return ""
    if not PEMISAH_MD.fullmatch(baris[mulai + 1].strip()):
        return ""
    return "\n".join(baris[mulai:mulai + 2])


def _potong(sel, n=56) -> str:
    s = " | ".join(str(x).strip() for x in sel)
    return s if len(s) <= n else s[: n - 1] + "…"


def _diagnosa_tak_berubah(a: dict, b: dict) -> str:
    """Kenapa teks B tidak berubah walau pasangannya disetujui."""
    teks_a = a.get("text_content") or ""
    header = baris_header_markdown(teks_a)
    if not header:
        fmt = a.get("table_format")
        return (f"A tidak berbentuk tabel Markdown (table_format={fmt!r}) — "
                f"tidak ada header untuk diulang")
    if (b.get("text_content") or "").strip().startswith(header.strip()):
        return "B SUDAH diawali header itu di dokumen asli — tidak digandakan"
    return "tidak diketahui — header ada dan B tidak diawali header itu"


def main() -> int:
    ap = argparse.ArgumentParser(description="Validasi kriteria header di data nyata")
    ap.add_argument("--keputusan", required=True)
    ap.add_argument("--chunks-v3", required=True)
    ap.add_argument("--chunks-v2", default=None,
                    help="Dump sebelum penggabungan, untuk diagnosis butir 6")
    ap.add_argument("--hanya-beda", action="store_true",
                    help="Cetak hanya pasangan yang kedua varian berbeda putusan")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    try:
        raw = json.loads(Path(args.keputusan).expanduser().read_text(encoding="utf-8"))
        pasangan_keputusan = (raw or {}).get("pasangan") or {}
        v3 = baca_jsonl(Path(args.chunks_v3).expanduser())
        v2 = baca_jsonl(Path(args.chunks_v2).expanduser()) if args.chunks_v2 else []
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"GAGAL membaca masukan: {e}")
        return 2

    per_id = {c["chunk_id"]: c for c in v3 if c.get("chunk_id")}
    sha_v2 = {c.get("text_sha") for c in v2 if c.get("text_sha")}

    disetujui = [k for k, v in pasangan_keputusan.items()
                 if isinstance(v, dict)
                 and str(v.get("keputusan", "")).strip().lower() == "terima"]

    print(f"Keputusan : {args.keputusan}")
    print(f"Dump v3   : {args.chunks_v3}  ({len(v3)} chunk)")
    if v2:
        print(f"Dump v2   : {args.chunks_v2}  ({len(v2)} chunk)")
    print(f"Disetujui : {len(disetujui)} pasangan\n")

    hasil, hilang = [], []
    for kunci in sorted(disetujui):
        id_a, _, id_b = kunci.partition("__")
        a, b = per_id.get(id_a), per_id.get(id_b)
        if a is None or b is None:
            hilang.append((kunci, "A" if a is None else "B"))
            continue

        t = urai(a.get("text_as_html"))
        if t is None:
            hasil.append({
                "kunci": kunci, "document_id": id_a.rsplit("_p", 1)[0],
                "halaman": [a.get("page_number"), b.get("page_number")],
                "kandidat": None, "tubuh": [],
                "A": {"header": False, "aturan": "html", "alasan": "text_as_html tidak dapat diurai"},
                "B": {"header": False, "aturan": "html", "alasan": "text_as_html tidak dapat diurai"},
                "tak_berubah": None,
            })
            continue

        pa = deteksi_header(t.baris, t.ada_th, t.ada_thead, pakai_kontras=True)
        pb = deteksi_header(t.baris, t.ada_th, t.ada_thead, pakai_kontras=False)
        tak_berubah = None
        if v2 and b.get("text_sha") in sha_v2:
            tak_berubah = _diagnosa_tak_berubah(a, b)

        hasil.append({
            "kunci": kunci, "document_id": id_a.rsplit("_p", 1)[0],
            "halaman": [a.get("page_number"), b.get("page_number")],
            "kandidat": list(t.baris[0]) if t.baris else None,
            "tubuh": [list(r) for r in t.baris[1:3]],
            "ada_th": t.ada_th or t.ada_thead,
            "A": {"header": pa.header, "aturan": pa.aturan, "alasan": pa.alasan},
            "B": {"header": pb.header, "aturan": pb.aturan, "alasan": pb.alasan},
            "tak_berubah": tak_berubah,
        })

    beda = [r for r in hasil if r["A"]["header"] != r["B"]["header"]]
    cetak = beda if args.hanya_beda else hasil

    for r in cetak:
        tanda = "  <-- BEDA" if r in beda else ""
        print(f"{r['document_id']}  hal {r['halaman'][0]}->{r['halaman'][1]}{tanda}")
        print(f"    kandidat : {_potong(r['kandidat'] or ['<tidak terurai>'])}")
        for i, baris in enumerate(r["tubuh"], 1):
            print(f"    tubuh {i}  : {_potong(baris)}")
        print(f"    Varian A : {'HEADER' if r['A']['header'] else 'tolak ':<7} "
              f"[{r['A']['aturan']}] {r['A']['alasan']}")
        print(f"    Varian B : {'HEADER' if r['B']['header'] else 'tolak ':<7} "
              f"[{r['B']['aturan']}] {r['B']['alasan']}")
        if r["tak_berubah"]:
            print(f"    butir 6  : {r['tak_berubah']}")
        print()

    print("=" * 78)
    print("RINGKASAN")
    print("=" * 78)
    n = len(hasil) or 1
    for v in ("A", "B"):
        ya = sum(1 for r in hasil if r[v]["header"])
        label = "dengan kontras kolom" if v == "A" else "tanpa kontras"
        print(f"  Varian {v} ({label:<21}) header={ya:>4}  tolak={len(hasil)-ya:>4}"
              f"  ({100*ya/n:.0f}% diulang)")
    print(f"\n  Putusan BERBEDA antar varian: {len(beda)}"
          f"   <-- ini yang perlu ditinjau manual")

    print("\n  Aturan penentu (Varian A):")
    for aturan, jml in Counter(r["A"]["aturan"] for r in hasil).most_common():
        contoh = next(r for r in hasil if r["A"]["aturan"] == aturan)
        print(f"    {aturan:<12}{jml:>4}   mis. {_potong(contoh['kandidat'] or ['-'], 40)}")

    per_dok = defaultdict(lambda: [0, 0])
    for r in hasil:
        per_dok[r["document_id"]][0 if r["A"]["header"] else 1] += 1
    print("\n  Per dokumen (header diulang / ditolak), Varian A:")
    for dok, (ya, tidak) in sorted(per_dok.items(), key=lambda kv: -sum(kv[1])):
        print(f"    {dok[:52]:<54}{ya:>4} / {tidak:<4}")

    tb = [r for r in hasil if r["tak_berubah"]]
    if tb:
        print(f"\n  BUTIR 6 — teks B tidak berubah antara v2 dan v3: {len(tb)}")
        for alasan, jml in Counter(r["tak_berubah"].split(" —")[0] for r in tb).most_common():
            print(f"    {jml:>4}  {alasan}")

    if hilang:
        print(f"\n  PERINGATAN: {len(hilang)} pasangan tidak ditemukan di dump v3")
        for kunci, sisi in hilang[:5]:
            print(f"    potongan {sisi} hilang: {kunci}")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"n_disetujui": len(disetujui), "hasil": hasil,
                        "beda": [r["kunci"] for r in beda]},
                       indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  Hasil lengkap: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
