"""bandingkan_klasifikasi.py — bandingkan dua run klasifikasi_gambar.py.

Label tiap gambar = jenis, ditambah "+tabel" bila memuat_tabel_data true
(v3). Rekaman v1/v2 tanpa jawaban kedua berlabel jenis saja.

Mencetak matriks perpindahan jenis (lama -> baru), daftar gambar yang
berpindah, dan memeriksa HARAPAN: kasus yang sudah ditinjau manusia dan
jenisnya diketahui. Kasus harapan dicocokkan lewat awalan document_id dan
akhiran _pN_cNN, karena nama lengkapnya panjang dan bisa bergeser antar dump;
kecocokan nol atau ganda dilaporkan, tidak ditebak.

Usage:
    python scripts/bandingkan_klasifikasi.py \\
        --lama ~/klasifikasi_v5/klasifikasi.jsonl \\
        --baru ~/klasifikasi_v5_p3/klasifikasi.jsonl --out ~/banding.csv \\
        --chunks <dump v4>/chunks.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

# Ada dua PDF pedoman laporan keuangan (HAANUDDIN dan HASANUDDIN); kasusnya
# dari yang pertama.
LK = "pedoman-penyusunan-laporan-keuangan-perguruan-tinggi-negeri-badan-hukum-universitas-haanuddin$"

# Label yang diterima per kasus. Label = jenis, ditambah "+tabel" bila
# memuat_tabel_data true (lihat lib.klasifikasi.Klasifikasi.label).
TABEL = frozenset({"tabel", "lainnya+tabel"})       # isi tabel tersalin
LAINNYA = frozenset({"lainnya", "lainnya+tabel"})   # narasi dipertahankan
NARASI_SAJA = frozenset({"lainnya"})
CAP = frozenset({"cap", "cap+tabel"})

# (awalan document_id, akhiran chunk_id, label yang diterima). Dari tinjauan
# daftar klasifikasi v1 dan v2 atas 1.441 gambar.
HARAPAN = (
    # tangkapan layar aplikasi dan formulir online
    ("buku-pedoman-skpi", "_p25_c03", LAINNYA),
    ("draft-panduan-teknis-kkn", "_p89_c01", LAINNYA),
    ("draft-panduan-teknis-kkn", "_p90_c02", LAINNYA),
    ("draft-panduan-teknis-kkn", "_p91_c01", LAINNYA),
    ("draft-panduan-teknis-kkn", "_p93_c01", LAINNYA),
    ("manual-dosen$", "_p19_c02", LAINNYA),
    ("manual$", "_p16_c00", LAINNYA),
    ("manual$", "_p27_c02", LAINNYA),
    ("manual$", "_p28_c00", LAINNYA),
    # flowchart bersiku kolom: kisi tata letak, bukan tabel data
    ("v2-sop-evaluasi-delapan-semester", "_p6_c02", NARASI_SAJA),
    ("v2-sop-evaluasi-empat-semester", "_p6_c02", NARASI_SAJA),
    # panduan dan matriks logo
    ("logo-universitas-hasanuddin", "_p9_c01", LAINNYA),
    ("tata-naskah-dinas", "_p141_c02", LAINNYA),
    # surat pernyataan syarat dan ketentuan DIPA
    (LK, "_p168_c05", LAINNYA),
    # cap BSrE di atas tabel
    ("ukt-tahun-2025", "_p5_c02", CAP),
    # tabel sungguhan: harus tersalin, sebagai tabel atau narasi+tabel
    (LK, "_p23_c00", TABEL),
    ("rubrik-2024", "_p50_c02", TABEL),
    ("sop-final-project", "_p7_c01", TABEL),
    ("sop12", "_p10_c02", TABEL),
    ("sop12", "_p11_c03", TABEL),
    ("pedoman-penyusunan-laporan-kinerja", "_p18_c00", TABEL),
    ("pedoman-penyusunan-laporan-kinerja", "_p47_c04", TABEL),
    ("peraturan-pedoman-penyusunan-laporan-kinerja", "_p47_c04", TABEL),
    ("panduan-dan-jurnal-kkn", "_p15_c00", TABEL),
    ("pedoman-penulisan-tesis", "_p33_c02", TABEL),
)


def label(h: dict) -> str:
    """Label rekaman; rekaman v1/v2 tanpa memuat_tabel_data berlabel jenis saja."""
    return f"{h.get('jenis')}{'+tabel' if h.get('memuat_tabel_data') else ''}"


def baca(path: Path) -> dict[str, dict]:
    return {h["chunk_id"]: h for h in map(json.loads, path.read_text(encoding="utf-8").splitlines())
            if h}


def cocokkan(ids, awalan: str, akhiran: str) -> list[str]:
    """chunk_id dengan document_id berawalan `awalan` dan berakhiran `akhiran`.

    document_id harus sama dengan awalan atau diawali awalan + "-". Awalan
    berakhiran "$" harus sama persis: "manual$" tidak menangkap manual-dosen.
    """
    persis = awalan.endswith("$")
    awalan = awalan.rstrip("$")
    hasil = []
    for cid in ids:
        if not cid.endswith(akhiran):
            continue
        doc = cid[: -len(akhiran)]
        if doc == awalan or (not persis and doc.startswith(awalan + "-")):
            hasil.append(cid)
    return sorted(hasil)


def periksa_harapan(baru: dict[str, dict]) -> list[dict]:
    keluar = []
    for awalan, akhiran, harap in HARAPAN:
        cocok = cocokkan(baru, awalan, akhiran)
        if len(cocok) != 1:
            keluar.append({"kasus": awalan + akhiran, "harap": "|".join(sorted(harap)), "dapat": "",
                           "status": f"{len(cocok)} kecocokan {cocok}"})
            continue
        dapat = label(baru[cocok[0]])
        keluar.append({"kasus": cocok[0], "harap": "|".join(sorted(harap)), "dapat": dapat,
                       "status": "OK" if dapat in harap else "MELESET"})
    return keluar


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lama", type=Path, required=True)
    ap.add_argument("--baru", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunks", type=Path, help="chunks.jsonl dump v4, untuk kolom deskripsi_v4")
    a = ap.parse_args()
    lama, baru = baca(a.lama), baca(a.baru)
    deskripsi = {}
    if a.chunks:
        for r in map(json.loads, a.chunks.read_text(encoding="utf-8").splitlines()):
            deskripsi[r["chunk_id"]] = " ".join((r.get("text_content") or "").split())[:400]

    pindah = Counter((label(lama[c]), label(baru[c])) for c in lama.keys() & baru.keys())
    print(f"lama {len(lama)}, baru {len(baru)}, sama-sama ada {len(lama.keys() & baru.keys())}")
    print("perpindahan jenis (lama -> baru):")
    for (x, y), n in sorted(pindah.items(), key=lambda t: -t[1]):
        print(f"  {x:14s} -> {y:14s} {n:5d}{'' if x == y else '  *'}")

    berubah = sorted((c for c in lama.keys() & baru.keys() if label(lama[c]) != label(baru[c])),
                     key=lambda c: (label(lama[c]), label(baru[c]), c))
    with a.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chunk_id", "lama", "baru", "jawaban_baru", "deskripsi_v4"])
        for c in berubah:
            w.writerow([c, label(lama[c]), label(baru[c]), baru[c].get("jawaban_mentah", "")[:80],
                        deskripsi.get(c, "")])
    print(f"{len(berubah)} gambar berpindah jenis -> {a.out}")

    print("\nkasus harapan:")
    hasil = periksa_harapan(baru)
    for h in hasil:
        print(f"  {h['status']:8s} harap {h['harap']:22s} dapat {h['dapat']:14s} {h['kasus']}")
    meleset = [h for h in hasil if h["status"] != "OK"]
    print(f"\n{len(hasil) - len(meleset)}/{len(hasil)} sesuai harapan")
    sys.exit(1 if meleset else 0)


if __name__ == "__main__":
    main()
