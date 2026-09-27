"""bandingkan_klasifikasi.py — bandingkan dua run klasifikasi_gambar.py.

Mencetak matriks perpindahan jenis (lama -> baru), daftar gambar yang
berpindah, dan memeriksa HARAPAN: kasus yang sudah ditinjau manusia dan
jenisnya diketahui. Kasus harapan dicocokkan lewat awalan document_id dan
akhiran _pN_cNN, karena nama lengkapnya panjang dan bisa bergeser antar dump;
kecocokan nol atau ganda dilaporkan, tidak ditebak.

Usage:
    python scripts/bandingkan_klasifikasi.py \\
        --lama ~/klasifikasi_v5/klasifikasi.jsonl \\
        --baru ~/klasifikasi_v5_p2/klasifikasi.jsonl --out ~/banding_klasifikasi.csv
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

# (awalan document_id, akhiran chunk_id, jenis yang diharapkan). Dari tinjauan
# daftar klasifikasi versi 1 atas 1.441 gambar.
HARAPAN = (
    # tangkapan layar aplikasi dan formulir online
    ("buku-pedoman-skpi", "_p25_c03", "lainnya"),
    ("draft-panduan-teknis-kkn", "_p89_c01", "lainnya"),
    ("draft-panduan-teknis-kkn", "_p90_c02", "lainnya"),
    ("draft-panduan-teknis-kkn", "_p91_c01", "lainnya"),
    ("draft-panduan-teknis-kkn", "_p93_c01", "lainnya"),
    ("manual-dosen$", "_p19_c02", "lainnya"),
    ("manual$", "_p16_c00", "lainnya"),
    ("manual$", "_p27_c02", "lainnya"),
    ("manual$", "_p28_c00", "lainnya"),
    # flowchart bersiku kolom
    ("v2-sop-evaluasi-delapan-semester", "_p6_c02", "lainnya"),
    ("v2-sop-evaluasi-empat-semester", "_p6_c02", "lainnya"),
    # panduan dan matriks logo
    ("logo-universitas-hasanuddin", "_p9_c01", "lainnya"),
    ("tata-naskah-dinas", "_p141_c02", "lainnya"),
    # surat pernyataan syarat dan ketentuan DIPA (dari deskripsi v2; periksa)
    (LK, "_p168_c05", "lainnya"),
    # cap BSrE di atas tabel
    ("ukt-tahun-2025", "_p5_c02", "cap"),
    # tabel sungguhan yang sudah terkonfirmasi
    (LK, "_p23_c00", "tabel"),
    ("rubrik-2024", "_p50_c02", "tabel"),
    ("sop-final-project", "_p7_c01", "tabel"),
    ("sop12", "_p10_c02", "tabel"),
    ("sop12", "_p11_c03", "tabel"),
    ("pedoman-penyusunan-laporan-kinerja", "_p18_c00", "tabel"),
)


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
            keluar.append({"kasus": awalan + akhiran, "harap": harap, "dapat": "",
                           "status": f"{len(cocok)} kecocokan {cocok}"})
            continue
        dapat = baru[cocok[0]].get("jenis")
        keluar.append({"kasus": cocok[0], "harap": harap, "dapat": dapat,
                       "status": "OK" if dapat == harap else "MELESET"})
    return keluar


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lama", type=Path, required=True)
    ap.add_argument("--baru", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    lama, baru = baca(a.lama), baca(a.baru)

    pindah = Counter((lama[c].get("jenis"), baru[c].get("jenis")) for c in lama.keys() & baru.keys())
    print(f"lama {len(lama)}, baru {len(baru)}, sama-sama ada {len(lama.keys() & baru.keys())}")
    print("perpindahan jenis (lama -> baru):")
    for (x, y), n in sorted(pindah.items(), key=lambda t: -t[1]):
        print(f"  {str(x):8s} -> {str(y):8s} {n:5d}{'' if x == y else '  *'}")

    berubah = sorted(c for c in lama.keys() & baru.keys() if lama[c].get("jenis") != baru[c].get("jenis"))
    with a.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chunk_id", "lama", "baru", "jawaban_baru"])
        for c in berubah:
            w.writerow([c, lama[c].get("jenis"), baru[c].get("jenis"), baru[c].get("jawaban_mentah", "")[:80]])
    print(f"{len(berubah)} gambar berpindah jenis -> {a.out}")

    print("\nkasus harapan:")
    hasil = periksa_harapan(baru)
    for h in hasil:
        print(f"  {h['status']:8s} harap {h['harap']:8s} dapat {str(h['dapat']):8s} {h['kasus']}")
    meleset = [h for h in hasil if h["status"] != "OK"]
    print(f"\n{len(hasil) - len(meleset)}/{len(hasil)} sesuai harapan")
    sys.exit(1 if meleset else 0)


if __name__ == "__main__":
    main()
