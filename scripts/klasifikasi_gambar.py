"""klasifikasi_gambar.py — klasifikasi tiga jenis (tabel / cap / lainnya) gambar
yang dideskripsikan, untuk DITINJAU sebelum reindex v5.

Read-only terhadap koleksi, cache, dan dump. Satu panggilan per chunk
ImageDescription: area bbox dirender 150 dpi lalu diperkecil ke sisi
terpanjang --sisi px (menentukan jenis tidak butuh detail sel).

Hasil ditulis per baris ke klasifikasi.jsonl begitu tiap gambar selesai, jadi
run yang terputus dilanjutkan dengan perintah yang sama. Melanjutkan dengan
prompt berbeda DITOLAK: satu berkas hanya boleh berisi jawaban satu prompt.

Gambar yang tetap gagal setelah coba ulang dicatat di gagal.jsonl, BUKAN di
klasifikasi.jsonl, lalu run berlanjut. Run berikutnya mencobanya lagi.
Pengecilan tidak pernah membuat sisi pendek di bawah batas model; sisanya
ditambal putih (manual_p23_c03 2087x118 -> 512x29 memicu Ollama 500).

Keluaran di --out:
    klasifikasi.jsonl       satu baris per gambar
    klasifikasi_tinjau.csv  terurut per jenis, dengan rasio tumpang tabel dan deskripsi v4
    gagal.jsonl             gambar yang gagal beserta alasannya (dicoba lagi di run berikut)
    gambar/<chunk_id>.png   gambar yang dikirim, untuk jenis tabel dan cap

Usage (server):
    python scripts/klasifikasi_gambar.py --chunks <dump v4>/chunks.jsonl \\
        --pdf-dir data/pdfs --out ~/klasifikasi_v5
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.transkripsi_ukur import (  # noqa: E402
    PROMPT_KLASIFIKASI, median, sebaran_tumpang, urai_klasifikasi,
)
from lib.ukur_io import (  # noqa: E402
    GagalVision, batas_model, buka_area, panggil, png_pemanasan, siapkan_png,
)

PROMPT_SHA = hashlib.sha256(PROMPT_KLASIFIKASI.encode("utf-8")).hexdigest()
DPI_RENDER = 150
NUM_PREDICT = 20
KOLOM_TINJAU = ("jenis", "chunk_id", "image_id", "halaman", "rasio_tumpang_tabel",
                "tabel_induk", "detik", "jawaban_mentah", "deskripsi_v4")


def baca_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def gambar_dideskripsi(rows: list[dict]) -> list[dict]:
    return sorted((r for r in rows if r.get("element_type") == "ImageDescription" and r.get("bbox")),
                  key=lambda r: r["chunk_id"])


def sudah_selesai(hasil_lama: list[dict]) -> set[str]:
    """chunk_id yang sudah diklasifikasi. Menolak berkas dari prompt lain."""
    lain = {h.get("prompt_sha256") for h in hasil_lama} - {PROMPT_SHA}
    if lain:
        raise SystemExit(f"klasifikasi.jsonl berisi jawaban prompt lain {sorted(lain)}; "
                         "pakai --out baru")
    return {h["chunk_id"] for h in hasil_lama}


def klasifikasi_satu(r: dict, pdf_dir: Path, sisi: int) -> tuple[dict, bytes]:
    a = buka_area(pdf_dir / r["file_name"], r["page_number"], r["bbox"], DPI_RENDER)
    kecil, _ = siapkan_png(a.png, sisi)
    h = panggil(kecil, PROMPT_KLASIFIKASI, NUM_PREDICT)
    return {"chunk_id": r["chunk_id"], "image_id": r.get("image_id"),
            "document_id": r.get("document_id"), "halaman": r.get("page_number"),
            "jenis": urai_klasifikasi(h["response"]), "jawaban_mentah": h["response"].strip(),
            "detik": h["detik"], "prompt_eval_count": h["prompt_eval_count"],
            "percobaan": h.get("percobaan", 1), "dipadding": h.get("dipadding", False),
            "sisi": sisi, "prompt_sha256": PROMPT_SHA}, kecil


def baris_tinjau(hasil: list[dict], rows: list[dict]) -> list[dict]:
    per_id = {r["chunk_id"]: r for r in rows}
    tumpang = {t.image_chunk: t for t in sebaran_tumpang(rows)}
    urutan = {"cap": 0, "tabel": 1, None: 2, "lainnya": 3}
    keluar = []
    for h in sorted(hasil, key=lambda h: (urutan.get(h["jenis"], 2), h["chunk_id"])):
        t = tumpang.get(h["chunk_id"])
        teks = " ".join((per_id.get(h["chunk_id"], {}).get("text_content") or "").split())
        keluar.append({
            "jenis": h["jenis"] or "TAK_DIKENALI", "chunk_id": h["chunk_id"],
            "image_id": h.get("image_id") or "", "halaman": h.get("halaman"),
            "rasio_tumpang_tabel": f"{t.rasio:.3f}" if t else "",
            "tabel_induk": t.table_chunk if t else "", "detik": h["detik"],
            "jawaban_mentah": h["jawaban_mentah"][:80], "deskripsi_v4": teks[:300],
        })
    return keluar


def ringkasan(hasil: list[dict], rows: list[dict], gagal: list[dict]) -> dict:
    tumpang = {t.image_chunk for t in sebaran_tumpang(rows)}
    selesai = {h["chunk_id"] for h in hasil}
    belum = {g["chunk_id"]: g["alasan"] for g in gagal if g["chunk_id"] not in selesai}
    jenis = Counter(h["jenis"] or "TAK_DIKENALI" for h in hasil)
    detik = [h["detik"] for h in hasil]
    return {
        "jumlah": len(hasil), "per_jenis": dict(jenis),
        "median_detik": median(detik), "total_jam": sum(detik) / 3600,
        # Dua kasus yang aturannya belum diputuskan:
        "cap_tanpa_tumpang_tabel": sum(1 for h in hasil if h["jenis"] == "cap"
                                       and h["chunk_id"] not in tumpang),
        "tabel_bertumpang_tabel": sum(1 for h in hasil if h["jenis"] == "tabel"
                                      and h["chunk_id"] in tumpang),
        "dipadding": sum(1 for h in hasil if h.get("dipadding")),
        "diulang": sum(1 for h in hasil if h.get("percobaan", 1) > 1),
        "gagal_belum_terklasifikasi": len(belum),
        "gagal": belum,
        "prompt_sha256": PROMPT_SHA,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunks", type=Path, required=True)
    ap.add_argument("--pdf-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sisi", type=int, default=512)
    ap.add_argument("--batas", type=int, default=0, help="0 = semua gambar")
    a = ap.parse_args()

    rows = baca_jsonl(a.chunks)
    (a.out / "gambar").mkdir(parents=True, exist_ok=True)
    berkas = a.out / "klasifikasi.jsonl"
    selesai = sudah_selesai(baca_jsonl(berkas))
    antre = [r for r in gambar_dideskripsi(rows) if r["chunk_id"] not in selesai]
    if a.batas:
        antre = antre[:a.batas]
    print(f"gambar: {len(gambar_dideskripsi(rows))}; sudah: {len(selesai)}; antre: {len(antre)}")
    berkas_gagal = a.out / "gagal.jsonl"
    if antre:
        print(f"batas model: {batas_model()}")
        panggil(png_pemanasan(), PROMPT_KLASIFIKASI, 5)
    with berkas.open("a", encoding="utf-8") as f, berkas_gagal.open("a", encoding="utf-8") as fg:
        for i, r in enumerate(antre, 1):
            try:
                h, png = klasifikasi_satu(r, a.pdf_dir, a.sisi)
            except GagalVision as e:
                fg.write(json.dumps({"chunk_id": r["chunk_id"], "alasan": str(e)}) + "\n")
                fg.flush()
                print(f"  [{i}/{len(antre)}] GAGAL {r['chunk_id']}: {e}")
                continue
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
            f.flush()
            if h["jenis"] != "lainnya":
                (a.out / "gambar" / f"{r['chunk_id']}.png").write_bytes(png)
            print(f"  [{i}/{len(antre)}] {h['detik']:5.1f}s {str(h['jenis']):8s} {r['chunk_id']}")

    hasil = baca_jsonl(berkas)
    with (a.out / "klasifikasi_tinjau.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KOLOM_TINJAU)
        w.writeheader()
        w.writerows(baris_tinjau(hasil, rows))
    ring = ringkasan(hasil, rows, baca_jsonl(berkas_gagal))
    (a.out / "ringkasan.json").write_text(json.dumps(ring, indent=1), encoding="utf-8")
    print(json.dumps(ring, indent=1))


if __name__ == "__main__":
    main()
