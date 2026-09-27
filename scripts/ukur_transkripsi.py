"""ukur_transkripsi.py — ukur transkripsi tabel oleh vision SEBELUM implementasi.

Read-only: tidak menulis ke koleksi, cache, atau dump. Keluaran hanya ke --out.

Dua pengukuran (klasifikasi gambar ada di scripts/klasifikasi_gambar.py):
1. --hanya-dampak (tanpa model, tanpa PDF): sebaran rasio tumpang tindih chunk
   gambar terhadap bbox tabel sehalaman, celah terbesar, dampak chunk_id pada
   beberapa ambang calon, dan deskripsi v4 gambar berasio tinggi.
2. Transkripsi sampel tabel pada tiap DPI: waktu, token, done_reason, proksi
   akurasi angka terhadap lapisan teks ASLI, penanda, dan berkas berdampingan
   (teks v4 | transkripsi) per sampel. --perluas menguji perluasan area render.

Usage (server):
    python scripts/ukur_transkripsi.py --chunks <dump v4>/chunks.jsonl --hanya-dampak
    python scripts/ukur_transkripsi.py --chunks <dump v4>/chunks.jsonl \\
        --pdf-dir data/pdfs --sampel 3 --dpi 200 --num-predict 9000 \\
        --pilih ukt-tahun-2025_p5_c01 --perluas --out ~/ukur_perluas
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.transkripsi_ukur import (  # noqa: E402
    PROMPT_KLASIFIKASI, PROMPT_TRANSKRIPSI, bersihkan, cakupan_angka,
    celah_terbesar, dampak_serapan, ketepatan_angka, median, perluas_bbox,
    peringatan, regresi_linear, sebaran_tumpang, urai_markdown,
)
from lib.ukur_io import buka_area, panggil, png_pemanasan  # noqa: E402

AMBANG_CALON = (0.5, 0.8, 0.95, 0.99)
AMBANG_TAMPIL_DESKRIPSI = 0.9
# Margin area render halaman pindai tanpa lapisan teks, pecahan sisi halaman.
# p90 kelebihan kata di luar bbox tabel berlapis teks: 0,025.
MARGIN_PINDAI = 0.025


def baca_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ── pengukuran 1: dampak serapan ────────────────────────────────────────────

def lapor_dampak(rows: list[dict]) -> dict:
    tumpang = sebaran_tumpang(rows)
    per_id = {r["chunk_id"]: r for r in rows}
    n_gambar = sum(1 for r in rows if r.get("element_type") == "ImageDescription")
    print("\n== Tumpang tindih gambar vs tabel sehalaman ==")
    print(f"chunk gambar: {n_gambar}; bertumpang (>0) dengan tabel: {len(tumpang)}")
    ember = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0001]
    for lo, hi in zip(ember, ember[1:]):
        n = sum(1 for t in tumpang if lo < t.rasio <= hi)
        print(f"  ({lo:.2f}, {min(hi, 1):.2f}] {n:5d} {'#' * min(n, 60)}")
    celah = celah_terbesar([t.rasio for t in tumpang])
    if celah:
        print(f"celah terbesar: {celah[0]:.3f} -> {celah[1]:.3f} (lebar {celah[1] - celah[0]:.3f})")
    print("rasio per gambar (terurut):")
    for t in tumpang:
        print(f"  {t.rasio:.3f}  {t.image_chunk}  dalam {t.table_chunk}")
    print(f"deskripsi v4 gambar berasio >= {AMBANG_TAMPIL_DESKRIPSI}:")
    for t in tumpang:
        if t.rasio >= AMBANG_TAMPIL_DESKRIPSI:
            teks = " ".join((per_id[t.image_chunk].get("text_content") or "").split())
            print(f"  {t.rasio:.3f}  {t.image_chunk}\n      {teks[:400]}")
    dampak = [dampak_serapan(rows, tumpang, a) for a in AMBANG_CALON]
    print("dampak per ambang calon (BUKAN pilihan):")
    for d in dampak:
        print(f"  >= {d['ambang']:.2f}: gambar {d['gambar_diserap']}, "
              f"halaman {d['halaman_terdampak']}, chunk bergeser {d['chunk_bergeser']}")
    return {"n_gambar": n_gambar, "celah": celah, "dampak": dampak,
            "tumpang": [t.__dict__ for t in tumpang]}


# ── pemilihan sampel ────────────────────────────────────────────────────────

def _jenis(pdf_dir: Path, r: dict) -> str | None:
    pdf = pdf_dir / (r.get("file_name") or "")
    if not pdf.is_file() or not r.get("bbox") or not r.get("page_number"):
        return None
    try:
        return buka_area(pdf, r["page_number"], r["bbox"], None).jenis_halaman
    except Exception:
        return None


def pilih_sampel(rows: list[dict], pdf_dir: Path, n: int, pilih: list[str]) -> list[tuple[str, dict]]:
    """Deterministik: pilihan eksplisit, lalu terpanjang, lanjutan, pindai, digital.

    Pilihan eksplisit boleh chunk gambar (calon gambar-tabel); pengisian
    otomatis hanya dari chunk Table.
    """
    per_id = {r["chunk_id"]: r for r in rows}
    hasil = [("pilihan", per_id[c]) for c in pilih if c in per_id]
    for c in pilih:
        if c not in per_id:
            print(f"PERINGATAN: --pilih {c} tidak ada di dump")
    tabel = sorted((r for r in rows if r.get("element_type") == "Table" and r.get("bbox")),
                   key=lambda r: r["chunk_id"])
    dipakai = {r["chunk_id"] for _, r in hasil}

    def ambil(label, calon, k):
        for r in calon:
            if len(hasil) >= n or k <= 0:
                return
            if r["chunk_id"] not in dipakai:
                hasil.append((label, r))
                dipakai.add(r["chunk_id"])
                k -= 1

    sisa = max(0, n - len(hasil))
    kuota = [sisa - 3 * (sisa // 4), sisa // 4, sisa // 4, sisa // 4]
    ambil("terpanjang", sorted(tabel, key=lambda r: -len(r.get("text_content") or "")), kuota[0])
    ambil("lanjutan", [r for r in tabel if (r.get("table_part") or 0) >= 1], kuota[1])
    ambil("pindai", (r for r in tabel if (_jenis(pdf_dir, r) or "").startswith("pindai")), kuota[2])
    ambil("digital", (r for r in tabel if _jenis(pdf_dir, r) == "digital_asli"), n)
    return hasil


# ── pengukuran 2: transkripsi ───────────────────────────────────────────────

def area_render(pdf: Path, r: dict, perluas: bool) -> list[float]:
    """bbox chunk, atau bbox yang diperluas ke kata beririsan / margin pindai."""
    if not perluas:
        return list(r["bbox"])
    a = buka_area(pdf, r["page_number"], r["bbox"], None)
    if a.jenis_halaman == "pindai_tanpa_lapisan":
        return perluas_bbox(r["bbox"], (), MARGIN_PINDAI)
    return perluas_bbox(r["bbox"], a.kata)


def ukur_satu(label: str, r: dict, pdf_dir: Path, dpi: int, num_predict: int,
              out: Path, perluas: bool = False) -> dict:
    pdf = pdf_dir / r["file_name"]
    bbox = area_render(pdf, r, perluas)
    a = buka_area(pdf, r["page_number"], bbox, dpi)
    hasil = panggil(a.png, PROMPT_TRANSKRIPSI, num_predict)
    md = bersihkan(hasil["response"])
    baris = urai_markdown(md)
    # Angka rujukan hanya dari lapisan teks ASLI. Lapisan OCR halaman pindai
    # bukan kebenaran (laporan-keuangan p23, bagan-akun p10-11).
    digital = a.jenis_halaman == "digital_asli"
    ada_lapisan = a.jenis_halaman != "pindai_tanpa_lapisan"
    rujukan = a.teks_halaman if ada_lapisan else (r.get("text_content") or "")
    rec = {
        "label": label, "chunk_id": r["chunk_id"], "element_type": r.get("element_type"),
        "dpi": dpi, "jenis_halaman": a.jenis_halaman, "digital": digital,
        "bbox_chunk": r["bbox"], "bbox_render": bbox,
        "piksel": len(a.png), "karakter_v4": len(r.get("text_content") or ""),
        "karakter_md": len(md), "terurai": baris is not None,
        "jumlah_baris": len(baris) - 1 if baris else 0,
        "jumlah_kolom": len(baris[0]) if baris else 0,
        "terpotong": hasil["done_reason"] == "length",
        "cakupan_angka": cakupan_angka(md, a.teks_area) if digital else None,
        "ketepatan_angka": ketepatan_angka(md, a.teks_halaman) if digital else None,
        "rujukan_peringatan": "lapisan_teks" if ada_lapisan else "teks_v4",
        "peringatan": list(peringatan(baris, rujukan)) if baris else [],
        **{k: v for k, v in hasil.items() if k != "response"},
    }
    nama = f"{r['chunk_id']}_{dpi}dpi"
    (out / f"{nama}.png").write_bytes(a.png)
    (out / f"{nama}.md").write_text(
        f"# {r['chunk_id']} ({label}, {dpi} dpi)\n\n"
        f"```json\n{json.dumps(rec, ensure_ascii=False, indent=1)}\n```\n\n"
        f"## Teks v4 (OCR)\n\n{r.get('text_content') or ''}\n\n"
        f"## Transkripsi (dibersihkan)\n\n{md}\n\n"
        f"## Keluaran mentah model\n\n{hasil['response']}\n", encoding="utf-8")
    return rec


def ringkas_transkripsi(recs: list[dict], rows: list[dict]) -> dict:
    tabel = [r for r in rows if r.get("element_type") == "Table"]
    panjang = [len(r.get("text_content") or "") for r in tabel]
    ringkas = {}
    for dpi in sorted({r["dpi"] for r in recs}):
        rs = [r for r in recs if r["dpi"] == dpi]
        detik = [r["detik"] for r in rs]
        # Regresi hanya atas chunk Table: panjang teks chunk gambar adalah
        # narasi, bukan ukuran tabel.
        rt = [r for r in rs if r["element_type"] == "Table"]
        fit = regresi_linear([r["karakter_v4"] for r in rt], [r["detik"] for r in rt])
        proyeksi_lin = sum(max(0.0, fit[0] + fit[1] * p) for p in panjang) / 3600 if fit else None
        cak = [r["cakupan_angka"] for r in rs if r["cakupan_angka"] is not None]
        tep = [r["ketepatan_angka"] for r in rs if r["ketepatan_angka"] is not None]
        ringkas[dpi] = {
            "median_detik": median(detik), "maks_detik": max(detik),
            "proyeksi_jam_median": median(detik) * len(tabel) / 3600,
            "proyeksi_jam_linear": proyeksi_lin,
            "median_cakupan_angka": median(cak), "median_ketepatan_angka": median(tep),
            "tak_terurai": sum(not r["terurai"] for r in rs),
            "terpotong": sum(r["terpotong"] for r in rs),
            "maks_konteks": max((r["prompt_eval_count"] or 0) + (r["eval_count"] or 0) for r in rs),
        }
        rasio = [r["eval_count"] / r["karakter_v4"] for r in rt if r["eval_count"] and r["karakter_v4"]]
        if rasio and panjang:
            ringkas[dpi]["saran_num_predict"] = int(max(rasio) * max(panjang) * 1.3)
    return {"jumlah_tabel": len(tabel), "tabel_terpanjang_karakter": max(panjang or [0]),
            "per_dpi": ringkas}


def ukur_semua(rows: list[dict], a, pilih: list[str]) -> dict:
    sampel = pilih_sampel(rows, a.pdf_dir, a.sampel, pilih)
    print(f"\n== Transkripsi {len(sampel)} sampel{' (area diperluas)' if a.perluas else ''} ==")
    recs = []
    for dpi in [int(d) for d in a.dpi.split(",")]:
        for label, r in sampel:
            rec = ukur_satu(label, r, a.pdf_dir, dpi, a.num_predict, a.out, a.perluas)
            recs.append(rec)
            print(f"  {dpi}dpi {rec['detik']:6.1f}s in={rec['prompt_eval_count']} "
                  f"out={rec['eval_count']} {rec['done_reason']} "
                  f"baris={rec['jumlah_baris']} cak={rec['cakupan_angka']} "
                  f"{rec['jenis_halaman']} {label:10s} {r['chunk_id']} {rec['peringatan']}")
    hasil = {"transkripsi": recs, "ringkas": ringkas_transkripsi(recs, rows)}
    print(json.dumps(hasil["ringkas"], indent=1))
    return hasil


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunks", type=Path, required=True)
    ap.add_argument("--pdf-dir", type=Path)
    ap.add_argument("--sampel", type=int, default=10)
    ap.add_argument("--pilih", default="", help="chunk_id dipisah koma, didahulukan")
    ap.add_argument("--dpi", default="200")
    ap.add_argument("--num-predict", type=int, default=9000)
    ap.add_argument("--perluas", action="store_true",
                    help="perluas area render ke kata beririsan (margin untuk pindaian)")
    ap.add_argument("--out", type=Path, default=Path("ukur_transkripsi"))
    ap.add_argument("--hanya-dampak", action="store_true")
    a = ap.parse_args()

    rows = baca_jsonl(a.chunks)
    a.out.mkdir(parents=True, exist_ok=True)
    laporan = {"chunks": str(a.chunks), "dampak": lapor_dampak(rows)}
    if not a.hanya_dampak:
        if not a.pdf_dir:
            raise SystemExit("--pdf-dir wajib tanpa --hanya-dampak")
        pilih = [c.strip() for c in a.pilih.split(",") if c.strip()]
        panggil(png_pemanasan(), PROMPT_KLASIFIKASI, 5)
        laporan.update(ukur_semua(rows, a, pilih))
    (a.out / "laporan.json").write_text(json.dumps(laporan, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nlaporan: {a.out / 'laporan.json'}")


if __name__ == "__main__":
    main()
