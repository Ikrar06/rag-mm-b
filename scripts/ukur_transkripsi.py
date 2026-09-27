"""ukur_transkripsi.py — ukur transkripsi tabel oleh vision SEBELUM implementasi.

Read-only: tidak menulis ke koleksi, cache, atau dump. Keluaran hanya ke --out.

Tiga pengukuran:
1. --hanya-dampak (tanpa model, tanpa PDF): sebaran rasio tumpang tindih chunk
   gambar terhadap bbox tabel sehalaman, celah terbesar, dan dampak chunk_id
   pada beberapa ambang calon. Ambang TIDAK dipilih di sini.
2. Transkripsi sampel tabel pada tiap DPI: waktu, token, done_reason, proksi
   akurasi angka terhadap lapisan teks, penanda, dan berkas berdampingan
   (teks v4 | transkripsi) per sampel.
3. --klasifikasi N: waktu dan jawaban "tabel atau bukan" untuk N chunk gambar,
   dirender lalu diperkecil ke sisi terpanjang --sisi-klasifikasi.

Usage (server):
    python scripts/ukur_transkripsi.py --chunks <dump v4>/chunks.jsonl --hanya-dampak
    python scripts/ukur_transkripsi.py --chunks <dump v4>/chunks.jsonl \\
        --pdf-dir data/pdfs --sampel 10 --dpi 150,200 \\
        --pilih <chunk UKT>,<chunk jadwal retensi>,<chunk gambar-tabel> \\
        --klasifikasi 20 --out ~/ukur_transkripsi
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.transkripsi_ukur import (  # noqa: E402
    PROMPT_KLASIFIKASI, PROMPT_TRANSKRIPSI, bersihkan, cakupan_angka,
    celah_terbesar, dampak_serapan, ketepatan_angka, median, peringatan,
    regresi_linear, sebaran_tumpang, urai_markdown,
)

NUM_CTX = 16384
SEED = 1337
TIMEOUT_DETIK = 1800.0
MIN_KARAKTER_LAPISAN_TEKS = 20
AMBANG_CALON = (0.5, 0.8, 0.95, 0.99)


def baca_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ── pengukuran 1: dampak serapan ────────────────────────────────────────────

def lapor_dampak(rows: list[dict]) -> dict:
    tumpang = sebaran_tumpang(rows)
    n_gambar = sum(1 for r in rows if r.get("element_type") == "ImageDescription")
    print(f"\n== Tumpang tindih gambar vs tabel sehalaman ==")
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
    dampak = [dampak_serapan(rows, tumpang, a) for a in AMBANG_CALON]
    print("dampak per ambang calon (BUKAN pilihan):")
    for d in dampak:
        print(f"  >= {d['ambang']:.2f}: gambar {d['gambar_diserap']}, "
              f"halaman {d['halaman_terdampak']}, chunk bergeser {d['chunk_bergeser']}")
    return {"n_gambar": n_gambar, "celah": celah, "dampak": dampak,
            "tumpang": [t.__dict__ for t in tumpang]}


# ── akses PDF dan model ─────────────────────────────────────────────────────

def _klip(page, bbox):
    import fitz
    r = page.rect
    return fitz.Rect(r.x0 + bbox[0] * r.width, r.y0 + bbox[1] * r.height,
                     r.x0 + bbox[2] * r.width, r.y0 + bbox[3] * r.height)


def buka_area(pdf: Path, halaman: int, bbox, dpi: int | None) -> tuple[bytes, str, str]:
    """(png area, teks lapisan area, teks lapisan halaman). dpi None: tanpa render."""
    import fitz
    with fitz.open(str(pdf)) as doc:
        page = doc[halaman - 1]
        klip = _klip(page, bbox)
        png = page.get_pixmap(dpi=dpi, clip=klip).tobytes("png") if dpi else b""
        return png, page.get_text("text", clip=klip), page.get_text("text")


def perkecil(png: bytes, sisi: int) -> bytes:
    from PIL import Image
    img = Image.open(io.BytesIO(png)).convert("RGB")
    skala = sisi / max(img.size)
    if skala < 1:
        img = img.resize((max(1, round(img.width * skala)), max(1, round(img.height * skala))))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def panggil(png: bytes, prompt: str, num_predict: int) -> dict:
    import httpx
    from backend.config import LLM_BASE_URL, LLM_PROVIDER, VISION_MODEL
    if LLM_PROVIDER != "ollama":
        raise SystemExit(f"alat ukur hanya untuk ollama, LLM_PROVIDER={LLM_PROVIDER}")
    payload = {"model": VISION_MODEL, "prompt": prompt, "stream": False,
               "images": [base64.b64encode(png).decode()],
               "options": {"temperature": 0, "seed": SEED, "num_ctx": NUM_CTX,
                           "num_predict": num_predict}}
    t0 = time.monotonic()
    with httpx.Client(timeout=TIMEOUT_DETIK) as c:
        data = c.post(f"{LLM_BASE_URL}/api/generate", json=payload).raise_for_status().json()
    return {"detik": round(time.monotonic() - t0, 1), "response": data.get("response", ""),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count": data.get("eval_count"), "done_reason": data.get("done_reason")}


# ── pemilihan sampel ────────────────────────────────────────────────────────

def ada_lapisan_teks(pdf_dir: Path, r: dict) -> bool | None:
    pdf = pdf_dir / (r.get("file_name") or "")
    if not pdf.is_file() or not r.get("bbox") or not r.get("page_number"):
        return None
    try:
        _, teks, _ = buka_area(pdf, r["page_number"], r["bbox"], None)
    except Exception:
        return None
    return len(teks.strip()) >= MIN_KARAKTER_LAPISAN_TEKS


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
    ambil("pindai", (r for r in tabel if ada_lapisan_teks(pdf_dir, r) is False), kuota[2])
    ambil("digital", (r for r in tabel if ada_lapisan_teks(pdf_dir, r)), n)
    return hasil


# ── pengukuran 2: transkripsi ───────────────────────────────────────────────

def ukur_satu(label: str, r: dict, pdf_dir: Path, dpi: int, num_predict: int, out: Path) -> dict:
    pdf = pdf_dir / r["file_name"]
    png, teks_bbox, teks_hal = buka_area(pdf, r["page_number"], r["bbox"], dpi)
    hasil = panggil(png, PROMPT_TRANSKRIPSI, num_predict)
    md = bersihkan(hasil["response"])
    baris = urai_markdown(md)
    digital = len(teks_bbox.strip()) >= MIN_KARAKTER_LAPISAN_TEKS
    rujukan = teks_hal if digital else (r.get("text_content") or "")
    rec = {
        "label": label, "chunk_id": r["chunk_id"], "element_type": r.get("element_type"),
        "dpi": dpi, "digital": digital,
        "piksel": len(png), "karakter_v4": len(r.get("text_content") or ""),
        "karakter_md": len(md), "terurai": baris is not None,
        "jumlah_baris": len(baris) - 1 if baris else 0,
        "jumlah_kolom": len(baris[0]) if baris else 0,
        "terpotong": hasil["done_reason"] == "length",
        "cakupan_angka": cakupan_angka(md, teks_bbox) if digital else None,
        "ketepatan_angka": ketepatan_angka(md, teks_hal) if digital else None,
        "peringatan": list(peringatan(baris, rujukan)) if baris else [],
        **{k: v for k, v in hasil.items() if k != "response"},
    }
    nama = f"{r['chunk_id']}_{dpi}dpi"
    (out / f"{nama}.png").write_bytes(png)
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


# ── pengukuran 3: klasifikasi gambar ────────────────────────────────────────

def ukur_klasifikasi(rows: list[dict], pdf_dir: Path, n: int, sisi: int, pilih: list[str]) -> dict:
    gambar = sorted((r for r in rows if r.get("element_type") == "ImageDescription" and r.get("bbox")),
                    key=lambda r: r["chunk_id"])
    langkah = max(1, len(gambar) // max(1, n))
    sampel = [r for r in gambar if r["chunk_id"] in pilih] + gambar[::langkah][:n]
    recs = []
    for r in sampel:
        png, _, _ = buka_area(pdf_dir / r["file_name"], r["page_number"], r["bbox"], 150)
        kecil = perkecil(png, sisi)
        h = panggil(kecil, PROMPT_KLASIFIKASI, 20)
        recs.append({"chunk_id": r["chunk_id"], "image_id": r.get("image_id"),
                     "jawaban": h["response"].strip(), "detik": h["detik"],
                     "prompt_eval_count": h["prompt_eval_count"]})
        print(f"  {h['detik']:6.1f}s  {h['response'].strip()[:30]:30s}  {r['chunk_id']}")
    detik = [r["detik"] for r in recs]
    n_gambar = len(gambar)
    return {"sisi": sisi, "sampel": recs, "median_detik": median(detik),
            "jumlah_gambar_dideskripsi": n_gambar,
            "proyeksi_jam": (median(detik) or 0) * n_gambar / 3600}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunks", type=Path, required=True)
    ap.add_argument("--pdf-dir", type=Path)
    ap.add_argument("--sampel", type=int, default=10)
    ap.add_argument("--pilih", default="", help="chunk_id dipisah koma, didahulukan")
    ap.add_argument("--dpi", default="150,200")
    ap.add_argument("--num-predict", type=int, default=8192)
    ap.add_argument("--klasifikasi", type=int, default=0)
    ap.add_argument("--sisi-klasifikasi", type=int, default=512)
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
        panggil(_png_pemanasan(), PROMPT_KLASIFIKASI, 5)
        laporan.update(ukur_semua(rows, a, pilih))
    (a.out / "laporan.json").write_text(json.dumps(laporan, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nlaporan: {a.out / 'laporan.json'}")


def _png_pemanasan() -> bytes:
    """Gambar kecil untuk memuat model ke memori; waktunya tidak dihitung."""
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(buf, format="PNG")
    return buf.getvalue()


def ukur_semua(rows: list[dict], a, pilih: list[str]) -> dict:
    sampel = pilih_sampel(rows, a.pdf_dir, a.sampel, pilih)
    print(f"\n== Transkripsi {len(sampel)} sampel ==")
    recs = []
    for dpi in [int(d) for d in a.dpi.split(",")]:
        for label, r in sampel:
            rec = ukur_satu(label, r, a.pdf_dir, dpi, a.num_predict, a.out)
            recs.append(rec)
            print(f"  {dpi}dpi {rec['detik']:6.1f}s in={rec['prompt_eval_count']} "
                  f"out={rec['eval_count']} {rec['done_reason']} "
                  f"baris={rec['jumlah_baris']} cak={rec['cakupan_angka']} "
                  f"{label:10s} {r['chunk_id']} {rec['peringatan']}")
    hasil = {"transkripsi": recs, "ringkas": ringkas_transkripsi(recs, rows)}
    print(json.dumps(hasil["ringkas"], indent=1))
    if a.klasifikasi:
        print(f"\n== Klasifikasi {a.klasifikasi} gambar, sisi {a.sisi_klasifikasi}px ==")
        k = ukur_klasifikasi(rows, a.pdf_dir, a.klasifikasi, a.sisi_klasifikasi,
                             [c for c in pilih if not _tabel(rows, c)])
        print(f"median {k['median_detik']}s; proyeksi {k['proyeksi_jam']:.1f} jam "
              f"untuk {k['jumlah_gambar_dideskripsi']} gambar")
        hasil["klasifikasi"] = k
    return hasil


def _tabel(rows: list[dict], chunk_id: str) -> bool:
    return any(r["chunk_id"] == chunk_id and r.get("element_type") == "Table" for r in rows)


if __name__ == "__main__":
    main()
