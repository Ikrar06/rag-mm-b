"""uji_transkripsi_pipeline.py — uji sampel transkripsi tabel LEWAT JALUR PIPELINE,
berdampingan dengan chunk v4.

Setiap dokumen sampel dijalankan utuh melalui fungsi indexing yang sama
(ekstraksi hi_res, simpan gambar, table_transcription.proses, deskripsi,
chunking) sehingga chunk_id dapat dibandingkan langsung dengan v4. Transkripsi
dan klasifikasi — yang mahal — dibatasi ke halaman sampel (ditambah halaman
kepala rantai untuk potongan lanjutan); halaman lain berjalan persis seperti
flag mati.

Tidak menyentuh Qdrant. Gambar ditulis ke --out/images, bukan data/images.
Cache vision dipakai bila aktif (deskripsi v4 menjadi hit).

Syarat: .env berisi konfigurasi riset (cp .env.research .env), termasuk
INDEX_TABLE_TRANSCRIPTION dan INDEX_IMAGE_TABLE_TRANSCRIPTION.

Usage (server):
    python scripts/uji_transkripsi_pipeline.py --chunks <dump v4>/chunks.jsonl \\
        --pdf-dir data/pdfs --out ~/uji_pipeline_v5
    # pemilih lain: --pilih "ukt-tahun-2025:5:Table,lanjutan,<chunk_id>"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.uji_pipeline import (  # noqa: E402
    SAMPEL_BAWAAN, baris_ringkas, halaman_uji, md_berdampingan, pasangkan, pilih,
)


def baca_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def periksa_konfigurasi(config) -> None:
    kurang = [n for n in ("INDEX_TABLE_TRANSCRIPTION", "INDEX_IMAGE_TABLE_TRANSCRIPTION",
                          "INDEX_STRUCTURAL_METADATA", "INDEX_PERSIST_IMAGES")
              if not getattr(config, n)]
    if kurang or config.PDF_EXTRACTION_STRATEGY != "hi_res":
        raise SystemExit(f"konfigurasi bukan riset: mati {kurang}, strategi "
                         f"{config.PDF_EXTRACTION_STRATEGY!r}. Jalankan: cp .env.research .env")


def jalankan_dokumen(pdf: Path, document_id: str, halaman: set[int], out: Path) -> tuple[list[dict], dict, float]:
    """(chunk v5 sebagai rekaman dump, info gambar, detik transkripsi)."""
    from backend.services import chunk_dump, table_transcription
    from backend.services import preprocessing as pre

    pre.IMAGES_DIR = str(out / "images")
    laporan = pre.ExtractionReport(strategy_requested="hi_res", strategy_used="hi_res")
    elements = pre._extract_hi_res(pdf, report=laporan)
    elements, rekaman = pre._persist_image_elements(elements, document_id, pdf.name)
    t0 = time.monotonic()
    elements, info = table_transcription.proses(elements, pdf, halaman_terpilih=halaman)
    detik = time.monotonic() - t0
    elements = pre._describe_image_elements(elements)
    chunks = pre.chunk_documents(elements, pdf.name, document_id=document_id)

    class _Doc:            # bentuk yang dibaca chunk_dump._record
        def __init__(self, c):
            self.text = c["text"]
            self.metadata = {**c, "page": c.get("page")}
    return [chunk_dump._record(_Doc(c)) for c in chunks], info, detik


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunks", type=Path, required=True, help="chunks.jsonl dump v4")
    ap.add_argument("--pdf-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pilih", default=",".join(SAMPEL_BAWAAN))
    a = ap.parse_args()

    from backend import config
    from backend.services import table_transcription
    periksa_konfigurasi(config)
    from backend.services.vision_io import batas_model
    print(f"batas model: {batas_model()}")      # gagal di sini, bukan setelah hi_res berjam-jam
    a.out.mkdir(parents=True, exist_ok=True)
    v4 = baca_jsonl(a.chunks)

    target = []
    for p in (x.strip() for x in a.pilih.split(",") if x.strip()):
        r = pilih(v4, p)
        print(f"{'OK ' if r else 'TAK ADA'} {p} -> {r['chunk_id'] if r else '-'}")
        if r:
            target.append((p, r))

    per_dok: dict[str, set[int]] = {}
    for _, r in target:
        per_dok.setdefault(r["document_id"], set()).update(halaman_uji(r))

    table_transcription.reset_stats()
    hasil_dok, waktu = {}, {}
    for doc_id, halaman in sorted(per_dok.items()):
        contoh = next(r for _, r in target if r["document_id"] == doc_id)
        pdf = a.pdf_dir / contoh["file_name"]
        print(f"\n== {doc_id} halaman {sorted(halaman)}")
        hasil_dok[doc_id] = jalankan_dokumen(pdf, doc_id, halaman, a.out)
        waktu[doc_id] = round(hasil_dok[doc_id][2], 1)
        print(f"   transkripsi+klasifikasi {waktu[doc_id]} s")

    ringkas = []
    for no, (p, r) in enumerate(target, 1):
        chunks, info, _ = hasil_dok[r["document_id"]]
        dibuang = {i for i, x in info.items() if x.get("perlakuan") == "buang"}
        v5, cara = pasangkan(r, chunks, dibuang)
        info_g = info.get(r.get("image_id") or "")
        (a.out / f"{no:02d}_{r['chunk_id']}.md").write_text(
            md_berdampingan(no, p, r, v5, cara, info_g), encoding="utf-8")
        ringkas.append(baris_ringkas(no, r, v5, cara, info_g))

    with (a.out / "ringkasan.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ringkas[0]))
        w.writeheader()
        w.writerows(ringkas)
    prov = table_transcription.provenance()
    (a.out / "ringkasan.json").write_text(json.dumps(
        {"sampel": ringkas, "detik_per_dokumen": waktu, "provenance": prov},
        ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for x in ringkas:
        print(f"{x['no']:2d} {x['pasangan']:9s} {str(x['jenis_v4']):17s}->{str(x['jenis_v5']):17s} "
              f"{x['sumber']:21s} {x['perlakuan_gambar']:12s} {x['chunk_v4']}  {x['peringatan']}")
    print(f"\nhitungan: {prov['hitungan'] if prov else {}}\nberkas: {a.out}")


if __name__ == "__main__":
    main()
