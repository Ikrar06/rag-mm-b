"""Uji alat uji sampel transkripsi lewat jalur pipeline (tanpa model, tanpa PDF)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.uji_pipeline import (  # noqa: E402
    baris_ringkas, halaman_uji, md_berdampingan, pasangkan, pilih,
)

V4 = [
    {"chunk_id": "ukt-tahun-2025_p5_c01", "document_id": "ukt-tahun-2025", "page_number": 5,
     "element_type": "Table", "text_content": "| OCR |", "file_name": "ukt.pdf"},
    {"chunk_id": "ukt-tahun-2025_p5_c02", "document_id": "ukt-tahun-2025", "page_number": 5,
     "element_type": "ImageDescription", "image_id": "ukt-tahun-2025_p5_img00",
     "text_content": "[Deskripsi Gambar] tabel biaya", "file_name": "ukt.pdf"},
    {"chunk_id": "ukt-tahun-2025_p5_c03", "document_id": "ukt-tahun-2025", "page_number": 5,
     "element_type": "NarrativeText", "text_content": "Ditetapkan", "file_name": "ukt.pdf"},
    {"chunk_id": "rubrik_p39_c00", "document_id": "rubrik", "page_number": 39,
     "element_type": "Table", "table_part": 2, "table_header_repeated": True,
     "table_group_id": "rubrik_p37_c01", "text_content": "| x |", "file_name": "r.pdf"},
    {"chunk_id": "sop12_p10_c02", "document_id": "sop12", "page_number": 10,
     "element_type": "ImageDescription", "image_id": "sop12_p10_img00",
     "text_content": "[Deskripsi Gambar] tabel", "file_name": "s.pdf"},
]


def test_pilih_pemilih():
    assert pilih(V4, "ukt-tahun-2025:5:Table")["chunk_id"] == "ukt-tahun-2025_p5_c01"
    assert pilih(V4, "ukt:5:ImageDescription")["chunk_id"] == "ukt-tahun-2025_p5_c02"
    assert pilih(V4, "lanjutan")["chunk_id"] == "rubrik_p39_c00"
    assert pilih(V4, "sop12_p10_c02")["image_id"] == "sop12_p10_img00"
    assert pilih(V4, "tidak:1:Table") is None and pilih(V4, "tidak_ada") is None


def test_halaman_uji_mencakup_kepala_rantai():
    assert halaman_uji(V4[3]) == {37, 38, 39}
    assert halaman_uji(V4[0]) == {5}


def test_pasangkan():
    v5 = [
        {"chunk_id": "ukt-tahun-2025_p5_c01", "page_number": 5, "element_type": "Table"},
        {"chunk_id": "ukt-tahun-2025_p5_c02", "page_number": 5, "element_type": "NarrativeText"},
        {"chunk_id": "sop12_p10_c02", "page_number": 10, "element_type": "Table",
         "image_id": "sop12_p10_img00", "table_origin": "image"},
    ]
    assert pasangkan(V4[0], v5, set())[1] == "chunk_id"
    # cap dibuang -> chunk teks bergeser ke c02; pasangan lewat urutan jenis
    assert pasangkan(V4[1], v5, {"ukt-tahun-2025_p5_img00"}) == (None, "dibuang")
    assert pasangkan(V4[2], v5, set()) == (v5[1], "urutan")
    assert pasangkan(V4[4], v5, set()) == (v5[2], "image_id")
    assert pasangkan(V4[3], v5, set()) == (None, "tidak_ada")


def test_md_dan_ringkas():
    v5 = {"chunk_id": "a", "element_type": "Table", "table_source": "vision_transcription",
          "transkripsi_peringatan": ["rujukan=lapisan_ocr", "angka_tak_ditemukan:12"],
          "text_content": "| v5 |"}
    md = md_berdampingan(1, "x", V4[0], v5, "chunk_id", {"perlakuan": "transkripsi"})
    assert "## v4\n\n| OCR |" in md and "## v5\n\n| v5 |" in md and "vision_transcription" in md
    assert "(tidak ada — dibuang)" in md_berdampingan(2, "y", V4[1], None, "dibuang", None)
    r = baris_ringkas(1, V4[0], v5, "chunk_id", None)
    assert r["peringatan"] == "rujukan=lapisan_ocr; angka_tak_ditemukan:12" and r["karakter_v5"] == 6


def test_main_dengan_pipeline_palsu(tmp_path, monkeypatch, capsys):
    import uji_transkripsi_pipeline as uji
    from backend import config
    from backend.services import table_transcription
    for n in ("INDEX_TABLE_TRANSCRIPTION", "INDEX_IMAGE_TABLE_TRANSCRIPTION",
              "INDEX_STRUCTURAL_METADATA", "INDEX_PERSIST_IMAGES"):
        monkeypatch.setattr(config, n, True)
    monkeypatch.setattr(config, "PDF_EXTRACTION_STRATEGY", "hi_res")
    monkeypatch.setattr(table_transcription, "provenance", lambda: {"hitungan": {"x": 1}})
    from backend.services import vision_io
    monkeypatch.setattr(vision_io, "batas_model", lambda: {"sisi_min": 32})
    dipanggil = []

    def palsu(pdf, doc_id, halaman, out):
        dipanggil.append((pdf.name, doc_id, sorted(halaman)))
        return ([{"chunk_id": "ukt-tahun-2025_p5_c01", "page_number": 5, "element_type": "Table",
                  "table_source": "vision_transcription", "text_content": "| v5 |",
                  "transkripsi_peringatan": ["kolom_tidak_konsisten:14/13"]},
                 {"chunk_id": "ukt-tahun-2025_p5_c04", "page_number": 5, "element_type": "Table",
                  "table_source": "ocr_fallback", "text_content": "| ocr |",
                  "transkripsi_peringatan": ["gagal:tak_terurai"]},
                 {"chunk_id": "ukt-tahun-2025_p6_c00", "page_number": 6, "element_type": "Table",
                  "table_source": "ocr_fallback"}],
                {"ukt-tahun-2025_p5_img00": {"perlakuan": "buang"}}, 12.5)

    monkeypatch.setattr(uji, "jalankan_dokumen", palsu)
    c = tmp_path / "c.jsonl"
    c.write_text("\n".join(json.dumps(r) for r in V4))
    monkeypatch.setattr(sys, "argv", ["x", "--chunks", str(c), "--pdf-dir", str(tmp_path),
                                      "--out", str(tmp_path / "o"),
                                      "--pilih", "ukt-tahun-2025:5:Table,ukt:5:ImageDescription,tak:1:Table"])
    uji.main()
    assert dipanggil == [("ukt.pdf", "ukt-tahun-2025", [5])]
    ring = json.loads((tmp_path / "o" / "ringkasan.json").read_text())
    assert [s["pasangan"] for s in ring["sampel"]] == ["chunk_id", "dibuang"]
    assert ring["detik_per_dokumen"] == {"ukt-tahun-2025": 12.5}
    assert ring["tabel_diproses"] == 2          # halaman 6 tidak diproses
    assert ring["tabel_fallback"] == [{"chunk_id": "ukt-tahun-2025_p5_c04",
                                       "alasan": ["gagal:tak_terurai"]}]
    assert ring["kolom_tidak_konsisten"] == [{"chunk_id": "ukt-tahun-2025_p5_c01",
                                              "tanda": "kolom_tidak_konsisten:14/13"}]
    assert (tmp_path / "o" / "01_ukt-tahun-2025_p5_c01.md").is_file()
    assert "TAK ADA tak:1:Table" in capsys.readouterr().out


def test_konfigurasi_bukan_riset_ditolak(monkeypatch):
    import uji_transkripsi_pipeline as uji
    from backend import config
    monkeypatch.setattr(config, "INDEX_TABLE_TRANSCRIPTION", False)
    with pytest.raises(SystemExit, match="cp .env.research"):
        uji.periksa_konfigurasi(config)
