"""Tag modality (pipeline, dump, payload index) dan paket dosen di build_eval_package."""

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from backend.services.modality import modality_dari  # noqa: E402
from lib.paket_dosen import (  # noqa: E402
    aman_spreadsheet, baris_semua_chunk, format_ukuran, judul_dokumen, path_gambar,
    tulis_semua_chunk,
)


@pytest.mark.parametrize("et,m", [
    ("ImageDescription", "image"), ("Table", "table"), ("NarrativeText", "text"),
    ("NarrativeText+Table", "text"),      # tabel kecil di buffer teks: TIDAK table
    ("Title+UncategorizedText", "text"), (None, "text"), ("", "text"),
])
def test_modality_pencocokan_tepat(et, m):
    assert modality_dari(et) == m


def test_modality_masuk_daftar_eksklusi_embedding():
    from backend.config import EMBED_EXCLUDED_METADATA_KEYS
    sys.path.insert(0, str(ROOT / "tests"))
    from test_eksklusi_embedding import kunci_struktural
    assert "modality" in kunci_struktural()            # masuk payload Qdrant
    assert "modality" in EMBED_EXCLUDED_METADATA_KEYS   # tapi tidak di-embed


def test_chunk_dibentuk_dengan_modality(monkeypatch):
    import importlib
    monkeypatch.setenv("INDEX_STRUCTURAL_METADATA", "true")
    monkeypatch.setenv("INDEX_TABLES_AS_OWN_CHUNKS", "true")
    monkeypatch.setenv("INDEX_MIN_CHUNK_TOKENS", "0")
    import backend.config as cfg
    importlib.reload(cfg)
    import backend.services.preprocessing as pre
    importlib.reload(pre)
    els = [{"category": "NarrativeText", "text": "Paragraf.", "page": 1, "metadata": {}},
           {"category": "Table", "text": "| a | b |\n|---|---|\n| 1 | 2 |", "page": 1,
            "metadata": {"raw_html": "<table/>", "table_format": "markdown", "bbox": [0, 0, 1, 1]}},
           {"category": "ImageDescription", "text": "Foto gedung.", "page": 2,
            "metadata": {"bbox": [0, 0, 1, 1], "image_id": "d_p2_img00"}}]
    got = {c["element_type"]: c["modality"] for c in pre._chunk_elements(els, "d.pdf", "d")}
    assert got == {"NarrativeText": "text", "Table": "table", "ImageDescription": "image"}


def test_dump_record_dan_csv_membawa_modality():
    pytest.importorskip("llama_index.core")
    from backend.services import chunk_dump

    class Doc:
        text = "isi"
        metadata = {"chunk_id": "d_p1_c00", "element_type": "Table", "page": 1}

    rec = chunk_dump._record(Doc())
    assert rec["modality"] == "table"                     # diturunkan bila tidak ada
    assert chunk_dump._row(rec)["modality"] == "table" and "modality" in chunk_dump._CSV_COLUMNS
    Doc.metadata = {**Doc.metadata, "modality": "image"}
    assert chunk_dump._record(Doc())["modality"] == "image"   # payload menang


def test_pasang_payload_index_idempoten_dan_tahan_galat(monkeypatch):
    sys.path.insert(0, str(ROOT / "tests"))
    import test_tahap_t_kait  # noqa: F401  (stub qdrant bila paket aslinya tak ada)
    from backend.services import indexing

    class Klien:
        def __init__(self, gagal=False):
            self.panggilan, self.gagal = [], gagal

        def create_payload_index(self, **kw):
            self.panggilan.append((kw["collection_name"], kw["field_name"]))
            if self.gagal:
                raise RuntimeError("qdrant mati")

    k = Klien()
    assert indexing._pasang_payload_index(k, "rag_mm_b_varian_b_v5") == ["modality"]
    assert k.panggilan == [("rag_mm_b_varian_b_v5", "modality")]
    assert indexing._pasang_payload_index(Klien(gagal=True), "x") == []


def test_fungsi_paket_dosen():
    assert judul_dokumen("3.-UKT-TAHUN-2025.pdf") == "3.-UKT-TAHUN-2025"
    fp = {"ukt-tahun-2025_p5_img00": "images/ukt-tahun-2025/p5_img00.jpg"}
    assert path_gambar("ukt-tahun-2025_p5_img00", fp) == "gambar/ukt-tahun-2025/p5_img00.jpg"
    assert path_gambar(None, fp) == "" and path_gambar("lain", fp) == ""
    assert aman_spreadsheet("- Database konsumen") == "'- Database konsumen"
    assert aman_spreadsheet("=SUM(A1)") == "'=SUM(A1)" and aman_spreadsheet("| a |") == "| a |"
    assert format_ukuran(512) == "512 B" and format_ukuran(3 * 1024 ** 3) == "3.0 GB"


CHUNKS = [
    {"chunk_id": "ukt-tahun-2025_p5_c01", "document_id": "ukt-tahun-2025", "page_number": 5,
     "element_type": "Table", "table_source": "vision_transcription", "file_name": "3.-UKT-TAHUN-2025.pdf",
     "text_content": "| NO | UKT |\n|---|---|\n| 1 | 500,000 |", "text_sha": "a"},
    {"chunk_id": "ukt-tahun-2025_p1_c00", "document_id": "ukt-tahun-2025", "page_number": 1,
     "element_type": "ImageDescription", "image_id": "ukt-tahun-2025_p1_img00",
     "file_name": "3.-UKT-TAHUN-2025.pdf", "text_content": "[Deskripsi Gambar] Logo, \"Unhas\"",
     "text_sha": "b"},
    {"chunk_id": "jadwal_p50_c03", "document_id": "jadwal", "page_number": 50,
     "element_type": "NarrativeText", "file_name": "JADWAL.pdf",
     "text_content": "- Database konsumen\n- Daftar nama", "text_sha": "c"},
]
IMAGES = [{"image_id": "ukt-tahun-2025_p1_img00", "document_id": "ukt-tahun-2025", "page_number": 1,
           "file_path": "images/ukt-tahun-2025/p1_img00.png", "narrative_summary": "Logo"}]


def test_semua_chunk_csv_terbaca_spreadsheet(tmp_path):
    ringkas = tulis_semua_chunk(tmp_path / "s.csv", baris_semua_chunk(CHUNKS, IMAGES))
    mentah = (tmp_path / "s.csv").read_bytes()
    assert mentah.startswith(b"\xef\xbb\xbf")                       # BOM untuk Excel
    with open(tmp_path / "s.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["chunk_id"] for r in rows] == ["jadwal_p50_c03", "ukt-tahun-2025_p1_c00",
                                             "ukt-tahun-2025_p5_c01"]
    assert rows[0]["text_content"] == "'- Database konsumen\n- Daftar nama"   # newline utuh
    assert rows[1]["text_content"] == '[Deskripsi Gambar] Logo, "Unhas"'       # koma & kutip
    assert rows[1]["path_gambar"] == "gambar/ukt-tahun-2025/p1_img00.png"
    assert rows[2]["modality"] == "table" and rows[2]["judul_dokumen"] == "3.-UKT-TAHUN-2025"
    assert ringkas == {"baris": 3, "sel_melebihi_batas_excel": 0, "sel_diberi_apostrof": 1,
                       "per_modality": {"text": 1, "table": 1, "image": 1}}


def test_build_eval_package_e2e(tmp_path, monkeypatch, capsys):
    import build_eval_package
    dump, root = tmp_path / "dump", tmp_path / "data"
    dump.mkdir()
    (root / "images/ukt-tahun-2025").mkdir(parents=True)
    (root / "images/ukt-tahun-2025/p1_img00.png").write_bytes(b"\x89PNG" + b"0" * 100)
    (dump / "chunks.jsonl").write_text("\n".join(json.dumps(c) for c in CHUNKS))   # tanpa modality
    (dump / "images.jsonl").write_text("\n".join(json.dumps(i) for i in IMAGES))
    (dump / "run_manifest.json").write_text(json.dumps({"provenance": {"qdrant_collection": "v5"}}))
    out = tmp_path / "paket"
    monkeypatch.setattr(sys, "argv", ["x", "--dump", str(dump), "--images-root", str(root),
                                      "--out", str(out)])
    build_eval_package.main()
    keluar = capsys.readouterr().out
    assert "semua_chunk.csv 3 baris" in keluar and "folder gambar" in keluar
    assert (out / "gambar/ukt-tahun-2025/p1_img00.png").is_file()      # path_gambar cocok
    readme = (out / "README_paket.md").read_text()
    for frasa in ("vision_transcription", "ocr_fallback", "kolom_tidak_konsisten",
                  "text_as_html", "`table`", "Google Drive"):
        assert frasa in readme
    per_dok = [json.loads(l) for l in (out / "chunks_jsonl/ukt-tahun-2025.jsonl").read_text().splitlines()]
    assert {c["modality"] for c in per_dok} == {"table", "image"}
