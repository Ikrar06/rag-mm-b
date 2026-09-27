"""Migrasi gold v4 -> v5: jangkar sidik raw_html, image_id, dan relevan_setara."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.gold_migrasi import (  # noqa: E402
    STATUS_AMBIGU, STATUS_HILANG, STATUS_IDENTIK, STATUS_ISI_BERUBAH, bangun_indeks,
    petakan_chunk,
)
from lib.relevan_setara import (  # noqa: E402
    kelompok_setara, luas_di_perluasan, terapkan_setara,
)
from backend.services.table_continuation import html_sha  # noqa: E402

HTML_A = "<table><tr><td>UKT</td><td>500,000</td></tr></table>"

# v5: tabel UKT (teks transkripsi, raw_html sama), cap p5 dibuang sehingga chunk
# teks sesudahnya bergeser dari c03 ke c02, gambar sop12 jadi Table.
V5 = [
    {"chunk_id": "ukt-tahun-2025_p5_c01", "document_id": "ukt-tahun-2025", "page_number": 5,
     "element_type": "Table", "text_sha": "baru1", "text_as_html": HTML_A,
     "text_content": "| NO | UKT PER SEMESTER KELOMPOK I |", "bbox": [0.03, 0.28, 0.95, 0.75],
     "render_bbox": [0.03, 0.28, 0.97, 0.75]},
    {"chunk_id": "ukt-tahun-2025_p5_c02", "document_id": "ukt-tahun-2025", "page_number": 5,
     "element_type": "NarrativeText", "text_sha": "teks_c03", "text_content": "Ditetapkan",
     "bbox": [0.74, 0.5, 0.89, 0.54]},
    {"chunk_id": "sop12_p10_c02", "document_id": "sop12", "page_number": 10,
     "element_type": "Table", "text_sha": "baru2", "image_id": "sop12_p10_img00",
     "text_content": "| Rentang | Huruf |", "table_origin": "image"},
    {"chunk_id": "sb_p19_c00", "document_id": "sb", "page_number": 19, "element_type": "Table",
     "text_sha": "baru3", "text_as_html": "<table/>", "bbox": [0.10, 0.04, 0.88, 0.69],
     "render_bbox": [0.10, 0.04, 0.95, 0.69], "text_content": "| URAIAN | BESARAN |"},
    {"chunk_id": "sb_p19_c12", "document_id": "sb", "page_number": 19,
     "element_type": "Title+UncategorizedText", "text_sha": "kolom", "bbox": [0.876, 0.05, 0.948, 0.88],
     "text_content": "# BESARAN 4 350.000 75.000"},
    {"chunk_id": "sb_p19_c01", "document_id": "sb", "page_number": 19, "element_type": "ListItem",
     "text_sha": "jauh", "bbox": [0.17, 0.70, 0.82, 0.71], "text_content": "Koordinator"},
]


@pytest.fixture
def indeks():
    return bangun_indeks(V5)


def test_tabel_dikenali_lewat_sidik_raw_html(indeks):
    h = petakan_chunk("ukt-tahun-2025_p5_c01", "sha_ocr_lama", indeks, sidik_lama=html_sha(HTML_A))
    assert (h.baru, h.status, h.jangkar) == ("ukt-tahun-2025_p5_c01", STATUS_ISI_BERUBAH, "sidik")


def test_sidik_hanya_di_dokumen_yang_sama(indeks):
    h = petakan_chunk("lain_p5_c01", "x", indeks, sidik_lama=html_sha(HTML_A))
    assert h.status != STATUS_ISI_BERUBAH or h.jangkar != "sidik"


def test_gambar_jadi_tabel_lewat_image_id(indeks):
    h = petakan_chunk("sop12_p10_c02", "narasi_lama", indeks, image_lama="sop12_p10_img00")
    assert (h.baru, h.status, h.jangkar) == ("sop12_p10_c02", STATUS_ISI_BERUBAH, "image_id")


def test_cap_dibuang_hilang_bukan_dipetakan_ke_chunk_teks(indeks):
    """ukt_p5_c02 di v4 = cap; di v5 id itu milik chunk teks yang bergeser."""
    h = petakan_chunk("ukt-tahun-2025_p5_c02", "narasi_cap", indeks,
                      image_lama="ukt-tahun-2025_p5_img00")
    assert h.status == STATUS_HILANG and h.baru is None and "cap" in h.alasan


def test_teks_bergeser_tetap_lewat_text_sha(indeks):
    h = petakan_chunk("ukt-tahun-2025_p5_c03", "teks_c03", indeks)
    assert h.baru == "ukt-tahun-2025_p5_c02"


def test_sidik_ganda_sedokumen_ambigu():
    v5 = [dict(V5[0]), {**V5[0], "chunk_id": "ukt-tahun-2025_p6_c00"}]
    h = petakan_chunk("ukt-tahun-2025_p5_c01", "x", bangun_indeks(v5), sidik_lama=html_sha(HTML_A))
    assert h.status == STATUS_AMBIGU


def test_relevan_setara_kolom_besaran():
    assert luas_di_perluasan([0.876, 0.05, 0.948, 0.88], [0.10, 0.04, 0.88, 0.69],
                             [0.10, 0.04, 0.95, 0.69]) > 0
    setara = kelompok_setara(["sb_p19_c12", "sb_p19_c01", "ukt-tahun-2025_p5_c02", "tak_ada"], V5)
    assert setara == {"sb_p19_c12": "sb_p19_c00"}          # c01 di luar daerah perluasan
    item = {"query_id": "q", "relevant_text_chunks": ["sb_p19_c12", "sb_p19_c01"]}
    assert terapkan_setara(item, setara)["relevan_setara"] == [["sb_p19_c12", "sb_p19_c00"]]
    assert "relevan_setara" not in terapkan_setara({"relevant_text_chunks": ["x"]}, setara)
    assert "relevan_setara" not in item                     # tidak memutasi


def test_cli_v4_ke_v5(tmp_path, monkeypatch, capsys):
    import migrasi_gold
    v4 = [
        {"chunk_id": "ukt-tahun-2025_p5_c01", "text_sha": "ocr1", "text_as_html": HTML_A,
         "text_content": "| OCR |"},
        {"chunk_id": "ukt-tahun-2025_p5_c02", "text_sha": "cap", "image_id": "ukt-tahun-2025_p5_img00",
         "text_content": "[Deskripsi Gambar] cap"},
        {"chunk_id": "sb_p19_c12", "text_sha": "kolom", "text_content": "# BESARAN 4 350.000 75.000"},
    ]
    gold = [
        {"query_id": "q1", "relevant_text_chunks": ["ukt-tahun-2025_p5_c01"], "human_verdict": "ok"},
        {"query_id": "q2", "relevant_text_chunks": ["ukt-tahun-2025_p5_c02"], "human_verdict": "ok"},
        {"query_id": "q3", "relevant_text_chunks": ["sb_p19_c12"], "human_verdict": "ok"},
    ]
    for nama, isi in (("v4.jsonl", v4), ("v5.jsonl", V5), ("gold.jsonl", gold)):
        (tmp_path / nama).write_text("\n".join(json.dumps(r) for r in isi))
    monkeypatch.setattr(sys, "argv", ["x", "--gold", str(tmp_path / "gold.jsonl"),
                                      "--chunks-baru", str(tmp_path / "v5.jsonl"),
                                      "--chunks-lama", str(tmp_path / "v4.jsonl"),
                                      "--out-dir", str(tmp_path / "o")])
    assert migrasi_gold.main() == 1                     # q2 (cap) tidak terpetakan
    lap = json.loads((tmp_path / "o" / "laporan_migrasi.json").read_text())
    assert lap["jangkar_v5_item"] == {"sidik": 1}
    assert lap["relevan_setara_pasangan"] == 1
    mig = [json.loads(l) for l in (tmp_path / "o" / "ground_truth_migrated.jsonl").read_text().splitlines()]
    assert {m["query_id"]: m.get("relevan_setara") for m in mig} == {
        "q1": None, "q3": [["sb_p19_c12", "sb_p19_c00"]]}
    assert STATUS_IDENTIK in lap["per_status"] and lap["per_status"][STATUS_HILANG] == 1
