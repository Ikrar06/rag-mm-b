"""Uji rantai penuh: berkas keputusan -> _chunk_elements -> chunk.

Membuktikan tiga hal yang menentukan sahnya Tahap B:
  1. Tanpa flag, hasilnya IDENTIK dengan sebelum tahap ini.
  2. Dengan flag tapi pasangan tidak disetujui, hasilnya juga identik.
  3. Dengan flag dan pasangan disetujui, header diulang TANPA menggeser chunk_id.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HDR = "| Persyaratan | S1 | S2 |\n| --- | --- | --- |"
TABEL_A = f"{HDR}\n| IPK minimum | 2,75 | 3,00 |"
TABEL_B = "| Masa studi | 14 | 8 |\n| TOEFL | 450 | 500 |"


def _elements():
    """Dua potongan tabel di halaman berurutan, plus teks di antaranya."""
    return [
        {"category": "Title", "text": "Persyaratan Ujian", "page": 4, "metadata": {}},
        {"category": "Table", "text": TABEL_A, "page": 4,
         "metadata": {"raw_html": "<table></table>", "table_format": "markdown",
                      "bbox": [0.1, 0.55, 0.9, 0.94]}},
        {"category": "Table", "text": TABEL_B, "page": 5,
         "metadata": {"raw_html": "<table></table>", "table_format": "markdown",
                      "bbox": [0.1, 0.08, 0.9, 0.35]}},
    ]


def _chunk(monkeypatch, aktif: bool, path_keputusan=None):
    """Jalankan _chunk_elements dengan konfigurasi tertentu."""
    import importlib

    monkeypatch.setenv("INDEX_TABLE_CONTINUATION", "true" if aktif else "false")
    monkeypatch.setenv("INDEX_STRUCTURAL_METADATA", "true")
    monkeypatch.setenv("INDEX_TABLES_AS_OWN_CHUNKS", "true")
    if path_keputusan:
        monkeypatch.setenv("TABLE_CONTINUATION_PATH", str(path_keputusan))

    import backend.config as cfg
    importlib.reload(cfg)
    import backend.services.table_continuation as tc
    importlib.reload(tc)
    tc.reload_keputusan()
    import backend.services.preprocessing as pre
    importlib.reload(pre)

    return pre._chunk_elements(_elements(), "sop.pdf", document_id="sop")


def _tabel(chunks):
    return [c for c in chunks if c.get("element_type") == "Table"]


@pytest.mark.integration
def test_flag_mati_identik_dengan_sebelumnya(monkeypatch):
    t = _tabel(_chunk(monkeypatch, aktif=False))
    assert len(t) == 2
    assert not t[1]["text"].startswith("| Persyaratan |")
    for c in t:
        assert "table_group_id" not in c
        assert "table_header_repeated" not in c


@pytest.mark.integration
def test_flag_hidup_tanpa_berkas_keputusan_juga_identik(monkeypatch, tmp_path):
    """Menyalakan flag tanpa berkas keputusan tidak boleh mengubah apa pun."""
    t = _tabel(_chunk(monkeypatch, aktif=True,
                      path_keputusan=tmp_path / "belum_ada.json"))
    assert len(t) == 2
    # Teks chunk selalu diawali "## {section}", jadi yang diperiksa adalah
    # BARIS HEADER tabel, bukan sekadar kemunculan katanya.
    assert "| Persyaratan | S1 | S2 |" not in t[1]["text"]
    assert t[1].get("table_header_repeated") is None


@pytest.mark.integration
def test_pasangan_tidak_disetujui_tidak_digabung(monkeypatch, tmp_path):
    p = tmp_path / "k.json"
    p.write_text(json.dumps({"pasangan": {
        "sop_p4_c00__sop_p5_c00": {"keputusan": "tolak"},
    }}), encoding="utf-8")
    t = _tabel(_chunk(monkeypatch, aktif=True, path_keputusan=p))
    assert "| Persyaratan | S1 | S2 |" not in t[1]["text"]


@pytest.mark.integration
def test_pasangan_disetujui_header_diulang_chunk_id_tetap(monkeypatch, tmp_path):
    tanpa = _tabel(_chunk(monkeypatch, aktif=False))
    kunci = f"{tanpa[0]['chunk_id']}__{tanpa[1]['chunk_id']}"

    p = tmp_path / "k.json"
    p.write_text(json.dumps({"pasangan": {kunci: {"keputusan": "terima"}}}),
                 encoding="utf-8")
    dengan = _tabel(_chunk(monkeypatch, aktif=True, path_keputusan=p))

    # chunk_id TIDAK bergeser — itu janji utama rancangan ini.
    assert [c["chunk_id"] for c in dengan] == [c["chunk_id"] for c in tanpa]
    assert len(dengan) == len(tanpa)

    # Potongan lanjutan kini membawa header dan berdiri sendiri.
    assert "| Persyaratan | S1 | S2 |" in dengan[1]["text"]
    assert dengan[1]["table_header_repeated"] is True
    assert dengan[1]["table_group_id"] == dengan[0]["chunk_id"]
    assert dengan[1]["table_part"] == 1 and dengan[0]["table_part"] == 0

    # text_sha potongan lanjutan BERUBAH; potongan pertama tidak.
    assert dengan[0]["text_sha"] == tanpa[0]["text_sha"]
    assert dengan[1]["text_sha"] != tanpa[1]["text_sha"]

    # raw_html tidak disentuh.
    assert dengan[1]["raw_html"] == tanpa[1]["raw_html"]


@pytest.mark.integration
def test_potongan_pertama_tidak_diubah(monkeypatch, tmp_path):
    tanpa = _tabel(_chunk(monkeypatch, aktif=False))
    kunci = f"{tanpa[0]['chunk_id']}__{tanpa[1]['chunk_id']}"
    p = tmp_path / "k.json"
    p.write_text(json.dumps({"pasangan": {kunci: {"keputusan": "terima"}}}),
                 encoding="utf-8")
    dengan = _tabel(_chunk(monkeypatch, aktif=True, path_keputusan=p))
    assert dengan[0]["text"] == tanpa[0]["text"]
