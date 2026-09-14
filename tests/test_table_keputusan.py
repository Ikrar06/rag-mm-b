"""Uji berkas keputusan terkurasi dan pengulangan baris header."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services import table_continuation as tc  # noqa: E402

TABEL = "| Persyaratan | S1 | S2 |\n| --- | --- | --- |\n| IPK | 2,75 | 3,00 |"


# ─── baris_header_markdown ───────────────────────────────────────────────────

@pytest.mark.unit
def test_header_diambil_dari_dua_baris_pertama():
    assert tc.baris_header_markdown(TABEL) == (
        "| Persyaratan | S1 | S2 |\n| --- | --- | --- |"
    )


@pytest.mark.unit
def test_prefiks_section_dilewati():
    """Teks chunk didahului '## {section}', header tetap ditemukan."""
    teks = f"## Persyaratan Ujian\n\n{TABEL}"
    assert tc.baris_header_markdown(teks).startswith("| Persyaratan |")


@pytest.mark.unit
@pytest.mark.parametrize("teks", [
    "", None, "tidak ada tabel di sini",
    "| hanya satu baris |",
    "| A | B |\n| bukan pemisah | tetap data |",
])
def test_tanpa_bentuk_tabel_mengembalikan_kosong(teks):
    assert tc.baris_header_markdown(teks) == ""


@pytest.mark.unit
def test_pemisah_dengan_penjajaran_tetap_dikenali():
    teks = "| A | B |\n|:---|---:|\n| 1 | 2 |"
    assert tc.baris_header_markdown(teks) != ""


# ─── ulangi_header memakai hasil di atas ─────────────────────────────────────

@pytest.mark.unit
def test_header_disisipkan_ke_potongan_lanjutan():
    header = tc.baris_header_markdown(TABEL)
    lanjutan = "| Masa studi | 14 | 8 |"
    hasil = tc.ulangi_header(header, lanjutan)
    assert hasil.splitlines()[0] == "| Persyaratan | S1 | S2 |"
    assert hasil.splitlines()[-1] == lanjutan
    # Potongan lanjutan kini tabel Markdown yang sah dan berdiri sendiri.
    assert tc.baris_header_markdown(hasil) == header


# ─── muat_keputusan ──────────────────────────────────────────────────────────

def _tulis(tmp_path, pasangan) -> Path:
    p = tmp_path / "table_continuation.json"
    p.write_text(json.dumps({"_meta": {"rule_version": 1}, "pasangan": pasangan}),
                 encoding="utf-8")
    return p


@pytest.mark.unit
def test_hanya_yang_diterima_yang_dimuat(tmp_path):
    tc.reload_keputusan()
    p = _tulis(tmp_path, {
        "a__b": {"keputusan": "terima"},
        "c__d": {"keputusan": "tolak", "alasan": "header dari OCR"},
        "e__f": {"keputusan": ""},          # belum ditinjau
        "g__h": {},                          # tanpa kolom keputusan
    })
    assert tc.muat_keputusan(p) == frozenset({"a__b"})


@pytest.mark.unit
def test_keputusan_tidak_peka_huruf_besar_dan_spasi(tmp_path):
    tc.reload_keputusan()
    p = _tulis(tmp_path, {"a__b": {"keputusan": "  TERIMA "}})
    assert tc.muat_keputusan(p) == frozenset({"a__b"})


@pytest.mark.unit
def test_berkas_tidak_ada_berarti_kosong_bukan_galat(tmp_path):
    """Menyalakan flag tanpa berkas keputusan = perilaku identik flag mati."""
    tc.reload_keputusan()
    assert tc.muat_keputusan(tmp_path / "belum_ada.json") == frozenset()


@pytest.mark.unit
@pytest.mark.parametrize("isi", ["{bukan json", "[]", '{"pasangan": "bukan dict"}', "null"])
def test_berkas_rusak_diperlakukan_kosong(tmp_path, isi):
    """Jangan menggabung berdasarkan berkas yang tidak dapat dibaca."""
    tc.reload_keputusan()
    p = tmp_path / "rusak.json"
    p.write_text(isi, encoding="utf-8")
    assert tc.muat_keputusan(p) == frozenset()


@pytest.mark.unit
def test_entri_bukan_dict_dilewati(tmp_path):
    tc.reload_keputusan()
    p = _tulis(tmp_path, {"a__b": "terima", "c__d": {"keputusan": "terima"}})
    assert tc.muat_keputusan(p) == frozenset({"c__d"})


@pytest.mark.unit
def test_hasil_di_cache_sampai_reload(tmp_path):
    tc.reload_keputusan()
    p = _tulis(tmp_path, {"a__b": {"keputusan": "terima"}})
    assert tc.muat_keputusan(p) == frozenset({"a__b"})
    _tulis(tmp_path, {"x__y": {"keputusan": "terima"}})
    assert tc.muat_keputusan(p) == frozenset({"a__b"})   # masih cache
    tc.reload_keputusan()
    assert tc.muat_keputusan(p) == frozenset({"x__y"})


@pytest.mark.unit
def test_kunci_pasangan_stabil_dan_berurutan():
    a, b = "dok_p4_c00", "dok_p5_c01"
    assert tc.kunci_pasangan(a, b) == f"{a}__{b}"
    assert tc.kunci_pasangan(a, b) != tc.kunci_pasangan(b, a)
