"""Gerbang berkas keputusan sebelum indexing.

Setiap kondisi di bawah menghasilkan index yang IDENTIK dengan fitur mati bila
dibiarkan lolos — berjam-jam indexing yang tampak berhasil tanpa perbaikan.
Karena itu semuanya harus gagal keras.
"""

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.services.table_continuation import periksa_berkas_keputusan  # noqa: E402


def tulis(tmp_path, isi) -> Path:
    p = tmp_path / "k.json"
    p.write_text(isi if isinstance(isi, str) else json.dumps(isi), encoding="utf-8")
    return p


BERSIDIK = {"keputusan": "terima", "html_sha_a": "a" * 16, "html_sha_b": "b" * 16}


@pytest.mark.unit
def test_berkas_tidak_ada_gagal_keras(tmp_path):
    with pytest.raises(ValueError, match="tidak ada"):
        periksa_berkas_keputusan(tmp_path / "salah_path.json")


@pytest.mark.unit
@pytest.mark.parametrize("isi", ["{bukan json", "[]", '{"pasangan": "bukan dict"}', "null"])
def test_berkas_rusak_gagal_keras(tmp_path, isi):
    with pytest.raises(ValueError):
        periksa_berkas_keputusan(tulis(tmp_path, isi))


@pytest.mark.unit
def test_nol_diterima_gagal_keras(tmp_path):
    p = tulis(tmp_path, {"pasangan": {"a__b": {"keputusan": "tolak"}, "c__d": {"keputusan": ""}}})
    with pytest.raises(ValueError, match="NOL yang diterima"):
        periksa_berkas_keputusan(p)


@pytest.mark.unit
def test_berkas_lama_tanpa_sidik_gagal_keras(tmp_path):
    """Berkas berkunci v2 yang belum dimigrasi — nilai bawaan config."""
    p = tulis(tmp_path, {"pasangan": {"a__b": {"keputusan": "terima"}, "c__d": BERSIDIK}})
    with pytest.raises(ValueError, match="TANPA sidik"):
        periksa_berkas_keputusan(p)


@pytest.mark.unit
def test_berkas_termigrasi_lolos_dan_meringkas(tmp_path):
    p = tulis(tmp_path, {"pasangan": {
        "a__b": BERSIDIK, "c__d": {"keputusan": "tolak"}, "e__f": {"keputusan": ""},
        "g__h": {"keputusan": "terima", "_migrasi": {"html_sha_a": "x", "html_sha_b": "y"}},
    }, "tidak_terpetakan": {}})
    r = periksa_berkas_keputusan(p)
    assert (r["n_diterima"], r["n_ditolak"], r["n_belum_ditinjau"]) == (2, 1, 1)
    assert len(r["sha256"]) == 64 and r["path"] == str(p)


@pytest.mark.unit
def test_gerbang_terpasang_di_index_documents():
    src = (ROOT / "backend" / "services" / "indexing.py").read_text(encoding="utf-8")
    badan = src[src.index("def index_documents("):]
    assert re.search(r"^\s+_check_table_continuation\(\)", badan, re.M)


@pytest.mark.unit
def test_flag_tabel_dibekukan_dan_sejalan_dengan_env_research():
    from backend.config import RESEARCH_EXPECTED_FLAGS as F
    assert F["INDEX_TABLE_CONTINUATION"] is True
    assert F["TABLE_HEADER_MAX_CELL_CHARS"] == 80
    env = (ROOT / ".env.research").read_text(encoding="utf-8")
    assert re.search(r"^INDEX_TABLE_CONTINUATION=true$", env, re.M)
    assert re.search(r"^TABLE_HEADER_MAX_CELL_CHARS=80$", env, re.M)
