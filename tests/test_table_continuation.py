"""Uji modul penaut tabel lintas halaman."""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services.table_continuation import (  # noqa: E402
    ENV_TABLE_AS_CELLS, AreaTeks, area_teks_halaman, ulangi_header,
)


# ─── area teks ───────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_tanpa_penanda_area_teks_seluruh_halaman():
    """Perilaku lama: dokumen tanpa kop diukur terhadap halaman penuh."""
    a = area_teks_halaman([])
    assert a == AreaTeks(0.0, 1.0)


@pytest.mark.unit
def test_header_mendorong_batas_atas_ke_bawah():
    a = area_teks_halaman([[0.1, 0.02, 0.9, 0.12]])
    assert a.atas == 0.12 and a.bawah == 1.0


@pytest.mark.unit
def test_footer_mendorong_batas_bawah_ke_atas():
    a = area_teks_halaman([[0.1, 0.93, 0.9, 0.97]])
    assert a.atas == 0.0 and a.bawah == 0.93


@pytest.mark.unit
def test_header_dan_footer_bersamaan():
    a = area_teks_halaman([[0.1, 0.02, 0.9, 0.14], [0.4, 0.94, 0.6, 0.98]])
    assert a.atas == 0.14 and a.bawah == 0.94


@pytest.mark.unit
def test_penanda_raksasa_tidak_melahap_seluruh_halaman():
    """Deteksi layout kadang salah menandai blok besar sebagai Header.
    Batasnya tidak boleh melewati tengah halaman."""
    a = area_teks_halaman([[0.0, 0.0, 1.0, 0.95]])
    assert a.atas == 0.5


@pytest.mark.unit
@pytest.mark.parametrize("b", [None, [], [0.1, 0.2], [0.1, 0.2, 0.3, 0.4, 0.5]])
def test_bbox_tidak_sah_dilewati(b):
    assert area_teks_halaman([b]) == AreaTeks(0.0, 1.0)


@pytest.mark.unit
def test_di_dasar_dan_di_puncak_mengikuti_area_teks():
    """Tabel yang sama dinilai berbeda tergantung tinggi kop halaman."""
    tabel_bawah = [0.1, 0.60, 0.9, 0.88]
    tabel_atas = [0.1, 0.16, 0.9, 0.40]

    penuh = area_teks_halaman([])
    assert not penuh.di_dasar(tabel_bawah)      # 0.88 vs batas 1.0
    assert not penuh.di_puncak(tabel_atas)      # 0.16 vs batas 0.0

    berkop = area_teks_halaman([[0.1, 0.02, 0.9, 0.14], [0.4, 0.90, 0.6, 0.97]])
    assert berkop.di_dasar(tabel_bawah)         # 0.88 vs batas 0.90
    assert berkop.di_puncak(tabel_atas)         # 0.16 vs batas 0.14


@pytest.mark.unit
@pytest.mark.parametrize("b", [None, [], [0.1, 0.2, 0.3]])
def test_di_dasar_aman_untuk_bbox_tidak_sah(b):
    a = area_teks_halaman([])
    assert not a.di_dasar(b) and not a.di_puncak(b)


@pytest.mark.unit
def test_area_teks_immutable():
    with pytest.raises(Exception):
        area_teks_halaman([]).atas = 0.5


# ─── ulangi_header ───────────────────────────────────────────────────────────

@pytest.mark.unit
def test_header_disisipkan_di_depan_potongan():
    hasil = ulangi_header("| Persyaratan | S1 | S2 |\n| --- | --- | --- |",
                          "| IPK | 2,75 | 3,00 |")
    assert hasil.startswith("| Persyaratan | S1 | S2 |")
    assert hasil.endswith("| IPK | 2,75 | 3,00 |")


@pytest.mark.unit
def test_header_tidak_digandakan_bila_sudah_ada():
    header = "| Persyaratan | S1 |"
    isi = f"{header}\n| IPK | 2,75 |"
    assert ulangi_header(header, isi) == isi


@pytest.mark.unit
@pytest.mark.parametrize("header", ["", "   ", None])
def test_header_kosong_mengembalikan_potongan_apa_adanya(header):
    assert ulangi_header(header, "| IPK | 2,75 |") == "| IPK | 2,75 |"


@pytest.mark.unit
@pytest.mark.parametrize("isi", ["", "   ", None])
def test_potongan_kosong_dikembalikan_apa_adanya(isi):
    assert ulangi_header("| A | B |", isi) == isi


# ─── siapkan_ekstraksi ───────────────────────────────────────────────────────

def _muat_ulang(monkeypatch, aktif: bool):
    """Impor ulang modul dengan INDEX_TABLE_CONTINUATION tertentu."""
    monkeypatch.setenv("INDEX_TABLE_CONTINUATION", "true" if aktif else "false")
    import backend.config as cfg
    importlib.reload(cfg)
    import backend.services.table_continuation as tc
    return importlib.reload(tc)


@pytest.mark.unit
def test_env_tidak_disentuh_saat_flag_mati(monkeypatch):
    monkeypatch.delenv(ENV_TABLE_AS_CELLS, raising=False)
    tc = _muat_ulang(monkeypatch, aktif=False)
    assert tc.siapkan_ekstraksi() == {}
    assert ENV_TABLE_AS_CELLS not in __import__("os").environ


@pytest.mark.unit
def test_env_disetel_saat_flag_hidup(monkeypatch):
    monkeypatch.delenv(ENV_TABLE_AS_CELLS, raising=False)
    tc = _muat_ulang(monkeypatch, aktif=True)
    assert tc.siapkan_ekstraksi() == {ENV_TABLE_AS_CELLS: "true"}


@pytest.mark.unit
def test_nilai_yang_sudah_ada_tidak_ditimpa(monkeypatch):
    """Peneliti yang sengaja mematikannya tetap menang, dan manifest merekam
    nilai EFEKTIF-nya, bukan yang kita inginkan."""
    monkeypatch.setenv(ENV_TABLE_AS_CELLS, "false")
    tc = _muat_ulang(monkeypatch, aktif=True)
    assert tc.siapkan_ekstraksi() == {ENV_TABLE_AS_CELLS: "false"}
