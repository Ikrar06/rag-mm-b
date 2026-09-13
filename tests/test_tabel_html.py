"""Uji logika murni penguraian tabel dan deteksi sel terpotong.

Angka dari modul ini dipakai untuk keputusan yang menyentuh 1.300 item gold,
jadi perilakunya dikunci di sini sebelum dipakai mengukur korpus.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from lib.tabel_html import (  # noqa: E402
    Tabel, urai, bandingkan_header, deteksi_sel_terpotong, rasio_angka,
)


def tbl(header, baris, pakai_th=True):
    h = ("<thead><tr>" + "".join(f"<th>{c}</th>" for c in header) + "</tr></thead>"
         if header and pakai_th else "")
    if header and not pakai_th:
        baris = [header] + list(baris)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in baris)
    return f"<table>{h}<tbody>{b}</tbody></table>"


HDR = ["Persyaratan", "S1", "S2", "S3"]
DATA = [["IPK minimum", "2,75", "3,00", "3,25"]]


# ─── urai ────────────────────────────────────────────────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("html", [None, "", "bukan tabel", "<p>teks</p>", "<table></table>"])
def test_urai_menolak_yang_bukan_tabel(html):
    assert urai(html) is None


@pytest.mark.unit
def test_urai_menangkap_baris_dan_penanda_header():
    t = urai(tbl(HDR, DATA))
    assert t.baris[0] == tuple(HDR)
    assert t.punya_header and t.ada_th and t.ada_thead
    assert t.n_kolom == 4


@pytest.mark.unit
def test_urai_tidak_mengubah_masukan_dan_immutable():
    html = tbl(HDR, DATA)
    t = urai(html)
    assert isinstance(t.baris, tuple) and isinstance(t.baris[0], tuple)
    with pytest.raises(Exception):
        t.baris = ()          # frozen dataclass
    assert html == tbl(HDR, DATA)


@pytest.mark.unit
def test_n_kolom_modal_tahan_baris_judul_colspan():
    html = ("<table><tr><td>JUDUL TABEL</td></tr>"
            "<tr><td>a</td><td>b</td><td>c</td></tr>"
            "<tr><td>d</td><td>e</td><td>f</td></tr></table>")
    assert urai(html).n_kolom == 3


@pytest.mark.unit
def test_urai_html_rusak_tidak_melempar():
    assert urai("<table><tr><td>a</tr>") is not None


# ─── bandingkan_header ───────────────────────────────────────────────────────

@pytest.mark.unit
def test_header_b_tanpa_th_eksplisit():
    a = urai(tbl(HDR, DATA))
    b = urai(tbl(None, [["Masa studi", "14", "8", "10"]]))
    assert bandingkan_header(a, b) == "b_tanpa_header"


@pytest.mark.unit
def test_header_tanpa_th_dibedakan_lewat_bentuk_baris():
    a = urai(tbl(HDR, DATA, pakai_th=False))
    b = urai(tbl(None, [["Masa studi", "14", "8", "10"]]))
    assert bandingkan_header(a, b) == "b_tanpa_header"


@pytest.mark.unit
def test_header_diulang():
    a = urai(tbl(HDR, DATA))
    b = urai(tbl(HDR, [["Masa studi", "14", "8", "10"]]))
    assert bandingkan_header(a, b) == "header_diulang"


@pytest.mark.unit
def test_header_berbeda_adalah_bukti_lawan():
    a = urai(tbl(HDR, DATA))
    b = urai(tbl(["Kode", "Mata Kuliah", "SKS", "Smt"], [["MK1", "Kalkulus", "3", "1"]]))
    assert bandingkan_header(a, b) == "header_berbeda"


@pytest.mark.unit
@pytest.mark.parametrize("a,b", [(None, None), (urai(tbl(HDR, DATA)), None)])
def test_header_tak_tentu_bila_ada_yang_kosong(a, b):
    assert bandingkan_header(a, b) == "tak_tentu"


@pytest.mark.unit
def test_rasio_angka():
    assert rasio_angka(["2,75", "3,00"]) == 1.0
    assert rasio_angka(["IPK", "S1"]) == 0.0
    assert rasio_angka([]) == 0.0


# ─── deteksi_sel_terpotong ───────────────────────────────────────────────────

@pytest.mark.unit
def test_sel_terpotong_kasus_nyata_uraian_terpenggal():
    """Kolom Uraian terpenggal: A berakhir 'dengan melampirkan',
    B mulai 'transkrip nilai dan'."""
    a = urai(tbl(["No", "Uraian"], [["1", "Mengajukan permohonan dengan melampirkan"]]))
    b = urai(tbl(None, [["", "transkrip nilai dan fotokopi KTM"]]))
    h = deteksi_sel_terpotong(a, b)
    assert len(h.kuat) == 1
    k = h.kuat[0]
    assert k.kolom == 1
    assert k.tanpa_tanda_baca and k.lanjutan_huruf_kecil and k.sel_lain_kosong


@pytest.mark.unit
def test_sel_terpotong_konjungsi_dikenali():
    a = urai(tbl(["No", "Uraian"], [["1", "Berkas diverifikasi admin prodi"]]))
    b = urai(tbl(None, [["", "dan diteruskan ke Wakil Dekan"]]))
    k = deteksi_sel_terpotong(a, b).kuat
    assert len(k) == 1 and k[0].lanjutan_konjungsi


@pytest.mark.unit
def test_baris_utuh_tidak_dianggap_terpotong():
    """Sel A berakhir titik, sel B diawali huruf besar, tidak ada sel kosong."""
    a = urai(tbl(["No", "Uraian"], [["1", "Mengajukan permohonan daring."]]))
    b = urai(tbl(None, [["2", "Melampirkan transkrip nilai."]]))
    assert deteksi_sel_terpotong(a, b).kuat == ()


@pytest.mark.unit
def test_lebar_baris_berbeda_dilaporkan_bukan_ditebak():
    a = urai(tbl(["No", "Uraian"], [["1", "teks"]]))
    b = urai(tbl(None, [["1", "teks", "lebih"]]))
    h = deteksi_sel_terpotong(a, b)
    assert h.kandidat == () and "lebar baris berbeda" in h.catatan


@pytest.mark.unit
@pytest.mark.parametrize("a,b", [(None, None), (urai(tbl(HDR, DATA)), None)])
def test_sel_terpotong_aman_bila_html_tak_terurai(a, b):
    h = deteksi_sel_terpotong(a, b)
    assert h.kandidat == () and h.catatan


@pytest.mark.unit
def test_sel_kosong_dilewati_bukan_dianggap_terpotong():
    a = urai(tbl(["No", "Uraian"], [["1", ""]]))
    b = urai(tbl(None, [["", "lanjutan teks"]]))
    assert deteksi_sel_terpotong(a, b).kandidat == ()


@pytest.mark.unit
def test_skor_satu_indikator_bukan_kandidat_kuat():
    """Tanpa tanda baca saja tidak cukup — angka tabel memang tanpa titik."""
    a = urai(tbl(["No", "Nilai"], [["1", "2,75"]]))
    b = urai(tbl(None, [["2", "3,00"]]))
    assert deteksi_sel_terpotong(a, b).kuat == ()
