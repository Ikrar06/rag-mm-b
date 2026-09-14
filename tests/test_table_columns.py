"""Uji pemulihan batas kolom dari posisi kata.

Sinyal ini menggantikan "koordinat kolom dari metadata Unstructured", yang
ternyata tidak ada. Ia hanya bekerja pada PDF berlapis teks, jadi perbedaan
antara "tidak cocok" dan "tidak dapat dinilai" harus tegas — yang kedua jatuh
ke adjudikasi vision, bukan dihitung sebagai bukti lawan.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services.table_columns import (  # noqa: E402
    BatasKolom, batas_dari_halaman, batas_dari_kata, mirip,
)


def kata_tiga_kolom(x0=0.0, lebar=300.0):
    """Kata di tiga rumpun: 0-60, 120-180, 240-300 (relatif terhadap x0)."""
    out = []
    for awal in (0, 120, 240):
        for k in range(3):
            out.append((x0 + awal + k * 18, x0 + awal + k * 18 + 16))
    return tuple(out), x0, x0 + lebar


@pytest.mark.unit
def test_tiga_rumpun_kata_menghasilkan_dua_pemisah():
    kata, a, b = kata_tiga_kolom()
    hasil = batas_dari_kata(kata, a, b)
    assert hasil.tersedia
    assert len(hasil.batas) == 2 and hasil.n_kolom == 3


@pytest.mark.unit
def test_spasi_antar_kata_bukan_pemisah_kolom():
    """Kata rapat berspasi normal harus jadi SATU kolom."""
    kata = tuple((k * 20, k * 20 + 18) for k in range(15))
    hasil = batas_dari_kata(kata, 0.0, 300.0)
    assert hasil.tersedia and hasil.n_kolom == 1


@pytest.mark.unit
def test_celah_di_tepi_bukan_pemisah():
    """Margin kiri dan kanan tabel tidak boleh dihitung sebagai kolom."""
    kata = tuple((100 + k * 20, 100 + k * 20 + 18) for k in range(5))
    hasil = batas_dari_kata(kata, 0.0, 400.0)
    assert hasil.tersedia and hasil.n_kolom == 1 and hasil.batas == ()


@pytest.mark.unit
def test_tanpa_kata_berarti_tidak_tersedia_bukan_nol_kolom():
    hasil = batas_dari_kata((), 0.0, 300.0)
    assert not hasil.tersedia and hasil.n_kolom == 0


@pytest.mark.unit
def test_kata_seluruhnya_di_luar_bbox_juga_tidak_tersedia():
    hasil = batas_dari_kata(((500.0, 520.0), (540.0, 560.0)), 0.0, 300.0)
    assert not hasil.tersedia


@pytest.mark.unit
@pytest.mark.parametrize("x0,x1", [(10.0, 10.0), (300.0, 0.0)])
def test_lebar_tabel_tidak_sah(x0, x1):
    assert not batas_dari_kata(((1.0, 2.0),), x0, x1).tersedia


@pytest.mark.unit
def test_batas_ternormalisasi_terhadap_lebar_tabel():
    kata, a, b = kata_tiga_kolom(x0=0.0, lebar=300.0)
    h1 = batas_dari_kata(kata, a, b)
    # Tabel yang sama pada halaman berukuran dua kali lipat.
    kata2 = tuple((x * 2, y * 2) for x, y in kata)
    h2 = batas_dari_kata(kata2, 0.0, 600.0)
    assert h1.batas == h2.batas


# ─── mirip() ─────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_dua_potongan_tabel_sama_cocok():
    kata, a, b = kata_tiga_kolom()
    assert mirip(batas_dari_kata(kata, a, b), batas_dari_kata(kata, a, b)) is True


@pytest.mark.unit
def test_geseran_kecil_masih_cocok():
    kata, a, b = kata_tiga_kolom()
    geser = tuple((x + 1.5, y + 1.5) for x, y in kata)
    assert mirip(batas_dari_kata(kata, a, b), batas_dari_kata(geser, a, b)) is True


@pytest.mark.unit
def test_lebar_kolom_berbeda_tidak_cocok():
    kata_a, a, b = kata_tiga_kolom()
    kata_b = tuple((x * 0.6, y * 0.6) for x, y in kata_a)
    assert mirip(batas_dari_kata(kata_a, a, b), batas_dari_kata(kata_b, a, b)) is False


@pytest.mark.unit
def test_jumlah_kolom_berbeda_tidak_cocok():
    kata_a, a, b = kata_tiga_kolom()
    kata_b = tuple((k * 20, k * 20 + 18) for k in range(15))
    assert mirip(batas_dari_kata(kata_a, a, b), batas_dari_kata(kata_b, a, b)) is False


@pytest.mark.unit
def test_tidak_tersedia_mengembalikan_none_bukan_false():
    """Halaman pindai: 'tidak dapat dinilai', HARUS dibedakan dari 'tidak cocok'.
    None diteruskan ke adjudikasi vision; False adalah bukti lawan."""
    kata, a, b = kata_tiga_kolom()
    ada = batas_dari_kata(kata, a, b)
    kosong = batas_dari_kata((), a, b)
    assert mirip(ada, kosong) is None
    assert mirip(kosong, ada) is None
    assert mirip(kosong, kosong) is None


@pytest.mark.unit
def test_batas_kolom_immutable():
    with pytest.raises(Exception):
        BatasKolom().batas = (0.5,)


# ─── jalur I/O ───────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_pdf_digital_tiga_kolom(tmp_path):
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "tabel.pdf"
    doc = fitz.open()
    page = doc.new_page()
    for y in (120, 140, 160):
        page.insert_text((72, y), "Persyaratan")
        page.insert_text((260, y), "2,75")
        page.insert_text((430, y), "3,00")
    doc.save(str(path)); doc.close()
    hasil = batas_dari_halaman(path, 1, [0.05, 0.10, 0.95, 0.30])
    assert hasil.tersedia and hasil.n_kolom == 3


@pytest.mark.integration
def test_pdf_tanpa_lapisan_teks_tidak_tersedia(tmp_path):
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "pindai.pdf"
    doc = fitz.open(); doc.new_page(); doc.save(str(path)); doc.close()
    assert not batas_dari_halaman(path, 1, [0.05, 0.1, 0.95, 0.9]).tersedia


@pytest.mark.integration
@pytest.mark.parametrize("bbox", [None, [], [0.1, 0.2], [0.1, 0.2, 0.3, 0.4, 0.5]])
def test_bbox_tidak_sah_aman(tmp_path, bbox):
    pytest.importorskip("fitz")
    assert not batas_dari_halaman(tmp_path / "apa.pdf", 1, bbox).tersedia


@pytest.mark.integration
def test_berkas_rusak_dan_halaman_di_luar_jangkauan(tmp_path):
    fitz = pytest.importorskip("fitz")
    rusak = tmp_path / "rusak.pdf"
    rusak.write_bytes(b"bukan pdf")
    assert not batas_dari_halaman(rusak, 1, [0.1, 0.1, 0.9, 0.9]).tersedia

    ok = tmp_path / "ok.pdf"
    doc = fitz.open(); doc.new_page(); doc.save(str(ok)); doc.close()
    assert not batas_dari_halaman(ok, 99, [0.1, 0.1, 0.9, 0.9]).tersedia
    assert not batas_dari_halaman(ok, 0, [0.1, 0.1, 0.9, 0.9]).tersedia
