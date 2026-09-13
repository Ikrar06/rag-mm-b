"""Uji pembedaan PDF digital vs pindai.

Aturan klasifikasinya diuji tanpa PDF (fungsi murni), lalu jalur I/O diuji
dengan PDF yang dibuat di tempat.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from lib.pdf_sumber import (  # noqa: E402
    AMBANG_KARAKTER, ProfilDokumen, profil_dari_cacah, profil_dari_pdf,
)

PANJANG = AMBANG_KARAKTER + 10
PENDEK = AMBANG_KARAKTER          # tepat di ambang = TIDAK berteks (perbandingan >)


@pytest.mark.unit
def test_ambang_menyalin_nilai_pipeline():
    """Kalau pipeline mengubah ambangnya, uji ini harus ikut diperbarui sadar."""
    assert AMBANG_KARAKTER == 50


@pytest.mark.unit
def test_dokumen_digital():
    p = profil_dari_cacah("a.pdf", {1: PANJANG, 2: PANJANG, 3: PANJANG})
    assert p.sumber == "digital" and p.rasio_berteks == 1.0


@pytest.mark.unit
def test_dokumen_pindai():
    p = profil_dari_cacah("a.pdf", {1: 0, 2: 3, 3: PENDEK})
    assert p.sumber == "pindai" and p.rasio_berteks == 0.0


@pytest.mark.unit
def test_dokumen_campuran():
    p = profil_dari_cacah("a.pdf", {1: PANJANG, 2: PANJANG, 3: 0, 4: 0})
    assert p.sumber == "campuran" and p.rasio_berteks == 0.5


@pytest.mark.unit
def test_tepat_di_ambang_dihitung_tanpa_teks():
    """Pipeline memakai `> AMBANG`, bukan `>=`. Perilakunya harus sama."""
    assert profil_dari_cacah("a.pdf", {1: AMBANG_KARAKTER}).n_halaman_berteks == 0
    assert profil_dari_cacah("a.pdf", {1: AMBANG_KARAKTER + 1}).n_halaman_berteks == 1


@pytest.mark.unit
def test_halaman_tabel_dinilai_terpisah_dari_dokumen():
    """Dokumen mayoritas digital, tapi halaman tabelnya sisipan pindai."""
    cacah = {1: PANJANG, 2: PANJANG, 3: PANJANG, 4: PANJANG,
             5: PANJANG, 6: PANJANG, 7: PANJANG, 8: PANJANG, 9: 0, 10: 0}
    p = profil_dari_cacah("a.pdf", cacah, halaman_tabel=(9, 10))
    assert p.sumber == "campuran"
    assert p.sumber_halaman_tabel == "pindai"
    assert p.rasio_tabel_berteks == 0.0


@pytest.mark.unit
def test_halaman_tabel_duplikat_dan_tak_urut_dinormalkan():
    p = profil_dari_cacah("a.pdf", {1: PANJANG, 2: 0}, halaman_tabel=(2, 1, 2))
    assert p.halaman_tabel == (1, 2)


@pytest.mark.unit
def test_dokumen_kosong_tidak_membagi_nol():
    p = profil_dari_cacah("a.pdf", {})
    assert p.sumber == "tak_terbaca"
    assert p.rasio_berteks == 0.0 and p.rasio_tabel_berteks == 0.0


@pytest.mark.unit
def test_profil_immutable():
    p = profil_dari_cacah("a.pdf", {1: PANJANG})
    with pytest.raises(Exception):
        p.n_halaman = 99


@pytest.mark.integration
def test_pdf_digital_nyata(tmp_path):
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "digital.pdf"
    doc = fitz.open()
    for _ in range(3):
        page = doc.new_page()
        page.insert_text((72, 100), "Persyaratan pengajuan izin ujian akhir "
                                    "bagi mahasiswa program sarjana Universitas.")
    doc.save(str(path)); doc.close()
    p = profil_dari_pdf(path, halaman_tabel=(1,))
    assert p.sumber == "digital" and p.n_halaman == 3
    assert p.sumber_halaman_tabel == "digital" and not p.error


@pytest.mark.integration
def test_pdf_tanpa_lapisan_teks(tmp_path):
    """Halaman kosong meniru hasil pindai: tidak ada teks yang dapat diambil."""
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "pindai.pdf"
    doc = fitz.open()
    for _ in range(2):
        doc.new_page()
    doc.save(str(path)); doc.close()
    p = profil_dari_pdf(path, halaman_tabel=(1, 2))
    assert p.sumber == "pindai" and p.sumber_halaman_tabel == "pindai"


@pytest.mark.integration
def test_berkas_rusak_jadi_error_bukan_lemparan(tmp_path):
    pytest.importorskip("fitz")
    path = tmp_path / "rusak.pdf"
    path.write_bytes(b"ini bukan PDF")
    p = profil_dari_pdf(path, halaman_tabel=(1,))
    assert p.error and p.sumber == "tak_terbaca"
    assert p.halaman_tabel == (1,)


@pytest.mark.integration
def test_berkas_tidak_ada_jadi_error(tmp_path):
    pytest.importorskip("fitz")
    p = profil_dari_pdf(tmp_path / "hilang.pdf")
    assert p.error and p.sumber == "tak_terbaca"


@pytest.mark.unit
def test_halaman_berlapis_teks_per_halaman():
    p = profil_dari_cacah("a.pdf", {1: PANJANG, 2: 0, 3: PANJANG})
    assert p.halaman_berlapis_teks(1) and p.halaman_berlapis_teks(3)
    assert not p.halaman_berlapis_teks(2)
    assert not p.halaman_berlapis_teks(99)
