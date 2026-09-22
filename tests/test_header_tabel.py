"""Uji kriteria "header kolom sungguhan, bukan baris data".

Kasus diambil dari run v3: baris data `921112 | BELANJA PENGADAAN BAHAN
MAKANAN` tersalin ke delapan chunk karena fallback "pakai baris pertama".
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from lib.header_tabel import deteksi_header  # noqa: E402


# ─── markup eksplisit menang atas segalanya ──────────────────────────────────

@pytest.mark.unit
def test_markup_eksplisit_langsung_diterima():
    """Kalau penghasil HTML menandainya header, tidak perlu menebak ulang."""
    p = deteksi_header([["921112", "BELANJA"]], ada_th=True)
    assert p and p.aturan == "markup"


# ─── contoh NYATA yang harus DITOLAK ─────────────────────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("kandidat,tubuh,aturan", [
    (["921112", "BELANJA PENGADAAN BAHAN MAKANAN"],
     [["921113", "BELANJA PENGADAAN OBAT"]], "b-numerik"),
    (["4141", "ALOKASI BELANJA APBN LAINNYA"],
     [["4142", "ALOKASI BELANJA MODAL"]], "b-numerik"),
    (["78", "Padang", "Kota Bukit Tinggi", "Orang/Kali", "215.000"],
     [["79", "Pangkalpinang", "Kab. Bangka", "Orang/Kali", "220.000"]], "b-numerik"),
    (["'00", "Manado", "Kab. Minahasa Tenggara", "Orang/Kali", "200.000"],
     [["101", "Palu", "Kab. Sigi", "Orang/Kali", "205.000"]], "b-numerik"),
    ([".", "Beban Belanja APBN - Gaji dan", "154.234.615.382", "138.209.484.846"],
     [["2", "Beban Belanja Barang", "45.120.000.000", "41.000.000.000"]], "b-numerik"),
])
def test_baris_data_ditolak(kandidat, tubuh, aturan):
    p = deteksi_header([kandidat, *tubuh])
    assert not p, f"baris data lolos jadi header: {p.alasan}"
    assert p.aturan == aturan


@pytest.mark.unit
def test_glosarium_ditolak_lewat_kontras():
    """`SLA | Subsidiary Loan Agreement` — baris pertama sejenis dengan tubuh."""
    p = deteksi_header([["SLA", "Subsidiary Loan Agreement"],
                        ["DIPA", "Daftar Isian Pelaksanaan Anggaran"],
                        ["SPM", "Surat Perintah Membayar"]])
    assert not p and p.aturan == "d-kontras"


@pytest.mark.unit
def test_judul_bagian_ditolak_lewat_kontras():
    """`Kata | Pengantar | Foreword` — judul bagian, bukan header kolom."""
    p = deteksi_header([["Kata", "Pengantar", "Foreword"],
                        ["Daftar", "Isi", "Table of Contents"],
                        ["Ringkasan", "Eksekutif", "Executive Summary"]])
    assert not p and p.aturan == "d-kontras"


@pytest.mark.unit
def test_daftar_isi_ocr_satu_sel_ditolak():
    p = deteksi_header([["IALAMAN JUDUL ............ i"], ["DAFTAR ISI ......... ii"]])
    assert not p and p.aturan == "a-bentuk"


# ─── contoh NYATA yang harus DITERIMA ────────────────────────────────────────

@pytest.mark.unit
def test_header_arsip():
    p = deteksi_header([
        ["NO.", "SERIES/JENIS ARSIP", "AKTIF", "INAKTIF", "KETERANGAN"],
        ["1", "Surat Keputusan Rektor", "2", "5", "Permanen"],
        ["2", "Notulen Rapat Senat", "2", "3", "Musnah"]])
    assert p and p.aturan == "d-kontras"


@pytest.mark.unit
def test_header_standar_biaya():
    p = deteksi_header([
        ["NO.", "PROVINSI", "SATUAN", "RODA 4", "RODA 6/BUS SEDANG"],
        ["1", "Aceh", "Per hari", "978.000", "2.427.000"]])
    assert p, "RODA 4 memuat digit tapi bukan sel numerik murni"


@pytest.mark.unit
def test_tabel_serba_teks_ditolak_dan_itu_disengaja():
    """`Item | Definisi | Dokumen Pendukung` adalah header sungguhan, tapi
    tanpa kolom numerik ia tidak dapat dibedakan dari glosarium.

    Ditolak SENGAJA: salah tolak hanya membiarkan potongan apa adanya,
    sedangkan salah terima menyisipkan isi dokumen ke chunk lain."""
    p = deteksi_header([["Item", "Definisi", "Dokumen Pendukung"],
                        ["KTM", "Kartu Tanda Mahasiswa", "Fotokopi KTM"]])
    assert not p and p.aturan == "d-kontras"


# ─── jadwal KKN: nasibnya ditentukan bentuk kolom JAM ────────────────────────

KKN = ["HARI/TGL", "JAM", "KEGIATAN", "KETERANGAN"]


@pytest.mark.unit
@pytest.mark.parametrize("jam,lolos", [
    ("08.00", True),               # jam telanjang -> numerik -> kontras
    ("08.00 - 10.00", True),       # rentang, masih numerik
    ("08:00-10:00", True),         # titik dua ikut pola numerik
    ("08.00 WITA", False),         # ada huruf -> bukan numerik
    ("Pagi", False),               # deskriptif
])
def test_jadwal_kkn_bergantung_bentuk_kolom_jam(jam, lolos):
    p = deteksi_header([KKN,
                        ["Senin, 12 Juli 2026", jam, "Registrasi ulang", "Gedung A"],
                        ["Selasa, 13 Juli 2026", jam, "Kuliah perdana", "Aula"]])
    assert bool(p) is lolos


@pytest.mark.unit
def test_jadwal_kkn_lolos_lewat_kolom_lain_bila_ada_nomor():
    """Bila tabelnya punya kolom NO., kontras datang dari sana walau JAM berhuruf."""
    p = deteksi_header([["NO.", *KKN],
                        ["1", "Senin, 12 Juli 2026", "08.00 WITA", "Registrasi", "-"],
                        ["2", "Selasa, 13 Juli 2026", "10.30 WITA", "Kuliah", "-"]])
    assert p and p.aturan == "d-kontras"


# ─── bentuk dan tepi ─────────────────────────────────────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("baris,aturan", [
    ([], "bentuk"),
    ([["satu"]], "a-bentuk"),
    ([["A", ""], ["1", "2"]], "a-bentuk"),
    ([["A", "B", "C"], ["1", "2"], ["3", "4"]], "a-bentuk"),   # != kolom modal
    ([["A", "B"]], "d-kontras"),                                # tanpa tubuh
])
def test_bentuk_tak_sah_ditolak(baris, aturan):
    p = deteksi_header(baris)
    assert not p and p.aturan == aturan


@pytest.mark.unit
def test_varian_b_meloloskan_yang_varian_a_tolak():
    """Pembanding: tanpa aturan kontras, glosarium lolos jadi header."""
    baris = [["SLA", "Subsidiary Loan Agreement"],
             ["DIPA", "Daftar Isian Pelaksanaan Anggaran"]]
    assert not deteksi_header(baris)
    assert deteksi_header(baris, pakai_kontras=False)


@pytest.mark.unit
def test_putusan_immutable():
    p = deteksi_header([["A", "B"], ["1", "2"]])
    with pytest.raises(Exception):
        p.header = False
