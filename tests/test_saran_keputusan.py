"""Uji apa yang menggerakkan kolom `saran` di table_continuation.json.

Kalibrasi `kualitas_teks` terbukti TERBALIK arahnya pada isi tabel, jadi ia
dicabut dari penggerak saran. Uji di bawah mengunci tiga hal: ia tidak lagi
berpengaruh, halaman tanpa lapisan teks TETAP menolak otomatis, dan sisanya
digerakkan sinyal struktural plus vision.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "analisis", Path(__file__).resolve().parent.parent
    / "scripts" / "analisis_tabel_lintas_halaman.py")
analisis = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analisis)

from lib.kualitas_teks import Kualitas, nilai_teks  # noqa: E402


class Profil:
    """Profil dokumen palsu dengan kendali penuh atas lapisan teks dan kualitas."""

    def __init__(self, berteks=True, kualitas=None, error=""):
        self._berteks = berteks
        self.kualitas = kualitas or nilai_teks("kata " * 40)
        self.error = error

    def halaman_berlapis_teks(self, _):
        return self._berteks


def rec(kategori="lanjutan_kuat", halaman=(35, 36)):
    return {"kategori": kategori, "halaman": list(halaman), "skor": 5}


# Teks rusak nyata dari korpus; heuristik menilainya SEHAT.
RUSAK_NYATA = nilai_teks("ATATAN ATAS LAPORAN KEUANGAN JUNI 2022 Umuk Tomggal " * 4)
# Header tabel sehat nyata; heuristik menghukumnya.
SEHAT_NYATA = nilai_teks(
    "NO. IBUKOTA PROVINSI KOTA/KABUPATEN TUJUAN SATUAN BESARAN " * 4)


@pytest.mark.unit
def test_kualitas_rusak_tidak_lagi_menolak():
    """standar-biaya-2023 hal 35->36: header terbaca sempurna, dilabeli rusak."""
    saran, alasan = analisis._saran(rec(), Profil(kualitas=SEHAT_NYATA), None)
    assert saran == "terima" and "sinyal struktural" in alasan


@pytest.mark.unit
def test_label_kualitas_apa_pun_tidak_menggerakkan_saran():
    hasil = {
        analisis._saran(rec(), Profil(kualitas=k), None)[0]
        for k in (RUSAK_NYATA, SEHAT_NYATA, Kualitas(), nilai_teks("kata " * 40))
    }
    assert hasil == {"terima"}, "kualitas_teks masih menggerakkan saran"


@pytest.mark.unit
def test_halaman_tanpa_lapisan_teks_TETAP_menolak():
    """Bukan penilaian kualitas — fakta bahwa isi selnya dari OCR."""
    saran, alasan = analisis._saran(rec(), Profil(berteks=False), None)
    assert saran == "tolak" and "tanpa lapisan teks" in alasan


@pytest.mark.unit
def test_penolak_lapisan_teks_menang_atas_vision():
    saran, _ = analisis._saran(
        rec(), Profil(berteks=False), {"verdict": "lanjutan"})
    assert saran == "tolak"


@pytest.mark.unit
@pytest.mark.parametrize("verdict,harapan", [
    ("lanjutan", "terima"), ("bukan_lanjutan", "tolak"), ("tidak_jelas", "tolak"),
    (None, "tolak"),
])
def test_kategori_mungkin_digerakkan_vision(verdict, harapan):
    saran, _ = analisis._saran(rec(kategori="mungkin"), Profil(),
                               {"verdict": verdict} if verdict else None)
    assert saran == harapan


@pytest.mark.unit
def test_vision_bukan_lanjutan_menolak_walau_kategori_kuat():
    saran, alasan = analisis._saran(rec(), Profil(), {"verdict": "bukan_lanjutan"})
    assert saran == "tolak" and "belum diverifikasi" in alasan


@pytest.mark.unit
def test_profil_error_tidak_menolak_otomatis():
    """PDF tidak terbaca bukan bukti bahwa tabelnya bukan lanjutan."""
    saran, _ = analisis._saran(rec(), Profil(berteks=False, error="rusak"), None)
    assert saran == "terima"


@pytest.mark.unit
def test_tanpa_profil_sama_sekali():
    saran, _ = analisis._saran(rec(), None, None)
    assert saran == "terima"


@pytest.mark.unit
def test_arah_sinyal_kualitas_memang_terbalik():
    """Mengunci temuan yang mendasari pencabutan ini.

    Kalau suatu saat heuristiknya diperbaiki sehingga uji ini gagal, itu sinyal
    untuk mempertimbangkan ulang apakah ia layak menggerakkan saran lagi.
    """
    assert RUSAK_NYATA.rasio_tanpa_vokal == 0.0      # rusak, tak terdeteksi
    assert RUSAK_NYATA.rasio_fungsi > SEHAT_NYATA.rasio_fungsi
    assert SEHAT_NYATA.rasio_fungsi == 0.0            # sehat, terhukum
