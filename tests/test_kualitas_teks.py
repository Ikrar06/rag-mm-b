"""Uji indikator kualitas lapisan teks.

Kasus penentu diambil dari korpus: `pedoman-penyusunan-laporan-keuangan`
ditandai "digital" oleh pdf_sumber karena PDF-nya punya lapisan teks, padahal
lapisan teks itu sendiri rusak. Indikator di sini harus menangkapnya.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from lib.kualitas_teks import (  # noqa: E402
    MIN_TOKEN, Kualitas, gabung, nilai_teks,
)

# Prosa Indonesia yang sehat, cukup panjang untuk melewati MIN_TOKEN.
SEHAT = (
    "Mahasiswa mengajukan permohonan izin ujian akhir melalui sistem akademik "
    "daring paling lambat empat belas hari kerja sebelum tanggal pelaksanaan. "
    "Berkas permohonan diverifikasi oleh admin program studi, kemudian "
    "diteruskan kepada Wakil Dekan Bidang Akademik untuk memperoleh "
    "persetujuan sebagaimana diatur dalam pedoman yang berlaku di fakultas."
)

# Potongan nyata dari korpus, diperpanjang dengan kerusakan sejenis agar
# melewati MIN_TOKEN. Huruf awal terpotong, huruf tersubstitusi, kata pecah.
RUSAK = (
    "ATATAN ATAS LAPORAN KEUANGAN JUNI 2022 Umuk Tomggal Ymg Berakhtr "
    "PADA TGL TSB DN LPRN PSISI KEUNGN KNSLDSN Ttl Jmlh Aktv Lncr "
    "Ktrngn Tmbhn Ats Lprn Kungn Knsldsn Prshn Dn Ntts Ank"
)


@pytest.mark.unit
def test_prosa_sehat_dinilai_baik():
    k = nilai_teks(SEHAT)
    assert k.cukup_sampel
    assert k.label == "baik"
    assert k.rasio_fungsi > 0.08


@pytest.mark.unit
def test_teks_ocr_rusak_dari_korpus_tertangkap():
    """Kasus yang lolos dari pdf_sumber: ada lapisan teks, tapi kacau."""
    k = nilai_teks(RUSAK)
    assert k.cukup_sampel
    assert k.label in ("rusak", "patut_dicurigai")
    assert k.skor is not None and k.skor < nilai_teks(SEHAT).skor


@pytest.mark.unit
def test_token_tanpa_vokal_terhitung():
    k = nilai_teks("Knsldsn Lprn Kngn " * 12)   # ketiganya tanpa vokal
    assert k.rasio_tanpa_vokal > 0.9


@pytest.mark.unit
def test_huruf_terisolasi_terhitung():
    k = nilai_teks("a b c d e f g h i j k l m n o p q r s t u v w x y z aa bb cc dd ee ff")
    assert k.rasio_terisolasi == 1.0


@pytest.mark.unit
def test_kapital_campur_di_tengah_kata():
    """'LAporan', 'UmUk' — transisi kapital di tengah kata, khas OCR."""
    k = nilai_teks("LAporan UmUk KEuangan TAnggal " * 10)
    assert k.rasio_kapital_campur > 0.9


@pytest.mark.unit
def test_kapital_wajar_tidak_dihitung_rusak():
    """Kata berkapital normal dan SINGKATAN tidak boleh dianggap rusak."""
    assert nilai_teks("Universitas Hasanuddin Fakultas Teknik " * 10).rasio_kapital_campur == 0.0
    assert nilai_teks("IPK SKS TOEFL KRS KHS UKT " * 10).rasio_kapital_campur == 0.0


@pytest.mark.unit
def test_sampel_kecil_tidak_dihakimi():
    k = nilai_teks("Terlalu pendek untuk dinilai")
    assert not k.cukup_sampel
    assert k.label == "sampel_kecil"
    # None, bukan 1.0: "tidak diketahui" tidak boleh terbaca "sehat".
    assert k.skor is None


@pytest.mark.unit
@pytest.mark.parametrize("teks", ["", None, "   ", "12345 678 90"])
def test_teks_kosong_atau_tanpa_huruf_aman(teks):
    k = nilai_teks(teks)
    assert k.n_token == 0 or not k.cukup_sampel
    assert k.rasio_tanpa_vokal == 0.0


@pytest.mark.unit
def test_ambang_min_token_konsisten():
    cukup = " ".join(["kata"] * MIN_TOKEN)
    kurang = " ".join(["kata"] * (MIN_TOKEN - 1))
    assert nilai_teks(cukup).cukup_sampel
    assert not nilai_teks(kurang).cukup_sampel


@pytest.mark.unit
def test_gabung_menjumlahkan_cacahan_bukan_merata_rasio():
    """Halaman pendek tidak boleh berbobot sama dengan halaman penuh."""
    panjang = nilai_teks(SEHAT)
    pendek = nilai_teks("Knsldsn")
    total = gabung([panjang, pendek])
    assert total.n_token == panjang.n_token + pendek.n_token
    # Satu kata rusak tidak boleh menjatuhkan penilaian seluruh dokumen.
    assert total.label == "baik"


@pytest.mark.unit
def test_gabung_kosong():
    assert gabung([]) == Kualitas()


@pytest.mark.unit
def test_kualitas_immutable():
    k = nilai_teks(SEHAT)
    with pytest.raises(Exception):
        k.n_token = 0
