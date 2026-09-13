"""Uji deteksi sel terpotong terhadap CONTOH NYATA dari korpus.

Empat kasus di bawah diambil dari keluaran instrumen atas
rag_mm_b_varian_b_v2, dokumen pedoman-penyusunan-laporan-keuangan. Tiga
pertama positif palsu yang lolos saringan versi pertama; yang terakhir positif
sejati. Keempatnya mengunci perilaku detektor.

Pola positif palsunya sama: angka diikuti tahun atau kata header. Indikator
"tidak berakhir dengan tanda baca penutup" terlalu longgar untuk sel berisi
angka — angka memang tidak pernah diakhiri titik.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from lib.tabel_html import deteksi_sel_terpotong, urai  # noqa: E402


def pasangan(baris_a, baris_b):
    """Bangun dua Tabel dari baris terakhir A dan baris pertama B."""
    def t(baris):
        return urai("<table><tbody>"
                    + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
                              for r in baris)
                    + "</tbody></table>")
    return t(baris_a), t(baris_b)


# ─── Positif palsu dari korpus — harus DITOLAK ───────────────────────────────

@pytest.mark.unit
def test_hal_25_26_angka_lalu_tahun_bukan_sel_terpotong():
    """A berakhir '5.829', B mulai '2022'. Itu tahun di header halaman baru."""
    a, b = pasangan([["Kas dan setara kas", "5.829"]], [["", "2022"]])
    assert deteksi_sel_terpotong(a, b).kuat == ()


@pytest.mark.unit
def test_hal_26_27_angka_panjang_lalu_tahun():
    a, b = pasangan([["Jumlah aset", "111.470.940.423"]], [["", "2022"]])
    assert deteksi_sel_terpotong(a, b).kuat == ()


@pytest.mark.unit
def test_hal_79_80_angka_lalu_kata_header():
    """B mulai 'URAIAN' — itu baris header halaman baru, bukan lanjutan."""
    a, b = pasangan([["Total", "138.209.484.846"]], [["", "URAIAN"]])
    assert deteksi_sel_terpotong(a, b).kuat == ()


@pytest.mark.unit
@pytest.mark.parametrize("kata", ["URAIAN", "DESCRIPTION", "NO.", "TOTAL",
                                  "Jumlah", "Keterangan", "2022", "2021"])
def test_kata_header_dan_tahun_selalu_ditolak(kata):
    a, b = pasangan([["1", "Pendapatan diterima dimuka sebesar"]], [["", kata]])
    assert deteksi_sel_terpotong(a, b).kuat == ()


@pytest.mark.unit
def test_sel_a_seluruhnya_numerik_ditolak():
    a, b = pasangan([["1", "1.234.567"]], [["", "lanjutan kalimat berikutnya"]])
    assert deteksi_sel_terpotong(a, b).kuat == ()


# ─── Positif sejati dari korpus — harus DITERIMA ─────────────────────────────

@pytest.mark.unit
def test_hal_83_84_kalimat_terbelah_adalah_sel_terpotong():
    """A berakhir '...dan 31 Desember 2021 sebesar', B mulai
    '1.837.275.237.398 and 31 December 2021'.

    B diawali angka TAPI tidak seluruhnya numerik — ada 'and 31 December'.
    A berakhir 'sebesar', kata yang menuntut pelengkap.
    """
    a, b = pasangan(
        [["Kas", "Saldo per 31 Desember 2022 dan 31 Desember 2021 sebesar"]],
        [["", "1.837.275.237.398 and 31 December 2021"]],
    )
    kuat = deteksi_sel_terpotong(a, b).kuat
    assert len(kuat) == 1
    assert kuat[0].ekor_menggantung


@pytest.mark.unit
def test_kalimat_indonesia_terbelah():
    a, b = pasangan([["2", "Berkas diverifikasi admin dengan melampirkan"]],
                    [["", "transkrip nilai dan fotokopi KTM"]])
    assert len(deteksi_sel_terpotong(a, b).kuat) == 1


@pytest.mark.unit
def test_konjungsi_inggris_dikenali_korpus_dwibahasa():
    a, b = pasangan([["3", "The statement of financial position"]],
                    [["", "and the related notes thereto"]])
    assert len(deteksi_sel_terpotong(a, b).kuat) == 1


@pytest.mark.unit
def test_tanpa_tanda_baca_saja_tidak_cukup():
    """Butuh minimal satu sinyal KEBAHASAAN, bukan sekadar tanpa titik."""
    a, b = pasangan([["1", "Aset Lancar"]], [["2", "Aset Tetap"]])
    assert deteksi_sel_terpotong(a, b).kuat == ()
