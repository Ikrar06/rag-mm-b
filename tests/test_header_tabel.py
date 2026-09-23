"""Uji kriteria "header kolom sungguhan, bukan baris data".

Dua putaran kegagalan membentuk kriteria ini:
- v3: fallback "pakai baris pertama" menyalin `921112 | BELANJA ...` ke 8 chunk
- validasi data nyata: markup <th> dipercaya tanpa syarat, dan table
  transformer memberi <th> ke baris pertama APA PUN isinya — 15 dari 51
  header markup ternyata data atau sampah
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from lib.header_tabel import deteksi_header, kepala_rantai  # noqa: E402


def sel(teks: str) -> list[str]:
    """'78 | | Padang' -> ['78', '', 'Padang']. Pipa di awal = sel pertama kosong,
    mengikuti cara baris itu ditampilkan (sel digabung dengan ' | ')."""
    return [s.strip() for s in teks.split("|")]


# ─── 15 kasus nyata yang LOLOS lewat markup di validasi pertama ──────────────
# Jalur markup tidak memakai tubuh tabel, jadi hasil di bawah ditentukan
# SEPENUHNYA oleh baris kandidat yang dikutip — bukan oleh tubuh karangan.

BOCOR_TERTANGKAP = [
    ("bagan-akun 20->21",          "5341 | BELANJA MODAL JALAN, IRIGASI", "b-numerik"),
    ("standar-biaya-2026 38->39",  "78 | | Padang | Kota Bukit Tinggi | 215.000", "b-numerik"),
    ("standar-biaya-2026 43->44",  "'00 | | Manado | Kab. Minahasa Tenggara", "b-numerik"),
    ("standar-biaya-2025 22->23",  "13.62 | Honorarium Tim Penyelenggara", "b-numerik"),
    ("laporan-keuangan 86->87",    ". | Beban Belanja APBN - Gaji dan | 154.234.615.382", "b-numerik"),
    ("rubrik 77->78",              "Durasi | 0-8 Jam | 0.5", "b-numerik"),
    ("laporan-keuangan 17->18",    "Kata | Pengantar | | | | Foreword", "a-terisi"),
    ("laporan-keuangan 38->39",    "| | | | | Lu | | ee |", "a-terisi"),
    ("standar-biaya-2023 18->19",  "| | | |", "a-terisi"),
    ("standar-biaya-2026 17->18",  "| | | 3", "a-terisi"),
    ("standar-biaya-2025 20->21",  "| 2 | | 3 4 | |", "a-terisi"),
    ("standar-biaya-2026 31->32",  "| ASIA TIMUR | | | |", "a-terisi"),
    ("kkn-covid 59->60",           "| Mensosialisasikan Pembelajaran", "a-terisi"),
]

SISA_RISIKO = [
    ("tata-cara-pembukuan 6->7",   "Tanggal Pembukuan | co & 2 KT MELEE"),
    ("SOP kop dokumen",            "PROGRAM STUDI TEKNIK INFORMATIKA | PROSEDUR"),
]


@pytest.mark.unit
@pytest.mark.parametrize("asal,baris,aturan", BOCOR_TERTANGKAP,
                         ids=[x[0] for x in BOCOR_TERTANGKAP])
def test_markup_tidak_otoritatif_bocor_kini_tertangkap(asal, baris, aturan):
    p = deteksi_header([sel(baris), ["x", "y"]], ada_th=True)
    assert not p, f"{asal} masih lolos: {p.alasan}"
    assert p.aturan.startswith(aturan), p.aturan


@pytest.mark.unit
@pytest.mark.parametrize("asal,baris", SISA_RISIKO, ids=[x[0] for x in SISA_RISIKO])
def test_sisa_risiko_yang_diterima_tetap_lolos(asal, baris):
    """Didokumentasikan sebagai sisa risiko: tidak ada aturan yang tidak mengarang
    untuk menolaknya. Uji ini mengunci bahwa keadaan itu disadari."""
    assert deteksi_header([sel(baris), ["a", "b"]], ada_th=True)


# ─── header markup yang BENAR harus tetap lolos ──────────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("baris", [
    "SERIES/JENIS ARSIP | AKTIF | INAKTIF | KETERANGAN",
    "| SERIES/JENIS ARSIP 2 | AKTIF 3 | INAKTIF 4 | KETERANGAN",   # satu sel kosong
    "No. | | Nama Duta K3 | Unit Kerja",                          # satu sel kosong
    "NO. | PROVINSI | SATUAN | RODA 4 | RODA 6/BUS SEDANG",
])
def test_header_markup_benar_lolos(baris):
    assert deteksi_header([sel(baris), ["1", "2", "3", "4"]], ada_th=True)


@pytest.mark.unit
def test_kontras_tidak_diterapkan_ke_markup():
    """Header jadwal retensi bertubuh TEKS. (d) akan membuang kedelapannya."""
    baris = [sel("SERIES/JENIS ARSIP | AKTIF | INAKTIF | KETERANGAN"),
             ["Surat Keputusan", "Dua tahun", "Lima tahun", "Permanen"],
             ["Notulen Rapat", "Dua tahun", "Tiga tahun", "Musnah"]]
    assert deteksi_header(baris, ada_th=True).aturan == "markup"
    assert not deteksi_header(baris, ada_th=False), "fallback memang menolak — (d)"


@pytest.mark.unit
def test_aturan_modal_markup_opsional_dan_mati_bawaan():
    baris = [sel("PROGRAM STUDI TEKNIK INFORMATIKA | PROSEDUR"),
             ["1", "Mengajukan", "Mahasiswa", "Formulir"],
             ["2", "Verifikasi", "Admin", "Berkas"]]
    assert deteksi_header(baris, ada_th=True)
    p = deteksi_header(baris, ada_th=True, modal_untuk_markup=True)
    assert not p and p.aturan == "a-modal/markup"


# ─── jalur fallback (tanpa markup) ───────────────────────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("kandidat,tubuh", [
    (["921112", "BELANJA PENGADAAN BAHAN MAKANAN"], [["921113", "BELANJA OBAT"]]),
    (["4141", "ALOKASI BELANJA APBN LAINNYA"], [["4142", "ALOKASI BELANJA MODAL"]]),
])
def test_fallback_menolak_kode_akun(kandidat, tubuh):
    assert not deteksi_header([kandidat, *tubuh])


@pytest.mark.unit
def test_fallback_glosarium_ditolak_lewat_kontras():
    p = deteksi_header([["SLA", "Subsidiary Loan Agreement"],
                        ["DIPA", "Daftar Isian Pelaksanaan Anggaran"]])
    assert not p and p.aturan == "d-kontras"


@pytest.mark.unit
def test_fallback_header_berkolom_numerik_lolos():
    p = deteksi_header([["NO.", "SERIES/JENIS ARSIP", "AKTIF", "INAKTIF", "KETERANGAN"],
                        ["1", "Surat Keputusan Rektor", "2", "5", "Permanen"]])
    assert p and p.aturan == "d-kontras"


@pytest.mark.unit
def test_fallback_satu_sel_kosong_kini_boleh():
    """Aturan lama menolak SEMUA sel kosong; kini cukup lebih dari separuh terisi."""
    assert deteksi_header([["NO.", "", "NAMA", "UNIT"], ["1", "", "Andi", "FT"],
                           ["2", "", "Budi", "FK"]])


@pytest.mark.unit
@pytest.mark.parametrize("jam,lolos", [
    ("08.00", True), ("08.00 - 10.00", True), ("08:00-10:00", True),
    ("08.00 WITA", False), ("Pagi", False),
])
def test_jadwal_kkn_fallback_bergantung_kolom_jam(jam, lolos):
    kkn = ["HARI/TGL", "JAM", "KEGIATAN", "KETERANGAN"]
    p = deteksi_header([kkn, ["Senin, 12 Juli", jam, "Registrasi", "Gedung A"],
                        ["Selasa, 13 Juli", jam, "Kuliah", "Aula"]])
    assert bool(p) is lolos


@pytest.mark.unit
@pytest.mark.parametrize("baris,awalan", [
    ([], "bentuk"),
    ([["satu"]], "a-bentuk"),
    ([["A", ""], ["1", "2"]], "a-terisi"),
    ([["A", "B", "C"], ["1", "2"], ["3", "4"]], "a-bentuk"),
])
def test_bentuk_tak_sah(baris, awalan):
    p = deteksi_header(baris)
    assert not p and p.aturan.startswith(awalan)


# ─── pewarisan rantai ────────────────────────────────────────────────────────

@pytest.mark.unit
def test_rantai_kkn_mewarisi_dari_kepala():
    """Potongan tengah rantai baris pertamanya kosong; header asli di kepala."""
    pas = [(f"kkn_p{n}_c00", f"kkn_p{n+1}_c00") for n in range(26, 35)]
    k = kepala_rantai(pas)
    assert all(k[b] == "kkn_p26_c00" for _, b in pas)
    assert k["kkn_p26_c00"] == "kkn_p26_c00"


@pytest.mark.unit
def test_rantai_putus_memulai_kepala_baru():
    k = kepala_rantai([("d_p1_c00", "d_p2_c00"), ("d_p5_c00", "d_p6_c00")])
    assert k["d_p2_c00"] == "d_p1_c00" and k["d_p6_c00"] == "d_p5_c00"


@pytest.mark.unit
def test_rantai_urutan_masukan_tidak_berpengaruh():
    pas = [("d_p3_c00", "d_p4_c00"), ("d_p1_c00", "d_p2_c00"), ("d_p2_c00", "d_p3_c00")]
    assert set(kepala_rantai(pas).values()) == {"d_p1_c00"}


@pytest.mark.unit
def test_rantai_siklus_tidak_menggantung():
    assert kepala_rantai([("a", "b"), ("b", "a")])       # berhenti, tidak loop


@pytest.mark.unit
def test_putusan_immutable():
    p = deteksi_header([["A", "B"], ["1", "2"]])
    with pytest.raises(Exception):
        p.header = False
