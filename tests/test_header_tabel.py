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
    ("standar-biaya-2026 17->18",  "| | | 3", "b-numerik"),
    ("standar-biaya-2025 20->21",  "| 2 | | 3 4 | |", "b-numerik"),
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


# ─── keadaan rantai: tidak diketahui vs terbukti data ────────────────────────

from lib.header_tabel import (  # noqa: E402
    STATUS_DIKETAHUI, STATUS_MATI, STATUS_TIDAK_DIKETAHUI, lanjutkan_rantai,
)
from lib.tabel_html import urai  # noqa: E402


def T(baris, th=True):
    h = "<thead><tr>" + "".join(f"<th>{c}</th>" for c in baris[0]) + "</tr></thead>" if th else ""
    body = baris[1:] if th else baris
    return urai("<table>" + h + "<tbody>" + "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in body) + "</tbody></table>")


def jalankan(potongan):
    """potongan: list (chunk_id, Tabel|None). Kembalikan keadaan setelah tiap potongan."""
    k, out = None, []
    for cid, t in potongan:
        p = deteksi_header(t.baris, t.ada_th, t.ada_thead) if t else None
        k = lanjutkan_rantai(k, cid, t, p)
        out.append(k)
    return out


AKTIVITAS = [["Aktivitas/Subaktivitas", "Volume", "Menit", "Total", "Bobot"],
             ["Persiapan", "2", "240", "480", "1"]]


@pytest.mark.unit
def test_kepala_tak_terurai_lalu_header_sah_menjadi_header_rantai():
    """rubrik 39->40: kepala p38_c07 tak terurai, header sah ada di potongan kedua."""
    k = jalankan([("r_p38_c07", None), ("r_p39_c00", T(AKTIVITAS)),
                  ("r_p40_c00", T([["", "", "", "", ""], ["Laporan", "1", "60", "60", "1"]]))])
    assert k[0].status == STATUS_TIDAK_DIKETAHUI
    assert k[1].status == STATUS_DIKETAHUI and k[1].sumber == "r_p39_c00"
    assert k[2].header == tuple(AKTIVITAS[0])       # diwarisi ke potongan ketiga


@pytest.mark.unit
def test_kepala_tak_terurai_lalu_baris_data_mematikan_rantai():
    """bagan-akun: kepala tak terurai, potongan terurai pertama 4141 — data."""
    k = jalankan([("a_p5_c00", None),
                  ("a_p6_c00", T([["4141", "ALOKASI"], ["4142", "MODAL"]])),
                  ("a_p7_c00", T([["Kode", "Uraian"], ["1", "x"]]))])   # tampak sah
    assert k[1].status == STATUS_MATI and k[1].sumber == "a_p6_c00"
    assert k[2].status == STATUS_MATI, "baris tampak sah di tengah tidak menghidupkan rantai"


@pytest.mark.unit
def test_kepala_ditolak_mematikan_seluruh_rantai():
    k = jalankan([("d_p1_c00", T([["78", "", "Padang", "Kota", "215.000"], ["1", "", "a", "b", "2"]])),
                  ("d_p2_c00", T([["NO.", "PROVINSI", "SATUAN"], ["1", "Aceh", "hari"]]))])
    assert all(x.status == STATUS_MATI for x in k)


@pytest.mark.unit
def test_header_diketahui_tidak_tertimpa_potongan_berikutnya():
    k = jalankan([("k_p26_c00", T([["HARI/TGL", "JAM", "KEGIATAN", "KETERANGAN"],
                                   ["Senin", "08.00", "Registrasi", "A"]])),
                  ("k_p27_c00", T([["", "", "", ""], ["Rabu", "09.00", "Kuliah", "B"]])),
                  ("k_p28_c00", None)])
    assert {x.header[0] for x in k} == {"HARI/TGL"}


@pytest.mark.unit
def test_rantai_seluruhnya_tak_terurai_tetap_tidak_diketahui():
    k = jalankan([("x_p1_c00", None), ("x_p2_c00", None)])
    assert all(x.status == STATUS_TIDAK_DIKETAHUI and x.header is None for x in k)


# ─── aturan (e) panjang sel ──────────────────────────────────────────────────

@pytest.mark.unit
def test_aturan_panjang_sel_mati_bawaan_dan_menolak_bila_dinyalakan():
    """kkn-covid 59->60: sel berupa kalimat panjang, lolos markup."""
    baris = [["No", "Mensosialisasikan Pembelajaran yang efektif pada Melakukan kegiatan"],
             ["1", "x"]]
    assert deteksi_header(baris, ada_th=True)
    p = deteksi_header(baris, ada_th=True, maks_panjang_sel=40)
    assert not p and p.aturan == "e-panjang/markup"


# ─── hanya penolakan yang membuktikan data yang mematikan rantai ─────────────

from lib.header_tabel import membuktikan_data  # noqa: E402


@pytest.mark.unit
def test_judul_tahap_satu_sel_netral_lalu_header_sah_jadi_header_rantai():
    """rubrik 39->40: p38_c08 berisi judul tahap satu sel di dalam tabel."""
    tahap = T([["Tahap 1: Pembinaan dan Penyusunan Usulan Konsep Desain"],
               ["Persiapan"]], th=True)
    k = jalankan([("r_p38_c08", tahap), ("r_p39_c00", T(AKTIVITAS)),
                  ("r_p40_c05", T([["", "", "", "", ""], ["Laporan", "1", "60", "60", "1"]]))])
    assert k[0].status == STATUS_TIDAK_DIKETAHUI, "satu sel bukan bukti baris data"
    assert k[1].status == STATUS_DIKETAHUI and k[2].header == tuple(AKTIVITAS[0])


@pytest.mark.unit
def test_mayoritas_kosong_netral():
    k = jalankan([("x_p1_c00", T([["", "", "Judul", ""], ["a", "b", "c", "d"]])),
                  ("x_p2_c00", T([["NO.", "PROVINSI", "SATUAN", "RODA 4"], ["1", "a", "b", "2"]]))])
    assert k[0].status == STATUS_TIDAK_DIKETAHUI and k[1].status == STATUS_DIKETAHUI


@pytest.mark.unit
def test_bagan_akun_4_pendapatan_tetap_mematikan_rantai():
    """Potongan terurai pertama '4 | | PENDAPATAN' — b-numerik, bukti data."""
    k = jalankan([("a_p5_c00", None),
                  ("a_p6_c00", T([["4", "", "PENDAPATAN"], ["41", "", "PENDAPATAN ASLI"]])),
                  ("a_p7_c00", T([["Kode", "Nama", "Uraian"], ["1", "x", "y"]]))])
    assert k[1].status == STATUS_MATI and k[2].status == STATUS_MATI


@pytest.mark.unit
def test_kkn_covid_panjang_sel_membuktikan_data_dan_memutus_warisan():
    """Dengan batas 80, kepala 59->60 (sel 161 karakter) tertolak sebagai data."""
    kalimat = ("Mensosialisasikan Pembelajaran yang efektif pada Melakukan kegiatan "
               "pendampingan belajar siswa selama masa pandemi dengan memperhatikan "
               "protokol kesehatan yang berlaku di lingkungan sekolah")
    t = T([["", kalimat, "x"], ["1", "a", "b"]])
    p = deteksi_header(t.baris, t.ada_th, t.ada_thead, maks_panjang_sel=80)
    assert not p and p.aturan == "e-panjang/markup" and membuktikan_data(p)
    k = lanjutkan_rantai(None, "kkn_p59_c00", t, p)
    assert k.status == STATUS_MATI


@pytest.mark.unit
@pytest.mark.parametrize("aturan,bukti", [
    ("b-numerik/markup", True), ("c-kode/fallback", True), ("d-kontras", True),
    ("e-panjang/markup", True), ("a-bentuk/markup", False), ("a-terisi/fallback", False),
    ("a-modal/markup", False), ("bentuk", False),
])
def test_klasifikasi_bukti_data(aturan, bukti):
    from lib.header_tabel import Putusan
    assert membuktikan_data(Putusan(False, aturan)) is bukti


@pytest.mark.unit
def test_header_lolos_tidak_pernah_membuktikan_data():
    from lib.header_tabel import Putusan
    assert not membuktikan_data(Putusan(True, "markup")) and not membuktikan_data(None)


@pytest.mark.unit
def test_bawaan_batas_panjang_sel_80():
    import importlib, os
    os.environ.pop("TABLE_HEADER_MAX_CELL_CHARS", None)
    import backend.config as cfg
    importlib.reload(cfg)
    assert cfg.TABLE_HEADER_MAX_CELL_CHARS == 80


# ─── urutan: bukti data tidak tertutup penolakan netral ──────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("markup", [True, False])
def test_numerik_di_baris_mayoritas_kosong_tetap_bukti_data(markup):
    """'4 | | | PENDAPATAN': dua dari empat sel terisi. a-terisi akan menyala
    bila diperiksa lebih dulu dan membuatnya netral — sel '4' tetap bukti data."""
    baris = [["4", "", "", "PENDAPATAN"], ["41", "", "", "PENDAPATAN ASLI"]]
    p = deteksi_header(baris, ada_th=markup)
    assert p.aturan.startswith("b-numerik"), p.aturan
    assert membuktikan_data(p)


@pytest.mark.unit
def test_rantai_mati_oleh_4_pendapatan_bermayoritas_kosong():
    k = jalankan([("a_p5_c00", None),
                  ("a_p6_c00", T([["4", "", "", "PENDAPATAN"], ["41", "", "", "ASLI"]])),
                  ("a_p7_c00", T([["Kode", "Nama", "Uraian"], ["1", "x", "y"]]))])
    assert k[1].status == STATUS_MATI and k[2].status == STATUS_MATI


@pytest.mark.unit
def test_satu_sel_numerik_bukti_data_satu_sel_teks_netral():
    assert deteksi_header([["5"], ["6"]]).aturan.startswith("b-numerik")
    assert deteksi_header([["Tahap 1: Pembinaan"], ["x"]]).aturan.startswith("a-bentuk")


@pytest.mark.unit
def test_sel_panjang_di_baris_mayoritas_kosong_tetap_bukti_data():
    p = deteksi_header([["", "", "x" * 120, ""], ["1", "2", "3", "4"]], ada_th=True,
                       maks_panjang_sel=80)
    assert p.aturan == "e-panjang/markup" and membuktikan_data(p)
