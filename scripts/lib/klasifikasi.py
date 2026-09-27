"""Klasifikasi gambar untuk transkripsi tabel: prompt, pengurai jawaban, dan
aturan perlakuan. Fungsi murni; pemanggilan model ada di scripts/lib/ukur_io.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Klasifikasi versi 3: DUA jawaban dalam satu panggilan.
# - jenis (tabel / cap / lainnya), "jika ragu" ke lainnya. Versi 1 condong ke
#   tabel dan mentranskripsi tangkapan layar, flowchart bersiku, dan matriks
#   logo sehingga alur dan petunjuk antarmuka hilang.
# - memuat_tabel_data, apa pun jenisnya. Versi 2 memaksa satu label sehingga
#   tabel sungguhan di halaman campuran ikut terbuang ke narasi (laporan
#   ekuitas laporan-keuangan p23, tabel penilaian sop-final-project p7, 56
#   gambar tabel -> lainnya). Dengan jawaban kedua, gambar yang keduanya
#   mendapat narasi DAN transkripsi; tidak ada arah salah yang menghapus isi.
PROMPT_KLASIFIKASI = """Gambar ini diambil dari dokumen PDF. Jawab DUA hal.

1. "jenis":
"cap": cap atau segel tanda tangan elektronik, yaitu logo Balai Sertifikasi
Elektronik (BSrE) bersama catatan "Dokumen ini telah ditandatangani secara
elektronik". Tetap jawab cap walaupun sebagian besar gambar berisi baris tabel
di belakangnya. Contoh: potongan tabel biaya UKT yang di tengahnya ada logo
BSrE dan catatan UU ITE adalah cap.

"tabel": gambar yang HANYA berisi tabel data, yaitu baris dan kolom berisi teks
atau angka: tabel biaya, daftar kode dan uraian, rekap angka, jadwal, termasuk
tabel hasil pindai.

"lainnya": semua yang lain, WALAUPUN berbentuk kisi atau memuat tabel di
dalamnya: tangkapan layar aplikasi, situs web, atau formulir online;
flowchart, bagan alir, dan bagan bersiku kolom (swimlane); panduan logo,
matriks logo, dan kumpulan ikon; surat, pernyataan, atau halaman teks yang
memiliki garis kotak; halaman berisi judul, paragraf, atau tanda tangan
bersama tabel; diagram, foto, logo lembaga, dan kode QR.
Jika ragu, jawab lainnya.

2. "memuat_tabel_data", APA PUN jenisnya:
true bila gambar berisi tabel data, yaitu baris dan kolom yang isinya teks
atau angka yang perlu disalin. Contoh: tangkapan layar yang menampilkan tabel
data, halaman pindai berisi judul dan tabel, laporan keuangan berkolom angka
walau tanpa garis, tabel SWOT, tabel rencana kerja, tabel penilaian.
false bila kisi hanya tata letak: kotak dan panah flowchart, lajur swimlane,
susunan logo atau ikon, kotak isian formulir kosong, bingkai surat.

Jawab HANYA dengan JSON satu baris, contoh:
{"jenis": "lainnya", "memuat_tabel_data": true}"""

JENIS_GAMBAR = ("tabel", "cap", "lainnya")
_JENIS_RE = re.compile(r'"jenis"\s*:\s*"([a-z]+)"')
_MEMUAT_RE = re.compile(r'"memuat_tabel_data"\s*:\s*(true|false)')

# Cap dibuang hanya bila sebagian besar luasnya di dalam Table. CALON: di run
# v2, rasio cap yang bertumpang tabel punya celah 0,023 -> 0,948
# (pengelolaan-dana-kelas-internasional p9_c02 vs UKT p5-p10). Ditetapkan
# setelah sebaran seluruh cap run v3 terlihat.
AMBANG_CAP = 0.5


@dataclass(frozen=True)
class Klasifikasi:
    jenis: str | None                   # tabel | cap | lainnya | None (tak dikenali)
    memuat_tabel_data: bool | None      # None bila jawaban tak memuatnya

    @property
    def label(self) -> str:
        """Label ringkas untuk matriks perbandingan: "lainnya+tabel", "cap", ..."""
        return f"{self.jenis}{'+tabel' if self.memuat_tabel_data else ''}"


def urai_klasifikasi(raw: str | None) -> Klasifikasi:
    """Jawaban klasifikasi -> Klasifikasi. Bagian yang tak dikenali jadi None."""
    teks = (raw or "").lower()
    m = _JENIS_RE.search(teks)
    b = _MEMUAT_RE.search(teks)
    return Klasifikasi(m.group(1) if m and m.group(1) in JENIS_GAMBAR else None,
                       (b.group(1) == "true") if b else None)


def perlakuan_gambar(k: Klasifikasi, rasio_tumpang: float, ambang_cap: float = AMBANG_CAP) -> str:
    """buang | transkripsi | narasi+tabel | narasi.

    Urutan aturannya:
    1. cap yang sebagian besar di dalam Table dibuang (isinya sudah ada di
       transkripsi tabel itu);
    2. gambar lain yang bertumpang Table tetap dinarasikan seperti v4 —
       mentranskripsinya menggandakan isi Table;
    3. jenis tabel ditranskripsi;
    4. gambar yang memuat tabel data (lainnya, atau cap yang berdiri sendiri)
       mendapat narasi dan transkripsi;
    5. sisanya, termasuk jawaban tak dikenali, dinarasikan seperti v4.
    """
    if k.jenis == "cap" and rasio_tumpang >= ambang_cap:
        return "buang"
    if rasio_tumpang > 0:
        return "narasi"
    if k.jenis == "tabel":
        return "transkripsi"
    if k.jenis is not None and k.memuat_tabel_data:
        return "narasi+tabel"
    return "narasi"
