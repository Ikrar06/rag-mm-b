"""Nilai kualitas lapisan teks sebuah dokumen.

Melengkapi `pdf_sumber`, yang hanya menjawab "ada atau tidak lapisan teks".
Ada kasus ketiga yang lolos dari pertanyaan itu: PDF yang PUNYA lapisan teks,
tetapi lapisan teksnya sendiri kacau — hasil pindai yang di-OCR lalu disimpan
sebagai PDF berteks. `pedoman-penyusunan-laporan-keuangan` menandai dirinya
"digital" padahal isinya "ATATAN ATAS LAPORAN KEUANGAN JUNI 2022 Umuk Tomggal".

Diukur atas TEKS HALAMAN, bukan atas sel tabel. Sel tabel berupa label pendek,
sehingga rasio kata fungsi padanya rendah secara alami dan tidak menandakan
kerusakan apa pun. Teks halaman cukup mirip prosa untuk dinilai.

BUKAN gerbang otomatis. Keluarannya jadi kolom di `table_continuation.json`
supaya peninjau tahu teks dokumen mana yang patut dicurigai. Ambang di bawah
adalah pilihan, bukan temuan — sub-indikator mentahnya selalu ikut dilaporkan.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

VOKAL = frozenset("aeiouAEIOU")

# Kelas tertutup bahasa Indonesia: daftarnya terbatas dan stabil, jadi bukan
# kamus yang perlu dipelihara. Dipakai sebagai ukuran CAKUPAN — prosa Indonesia
# yang sehat memuat kata-kata ini dalam proporsi yang cukup tetap.
KATA_FUNGSI = frozenset({
    "yang", "dan", "di", "ke", "dari", "untuk", "pada", "dengan", "ini", "itu",
    "atau", "dalam", "oleh", "akan", "tidak", "adalah", "sebagai", "telah",
    "dapat", "juga", "serta", "bahwa", "karena", "jika", "bila", "agar",
    "atas", "antara", "tersebut", "para", "setiap", "sudah", "belum", "masih",
    "harus", "kepada", "tentang", "hingga", "sampai", "namun",
    "the", "of", "and", "in", "to", "for", "on", "as", "at", "by", "with",
    "is", "are", "was", "were", "be", "been", "from", "that", "this",
})

# Ambang pelabelan. Pilihan, bukan temuan.
BATAS_RUSAK = 0.55
BATAS_CURIGA = 0.75
MIN_TOKEN = 30          # di bawah ini sampelnya terlalu kecil untuk dinilai

_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _kapital_campur(token: str) -> bool:
    """Token dengan pola kapital yang tidak wajar.

    Wajar: seluruhnya kecil ("kata"), seluruhnya besar ("IPK"), atau kapital
    di awal saja ("Universitas"). Selain itu anomali khas OCR — "LAporan",
    "UmUk", "KEuangan". Diperiksa sebagai tiga bentuk sah, bukan lewat regex
    pola rusak, karena bentuk rusaknya tak terbatas sedangkan yang sah hanya tiga.
    """
    if len(token) < 2 or not token.isalpha():
        return False
    return not (token.islower() or token.isupper() or
                (token[0].isupper() and token[1:].islower()))


def _tokenkan(teks: str) -> tuple[str, ...]:
    bersih = unicodedata.normalize("NFKC", teks or "")
    return tuple(_TOKEN_RE.findall(bersih))


@dataclass(frozen=True)
class Kualitas:
    """Penilaian kualitas teks. Immutable."""

    n_token: int = 0
    n_tanpa_vokal: int = 0
    n_terisolasi: int = 0
    n_kapital_campur: int = 0
    n_fungsi: int = 0

    @property
    def cukup_sampel(self) -> bool:
        return self.n_token >= MIN_TOKEN

    @property
    def rasio_tanpa_vokal(self) -> float:
        return self.n_tanpa_vokal / self.n_token if self.n_token else 0.0

    @property
    def rasio_terisolasi(self) -> float:
        return self.n_terisolasi / self.n_token if self.n_token else 0.0

    @property
    def rasio_kapital_campur(self) -> float:
        return self.n_kapital_campur / self.n_token if self.n_token else 0.0

    @property
    def rasio_fungsi(self) -> float:
        return self.n_fungsi / self.n_token if self.n_token else 0.0

    @property
    def skor(self) -> float | None:
        """0..1, makin tinggi makin sehat. None bila sampelnya terlalu kecil.

        Sengaja None, bukan 1.0. Mengembalikan 1.0 untuk sampel kecil berarti
        "tidak diketahui" terbaca sebagai "sehat", dan siapa pun yang mengurutkan
        dokumen berdasarkan skor akan menempatkannya di posisi teraman. None
        memaksa pemanggil menanganinya secara sadar.

        Tiga indikator kerusakan mengurangi skor; cakupan kata fungsi yang
        terlalu rendah juga, karena prosa Indonesia yang utuh selalu memuatnya.
        Bobotnya rata — tidak ada dasar empiris untuk membobotinya berbeda, dan
        mengarang bobot akan menyembunyikan bahwa ini heuristik.
        """
        if not self.cukup_sampel:
            return None
        rusak = (min(self.rasio_tanpa_vokal * 4, 1.0)
                 + min(self.rasio_terisolasi * 4, 1.0)
                 + min(self.rasio_kapital_campur * 4, 1.0)
                 + (1.0 if self.rasio_fungsi < 0.08 else 0.0))
        return max(0.0, 1.0 - rusak / 4)

    @property
    def label(self) -> str:
        """'baik' | 'patut_dicurigai' | 'rusak' | 'sampel_kecil'."""
        skor = self.skor
        if skor is None:
            return "sampel_kecil"
        if skor < BATAS_RUSAK:
            return "rusak"
        if skor < BATAS_CURIGA:
            return "patut_dicurigai"
        return "baik"


def nilai_teks(teks: str) -> Kualitas:
    """Hitung indikator kerusakan OCR atas sepotong teks. Fungsi murni."""
    token = _tokenkan(teks)
    if not token:
        return Kualitas()

    return Kualitas(
        n_token=len(token),
        n_tanpa_vokal=sum(1 for t in token if len(t) >= 3 and not (set(t) & VOKAL)),
        n_terisolasi=sum(1 for t in token if len(t) <= 2),
        n_kapital_campur=sum(1 for t in token if _kapital_campur(t)),
        n_fungsi=sum(1 for t in token if t.lower() in KATA_FUNGSI),
    )


def gabung(bagian) -> Kualitas:
    """Jumlahkan penilaian beberapa halaman jadi satu penilaian dokumen.

    Menjumlahkan CACAHAN, bukan merata-rata rasio: halaman pendek tidak boleh
    berbobot sama dengan halaman penuh.
    """
    total = Kualitas()
    for k in bagian:
        total = Kualitas(
            n_token=total.n_token + k.n_token,
            n_tanpa_vokal=total.n_tanpa_vokal + k.n_tanpa_vokal,
            n_terisolasi=total.n_terisolasi + k.n_terisolasi,
            n_kapital_campur=total.n_kapital_campur + k.n_kapital_campur,
            n_fungsi=total.n_fungsi + k.n_fungsi,
        )
    return total
