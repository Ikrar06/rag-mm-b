"""Bedakan PDF digital dari PDF hasil pindai, per dokumen dan per halaman.

Dipakai untuk dua keputusan sekaligus:

1. Apakah batas kolom lewat PyMuPDF layak dikerjakan. `page.get_text("words")`
   hanya memberi kata pada PDF yang punya lapisan teks; pada hasil pindai ia
   kosong, sehingga sinyal itu mustahil di sana.
2. Penyaringan kualitas OCR per dokumen. Isi sel tabel pada dokumen pindai
   berasal dari OCR, jadi kerusakannya sistemik per dokumen — bukan per pasangan.

Ambangnya BUKAN karangan baru: `AMBANG_KARAKTER` menyalin
`preprocessing.OCR_TEXT_THRESHOLD_CHARS`, aturan yang sudah dipakai pipeline
untuk memutuskan sebuah halaman perlu di-OCR atau tidak. Memakai ambang berbeda
di sini berarti mengukur sesuatu yang bukan perilaku pipeline.

Logika murni dipisah dari I/O supaya dapat diuji tanpa PDF.
"""

from __future__ import annotations

from dataclasses import dataclass

# Cermin preprocessing.OCR_TEXT_THRESHOLD_CHARS (backend/services/preprocessing.py:201).
# Halaman dengan teks <= ini dianggap tidak punya lapisan teks yang berguna.
AMBANG_KARAKTER = 50

# Titik potong putusan tingkat dokumen. Eksplisit sebagai konstanta karena ini
# pilihan, bukan temuan — proporsi mentahnya selalu ikut dilaporkan supaya
# putusan ini dapat diabaikan pembaca.
BATAS_DIGITAL = 0.90
BATAS_PINDAI = 0.10


@dataclass(frozen=True)
class ProfilDokumen:
    """Profil sumber teks sebuah dokumen. Immutable."""

    file_name: str
    n_halaman: int = 0
    halaman_berteks: frozenset[int] = frozenset()
    halaman_tabel: tuple[int, ...] = ()
    error: str = ""

    @property
    def n_halaman_berteks(self) -> int:
        return len(self.halaman_berteks)

    @property
    def n_tabel_berteks(self) -> int:
        return sum(1 for h in self.halaman_tabel if h in self.halaman_berteks)

    def halaman_berlapis_teks(self, halaman: int) -> bool:
        """Apakah satu halaman punya lapisan teks. Dipakai silang-tabulasi
        antara kategori pasangan dan sumber halamannya."""
        return halaman in self.halaman_berteks

    @property
    def rasio_berteks(self) -> float:
        return self.n_halaman_berteks / self.n_halaman if self.n_halaman else 0.0

    @property
    def rasio_tabel_berteks(self) -> float:
        """Rasio pada HALAMAN BERTABEL saja.

        Ini yang menentukan, bukan rasio seluruh dokumen: sebuah dokumen bisa
        90% digital sementara justru halaman tabelnya sisipan hasil pindai.
        """
        n = len(self.halaman_tabel)
        return self.n_tabel_berteks / n if n else 0.0

    @property
    def sumber(self) -> str:
        """'digital' | 'pindai' | 'campuran' | 'tak_terbaca'."""
        if self.error or not self.n_halaman:
            return "tak_terbaca"
        r = self.rasio_berteks
        if r >= BATAS_DIGITAL:
            return "digital"
        if r <= BATAS_PINDAI:
            return "pindai"
        return "campuran"

    @property
    def sumber_halaman_tabel(self) -> str:
        """Putusan yang sama, dihitung hanya atas halaman bertabel."""
        if self.error or not self.halaman_tabel:
            return "tak_terbaca"
        r = self.rasio_tabel_berteks
        if r >= BATAS_DIGITAL:
            return "digital"
        if r <= BATAS_PINDAI:
            return "pindai"
        return "campuran"


def profil_dari_cacah(
    file_name: str,
    char_per_halaman: dict[int, int],
    halaman_tabel: tuple[int, ...] = (),
) -> ProfilDokumen:
    """Bangun profil dari cacahan karakter per halaman (1-indexed).

    Fungsi murni — tidak membuka berkas. Dipisah supaya aturan klasifikasinya
    dapat diuji tanpa PDF.
    """
    return ProfilDokumen(
        file_name=file_name,
        n_halaman=len(char_per_halaman),
        halaman_berteks=frozenset(
            h for h, n in char_per_halaman.items() if n > AMBANG_KARAKTER
        ),
        halaman_tabel=tuple(sorted(set(halaman_tabel))),
    )


def profil_dari_pdf(path, halaman_tabel: tuple[int, ...] = ()) -> ProfilDokumen:
    """Buka PDF dan hitung profilnya. Satu-satunya fungsi yang menyentuh disk.

    Kegagalan membuka berkas TIDAK melempar — ia jadi `error` pada profil,
    supaya satu PDF rusak tidak menjatuhkan pengukuran 79 dokumen.
    """
    import fitz

    nama = getattr(path, "name", str(path))
    try:
        with fitz.open(str(path)) as doc:
            cacah = {i + 1: len(doc[i].get_text().strip()) for i in range(doc.page_count)}
    except Exception as e:
        return ProfilDokumen(file_name=nama, halaman_tabel=tuple(sorted(set(halaman_tabel))),
                             error=f"{type(e).__name__}: {e}")
    return profil_dari_cacah(nama, cacah, halaman_tabel)
