"""Pulihkan batas kolom sebuah tabel dari posisi kata di halaman PDF.

Kenapa dari PyMuPDF dan bukan dari metadata Unstructured
--------------------------------------------------------
`element.metadata.coordinates.points` hanya memuat EMPAT SUDUT persegi panjang
element (`partition/pdf.py:475`), jadi untuk element Table ia kotak seluruh
tabel — tidak ada koordinat per kolom. `table_as_cells` pun tidak menolong:
`SimpleTableCell` menyimpan `x, y, w, h` sebagai INDEKS GRID dan span, bukan
piksel (`metrics/table/table_formats.py`), dan bbox sel dari Table Transformer
dibuang saat konversi.

Satu-satunya sumber posisi kolom yang sebenarnya adalah lapisan teks PDF.
`page.get_text("words")` memberi bbox per kata; celah vertikal yang tidak
dilewati satu kata pun adalah pemisah kolom.

Batasnya: ini HANYA bekerja pada PDF yang punya lapisan teks. Pada hasil
pindai, `get_text("words")` kosong dan fungsi ini mengembalikan None —
"tidak tersedia", bukan "tidak ada kolom". Pemanggil wajib membedakan keduanya.

Logika murni dipisah dari I/O supaya dapat diuji tanpa PDF.
"""

from __future__ import annotations

from dataclasses import dataclass

# Resolusi proyeksi sumbu-x. 200 bin atas lebar tabel ~ 0,5% lebar per bin,
# cukup halus untuk memisahkan kolom tanpa terpancing spasi antar kata.
N_BIN = 200

# Celah minimum yang dianggap pemisah kolom, dalam satuan bin. Tiga bin ~ 1,5%
# lebar tabel. Spasi antar kata dalam satu sel jauh lebih sempit dari itu.
MIN_BIN_CELAH = 3

# Toleransi saat membandingkan dua himpunan batas kolom, dalam satuan lebar
# tabel ternormalisasi. Dua potongan tabel yang sama tidak akan identik sampai
# digit terakhir karena OCR dan pembulatan layout.
TOLERANSI = 0.02


@dataclass(frozen=True)
class BatasKolom:
    """Batas kolom ternormalisasi terhadap lebar tabel (0..1). Immutable."""

    batas: tuple[float, ...] = ()
    n_kata: int = 0
    tersedia: bool = False

    @property
    def n_kolom(self) -> int:
        """Jumlah kolom = jumlah pemisah + 1."""
        return len(self.batas) + 1 if self.tersedia else 0


def batas_dari_kata(
    kata_x: tuple[tuple[float, float], ...],
    x0_tabel: float,
    x1_tabel: float,
) -> BatasKolom:
    """Hitung batas kolom dari rentang-x kata yang ada di dalam tabel.

    `kata_x` adalah pasangan (x0, x1) tiap kata dalam satuan halaman. Fungsi
    murni: tidak membuka berkas, tidak mengubah argumennya.

    Mengembalikan `tersedia=False` bila tidak ada kata sama sekali — itu
    kondisi "halaman tanpa lapisan teks", bukan "tabel tanpa kolom".
    """
    lebar = x1_tabel - x0_tabel
    if lebar <= 0 or not kata_x:
        return BatasKolom(n_kata=len(kata_x), tersedia=False)

    terisi = [False] * N_BIN
    n_dipakai = 0
    for x0, x1 in kata_x:
        if x1 < x0_tabel or x0 > x1_tabel:
            continue
        n_dipakai += 1
        b0 = max(0, min(N_BIN - 1, int((x0 - x0_tabel) / lebar * N_BIN)))
        b1 = max(0, min(N_BIN - 1, int((x1 - x0_tabel) / lebar * N_BIN)))
        for b in range(b0, b1 + 1):
            terisi[b] = True

    if not n_dipakai:
        return BatasKolom(n_kata=0, tersedia=False)

    # Celah di TEPI tabel bukan pemisah kolom — itu margin. Hanya celah yang
    # diapit bin terisi di kedua sisi yang dihitung.
    batas: list[float] = []
    i = 0
    while i < N_BIN:
        if terisi[i]:
            i += 1
            continue
        awal = i
        while i < N_BIN and not terisi[i]:
            i += 1
        if awal == 0 or i >= N_BIN:
            continue                      # celah tepi
        if i - awal >= MIN_BIN_CELAH:
            batas.append((awal + i) / 2 / N_BIN)

    return BatasKolom(batas=tuple(round(b, 4) for b in batas),
                      n_kata=n_dipakai, tersedia=True)


def mirip(a: BatasKolom, b: BatasKolom, toleransi: float = TOLERANSI) -> bool | None:
    """Apakah dua himpunan batas kolom cocok.

    None berarti TIDAK DAPAT DINILAI — salah satu potongan tidak punya lapisan
    teks. Pemanggil harus meneruskannya ke adjudikasi vision, bukan
    memperlakukannya sebagai ketidakcocokan.
    """
    if not a.tersedia or not b.tersedia:
        return None
    if len(a.batas) != len(b.batas):
        return False
    return all(abs(x - y) <= toleransi for x, y in zip(a.batas, b.batas))


def batas_dari_halaman(pdf_path, nomor_halaman: int, bbox_ternormalisasi) -> BatasKolom:
    """Buka satu halaman PDF dan hitung batas kolom di dalam bbox tabel.

    `bbox_ternormalisasi` adalah [x0, y0, x1, y1] dalam 0..1 seperti yang
    diproduksi `preprocessing._element_bbox`.

    Kegagalan membuka berkas TIDAK melempar — ia jadi `tersedia=False`, supaya
    satu PDF rusak tidak menjatuhkan pengukuran seluruh korpus.
    """
    import fitz

    if not bbox_ternormalisasi or len(bbox_ternormalisasi) != 4:
        return BatasKolom()
    try:
        with fitz.open(str(pdf_path)) as doc:
            if not 1 <= nomor_halaman <= doc.page_count:
                return BatasKolom()
            page = doc[nomor_halaman - 1]
            r = page.rect
            x0 = r.x0 + bbox_ternormalisasi[0] * r.width
            y0 = r.y0 + bbox_ternormalisasi[1] * r.height
            x1 = r.x0 + bbox_ternormalisasi[2] * r.width
            y1 = r.y0 + bbox_ternormalisasi[3] * r.height
            kata = tuple(
                (w[0], w[2]) for w in page.get_text("words")
                if w[1] >= y0 - 1 and w[3] <= y1 + 1
            )
    except Exception:
        return BatasKolom()
    return batas_dari_kata(kata, x0, x1)
