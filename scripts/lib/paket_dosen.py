"""Bagian paket eval untuk peninjauan langsung oleh dosen: semua_chunk.csv yang
terbuka benar di Excel dan Google Sheets, path gambar yang cocok setelah folder
`gambar/` diunggah utuh ke Google Drive, README_paket.md, dan ukuran paket.

Fungsi murni kecuali `tulis_semua_chunk` dan `ukuran_folder`.
"""

from __future__ import annotations

import csv
from pathlib import Path

from backend.services.modality import modality_dari

KOLOM_SEMUA_CHUNK = ("chunk_id", "document_id", "judul_dokumen", "halaman", "modality",
                     "element_type", "table_source", "image_id", "path_gambar", "text_content")
# Batas isi satu sel Excel. Sel yang lebih panjang terpotong saat dibuka di Excel;
# tidak dipotong di sini (chunks_jsonl tetap sumber otoritatif), hanya dihitung.
BATAS_SEL_EXCEL = 32767
# Awalan yang dibaca Excel/Sheets sebagai rumus ("- Database" -> #NAME?).
_AWALAN_RUMUS = ("=", "+", "-", "@", "\t", "\r")


def modality_chunk(c: dict) -> str:
    """modality dari dump, atau diturunkan dari element_type untuk dump lama."""
    return c.get("modality") or modality_dari(c.get("element_type"))


def judul_dokumen(file_name: str | None) -> str:
    """Judul dari nama berkas PDF. Registry belum punya judul terkurasi (lapis 4)."""
    nama = Path(file_name or "").name
    return nama[:-4] if nama.lower().endswith(".pdf") else nama


def path_gambar(image_id: str | None, file_path_per_image: dict[str, str]) -> str:
    """'gambar/<document_id>/pN_imgNN.ext' — sama dengan letak salinan di paket."""
    fp = file_path_per_image.get(image_id or "")
    return "gambar/" + fp.replace("images/", "", 1) if fp else ""


def aman_spreadsheet(teks: str | None) -> str:
    """Cegah sel dibaca sebagai rumus: awali apostrof bila diawali = + - @."""
    teks = teks or ""
    return "'" + teks if teks.startswith(_AWALAN_RUMUS) else teks


def baris_semua_chunk(chunks: list[dict], images: list[dict]) -> list[dict]:
    """Satu baris per chunk, urut dokumen lalu halaman lalu chunk_id."""
    fp = {i["image_id"]: i.get("file_path") or "" for i in images if i.get("image_id")}
    urut = sorted(chunks, key=lambda c: (c.get("document_id") or "", c.get("page_number") or 0,
                                         c.get("chunk_id") or ""))
    return [{
        "chunk_id": c.get("chunk_id") or "",
        "document_id": c.get("document_id") or "",
        "judul_dokumen": aman_spreadsheet(judul_dokumen(c.get("file_name"))),
        "halaman": "" if c.get("page_number") is None else c["page_number"],
        "modality": modality_chunk(c),
        "element_type": c.get("element_type") or "",
        "table_source": c.get("table_source") or "",
        "image_id": c.get("image_id") or "",
        "path_gambar": path_gambar(c.get("image_id"), fp),
        "text_content": aman_spreadsheet(c.get("text_content")),
    } for c in urut]


def tulis_semua_chunk(path: Path, baris: list[dict]) -> dict:
    """Tulis CSV (UTF-8 dengan BOM, koma, kutip standar). -> ringkasan."""
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KOLOM_SEMUA_CHUNK, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        w.writerows(baris)
    return {
        "baris": len(baris),
        "sel_melebihi_batas_excel": sum(1 for b in baris if len(b["text_content"]) > BATAS_SEL_EXCEL),
        "sel_diberi_apostrof": sum(1 for b in baris if b["text_content"].startswith("'")),
        "per_modality": {m: sum(1 for b in baris if b["modality"] == m)
                         for m in ("text", "table", "image")},
    }


def ukuran_folder(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def format_ukuran(n: int) -> str:
    for satuan in ("B", "KB", "MB", "GB"):
        if n < 1024 or satuan == "GB":
            return f"{n:.0f} {satuan}" if satuan == "B" else f"{n:.1f} {satuan}"
        n /= 1024
    return f"{n:.1f} GB"


def readme_paket(ringkas: dict, sumber: str, collection: str) -> str:
    pm = ringkas["per_modality"]
    return f"""# Paket Chunk dan Gambar — v5

Paket ini berisi seluruh potongan teks (chunk) dan gambar yang dipakai sistem
pencarian, dalam bentuk yang bisa dibuka langsung tanpa perangkat lunak khusus.

| | |
|---|---|
| Jumlah chunk | {ringkas['baris']:,} |
| Teks / tabel / gambar | {pm['text']:,} / {pm['table']:,} / {pm['image']:,} |
| Sumber data | `{sumber}` |
| Collection | `{collection}` |

## Isi paket

| Berkas / folder | Isi |
|---|---|
| `semua_chunk.csv` | Semua chunk dalam satu tabel. Buka di Excel atau Google Sheets. |
| `gambar/` | Berkas gambar asli, satu subfolder per dokumen. |
| `chunks_jsonl/`, `chunks_markdown/` | Data yang sama per dokumen, untuk tim evaluasi dan AI lokal. |
| `daftar_gambar.csv`, `daftar_dokumen.csv` | Inventaris gambar dan dokumen. |

## Kolom `semua_chunk.csv`

| Kolom | Arti |
|---|---|
| `chunk_id` | Identitas chunk: `<document_id>_p<halaman>_c<urutan>`. Jangan diubah. |
| `document_id` | Identitas dokumen. |
| `judul_dokumen` | Diambil dari nama berkas PDF (judul resmi belum dikurasi). |
| `halaman` | Halaman PDF tempat chunk berada. |
| `modality` | Jenis isi: lihat di bawah. |
| `element_type` | Jenis elemen hasil ekstraksi (rinci, untuk keperluan teknis). |
| `table_source` | Asal teks tabel: lihat di bawah. Kosong untuk chunk bukan tabel. |
| `image_id` | Identitas gambar, bila chunk berasal dari gambar. |
| `path_gambar` | Letak berkas gambar di paket, misalnya `gambar/ukt-tahun-2025/p5_img00.jpg`. |
| `text_content` | Isi chunk. |

### Nilai `modality`

- `text` — teks biasa (paragraf, butir, judul).
- `table` — tabel. Termasuk gambar yang isinya tabel dan sudah disalin menjadi
  tabel (`image_id` terisi).
- `image` — deskripsi gambar dalam kalimat. Gambar yang memuat tabel di
  dalamnya (misalnya tangkapan layar) berisi deskripsi lalu tabel salinannya.

### Nilai `table_source`

- `vision_transcription` — tabel disalin oleh model vision dari gambar halaman
  dalam format tabel Markdown. Ini isi tabel yang dipakai sistem.
- `ocr_fallback` — penyalinan oleh model gagal, jadi isinya teks hasil OCR
  (biasanya lebih berantakan).

### Catatan `transkripsi_peringatan` (ada di `chunks_jsonl/`)

Penanda otomatis, **tidak pernah mengubah isi chunk**:

- `rujukan=lapisan_teks` / `rujukan=lapisan_ocr` / `rujukan=teks_ocr` — sumber
  pembanding angka. Penanda dengan `lapisan_ocr` banyak yang palsu, karena
  lapisan OCR halaman pindaian sering salah baca.
- `angka_tak_ditemukan:...` — angka di tabel salinan yang tidak ditemukan di
  pembanding. Perlu dicek, bukan pasti salah.
- `baris_berturut_identik:N` — ada baris yang berulang persis (kemungkinan model mengulang).
- `kolom_tidak_konsisten:X/Y` — jumlah kolom judul X tidak sama dengan baris
  data Y; nilai bisa berada di bawah judul kolom yang salah.
- `gagal:...` — penyalinan gagal; alasannya tertulis.

## Yang dibaca untuk evaluasi

Kolom **`text_content`** — isi yang sama persis dengan yang dicari sistem.
Di `chunks_jsonl/` ada juga `text_as_html`: itu HTML hasil OCR lama yang hanya
dipakai sebagai identitas tabel, **bukan** isi yang dievaluasi. Untuk tabel,
keduanya memang berbeda.

## Membuka dan mengunggah

- **Gambar:** unggah folder `gambar/` **utuh** ke Google Drive, di folder yang
  sama dengan `semua_chunk.csv`, supaya `path_gambar` langsung cocok.
- **Excel:** isi yang diawali `=`, `+`, `-`, atau `@` diberi tanda `'` di depan
  agar tidak dibaca sebagai rumus ({ringkas['sel_diberi_apostrof']:,} sel). Satu
  sel Excel maksimal {BATAS_SEL_EXCEL:,} karakter;
  {ringkas['sel_melebihi_batas_excel']} chunk lebih panjang dari itu dan akan
  terpotong di Excel — isi lengkapnya ada di `chunks_jsonl/`.
"""
