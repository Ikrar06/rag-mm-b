"""relevan_setara: chunk yang saling MENGGANTIKAN sebagai bukti relevan.

Asal kebutuhannya (Tahap T, keputusan 3). Area render tabel v5 ditarik ke kata
yang terpotong tepinya, termasuk kata milik chunk TEKS. Isi yang sama lalu
ada di dua chunk: chunk teks lama dan chunk tabel baru. Bila gold merujuk
chunk teks tetapi retrieval mengambil chunk tabel, itu bukan meleset.

Skema gold
----------
    "relevan_setara": [["<chunk teks>", "<chunk tabel>"], ...]

Setiap kelompok berisi chunk yang saling menggantikan. `relevant_text_chunks`
TIDAK diubah, jadi evaluator lama tetap berjalan.

Cara evaluator membacanya
-------------------------
1. Chunk relevan yang punya kelompok diganti kelompoknya; yang tidak punya
   kelompok menjadi kelompok berisi dirinya sendiri.
2. Recall: per KELOMPOK. Kelompok terpenuhi bila salah satu anggotanya
   terambil; penyebutnya jumlah kelompok, bukan jumlah chunk.
3. Precision/MRR: chunk terambil relevan bila anggota salah satu kelompok.
4. nDCG: gain diberikan SEKALI per kelompok, pada kemunculan pertama.

Pengisian otomatis (untuk ditinjau)
-----------------------------------
Dua syarat, KEDUANYA wajib:
1. letak — bbox chunk teks beririsan dengan DAERAH PERLUASAN tabel sehalaman
   (bagian `render_bbox` di luar `bbox` asli);
2. isi — `containment` token chunk teks di chunk tabel >= ambang.

Syarat letak saja terbukti salah: ketiga pasangan migrasi v5 pertama
(manual-dosen p17 catatan tombol vs tabel status; penyelenggaraan-prodi p10
ayat 7-9 vs tabel konversi; kalender catatan 8 Aug vs grid Agustus) hanya
bertetangga, isinya berbeda. Ambang TIDAK punya nilai bawaan: ditetapkan dari
sebaran (scripts/sebaran_setara.py) dan diberikan eksplisit.

Fungsi murni.
"""

from __future__ import annotations

import re

# Luas irisan minimum (pecahan luas halaman). bbox ternormalisasi 4 desimal;
# irisan di bawah ini hanya derau pembulatan tepi.
LUAS_MIN = 1e-5
_ANGKA = re.compile(r"\d[\d.,]*\d")
_TOKEN = re.compile(r"[a-z0-9]+")


def token_isi(teks: str | None) -> frozenset[str]:
    """Token isi: tanpa baris judul Markdown, angka tanpa pemisah ribuan.

    Baris `#`/`##` dibuang karena chunk teks dan chunk tabel berbagi prefiks
    section yang sama — tanpa itu dua chunk berbeda tampak mirip. Huruf tunggal
    dibuang (butir a., b.); angka satu digit dipertahankan.
    """
    isi = "\n".join(l for l in (teks or "").splitlines() if not l.lstrip().startswith("#"))
    isi = _ANGKA.sub(lambda m: re.sub(r"[.,]", "", m.group(0)), isi.lower())
    return frozenset(w for w in _TOKEN.findall(isi) if len(w) >= 2 or w.isdigit())


def containment(teks: str | None, tabel: str | None) -> float | None:
    """Bagian token chunk teks yang muncul di chunk tabel. None bila teks tanpa token.

    Containment, bukan Jaccard: chunk tabel jauh lebih panjang daripada bagian
    teks yang tergandakan, sehingga Jaccard selalu kecil walau duplikat penuh.
    """
    a = token_isi(teks)
    return len(a & token_isi(tabel)) / len(a) if a else None


def _iris(a, b):
    if not a or not b:
        return None
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _luas(b) -> float:
    return (b[2] - b[0]) * (b[3] - b[1]) if b else 0.0


def luas_di_perluasan(teks_bbox, tabel_bbox, render_bbox) -> float:
    """Luas bagian bbox teks yang jatuh di render_bbox TAPI di luar bbox tabel."""
    di_render = _iris(teks_bbox, render_bbox)
    return _luas(di_render) - _luas(_iris(di_render, tabel_bbox))


def kandidat_letak(chunk_ids, rows_baru: list[dict]) -> list[tuple[str, str, float]]:
    """(chunk teks, chunk tabel, containment) untuk setiap pasangan yang lolos
    syarat LETAK. `chunk_ids` None = semua chunk teks (untuk sebaran)."""
    per_id = {r.get("chunk_id"): r for r in rows_baru}
    tabel: dict[tuple, list[dict]] = {}
    for r in rows_baru:
        if r.get("element_type") == "Table" and r.get("render_bbox") and r.get("bbox"):
            tabel.setdefault((r.get("document_id"), r.get("page_number")), []).append(r)
    hasil = []
    for cid in (chunk_ids if chunk_ids is not None else list(per_id)):
        r = per_id.get(cid)
        if not r or r.get("element_type") in ("Table", "ImageDescription") or not r.get("bbox"):
            continue
        for t in tabel.get((r.get("document_id"), r.get("page_number")), []):
            if luas_di_perluasan(r["bbox"], t["bbox"], t["render_bbox"]) > LUAS_MIN:
                c = containment(r.get("text_content"), t.get("text_content"))
                if c is not None:
                    hasil.append((cid, t["chunk_id"], c))
    return hasil


def kelompok_setara(chunk_ids, rows_baru: list[dict], ambang: float) -> dict[str, str]:
    """chunk teks -> chunk tabel yang lolos syarat letak DAN isi (containment >= ambang).

    Bila satu chunk teks lolos untuk beberapa tabel, yang containment-nya
    tertinggi dipilih.
    """
    terbaik: dict[str, tuple[float, str]] = {}
    for cid, tid, c in kandidat_letak(chunk_ids, rows_baru):
        if c >= ambang and c > terbaik.get(cid, (-1.0, ""))[0]:
            terbaik[cid] = (c, tid)
    return {cid: tid for cid, (_, tid) in terbaik.items()}


def terapkan_setara(item: dict, setara: dict[str, str]) -> dict:
    """Salinan item dengan `relevan_setara`. Item tanpa pasangan tidak berubah."""
    kelompok = [[cid, setara[cid]] for cid in (item.get("relevant_text_chunks") or [])
                if cid in setara]
    return {**item, "relevan_setara": kelompok} if kelompok else dict(item)
