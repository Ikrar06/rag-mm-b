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
Chunk teks yang dirujuk gold dikelompokkan dengan chunk tabel di halaman yang
sama bila bbox-nya beririsan dengan DAERAH PERLUASAN tabel itu — bagian
`render_bbox` di luar `bbox` asli. bbox chunk teks adalah gabungan element-nya,
jadi irisan kecil bisa berupa kebetulan; setiap pasangan dilaporkan.

Fungsi murni.
"""

from __future__ import annotations

# Luas irisan minimum (pecahan luas halaman). bbox ternormalisasi 4 desimal;
# irisan di bawah ini hanya derau pembulatan tepi.
LUAS_MIN = 1e-5


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


def kelompok_setara(chunk_ids, rows_baru: list[dict]) -> dict[str, str]:
    """chunk teks -> chunk tabel yang daerah perluasannya menelan isi teks itu."""
    per_id = {r.get("chunk_id"): r for r in rows_baru}
    tabel: dict[tuple, list[dict]] = {}
    for r in rows_baru:
        if r.get("element_type") == "Table" and r.get("render_bbox") and r.get("bbox"):
            tabel.setdefault((r.get("document_id"), r.get("page_number")), []).append(r)
    hasil = {}
    for cid in chunk_ids:
        r = per_id.get(cid)
        if not r or r.get("element_type") in ("Table", "ImageDescription") or not r.get("bbox"):
            continue
        calon = [(luas_di_perluasan(r["bbox"], t["bbox"], t["render_bbox"]), t["chunk_id"])
                 for t in tabel.get((r.get("document_id"), r.get("page_number")), [])]
        calon = [c for c in calon if c[0] > LUAS_MIN]
        if calon:
            hasil[cid] = max(calon)[1]
    return hasil


def terapkan_setara(item: dict, setara: dict[str, str]) -> dict:
    """Salinan item dengan `relevan_setara`. Item tanpa pasangan tidak berubah."""
    kelompok = [[cid, setara[cid]] for cid in (item.get("relevant_text_chunks") or [])
                if cid in setara]
    return {**item, "relevan_setara": kelompok} if kelompok else dict(item)
