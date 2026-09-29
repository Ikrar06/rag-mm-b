"""Perbandingan dua dump chunks.jsonl per dokumen dan per halaman. Fungsi murni."""

from __future__ import annotations

from collections import Counter, defaultdict

# Teks OCR tabel sependek ini (karakter) hampir pasti di bawah
# INDEX_MIN_CHUNK_TOKENS=8 di v4. Hanya untuk melabeli sebab, bukan menyaring.
TEKS_TABEL_PENDEK = 40


def _jenis(r: dict) -> str:
    if r.get("table_origin") == "image":
        return "Table(gambar)"
    return r.get("element_type") or "?"


def _kunci_isi(r: dict) -> str:
    """Identitas isi chunk untuk mencari padanan lintas dump.

    Tabel: sidik HTML (teks v5 transkripsi, v4 OCR — text_sha pasti beda).
    Gambar: image_id. Lainnya: text_sha.
    """
    if r.get("image_id"):
        return "img:" + r["image_id"]
    if r.get("element_type") == "Table" and r.get("text_as_html"):
        return "html:" + " ".join(r["text_as_html"].split())
    return "sha:" + (r.get("text_sha") or r.get("chunk_id") or "")


def _sebab(tambahan: list[dict], hilang: list[dict]) -> str:
    sebab = []
    for r in tambahan:
        if r.get("element_type") == "Table" and not r.get("image_id"):
            pendek = len((r.get("teks_ocr") or "").strip()) < TEKS_TABEL_PENDEK
            sebab.append("tabel_kecil_lolos" if pendek else "tabel_baru")
        elif r.get("image_id"):
            sebab.append("gambar_baru")
        else:
            sebab.append("teks_baru")
    for r in hilang:
        sebab.append("gambar_hilang" if r.get("image_id") else
                     "tabel_hilang" if r.get("element_type") == "Table" else "teks_hilang")
    return ",".join(sorted(set(sebab))) or "tidak_diketahui"


def bandingkan(lama: list[dict], baru: list[dict]) -> dict:
    per_dok_l, per_dok_b = defaultdict(list), defaultdict(list)
    for r in lama:
        per_dok_l[r.get("document_id")].append(r)
    for r in baru:
        per_dok_b[r.get("document_id")].append(r)

    dokumen, urutan_berubah, n_bergeser, sebab_total = [], [], 0, Counter()
    for doc in sorted(set(per_dok_l) & set(per_dok_b)):
        hal_l, hal_b = defaultdict(list), defaultdict(list)
        for r in per_dok_l[doc]:
            hal_l[r.get("page_number")].append(r)
        for r in per_dok_b[doc]:
            hal_b[r.get("page_number")].append(r)
        halaman = []
        for hal in sorted(set(hal_l) | set(hal_b), key=lambda x: (x is None, x or 0)):
            L = sorted(hal_l[hal], key=lambda r: r.get("chunk_id") or "")
            B = sorted(hal_b[hal], key=lambda r: r.get("chunk_id") or "")
            kl, kb = Counter(map(_kunci_isi, L)), Counter(map(_kunci_isi, B))
            tambahan = [r for r in B if kb[_kunci_isi(r)] > kl[_kunci_isi(r)]]
            hilang = [r for r in L if kl[_kunci_isi(r)] > kb[_kunci_isi(r)]]
            id_l = {_kunci_isi(r): r["chunk_id"] for r in L}
            bergeser = [(id_l[_kunci_isi(r)], r["chunk_id"]) for r in B
                        if _kunci_isi(r) in id_l and id_l[_kunci_isi(r)] != r["chunk_id"]]
            n_bergeser += len(bergeser)
            if len(L) != len(B):
                s = _sebab(tambahan, hilang)
                sebab_total[s] += 1
                halaman.append({
                    "halaman": hal, "selisih": len(B) - len(L), "sebab": s,
                    "tambahan": [f"{r['chunk_id']} {_jenis(r)} {(r.get('text_content') or '')[:60]!r}"
                                 for r in tambahan],
                    "hilang": [f"{r['chunk_id']} {_jenis(r)} {(r.get('text_content') or '')[:60]!r}"
                               for r in hilang],
                    "bergeser": bergeser,
                })
            elif [_jenis(r) for r in L] != [_jenis(r) for r in B]:
                urutan_berubah.append(f"{doc} hal {hal}: {[_jenis(r) for r in L]} -> "
                                      f"{[_jenis(r) for r in B]}")
        if len(per_dok_l[doc]) != len(per_dok_b[doc]):
            jl, jb = Counter(map(_jenis, per_dok_l[doc])), Counter(map(_jenis, per_dok_b[doc]))
            dokumen.append({
                "document_id": doc, "lama": len(per_dok_l[doc]), "baru": len(per_dok_b[doc]),
                "selisih": len(per_dok_b[doc]) - len(per_dok_l[doc]),
                "per_jenis": {j: jb[j] - jl[j] for j in sorted(set(jl) | set(jb)) if jb[j] != jl[j]},
                "halaman": halaman,
            })
    # Chunk baru yang berasal dari gambar yang di dump lama BUKAN chunk — gambar
    # dekoratif/tidak jelas yang di run utama v5 ikut diklasifikasi dan lolos
    # sebagai gambar-tabel atau narasi+tabel. Saringan deskripsi (perbaikan a)
    # meniadakan chunk ini; bila tidak nol, run utama tidak konsisten dengan
    # run susulan dan harus diulang penuh.
    gambar_lama = {r.get("image_id") for r in lama if r.get("image_id")}
    gambar_tanpa_padanan = sorted(
        f"{r['chunk_id']} {_jenis(r)} {r.get('image_content') or ''}" for r in baru
        if r.get("image_id") and r["image_id"] not in gambar_lama
        and r.get("document_id") in per_dok_l)
    return {
        "gambar_tanpa_padanan": gambar_tanpa_padanan,
        "hanya_lama": sorted(set(per_dok_l) - set(per_dok_b)),
        "hanya_baru": sorted(set(per_dok_b) - set(per_dok_l)),
        "dokumen": sorted(dokumen, key=lambda d: -abs(d["selisih"])),
        "urutan_berubah": urutan_berubah, "n_bergeser": n_bergeser, "sebab": sebab_total,
    }
