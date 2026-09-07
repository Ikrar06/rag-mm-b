"""analisis_tabel_lintas_halaman.py — ukur berapa banyak tabel yang terpotong
di batas halaman benar-benar merupakan SATU tabel yang berlanjut.

READ-ONLY. Tidak mengindeks, tidak menulis ke Qdrant, tidak mengubah pipeline.
Instrumen pengukuran untuk Tahap A; bukan perbaikan.

Latar
-----
`_chunk_elements` tidak punya logika batas halaman sama sekali. Tabel yang
melintasi halaman sudah tiba sebagai DUA element Table terpisah dari
`partition_pdf` (satu per halaman), dan dengan INDEX_TABLES_AS_OWN_CHUNKS=true
masing-masing menjadi chunk sendiri. Tidak ada penanda bahwa keduanya satu tabel.

"Punya tabel di halaman berurutan" saja tidak cukup untuk menyimpulkan
kelanjutan — dua tabel berbeda yang kebetulan bersebelahan menghasilkan pola
yang sama. Skrip ini memisahkan keduanya dengan lima sinyal.

Lima sinyal
-----------
1. `kolom_sama`      jumlah kolom modal kedua potongan sama
2. `header`          `b_tanpa_header` (A punya <th>/<thead>, B tidak) = kuat;
                     `header_diulang` (baris pertama B sama dengan A) = lemah;
                     `header_berbeda` (keduanya punya header, isinya beda) = bukti LAWAN
3. `bawah_atas`      A berakhir di bawah halaman N (bbox y1 tinggi) DAN
                     B mulai di atas halaman N+1 (bbox y0 rendah)
4. `section_sama`    nilai `section` kedua potongan sama
5. `tanpa_sisipan`   tidak ada chunk non-tabel di antara keduanya (chunk_index)

Sinyal 5 tidak diminta di daftar awal tapi paling tajam memisahkan kasus:
dua tabel berbeda hampir selalu dipisahkan prosa atau judul.

Jebakan: `document_id` di payload Qdrant BUKAN slug
---------------------------------------------------
LlamaIndex menimpanya dengan UUID node saat menulis ke vector store:

    llama_index/core/vector_stores/utils.py, node_to_metadata_dict()
        metadata["document_id"] = node.ref_doc_id or "None"

`indexing.py` membuat satu Document per chunk, jadi setiap titik memperoleh UUID
yang berbeda. Mengelompokkan dengan field itu menghasilkan "1.160 chunk tabel di
1.160 dokumen" dan NOL pasangan — bukan nol karena tidak ada tabel bersambung,
tapi karena tiap chunk jadi dokumen sendiri sehingga tidak ada yang bertetangga.

Slug diambil dari `chunk_id`, yang tidak ditimpa. Ia juga selamat di dalam
`_node_content` (di-dump SEBELUM penimpaan) dan dipakai sebagai silang-periksa.

Sumber data
-----------
    --chunks-jsonl PATH   baca dump chunks.jsonl (tidak perlu Qdrant)
    (default)             scroll koleksi Qdrant

Usage:
    python scripts/analisis_tabel_lintas_halaman.py --collection rag_mm_b_varian_b_v2
    python scripts/analisis_tabel_lintas_halaman.py --chunks-jsonl dump/chunks.jsonl
    python scripts/analisis_tabel_lintas_halaman.py --contoh 10 --json hasil.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ambang posisi vertikal. bbox ternormalisasi [0,1] dengan y dari ATAS
# (preprocessing._element_bbox). Longgar sengaja: tabel bawah halaman sering
# menyisakan ruang untuk footer, dan tabel atas halaman sering di bawah header.
AMBANG_BAWAH = 0.72   # A berakhir di bawah ambang ini = "menempel dasar halaman"
AMBANG_ATAS = 0.30    # B mulai di atas ambang ini = "menempel puncak halaman"


class _Tabel(HTMLParser):
    """Ambil baris tabel sebagai list-of-list, plus tahu mana sel header."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.baris: list[list[str]] = []
        self.ada_th = False
        self.ada_thead = False
        self._baris: list[str] | None = None
        self._sel: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "thead":
            self.ada_thead = True
        elif tag == "tr":
            self._baris = []
        elif tag in ("td", "th"):
            if tag == "th":
                self.ada_th = True
            self._sel = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._sel is not None and self._baris is not None:
            self._baris.append(" ".join("".join(self._sel).split()))
            self._sel = None
        elif tag == "tr" and self._baris is not None:
            if self._baris:
                self.baris.append(self._baris)
            self._baris = None

    def handle_data(self, data):
        if self._sel is not None:
            self._sel.append(data)


def urai(html: str) -> _Tabel | None:
    if not html or "<t" not in html.lower():
        return None
    p = _Tabel()
    try:
        p.feed(html)
        p.close()
    except Exception:
        return None
    return p if p.baris else None


def n_kolom(t: _Tabel) -> int:
    """Jumlah kolom modal — tahan terhadap baris judul yang di-colspan."""
    c = Counter(len(b) for b in t.baris if b)
    return c.most_common(1)[0][0] if c else 0


def _norm(sel: list[str]) -> str:
    return "|".join(re.sub(r"\s+", " ", s).strip().lower() for s in sel)


def _angka(sel: list[str]) -> float:
    """Proporsi sel yang berupa angka. Baris data biasanya tinggi, header rendah."""
    if not sel:
        return 0.0
    n = sum(1 for s in sel if re.fullmatch(r"[\d.,%()\-\s/]+", s.strip()) and s.strip())
    return n / len(sel)


def bandingkan_header(a: _Tabel, b: _Tabel) -> str:
    """'b_tanpa_header' | 'header_diulang' | 'header_berbeda' | 'tak_tentu'."""
    if not a.baris or not b.baris:
        return "tak_tentu"
    if _norm(a.baris[0]) == _norm(b.baris[0]):
        return "header_diulang"
    # A punya penanda header eksplisit, B tidak → B potongan lanjutan.
    if (a.ada_th or a.ada_thead) and not (b.ada_th or b.ada_thead):
        return "b_tanpa_header"
    # Tanpa penanda eksplisit: pakai bentuk baris pertama. Baris pertama B yang
    # didominasi angka adalah baris DATA, artinya headernya tertinggal di A.
    if _angka(b.baris[0]) >= 0.5 and _angka(a.baris[0]) < 0.5:
        return "b_tanpa_header"
    if (a.ada_th or a.ada_thead) and (b.ada_th or b.ada_thead):
        return "header_berbeda"
    return "tak_tentu"


# chunk_id berbentuk "{document_id}_p{N}_c{NN}" atau "{document_id}_pNA_c{NN}"
# (preprocessing.emit). Jangkar di UJUNG supaya slug yang kebetulan memuat "_p"
# tidak terpotong di tempat yang salah.
_CHUNK_ID_RE = re.compile(r"^(?P<doc>.+)_p(?:\d+|NA)_c\d+$")


def slug_dari_chunk_id(chunk_id) -> str | None:
    if not isinstance(chunk_id, str):
        return None
    m = _CHUNK_ID_RE.match(chunk_id)
    return m.group("doc") if m else None


def kunci_dokumen(r: dict) -> str:
    """Slug dokumen sebuah chunk.

    JANGAN memakai payload['document_id'] dari Qdrant. LlamaIndex MENIMPA field
    itu dengan UUID node saat menulis ke vector store:

        llama_index/core/vector_stores/utils.py, node_to_metadata_dict()
            metadata["document_id"] = node.ref_doc_id or "None"

    indexing.py membuat satu Document per chunk, jadi tiap titik memperoleh UUID
    yang berbeda. Mengelompokkan dengan field itu menghasilkan "N chunk di N
    dokumen" dan nol pasangan — gejala yang persis pernah terjadi.

    Urutan sumber: chunk_id (selalu utuh), lalu slug yang selamat di dalam
    _node_content (di-dump SEBELUM penimpaan), lalu file_name sebagai upaya
    terakhir.
    """
    return (slug_dari_chunk_id(r.get("chunk_id"))
            or r.get("_slug_node_content")
            or r.get("file_name")
            or "?")


def muat_jsonl(path: Path) -> list[dict]:
    out = []
    for i, ln in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not ln.strip():
            continue
        try:
            r = json.loads(ln)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} baris {i}: {e}") from e
        # Dump bisa berisi payload mentah Qdrant. Pulihkan slug dari
        # _node_content di sini juga supaya silang-periksa bekerja di kedua jalur.
        if isinstance(r, dict) and "_node_content" in r:
            try:
                dalam = (json.loads(r["_node_content"]) or {}).get("metadata") or {}
            except Exception:
                dalam = {}
            r = {**dalam, **r, "_slug_node_content": dalam.get("document_id")}
        out.append(r)
    return out


def muat_qdrant(collection: str) -> list[dict]:
    from backend.services.indexing import get_qdrant_client

    client = get_qdrant_client()
    out, offset = [], None
    while True:
        titik, offset = client.scroll(
            collection_name=collection, limit=1000, offset=offset,
            with_payload=True, with_vectors=False,
        )
        if not titik:
            break
        for p in titik:
            pl = p.payload or {}
            if not isinstance(pl, dict):
                continue
            # _node_content memuat metadata APA ADANYA sebelum LlamaIndex
            # menimpa document_id — slug aslinya selamat di sana.
            if "_node_content" in pl:
                try:
                    dalam = (json.loads(pl["_node_content"]) or {}).get("metadata") or {}
                except Exception:
                    dalam = {}
                pl = {**dalam, **pl, "_slug_node_content": dalam.get("document_id")}
            out.append(pl)
        if offset is None:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Ukur tabel terpotong lintas halaman")
    ap.add_argument("--collection", default=None)
    ap.add_argument("--chunks-jsonl", default=None)
    ap.add_argument("--contoh", type=int, default=5, help="Contoh nyata per kategori")
    ap.add_argument("--json", default=None, help="Tulis hasil lengkap ke berkas JSON")
    args = ap.parse_args()

    if args.chunks_jsonl:
        src = Path(args.chunks_jsonl)
        if not src.exists():
            print(f"GAGAL: {src} tidak ditemukan")
            return 2
        rows = muat_jsonl(src)
        asal = str(src)
    else:
        from backend.config import QDRANT_COLLECTION_NAME
        koleksi = args.collection or QDRANT_COLLECTION_NAME
        try:
            rows = muat_qdrant(koleksi)
        except Exception as e:
            print(f"GAGAL membaca Qdrant: {e}")
            return 2
        asal = f"qdrant:{koleksi}"

    print(f"Sumber : {asal}")
    print(f"Chunk  : {len(rows)}")
    print(f"Dokumen: {len({kunci_dokumen(r) for r in rows})} unik "
          f"(dari chunk_id, BUKAN dari payload document_id yang ditimpa LlamaIndex)")

    # Kelompokkan per dokumen. chunk_index memberi urutan dokumen yang sama
    # dengan urutan element dari partition_pdf.
    per_dok: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        per_dok[kunci_dokumen(r)].append(r)

    # Deteksi dini kalau pengelompokan runtuh lagi: satu chunk per "dokumen"
    # berarti kuncinya unik per titik, bukan per dokumen.
    if len(rows) >= 20 and len(per_dok) > 0.9 * len(rows):
        print("  PERINGATAN: hampir tiap chunk jadi dokumen sendiri "
              f"({len(per_dok)} kunci / {len(rows)} chunk). chunk_id mungkin "
              "tidak berbentuk {document_id}_p{N}_c{NN}.")

    tak_cocok = [r for r in rows if not slug_dari_chunk_id(r.get("chunk_id"))]
    if tak_cocok:
        contoh_id = tak_cocok[0].get("chunk_id")
        print(f"  PERINGATAN: {len(tak_cocok)}/{len(rows)} chunk_id tidak cocok pola "
              f"{{document_id}}_p{{N}}_c{{NN}}, mis. {contoh_id!r}. Slug-nya jatuh ke "
              f"_node_content atau file_name — pengelompokan bisa salah.")

    # Silang-periksa dua jalur pemulihan slug.
    beda = [r for r in rows
            if r.get("_slug_node_content")
            and slug_dari_chunk_id(r.get("chunk_id"))
            and r["_slug_node_content"] != slug_dari_chunk_id(r["chunk_id"])]
    if beda:
        print(f"  PERINGATAN: {len(beda)} chunk punya slug berbeda antara chunk_id "
              f"dan _node_content, mis. {beda[0].get('chunk_id')!r} vs "
              f"{beda[0]['_slug_node_content']!r}")

    tabel_total = sum(1 for r in rows if r.get("element_type") == "Table")
    dok_bertabel = {d for d, v in per_dok.items() if any(x.get("element_type") == "Table" for x in v)}
    print(f"Tabel  : {tabel_total} chunk di {len(dok_bertabel)} dokumen\n")

    KATEGORI = ("lanjutan_kuat", "mungkin", "tabel_berbeda")
    hitung = Counter()
    dok_per_kat: dict[str, set] = defaultdict(set)
    contoh: dict[str, list] = defaultdict(list)
    semua: list[dict] = []

    for dok, chunks in per_dok.items():
        chunks = sorted(chunks, key=lambda r: (r.get("chunk_index") if isinstance(r.get("chunk_index"), int) else 0))
        tabel = [(i, r) for i, r in enumerate(chunks) if r.get("element_type") == "Table"]

        for (ia, a), (ib, b) in zip(tabel, tabel[1:]):
            pa, pb = a.get("page"), b.get("page")
            if not isinstance(pa, int) or not isinstance(pb, int) or pb != pa + 1:
                continue

            ta, tb = urai(a.get("raw_html") or ""), urai(b.get("raw_html") or "")
            ka = n_kolom(ta) if ta else 0
            kb = n_kolom(tb) if tb else 0
            kolom_sama = bool(ka and kb and ka == kb)
            header = bandingkan_header(ta, tb) if (ta and tb) else "tak_tentu"

            ba, bb = a.get("bbox"), b.get("bbox")
            bawah_atas = bool(
                isinstance(ba, list) and len(ba) == 4 and isinstance(bb, list) and len(bb) == 4
                and ba[3] >= AMBANG_BAWAH and bb[1] <= AMBANG_ATAS
            )
            section_sama = bool(a.get("section") and a.get("section") == b.get("section"))
            # Sisipan = chunk apa pun di antara keduanya yang BUKAN tabel.
            sisipan = [c for c in chunks[ia + 1:ib] if c.get("element_type") != "Table"]
            tanpa_sisipan = not sisipan

            skor = sum([kolom_sama, header == "b_tanpa_header", bawah_atas,
                        section_sama, tanpa_sisipan])
            if header == "header_berbeda":
                kat = "tabel_berbeda"
            elif kolom_sama and header == "b_tanpa_header" and (bawah_atas or tanpa_sisipan):
                kat = "lanjutan_kuat"
            elif kolom_sama and skor >= 3:
                kat = "mungkin"
            elif skor <= 1:
                kat = "tabel_berbeda"
            else:
                kat = "mungkin"

            hitung[kat] += 1
            dok_per_kat[kat].add(dok)
            rec = {
                "document_id": dok, "kategori": kat, "skor": skor,
                "chunk_id_a": a.get("chunk_id"), "chunk_id_b": b.get("chunk_id"),
                "halaman": [pa, pb], "kolom": [ka, kb],
                "sinyal": {"kolom_sama": kolom_sama, "header": header,
                           "bawah_atas": bawah_atas, "section_sama": section_sama,
                           "tanpa_sisipan": tanpa_sisipan,
                           "n_sisipan": len(sisipan)},
                "bbox_a": ba, "bbox_b": bb,
                "baris_pertama_a": (ta.baris[0] if ta and ta.baris else None),
                "baris_pertama_b": (tb.baris[0] if tb and tb.baris else None),
            }
            semua.append(rec)
            if len(contoh[kat]) < args.contoh:
                contoh[kat].append(rec)

    total = sum(hitung.values())
    print("=" * 78)
    print(f"PASANGAN TABEL DI HALAMAN BERURUTAN: {total}")
    print("=" * 78)
    if not total:
        print("\nTidak ada pasangan. Tidak ada yang perlu diperbaiki.")
        return 0

    for kat in KATEGORI:
        n = hitung[kat]
        print(f"  {kat:<16}{n:>5}  {100 * n / total:>5.1f}%   di {len(dok_per_kat[kat])} dokumen")

    print("\nSEBARAN SINYAL (atas seluruh pasangan)")
    for nama in ("kolom_sama", "bawah_atas", "section_sama", "tanpa_sisipan"):
        n = sum(1 for r in semua if r["sinyal"][nama])
        print(f"  {nama:<16}{n:>5}  {100 * n / total:>5.1f}%")
    hc = Counter(r["sinyal"]["header"] for r in semua)
    print("  header:")
    for k, n in hc.most_common():
        print(f"    {k:<18}{n:>5}  {100 * n / total:>5.1f}%")

    for kat in KATEGORI:
        if not contoh[kat]:
            continue
        print(f"\n{'=' * 78}\nCONTOH — {kat}\n{'=' * 78}")
        for r in contoh[kat]:
            print(f"  {r['document_id']}  hal {r['halaman'][0]}->{r['halaman'][1]}  "
                  f"kolom {r['kolom'][0]}/{r['kolom'][1]}  skor {r['skor']}/5")
            print(f"    A {r['chunk_id_a']}   bbox_y {r['bbox_a'][1] if r['bbox_a'] else '?'}"
                  f"..{r['bbox_a'][3] if r['bbox_a'] else '?'}")
            print(f"    B {r['chunk_id_b']}   bbox_y {r['bbox_b'][1] if r['bbox_b'] else '?'}"
                  f"..{r['bbox_b'][3] if r['bbox_b'] else '?'}")
            print(f"    baris-1 A: {r['baris_pertama_a']}")
            print(f"    baris-1 B: {r['baris_pertama_b']}")
            print(f"    sinyal   : {r['sinyal']}")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "sumber": asal, "n_chunk": len(rows), "n_tabel": tabel_total,
            "n_dokumen_bertabel": len(dok_bertabel),
            "ringkasan": {k: {"pasangan": hitung[k], "dokumen": len(dok_per_kat[k])}
                          for k in KATEGORI},
            "ambang": {"AMBANG_BAWAH": AMBANG_BAWAH, "AMBANG_ATAS": AMBANG_ATAS},
            "pasangan": semua,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nHasil lengkap: {args.json}")

    print("\nCATATAN: ini pengukuran, bukan perbaikan. Ambang bbox "
          f"({AMBANG_BAWAH}/{AMBANG_ATAS}) heuristik — periksa contoh 'mungkin' "
          "sebelum menyetel ulang.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
