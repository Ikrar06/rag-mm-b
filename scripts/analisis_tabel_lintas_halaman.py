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

Dua angka yang tidak ada di payload
-----------------------------------
`--pdf-dir` membuka PDF aslinya untuk membedakan dokumen digital dari hasil
pindai. Lapisan teks hanya ada di berkas asli, tidak ikut ke Qdrant. Tanpa
argumen itu, seluruh bagian KOMPOSISI SUMBER dilewati.

Ambangnya menyalin `preprocessing.OCR_TEXT_THRESHOLD_CHARS`, aturan yang sudah
dipakai pipeline untuk memutuskan sebuah halaman perlu di-OCR.

Usage:
    # satu jalan, seluruh angka:
    python scripts/analisis_tabel_lintas_halaman.py \
        --collection rag_mm_b_varian_b_v2 --pdf-dir data/pdfs \
        --contoh 10 --json hasil.json

    python scripts/analisis_tabel_lintas_halaman.py --chunks-jsonl dump/chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.pdf_sumber import profil_dari_pdf  # noqa: E402
from lib.tabel_html import (  # noqa: E402
    bandingkan_header, deteksi_sel_terpotong, urai,
)

# Ambang posisi vertikal. bbox ternormalisasi [0,1] dengan y dari ATAS
# (preprocessing._element_bbox). Longgar sengaja: tabel bawah halaman sering
# menyisakan ruang untuk footer, dan tabel atas halaman sering di bawah header.
AMBANG_BAWAH = 0.72   # A berakhir di bawah ambang ini = "menempel dasar halaman"
AMBANG_ATAS = 0.30    # B mulai di atas ambang ini = "menempel puncak halaman"


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



def _ringkas_sel(h) -> dict:
    """HasilSel -> dict siap-JSON. Tidak mengubah argumennya."""
    return {
        "n_kandidat": len(h.kandidat),
        "n_kuat": len(h.kuat),
        "catatan": h.catatan,
        "kuat": [
            {"kolom": k.kolom, "skor": k.skor,
             "ekor_a": k.ekor_a[-80:], "kepala_b": k.kepala_b[:80],
             "tanpa_tanda_baca": k.tanpa_tanda_baca,
             "lanjutan_huruf_kecil": k.lanjutan_huruf_kecil,
             "lanjutan_konjungsi": k.lanjutan_konjungsi,
             "sel_lain_kosong_a": k.sel_lain_kosong_a,
             "sel_lain_kosong_b": k.sel_lain_kosong_b}
            for k in h.kuat
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Ukur tabel terpotong lintas halaman")
    ap.add_argument("--collection", default=None)
    ap.add_argument("--chunks-jsonl", default=None)
    ap.add_argument("--contoh", type=int, default=5, help="Contoh nyata per kategori")
    ap.add_argument("--json", default=None, help="Tulis hasil lengkap ke berkas JSON")
    ap.add_argument("--pdf-dir", default=None,
                    help="Folder PDF sumber. Tanpa ini, komposisi digital/pindai "
                         "dilewati (payload Qdrant tidak memuat lapisan teks).")
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

    nama_berkas: dict[str, str] = {}
    for dok, chunks in per_dok.items():
        for c in chunks:
            if isinstance(c.get("file_name"), str) and c["file_name"]:
                nama_berkas[dok] = c["file_name"]
                break

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

            ta, tb = urai(a.get("raw_html")), urai(b.get("raw_html"))
            ka = ta.n_kolom if ta else 0
            kb = tb.n_kolom if tb else 0
            kolom_sama = bool(ka and kb and ka == kb)
            header = bandingkan_header(ta, tb)

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
                "baris_pertama_a": (list(ta.baris[0]) if ta and ta.baris else None),
                "baris_pertama_b": (list(tb.baris[0]) if tb and tb.baris else None),
                "sel_terpotong": _ringkas_sel(deteksi_sel_terpotong(ta, tb)),
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

    # ── Komposisi sumber: PDF digital vs pindai ─────────────────────────────
    #
    # Tidak dapat dijawab dari payload Qdrant: lapisan teks hanya ada di PDF
    # aslinya. Dilewati bila --pdf-dir tidak diberikan.
    profil: dict[str, object] = {}
    if args.pdf_dir:
        pdf_dir = Path(args.pdf_dir)
        halaman_tabel_per_dok: dict[str, set] = defaultdict(set)
        for dok, chunks in per_dok.items():
            for c in chunks:
                if c.get("element_type") == "Table" and isinstance(c.get("page"), int):
                    halaman_tabel_per_dok[dok].add(c["page"])

        for dok in sorted(dok_bertabel):
            nama = nama_berkas.get(dok)
            if not nama:
                continue
            profil[dok] = profil_dari_pdf(
                pdf_dir / nama, tuple(sorted(halaman_tabel_per_dok[dok]))
            )

        print("\n" + "=" * 78)
        print("KOMPOSISI SUMBER — PDF DIGITAL vs PINDAI")
        print("=" * 78)
        print(f"  Folder PDF: {pdf_dir}")
        print(f"  Dokumen bertabel diperiksa: {len(profil)} dari {len(dok_bertabel)}")
        if len(profil) < len(dok_bertabel):
            print(f"  {len(dok_bertabel) - len(profil)} dokumen dilewati "
                  f"(file_name tidak ada di payload)")

        for label, atribut in (("seluruh halaman", "sumber"),
                               ("HALAMAN BERTABEL saja", "sumber_halaman_tabel")):
            c = Counter(getattr(v, atribut) for v in profil.values())
            n = sum(c.values()) or 1
            print(f"\n  Dinilai atas {label}:")
            for k in ("digital", "campuran", "pindai", "tak_terbaca"):
                if c[k]:
                    print(f"    {k:<14}{c[k]:>4}  {100 * c[k] / n:>5.1f}%")

        print("\n  Angka yang menentukan adalah HALAMAN BERTABEL: sebuah dokumen")
        print("  bisa mayoritas digital sementara justru halaman tabelnya sisipan pindai.")

        buruk = sorted((v for v in profil.values()
                        if v.sumber_halaman_tabel in ("pindai", "campuran")),
                       key=lambda v: v.rasio_tabel_berteks)
        if buruk:
            print(f"\n  DOKUMEN YANG HALAMAN TABELNYA TANPA LAPISAN TEKS "
                  f"({len(buruk)}) — isi selnya dari OCR:")
            for v in buruk[: max(args.contoh, 5)]:
                print(f"    {v.file_name:<52}{v.rasio_tabel_berteks:>5.0%} berteks  "
                      f"({len(v.halaman_tabel)} hal tabel)")

        # Silang-tabulasi: kategori pasangan x sumber halamannya.
        print("\n  SILANG-TABULASI kategori pasangan x lapisan teks halaman:")
        print(f"    {'kategori':<16}{'berteks':>9}{'tanpa teks':>12}{'tak jelas':>11}")
        for kat in KATEGORI:
            b = t_ = x = 0
            for r in semua:
                if r["kategori"] != kat:
                    continue
                pr = profil.get(r["document_id"])
                if pr is None or pr.error:
                    x += 1
                elif (pr.halaman_berlapis_teks(r["halaman"][0])
                      and pr.halaman_berlapis_teks(r["halaman"][1])):
                    b += 1
                else:
                    t_ += 1
            print(f"    {kat:<16}{b:>9}{t_:>12}{x:>11}")
        print("    -> baris 'lanjutan_kuat' kolom 'tanpa teks' adalah kandidat")
        print("       yang headernya berasal dari OCR; itu yang perlu ditolak.")

    # ── Kasus 3: sel terpotong di tengah ────────────────────────────────────
    #
    # Berbeda dari baris terpotong dan berbeda penanganannya: header yang
    # diulang TIDAK menolong, karena pemetaan baris-kolomnya sudah benar —
    # yang terpenggal adalah isi selnya.
    dgn_kandidat = [r for r in semua if r["sel_terpotong"]["n_kandidat"]]
    dgn_kuat = [r for r in semua if r["sel_terpotong"]["n_kuat"]]
    tak_terurai = [r for r in semua if r["sel_terpotong"]["catatan"]]

    print("\n" + "=" * 78)
    print("KASUS 3 — SEL TERPOTONG DI TENGAH")
    print("=" * 78)
    print(f"  {'pasangan dengan kandidat':<34}{len(dgn_kandidat):>5}  "
          f"{100 * len(dgn_kandidat) / total:>5.1f}%")
    print(f"  {'pasangan dengan kandidat KUAT':<34}{len(dgn_kuat):>5}  "
          f"{100 * len(dgn_kuat) / total:>5.1f}%   (>= 2 dari 3 indikator)")
    print(f"  {'tidak dapat dinilai':<34}{len(tak_terurai):>5}  "
          f"{100 * len(tak_terurai) / total:>5.1f}%   (raw_html/lebar baris)")
    per_kat = Counter(r["kategori"] for r in dgn_kuat)
    if per_kat:
        print("  sebaran kandidat kuat per kategori:")
        for k in KATEGORI:
            if per_kat[k]:
                print(f"    {k:<16}{per_kat[k]:>5}")

    if dgn_kuat:
        print(f"\n  CONTOH (maks {args.contoh}):")
        for r in dgn_kuat[: max(args.contoh, 1)]:
            print(f"    {r['document_id']}  hal {r['halaman'][0]}->{r['halaman'][1]}  "
                  f"[{r['kategori']}]")
            for k in r["sel_terpotong"]["kuat"]:
                print(f"      kolom {k['kolom']}  skor {k['skor']}/3")
                print(f"        A ...{k['ekor_a']}")
                print(f"        B {k['kepala_b']}...")
    else:
        print("\n  Tidak ada kandidat kuat. Kasus ini tidak perlu ditangani.")

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
            "sumber_dokumen": {
                dok: {"file_name": v.file_name, "n_halaman": v.n_halaman,
                      "sumber": v.sumber, "sumber_halaman_tabel": v.sumber_halaman_tabel,
                      "rasio_berteks": round(v.rasio_berteks, 4),
                      "rasio_tabel_berteks": round(v.rasio_tabel_berteks, 4),
                      "n_halaman_tabel": len(v.halaman_tabel), "error": v.error}
                for dok, v in profil.items()
            },
            "pasangan": semua,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nHasil lengkap: {args.json}")

    print("\nCATATAN: ini pengukuran, bukan perbaikan. Ambang bbox "
          f"({AMBANG_BAWAH}/{AMBANG_ATAS}) heuristik — periksa contoh 'mungkin' "
          "sebelum menyetel ulang.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
