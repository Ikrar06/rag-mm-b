"""build_eval_package.py — susun paket lengkap untuk tim evaluasi.

Menghasilkan satu folder berisi chunk per dokumen (JSONL + Markdown terbaca),
gambar beserta deskripsinya, inventaris dokumen, kandidat per strata visual,
template ground truth, dan validator agar keluaran mereka bisa dicek sendiri.

Read-only terhadap artefak riset. Tidak menyentuh Qdrant.

Pemakaian:
  python scripts/build_eval_package.py --dump data/dumps/<run_id>
  python scripts/build_eval_package.py --dump data/dumps/<run_id> --no-images
  python scripts/build_eval_package.py --dump data/dumps/<run_id> --docs daftar.txt
"""
import argparse, collections, csv, json, os, re, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.paket_dosen import (  # noqa: E402
    baris_semua_chunk, format_ukuran, modality_chunk, readme_paket, tulis_semua_chunk,
    ukuran_folder,
)

FLOW = re.compile(r"diagram alir|flowchart|bagan alir|alur proses|diagram alur", re.I)
TABEL = re.compile(r"\btabel\b|baris dan kolom|kolom.*baris", re.I)
FORM = re.compile(r"formulir|blanko|template|kop surat|tanda tangan|lembar pengesahan", re.I)
ORG = re.compile(r"struktur organisasi|bagan organisasi|hierarki", re.I)


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def strata(summary, w, h):
    """Tebakan awal tipe visual. HANYA saran; label final tetap dari anotator."""
    if not summary:
        return ""
    if w and h and w < 120 and h < 120:
        return ""                      # terlalu kecil, kemungkinan ikon
    if FLOW.search(summary):
        return "flowchart_prosedural"
    if ORG.search(summary):
        return "figur_deskriptif"
    if FORM.search(summary):
        return "formulir_template"
    if TABEL.search(summary):
        return "tabel_persyaratan"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="Direktori dump hasil indexing")
    ap.add_argument("--images-root", default="data",
                    help="Root untuk file_path di images.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-images", action="store_true",
                    help="Jangan salin berkas gambar (paket jadi jauh lebih kecil)")
    ap.add_argument("--docs", default=None,
                    help="Berkas berisi daftar document_id (satu per baris) untuk membatasi korpus")
    a = ap.parse_args()

    dump = Path(a.dump)
    out = Path(a.out or (Path.home() / "rag_mm_b_shared" / "paket_tim_eval"))
    if out.exists():
        shutil.rmtree(out)
    (out / "chunks_jsonl").mkdir(parents=True)
    (out / "chunks_markdown").mkdir(parents=True)
    (out / "gambar").mkdir(parents=True)

    # modality diturunkan dari element_type bila dump lama tidak membawanya,
    # supaya paket v5 dapat dibangun ulang dari dump yang sama.
    chunks = [{**c, "modality": modality_chunk(c)} for c in read_jsonl(dump / "chunks.jsonl")]
    images = read_jsonl(dump / "images.jsonl")
    manifest = json.load(open(dump / "run_manifest.json", encoding="utf-8"))

    filter_docs = None
    if a.docs:
        filter_docs = {l.strip() for l in open(a.docs, encoding="utf-8") if l.strip()}
        chunks = [c for c in chunks if c.get("document_id") in filter_docs]
        images = [i for i in images if i.get("document_id") in filter_docs]

    by_doc = collections.defaultdict(list)
    for c in chunks:
        by_doc[c.get("document_id") or "(tanpa-id)"].append(c)
    img_by_doc = collections.defaultdict(list)
    for i in images:
        img_by_doc[i.get("document_id") or "(tanpa-id)"].append(i)
    img_by_id = {i["image_id"]: i for i in images if i.get("image_id")}

    # ---------- per dokumen ----------
    for doc, cs in sorted(by_doc.items()):
        cs.sort(key=lambda x: (x.get("page_number") or 0, x.get("chunk_id") or ""))
        with open(out / "chunks_jsonl" / f"{doc}.jsonl", "w", encoding="utf-8") as f:
            for c in cs:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        src = cs[0].get("file_name", "")
        lines = [f"# {doc}", "", f"Berkas asal: `{src}`",
                 f"Jumlah chunk: {len(cs)}", "",
                 "> Setiap blok di bawah punya `chunk_id`. Saat membuat pertanyaan,",
                 "> salin `chunk_id` itu apa adanya ke kolom `relevant_text_chunks`.", "",
                 "---", ""]
        for c in cs:
            et = c.get("element_type") or "?"
            pg = c.get("page_number")
            lines.append(f"## `{c.get('chunk_id')}`")
            lines.append(f"halaman {pg} · {c.get('modality')} · tipe {et}"
                         + (f" · gambar `{c['image_id']}`" if c.get("image_id") else ""))
            lines.append("")
            lines.append((c.get("text_content") or "").strip())
            lines.append("")
        (out / "chunks_markdown" / f"{doc}.md").write_text("\n".join(lines), encoding="utf-8")

    # ---------- gambar ----------
    root = Path(a.images_root)
    disalin = gagal = 0
    with open(out / "daftar_gambar.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "document_id", "halaman", "lebar", "tinggi",
                    "berkas_di_paket", "saran_visual_type", "deskripsi"])
        for i in sorted(images, key=lambda x: (x.get("document_id") or "",
                                               x.get("page_number") or 0)):
            rel = ""
            if not a.no_images and i.get("file_path"):
                s = root / i["file_path"]
                if s.exists():
                    d = out / "gambar" / i["file_path"].replace("images/", "", 1)
                    d.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(s, d)
                    rel = str(d.relative_to(out)); disalin += 1
                else:
                    gagal += 1
            w.writerow([i.get("image_id"), i.get("document_id"), i.get("page_number"),
                        i.get("width"), i.get("height"), rel,
                        strata(i.get("narrative_summary"), i.get("width"), i.get("height")),
                        (i.get("narrative_summary") or "").replace("\n", " ")])

    # ---------- inventaris dokumen ----------
    with open(out / "daftar_dokumen.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["document_id", "berkas_asal", "jumlah_chunk", "chunk_teks",
                    "chunk_tabel", "chunk_gambar", "jumlah_gambar", "halaman_maks"])
        for doc, cs in sorted(by_doc.items()):
            et = collections.Counter(c.get("element_type") for c in cs)
            img = sum(v for k, v in et.items() if k == "ImageDescription")
            tbl = sum(v for k, v in et.items() if k == "Table")
            w.writerow([doc, cs[0].get("file_name", ""), len(cs),
                        len(cs) - img - tbl, tbl, img, len(img_by_doc.get(doc, [])),
                        max((c.get("page_number") or 0) for c in cs)])

    # ---------- kandidat per strata ----------
    with open(out / "kandidat_strata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["saran_visual_type", "image_id", "document_id", "halaman",
                    "chunk_id_terkait", "deskripsi"])
        chunk_of_img = {c["image_id"]: c.get("chunk_id")
                        for c in chunks if c.get("image_id")}
        rows = []
        for i in images:
            s = strata(i.get("narrative_summary"), i.get("width"), i.get("height"))
            if s:
                rows.append([s, i["image_id"], i.get("document_id"), i.get("page_number"),
                             chunk_of_img.get(i["image_id"], ""),
                             (i.get("narrative_summary") or "")[:300].replace("\n", " ")])
        rows.sort(key=lambda r: (r[0], r[2] or "", r[3] or 0))
        w.writerows(rows)
    saran = collections.Counter(r[0] for r in rows)

    # ---------- template ground truth ----------
    contoh_img = next((i for i in images if i.get("narrative_summary")), None)
    contoh_chunk = next((c for c in chunks if (c.get("element_type") or "").startswith("Narrative")), None)
    tpl = [
        {"query_id": "q0001",
         "document_id": (contoh_chunk or {}).get("document_id", "isi-dengan-document_id"),
         "query_type": "text_only",
         "visual_type": "",
         "question": "contoh pertanyaan berbasis teks",
         "reference_answer": "jawaban rujukan, disalin dari isi chunk",
         "relevant_text_chunks": [(contoh_chunk or {}).get("chunk_id", "")],
         "relevant_images": [],
         "annotator": "nama-anotator",
         "catatan": ""},
        {"query_id": "q0002",
         "document_id": (contoh_img or {}).get("document_id", ""),
         "query_type": "image_only",
         "visual_type": "flowchart_prosedural",
         "question": "contoh pertanyaan berbasis gambar",
         "reference_answer": "jawaban rujukan",
         "relevant_text_chunks": [],
         "relevant_images": [(contoh_img or {}).get("image_id", "")],
         "annotator": "nama-anotator",
         "catatan": ""},
    ]
    with open(out / "template_ground_truth.jsonl", "w", encoding="utf-8") as f:
        for t in tpl:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    # ---------- validator ----------
    (out / "validasi_ground_truth.py").write_text('''#!/usr/bin/env python3
"""validasi_ground_truth.py — cek ground truth SEBELUM dikirim balik.

  python3 validasi_ground_truth.py ground_truth.jsonl

Memastikan setiap chunk_id dan image_id yang dirujuk benar-benar ada.
"""
import csv, json, sys, collections
from pathlib import Path

here = Path(__file__).parent
if len(sys.argv) < 2:
    sys.exit("pemakaian: python3 validasi_ground_truth.py <berkas.jsonl>")

chunk_ids, img_ids = set(), set()
for p in (here / "chunks_jsonl").glob("*.jsonl"):
    for l in open(p, encoding="utf-8"):
        if l.strip():
            chunk_ids.add(json.loads(l)["chunk_id"])
with open(here / "daftar_gambar.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        img_ids.add(r["image_id"])

rows, bad_c, bad_i, kosong = [], [], [], 0
dup = collections.Counter()
for n, l in enumerate(open(sys.argv[1], encoding="utf-8"), 1):
    if not l.strip():
        continue
    try:
        r = json.loads(l)
    except Exception as e:
        print(f"  baris {n}: JSON rusak — {e}"); continue
    rows.append(r); dup[r.get("query_id")] += 1
    cs = r.get("relevant_text_chunks") or []
    ims = r.get("relevant_images") or []
    if isinstance(cs, str): cs = [x.strip() for x in cs.replace(";", ",").split(",") if x.strip()]
    if isinstance(ims, str): ims = [x.strip() for x in ims.replace(";", ",").split(",") if x.strip()]
    if not cs and not ims: kosong += 1
    bad_c += [(r.get("query_id"), c) for c in cs if c not in chunk_ids]
    bad_i += [(r.get("query_id"), i) for i in ims if i not in img_ids]

print(f"item              : {len(rows)}")
print(f"chunk_id dikenal  : {len(chunk_ids):,}")
print(f"image_id dikenal  : {len(img_ids):,}")
print()
print(f"query_id duplikat : {sum(1 for v in dup.values() if v > 1)}")
print(f"tanpa rujukan     : {kosong}")
print(f"chunk_id tak ada  : {len(bad_c)}")
for q, c in bad_c[:10]: print(f"    {q}: {c}")
print(f"image_id tak ada  : {len(bad_i)}")
for q, i in bad_i[:10]: print(f"    {q}: {i}")
print()
by = collections.Counter(r.get("visual_type") or "(kosong)" for r in rows)
print("sebaran visual_type:")
for k, v in by.most_common(): print(f"    {v:>4}  {k}")
ok = not bad_c and not bad_i and kosong == 0
print()
print("LOLOS — siap dikirim" if ok else "PERBAIKI dulu sebelum dikirim")
sys.exit(0 if ok else 1)
''', encoding="utf-8")

    # ---------- README ----------
    n_doc, n_chunk, n_img = len(by_doc), len(chunks), len(images)
    readme = f"""# Paket untuk Tim Evaluasi

Semua bahan untuk membuat ground truth, memakai penomoran yang **sama persis**
dengan sistem yang akan dievaluasi. Tidak perlu mengekstrak PDF lagi.

| | |
|---|---|
| Dokumen | {n_doc} |
| Chunk | {n_chunk:,} |
| Gambar | {n_img:,} |
| Sumber | `{dump.name}` |
| Collection | `{manifest.get('provenance',{}).get('qdrant_collection','')}` |

## Kenapa harus pakai penomoran ini

Percobaan sebelumnya memakai hasil ekstraksi terpisah, dan hanya 25% id yang
bisa dicocokkan. Penyebabnya bukan penamaan, tapi ekstraksinya memang
menghasilkan gambar yang berbeda: halaman dokumen hasil scan terekstrak
sebagai gambar utuh di satu sisi, dan sebagai teks hasil OCR di sisi lain.

Dengan memakai paket ini, `chunk_id` dan `image_id` yang Anda tulis pasti
menunjuk ke chunk yang sama dengan yang ada di sistem.

## Isi paket

| Berkas / folder | Isi |
|---|---|
| `chunks_markdown/` | Satu berkas per dokumen, teks penuh, tiap blok diawali `chunk_id`. **Ini yang paling nyaman diberikan ke AI lokal.** |
| `chunks_jsonl/` | Data yang sama dalam JSONL, satu berkas per dokumen |
| `gambar/` | Berkas gambar asli, tersusun per dokumen |
| `daftar_gambar.csv` | `image_id`, halaman, ukuran, deskripsi, saran tipe visual |
| `daftar_dokumen.csv` | Inventaris: jumlah chunk teks/tabel/gambar per dokumen |
| `kandidat_strata.csv` | Gambar yang terindikasi flowchart / tabel / formulir / figur, untuk memenuhi target strata |
| `template_ground_truth.jsonl` | Contoh format keluaran |
| `validasi_ground_truth.py` | Pemeriksa mandiri sebelum berkas dikirim balik |

## Cara kerja yang disarankan

1. Pilih dokumen dari `daftar_dokumen.csv`.
2. Berikan `chunks_markdown/<document_id>.md` ke AI lokal Anda, minta ia
   membuat pertanyaan **beserta `chunk_id` sumbernya**. `chunk_id` sudah
   tertulis di atas tiap blok, jadi tinggal disalin.
3. Untuk pertanyaan berbasis gambar, buka berkasnya di `gambar/`, dan salin
   `image_id` dari `daftar_gambar.csv`.
4. Tulis hasilnya mengikuti `template_ground_truth.jsonl`.
5. Jalankan `python3 validasi_ground_truth.py <berkas-anda>.jsonl` sampai
   statusnya LOLOS.

## Format keluaran

```json
{{"query_id": "q0001", "document_id": "...", "query_type": "text_only",
 "visual_type": "", "question": "...", "reference_answer": "...",
 "relevant_text_chunks": ["chunk_id_1"], "relevant_images": [],
 "annotator": "...", "catatan": ""}}
```

- `query_type`: `text_only` · `image_only` · `table_only` · `step_order`
- `visual_type` (untuk pertanyaan berbasis visual): `flowchart_prosedural` ·
  `tabel_persyaratan` · `formulir_template` · `figur_deskriptif`
- `relevant_text_chunks` dan `relevant_images`: **daftar**, boleh lebih dari satu
- `reference_answer`: jawaban rujukan. Kalau ada catatan koreksi, taruh di
  `catatan`, jangan di kolom jawaban.

## Target sebaran strata

25 pertanyaan per strata. Kandidatnya sudah disaring di `kandidat_strata.csv`:

{chr(10).join(f'- {k}: {v} kandidat' for k, v in saran.most_common()) or '- (belum ada saran otomatis)'}

Kolom `saran_visual_type` hanya tebakan dari deskripsi otomatis. **Label final
tetap ditentukan anotator** setelah melihat gambarnya.

## Yang tidak boleh diubah

`chunk_id` dan `image_id` disalin apa adanya. Jangan dibuat penomoran baru,
jangan dipotong, jangan diubah huruf besar-kecilnya. Kalau id-nya berbeda,
ground truth tidak bisa dicocokkan dengan hasil sistem.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    shutil.copy2(dump / "run_manifest.json", out / "run_manifest.json")

    # ---------- paket dosen: satu CSV gabungan + README ----------
    ringkas = tulis_semua_chunk(out / "semua_chunk.csv", baris_semua_chunk(chunks, images))
    (out / "README_paket.md").write_text(readme_paket(
        ringkas, dump.name, manifest.get("provenance", {}).get("qdrant_collection", "")),
        encoding="utf-8")

    print(f"Paket ditulis: {out}")
    print(f"  dokumen {n_doc} · chunk {n_chunk:,} · gambar {n_img:,}")
    print(f"  gambar disalin {disalin}" + (f", gagal {gagal}" if gagal else ""))
    print(f"  semua_chunk.csv {ringkas['baris']:,} baris · per modality {ringkas['per_modality']}")
    if ringkas["sel_melebihi_batas_excel"]:
        print(f"  PERINGATAN: {ringkas['sel_melebihi_batas_excel']} chunk > 32.767 karakter "
              "(terpotong bila dibuka di Excel)")
    print(f"  ukuran paket {format_ukuran(ukuran_folder(out))} · "
          f"folder gambar {format_ukuran(ukuran_folder(out / 'gambar'))}")
    print()
    print("  kandidat strata:")
    for k, v in saran.most_common():
        print(f"    {v:>4}  {k}")


if __name__ == "__main__":
    main()
