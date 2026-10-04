"""map_eval_ids.py — petakan image_id tim eval ke image_id pipeline riset.

Read-only. Menghasilkan berkas pemetaan + laporan cakupan.

Tiga lapis pencocokan, berurutan:
  1. sha256 bytes gambar  (eksak, kalau ekstraksinya menghasilkan bytes sama)
  2. dokumen + halaman + dimensi piksel  (kuat, toleran perbedaan encoding)
  3. dokumen + halaman, hanya bila kandidatnya tunggal

Pemakaian:
  python scripts/map_eval_ids.py \
      --eval-meta ~/rag_mm_b_shared/eval_meta \
      --our-images data/dumps/<run>/images.jsonl \
      --ground-truth ~/rag_mm_b_shared/ground_truth_final.json \
      --out ~/rag_mm_b_shared/id_mapping
"""
import argparse, collections, glob, hashlib, json, os, sys
from pathlib import Path


def load_ours(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    by_sha, by_key, by_docpage = {}, {}, collections.defaultdict(list)
    for r in rows:
        if r.get("sha256"):
            by_sha.setdefault(r["sha256"], r)
        k = (r.get("source_file"), r.get("page_number"), r.get("width"), r.get("height"))
        by_key.setdefault(k, []).append(r)
        by_docpage[(r.get("source_file"), r.get("page_number"))].append(r)
    return rows, by_sha, by_key, by_docpage


def sha_of(p):
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-meta", required=True)
    ap.add_argument("--our-images", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    meta_root = Path(os.path.expanduser(a.eval_meta))
    out_dir = Path(os.path.expanduser(a.out)); out_dir.mkdir(parents=True, exist_ok=True)

    ours, by_sha, by_key, by_docpage = load_ours(os.path.expanduser(a.our_images))
    print(f"gambar kami        : {len(ours)}")

    # peta pdf_id -> source_file dari all_summaries.json
    summ_path = meta_root / "all_summaries.json"
    pdfid2src = {}
    if summ_path.exists():
        for s in json.load(open(summ_path, encoding="utf-8")):
            pdfid2src[s["pdf_id"]] = s["source_file"]
    print(f"dokumen tim eval   : {len(pdfid2src)}")

    our_sources = {r.get("source_file") for r in ours}
    tak_ada_dok = sorted(v for v in pdfid2src.values() if v not in our_sources)

    mapping, unmatched = {}, []
    cara = collections.Counter()

    for mf in sorted(glob.glob(str(meta_root / "*" / "images_meta.json"))):
        folder = Path(mf).parent
        for e in json.load(open(mf, encoding="utf-8")):
            eid, page = e["image_id"], e.get("page")
            w, h = e.get("width"), e.get("height")
            pdf_id = eid.rsplit("_img_", 1)[0]
            src = pdfid2src.get(pdf_id)
            hit = None; how = None

            img_path = folder / e.get("file", "")
            if img_path.exists():
                s = sha_of(img_path)
                if s and s in by_sha:
                    hit, how = by_sha[s], "sha256"

            if hit is None and src:
                c = by_key.get((src, page, w, h), [])
                if len(c) == 1:
                    hit, how = c[0], "dok+hal+dimensi"

            if hit is None and src:
                c = by_docpage.get((src, page), [])
                if len(c) == 1:
                    hit, how = c[0], "dok+hal (tunggal)"

            if hit:
                cara[how] += 1
                mapping[eid] = {
                    "eval_image_id": eid, "our_image_id": hit["image_id"],
                    "document_id": hit["document_id"], "source_file": src,
                    "page_eval": page, "page_ours": hit.get("page_number"),
                    "cara": how,
                    "eval_decorative": e.get("is_likely_decorative"),
                    "our_has_summary": bool(hit.get("narrative_summary")),
                }
            else:
                unmatched.append({"eval_image_id": eid, "source_file": src,
                                  "page": page, "width": w, "height": h,
                                  "dokumen_ada_di_kami": src in our_sources if src else False})

    total = len(mapping) + len(unmatched)
    print(f"gambar tim eval    : {total}")
    print()
    print("HASIL PEMETAAN")
    for k, v in cara.most_common():
        print(f"  {v:>5}  {k}")
    print(f"  {len(unmatched):>5}  GAGAL")
    print(f"  cakupan: {len(mapping)}/{total} = {len(mapping)/total*100:.1f}%")

    if tak_ada_dok:
        print()
        print(f"dokumen tim eval yang TIDAK ada di korpus kami: {len(tak_ada_dok)}")
        for d in tak_ada_dok[:10]: print("   ", d)

    # dampak ke ground truth
    gt = json.load(open(os.path.expanduser(a.ground_truth), encoding="utf-8"))
    bisa = tidak = tanpa_img = 0
    per_visual = collections.Counter()
    for q in gt:
        ids = [x for x in str(q.get("relevant_images", "")).replace(";", ",").split(",") if x.strip()]
        ids = [x.strip() for x in ids]
        if not ids:
            tanpa_img += 1; continue
        if all(i in mapping for i in ids):
            bisa += 1; per_visual[q.get("visual_type") or "(kosong)"] += 1
        else:
            tidak += 1
    print()
    print("DAMPAK KE GROUND TRUTH")
    print(f"  total item            : {len(gt)}")
    print(f"  punya relevant_images : {len(gt)-tanpa_img}")
    print(f"    dapat dipetakan     : {bisa}")
    print(f"    TIDAK dapat         : {tidak}")
    print(f"  tanpa relevant_images : {tanpa_img}")
    print()
    print("  item terpetakan per visual_type:")
    for k, v in per_visual.most_common(): print(f"    {v:>4}  {k}")

    with open(out_dir / "image_id_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    with open(out_dir / "image_id_unmatched.json", "w", encoding="utf-8") as f:
        json.dump(unmatched, f, indent=2, ensure_ascii=False)
    print()
    print("Ditulis:", out_dir / "image_id_mapping.json")
    print("        ", out_dir / "image_id_unmatched.json")


if __name__ == "__main__":
    main()
