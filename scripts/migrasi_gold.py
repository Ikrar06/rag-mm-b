"""migrasi_gold.py — petakan ground truth lama ke index baru.

Berjangkar `text_sha`: chunk yang isinya tidak berubah tetap dikenali meski
penomorannya bergeser.

Kenapa ini sebagian besar VERIFIKASI
------------------------------------
Tahap B tidak menggeser `chunk_id`. Mengulang baris header tidak menambah atau
mengurangi chunk, dan `ordinal` hanya naik saat sebuah chunk benar-benar
di-`append`. Yang diharapkan: mayoritas item terpetakan identik, sebagian kecil
`isi_berubah` (potongan lanjutan yang headernya diulang), dan nol hilang.

Dua kasus tetap dideteksi, bukan diasumsikan tidak ada:
  - chunk yang sebelumnya dibuang filter INDEX_MIN_CHUNK_TOKENS lalu lolos
    setelah ditambahi header, sehingga menggeser chunk_id sesudahnya
  - chunk yang benar-benar tidak lagi diproduksi

Keluaran
--------
    ground_truth_migrated.jsonl       item yang terpetakan, SELURUH kolom
                                      apa adanya, siap untuk
                                      validasi_ground_truth.py
    ground_truth_perlu_tinjau.jsonl   bagian dari yang di atas yang isinya
                                      berubah — butuh tinjauan ulang manusia.
                                      Membawa field diagnostik `_migrasi`.
    ground_truth_tidak_terpetakan.jsonl  item yang TIDAK dibuang, dipisahkan
                                      beserta alasannya per chunk
    laporan_migrasi.json              ringkasan angka

Usage:
    python scripts/migrasi_gold.py \\
        --gold ~/rag_mm_b/data/eval/ground_truth_final_20260906.jsonl \\
        --chunks-baru dump/<run_baru>/chunks.jsonl \\
        --chunks-lama dump/<run_lama>/chunks.jsonl \\
        --out-dir data/eval/migrasi

`--chunks-lama` opsional tapi SANGAT disarankan: tanpa itu perubahan isi tidak
dapat dideteksi, dan item yang chunk_id-nya bergeser tidak punya jangkar.
Ketiadaannya dilaporkan, bukan disamarkan.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.gold_migrasi import (  # noqa: E402
    STATUS_AMBIGU, STATUS_HILANG, STATUS_IDENTIK, STATUS_ISI_BERUBAH,
    STATUS_PINDAH, bangun_indeks, petakan_item, query_id_usang, slug_dokumen,
    terapkan,
)

URUTAN_STATUS = (STATUS_IDENTIK, STATUS_ISI_BERUBAH, STATUS_PINDAH,
                 STATUS_AMBIGU, STATUS_HILANG)


def baca_jsonl(path: Path) -> list[dict]:
    out = []
    for i, ln in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not ln.strip():
            continue
        try:
            rec = json.loads(ln)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} baris {i}: JSON tidak valid — {e}") from e
        if not isinstance(rec, dict):
            raise ValueError(f"{path} baris {i}: baris harus object")
        out.append(rec)
    return out


def tulis_jsonl(path: Path, baris) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in baris:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def _diagnostik(h) -> dict:
    return {
        "status": h.status,
        "chunks": [{"lama": c.lama, "baru": c.baru, "status": c.status,
                    "alasan": c.alasan} for c in h.chunks],
        "image_hilang": list(h.image_hilang),
        "masalah": list(h.masalah),
        "query_id_usang": query_id_usang(h),
    }


def muat_chunk(args) -> tuple[list[dict], list[str], str]:
    """Rekaman chunk index baru + daftar image_id + asal datanya."""
    if args.chunks_baru:
        p = Path(args.chunks_baru).expanduser()
        rows = baca_jsonl(p)
        images = []
        if args.images_baru:
            images = [r.get("image_id") for r in
                      baca_jsonl(Path(args.images_baru).expanduser())]
        return rows, images, str(p)

    from backend.config import QDRANT_COLLECTION_NAME
    from backend.services.index_verify import scan_chunk_ids  # noqa: F401
    from backend.services.indexing import get_qdrant_client

    koleksi = args.collection or QDRANT_COLLECTION_NAME
    client = get_qdrant_client()
    rows, images, offset = [], [], None
    while True:
        titik, offset = client.scroll(collection_name=koleksi, limit=1000,
                                      offset=offset, with_payload=True,
                                      with_vectors=False)
        if not titik:
            break
        for pt in titik:
            pl = pt.payload or {}
            if "_node_content" in pl:
                try:
                    dalam = (json.loads(pl["_node_content"]) or {}).get("metadata") or {}
                except Exception:
                    dalam = {}
                pl = {**dalam, **pl}
            rows.append({"chunk_id": pl.get("chunk_id"), "text_sha": pl.get("text_sha")})
            if pl.get("image_id"):
                images.append(pl["image_id"])
        if offset is None:
            break
    return rows, images, f"qdrant:{koleksi}"



def _hitung_tabrakan(path: Path) -> int:
    """Hitung text_sha yang dipakai lebih dari satu chunk.

    Menjawab satu pertanyaan sebelum migrasi dijalankan: berapa banyak chunk
    yang isinya tidak unik, sehingga `text_sha` tidak dapat dipakai sebagai
    jangkar tunggal.

    Yang menentukan bukan angka totalnya, melainkan pemecahannya: tabrakan
    LINTAS dokumen dapat diselesaikan jangkar pembeda `document_id`, sedangkan
    tabrakan DI DALAM satu dokumen tidak — teks identik berulang di dokumen yang
    sama tidak dapat dipilih tanpa menebak.
    """
    try:
        rows = baca_jsonl(path)
    except (ValueError, OSError) as e:
        print(f"GAGAL membaca {path}: {e}")
        return 2

    per_sha: dict[str, list[str]] = {}
    tanpa_sha = 0
    for r in rows:
        cid, sha = r.get("chunk_id"), r.get("text_sha")
        if not isinstance(cid, str) or not cid:
            continue
        if not isinstance(sha, str) or not sha:
            tanpa_sha += 1
            continue
        per_sha.setdefault(sha, []).append(cid)

    tabrakan = {s: ids for s, ids in per_sha.items() if len(ids) > 1}
    n_chunk_kena = sum(len(v) for v in tabrakan.values())
    total = sum(len(v) for v in per_sha.values())

    # Pisahkan yang terselesaikan jangkar dokumen dari yang tidak.
    lintas, dalam, dalam_chunk = 0, 0, 0
    for ids in tabrakan.values():
        per_dok: dict[str, int] = {}
        for cid in ids:
            d = slug_dokumen(cid) or "?"
            per_dok[d] = per_dok.get(d, 0) + 1
        if max(per_dok.values()) == 1:
            lintas += 1                      # tiap dokumen punya satu -> terselesaikan
        else:
            dalam += 1
            dalam_chunk += sum(n for n in per_dok.values() if n > 1)

    print(f"Sumber : {path}")
    print(f"Chunk  : {total} ber-text_sha" + (f", {tanpa_sha} TANPA text_sha" if tanpa_sha else ""))
    print(f"text_sha unik : {len(per_sha)}")
    print()
    print("=" * 72)
    print("TABRAKAN text_sha")
    print("=" * 72)
    print(f"  {'text_sha dipakai >1 chunk':<38}{len(tabrakan):>7}"
          f"{100 * len(tabrakan) / max(len(per_sha), 1):>7.1f}%")
    print(f"  {'chunk yang isinya tidak unik':<38}{n_chunk_kena:>7}"
          f"{100 * n_chunk_kena / max(total, 1):>7.1f}%")
    print()
    print(f"  {'terselesaikan jangkar document_id':<38}{lintas:>7}"
          f"   (tersebar lintas dokumen)")
    print(f"  {'TETAP ambigu':<38}{dalam:>7}"
          f"   ({dalam_chunk} chunk, teks identik berulang di dokumen yang sama)")

    if tabrakan:
        print("\n  TERBESAR:")
        for sha, ids in sorted(tabrakan.items(), key=lambda kv: -len(kv[1]))[:5]:
            dok = {slug_dokumen(c) or "?" for c in ids}
            sifat = "lintas dokumen" if len(dok) == len(ids) else f"{len(dok)} dokumen"
            print(f"    {len(ids):>5} chunk  sha={sha[:16]}  {sifat}")
            print(f"           contoh: {', '.join(ids[:3])}")

    print("\n  Angka yang menentukan adalah baris 'TETAP ambigu': hanya itu yang")
    print("  akan masuk ground_truth_tidak_terpetakan.jsonl, dan hanya bila ada")
    print("  item gold yang merujuknya.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Migrasi ground truth ke index baru")
    ap.add_argument("--gold", default=None, help="ground_truth_final_*.jsonl")
    ap.add_argument("--chunks-baru", default=None,
                    help="chunks.jsonl dari run BARU (alternatif --collection)")
    ap.add_argument("--images-baru", default=None, help="images.jsonl dari run BARU")
    ap.add_argument("--collection", default=None, help="Baca index baru dari Qdrant")
    ap.add_argument("--chunks-lama", default=None,
                    help="chunks.jsonl run LAMA — jangkar text_sha. Sangat disarankan.")
    ap.add_argument("--out-dir", default="migrasi_gold")
    ap.add_argument("--hitung-tabrakan", action="store_true",
                    help="Hitung text_sha yang muncul lebih dari sekali lalu "
                         "berhenti. Tidak menjalankan migrasi, tidak butuh --gold.")
    args = ap.parse_args()

    if args.hitung_tabrakan:
        sumber = args.chunks_lama or args.chunks_baru
        if not sumber:
            print("GAGAL: --hitung-tabrakan butuh --chunks-lama atau --chunks-baru")
            return 2
        return _hitung_tabrakan(Path(sumber).expanduser())

    if not args.gold:
        print("GAGAL: --gold wajib kecuali memakai --hitung-tabrakan")
        return 2
    if not args.chunks_baru and not args.collection:
        print("GAGAL: beri --chunks-baru atau --collection")
        return 2

    gold_path = Path(args.gold).expanduser()
    if not gold_path.exists():
        print(f"GAGAL: {gold_path} tidak ditemukan")
        return 2

    try:
        gold = baca_jsonl(gold_path)
        rows, images, asal = muat_chunk(args)
    except (ValueError, OSError) as e:
        print(f"GAGAL membaca masukan: {e}")
        return 2

    sha_lama: dict[str, str] = {}
    teks_lama: dict[str, str] = {}
    if args.chunks_lama:
        for r in baca_jsonl(Path(args.chunks_lama).expanduser()):
            cid = r.get("chunk_id")
            if not isinstance(cid, str) or not cid:
                continue
            if isinstance(r.get("text_sha"), str):
                sha_lama[cid] = r["text_sha"]
            isi = r.get("text_content") or r.get("text")
            if isinstance(isi, str) and isi:
                teks_lama[cid] = isi

    indeks = bangun_indeks(rows, images)

    print(f"Gold        : {gold_path}  ({len(gold)} item)")
    print(f"Index baru  : {asal}  ({indeks.n_chunk} chunk, {len(indeks.image_ids)} gambar)")
    print(f"Jangkar lama: {len(sha_lama)} text_sha, {len(teks_lama)} teks"
          + ("" if sha_lama else "  <-- TIDAK ADA: perubahan isi tidak terdeteksi, "
                                "dan chunk yang bergeser tidak punya jangkar"))

    hasil = [petakan_item(it, sha_lama, indeks, teks_lama) for it in gold]

    out = Path(args.out_dir).expanduser()
    terpetakan = [h for h in hasil if h.terpetakan]
    n_mig = tulis_jsonl(out / "ground_truth_migrated.jsonl",
                        (terapkan(h) for h in terpetakan))
    perlu_tinjau = [h for h in terpetakan
                    if h.status in (STATUS_ISI_BERUBAH, STATUS_PINDAH)
                    or query_id_usang(h) or h.masalah]
    n_tinjau = tulis_jsonl(out / "ground_truth_perlu_tinjau.jsonl",
                           ({**terapkan(h), "_migrasi": _diagnostik(h)}
                            for h in perlu_tinjau))
    gagal = [h for h in hasil if not h.terpetakan]
    n_gagal = tulis_jsonl(out / "ground_truth_tidak_terpetakan.jsonl",
                          ({**h.item, "_migrasi": _diagnostik(h)} for h in gagal))

    per_status = Counter(h.status for h in hasil)
    per_verdict = Counter(str(h.item.get("human_verdict", "")) for h in hasil)
    verdict_mig = Counter(str(h.item.get("human_verdict", "")) for h in terpetakan)

    print("\n" + "=" * 72)
    print("HASIL MIGRASI")
    print("=" * 72)
    for s in URUTAN_STATUS:
        if per_status[s]:
            print(f"  {s:<16}{per_status[s]:>6}  {100 * per_status[s] / len(gold):>5.1f}%")

    print(f"\n  ground_truth_migrated.jsonl        {n_mig:>6}")
    print(f"  ground_truth_perlu_tinjau.jsonl   {n_tinjau:>6}   (isi berubah / bergeser)")
    print(f"  ground_truth_tidak_terpetakan.jsonl {n_gagal:>4}   TIDAK DIBUANG")

    print("\n  human_verdict — sebelum vs setelah migrasi:")
    for v in sorted(set(per_verdict) | set(verdict_mig)):
        label = v or "<kosong>"
        tanda = "" if per_verdict[v] == verdict_mig[v] else "   <-- ADA YANG HILANG"
        print(f"    {label:<12}{per_verdict[v]:>6} -> {verdict_mig[v]:>6}{tanda}")

    n_qid = sum(1 for h in hasil if query_id_usang(h))
    if n_qid:
        print(f"\n  {n_qid} item ber-query_id memuat chunk_id yang bergeser.")
        print("  query_id SENGAJA tidak diubah — ia identitas item, bukan rujukan")
        print("  ke chunk, dan mengubahnya memutus jejak ke hasil tinjauan manusia.")

    masalah = [h for h in hasil if h.masalah]
    if masalah:
        print(f"\n  {len(masalah)} item melanggar invarian yang dinyatakan tim eval:")
        for m, n in Counter(x for h in masalah for x in h.masalah).most_common():
            print(f"    {n:>4}  {m}")

    if gagal:
        print(f"\n  ALASAN item tidak terpetakan:")
        alasan = Counter(c.alasan for h in gagal for c in h.chunks if not c.terpetakan)
        for a, n in alasan.most_common(6):
            print(f"    {n:>4}  {a[:96]}")
        if any(h.image_hilang for h in gagal):
            n_img = sum(len(h.image_hilang) for h in gagal)
            print(f"    {n_img:>4}  image_id tidak ada di index baru")

    (out / "laporan_migrasi.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gold": str(gold_path), "index_baru": asal,
        "jangkar_text_sha": len(sha_lama), "jangkar_teks": len(teks_lama),
        "n_item": len(gold), "per_status": dict(per_status),
        "n_migrated": n_mig, "n_perlu_tinjau": n_tinjau, "n_tidak_terpetakan": n_gagal,
        "human_verdict_sebelum": dict(per_verdict),
        "human_verdict_setelah": dict(verdict_mig),
        "query_id_usang": n_qid,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n  laporan_migrasi.json: {out / 'laporan_migrasi.json'}")
    print("\n  Langkah berikutnya: jalankan validasi_ground_truth.py atas")
    print(f"  {out / 'ground_truth_migrated.jsonl'} — harus lolos tanpa galat.")
    return 1 if gagal else 0


if __name__ == "__main__":
    sys.exit(main())
