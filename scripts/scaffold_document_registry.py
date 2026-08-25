"""scaffold_document_registry.py — generate/perbarui document_registry.json.

Memindai folder PDF, menghitung sha256 tiap berkas, dan mengisi `document_id`
OTOMATIS dari nama berkas lewat `document_registry.slugify()`.

Registry adalah ARTEFAK BERSAMA, bukan hasil algoritma yang dijalankan
masing-masing fork. `document_id` menentukan `chunk_id`, `image_id`, dan nama
direktori gambar sekaligus — jadi konsistensi harus datang dari berkas yang
sama, bukan dari dua eksekusi yang kebetulan sepakat.

JAMINAN: ENTRI LAMA TIDAK PERNAH DITIMPA
----------------------------------------
Slug hanya dihitung untuk nama berkas yang BELUM punya `document_id`. Entri yang
sudah terisi disalin apa adanya — termasuk bila aturan slug berubah di kemudian
hari, dan termasuk bila seseorang mengeditnya manual untuk menyelesaikan
tabrakan.

Yang membuat jaminan itu terlihat: tiap entri auto membawa `slug_rule_version`
yang berlaku saat ia dibuat. Registry yang memuat entri dari beberapa versi
aturan tetap dapat ditelusuri, dan `document_registry_notes.md` melaporkan entri
yang `document_id`-nya tidak lagi sama dengan slug aturan sekarang.

TABRAKAN GAGAL KERAS
--------------------
Dua berkas yang menghasilkan slug sama menghentikan proses, dengan daftar
berkasnya. TIDAK ada sufiks otomatis: sufiks berbasis urutan tidak stabil, dan
`document_id` yang bergeser membatalkan seluruh anotasi gold. Tabrakan
diselesaikan manual dengan mengedit registry — satu-satunya bagian yang manual.

Aman dijalankan berulang: berkas baru ditambahkan, `sha256` diperbarui bila isi
berubah, entri untuk berkas yang sudah tidak ada dipertahankan dan ditandai.

Usage:
    python scripts/scaffold_document_registry.py
    python scripts/scaffold_document_registry.py --pdf-dir data/pdfs
    python scripts/scaffold_document_registry.py --out data/document_registry.json
    python scripts/scaffold_document_registry.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import DATA_DIR, DOCUMENT_REGISTRY_PATH
from backend.services.document_registry import (
    SLUG_RULE_VERSION, slugify, split_document_block,
)
from backend.services.preprocessing import file_sha256

NOTES_NAME = "document_registry_notes.md"


def _load_existing(path: Path) -> tuple[dict, dict]:
    if not path.exists():
        return {}, {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"[ERROR] {path} bukan JSON valid — {e}")
        print("        Perbaiki atau pindahkan dulu; scaffolder menolak menimpanya.")
        sys.exit(1)
    if not isinstance(raw, dict):
        print(f"[ERROR] {path}: akar JSON harus object.")
        sys.exit(1)
    meta, docs = split_document_block(raw)
    return meta, {k: dict(v) if isinstance(v, dict) else {} for k, v in docs.items()}


def _write_notes(path: Path, ringkasan: dict) -> None:
    """Catatan yang ikut dibagikan bersama registry.

    Peringatan TIDAK cukup di stdout: berkas inilah yang dibaca peneliti lain
    sebelum mulai menganotasi, dan mereka perlu tahu ada `document_id` yang
    tidak terbaca manusia sebelum, bukan sesudah.
    """
    L: list[str] = []
    A = L.append
    A("# Catatan document_registry.json")
    A("")
    A("Dibuat otomatis oleh `scripts/scaffold_document_registry.py`. **Jangan diedit")
    A("manual** — berkas ini ditulis ulang setiap scaffolder dijalankan.")
    A("")
    A(f"| | |")
    A(f"|---|---|")
    A(f"| Registry | `{ringkasan['registry_path']}` |")
    A(f"| Di-generate | {ringkasan['generated_at']} |")
    A(f"| Versi aturan slug | {ringkasan['slug_rule_version']} |")
    A(f"| Folder PDF | `{ringkasan['pdf_dir']}` |")
    A(f"| PDF terbaca | {ringkasan['n_pdf']} |")
    A(f"| Entri registry | {ringkasan['n_entries']} |")
    A(f"| Entri baru run ini | {ringkasan['n_new']} |")
    A("")

    if ringkasan["digit_only"]:
        A("## ⚠ document_id yang hanya berisi angka")
        A("")
        A("Slug ini valid dan unik, tapi **tidak terbaca manusia**. `chunk_id` dan")
        A("`image_id` yang diturunkan darinya juga tidak. Pertimbangkan mengedit")
        A("registry sebelum anotasi dimulai — setelah anotasi berjalan, mengubahnya")
        A("membatalkan seluruh `gold_chunk_ids` dan `gold_image_ids` dokumen itu.")
        A("")
        A("| Berkas | document_id |")
        A("|---|---|")
        for f, d in ringkasan["digit_only"]:
            A(f"| `{f}` | `{d}` |")
        A("")

    if ringkasan["drifted"]:
        A("## Entri yang tidak lagi sama dengan slug aturan sekarang")
        A("")
        A("Sengaja **tidak ditimpa**. Penyebabnya salah satu dari dua: entri dibuat")
        A("di bawah versi aturan slug yang lebih lama, atau seseorang mengeditnya")
        A("manual untuk menyelesaikan tabrakan. Keduanya harus dipertahankan —")
        A("`document_id` yang berubah membatalkan anotasi gold.")
        A("")
        A("| Berkas | document_id tersimpan | slug aturan sekarang | versi aturan |")
        A("|---|---|---|---|")
        for f, tersimpan, sekarang, versi in ringkasan["drifted"]:
            A(f"| `{f}` | `{tersimpan}` | `{sekarang}` | {versi} |")
        A("")

    if ringkasan["changed_sha"]:
        A("## ⚠ Isi berkas berubah")
        A("")
        A("Nama berkas sama, sha256 berbeda. Bila ini revisi dokumen, revisinya")
        A("biasanya butuh `document_id` BARU — bukan memakai ulang yang lama.")
        A("")
        A("| Berkas | sha256 lama | sha256 baru |")
        A("|---|---|---|")
        for f, lama, baru in ringkasan["changed_sha"]:
            A(f"| `{f}` | `{lama[:16]}` | `{baru[:16]}` |")
        A("")

    if ringkasan["missing"]:
        A("## Entri tanpa berkas di folder PDF")
        A("")
        A("Dipertahankan, tidak dihapus — anotasi gold bisa jadi masih menunjuk ke")
        A("sana.")
        A("")
        for f in ringkasan["missing"]:
            A(f"- `{f}`")
        A("")

    A("## Siapa yang boleh menjalankan generate ulang")
    A("")
    A("Registry ini **satu berkas di lokasi bersama** — tidak dikirim-kirim antar")
    A("peneliti. Yang perlu disepakati bukan cara mengirimnya, tapi siapa yang")
    A("boleh menjalankan ulang dan kapan.")
    A("")
    A("**Aman dijalankan ulang kapan saja** — menambah entri untuk PDF baru tanpa")
    A("menyentuh yang lama. Menjalankannya dua kali berturut-turut tanpa PDF baru")
    A("tidak mengubah apa pun kecuali stempel waktu.")
    A("")
    A("**Butuh kesepakatan lebih dulu:**")
    A("")
    A("- Setelah anotasi gold dimulai. Entri lama memang tidak ditimpa, tapi PDF")
    A("  baru menambah dokumen ke korpus, dan itu mengubah komposisi strata.")
    A("- Bila ada tabrakan slug yang harus diselesaikan manual. Suntingan itu")
    A("  menentukan `document_id` permanen dokumen tersebut.")
    A("- Bila `sha256` sebuah berkas berubah. Perlu diputuskan apakah itu revisi")
    A("  yang butuh `document_id` baru.")
    A("")
    A("**Jangan pernah:** menghapus registry lalu generate ulang dari nol setelah")
    A("indexing berjalan. Urutan pemindaian folder dapat berbeda, dan walau slug")
    A("bersifat deterministik terhadap nama berkas, entri hasil suntingan manual")
    A("akan hilang — `document_id` berubah, dan seluruh `gold_chunk_ids` serta")
    A("`gold_image_ids` yang menunjuknya jadi tidak valid.")
    A("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate/perbarui document_registry.json")
    ap.add_argument("--pdf-dir", default=DATA_DIR, help="Folder PDF (default: DATA_DIR)")
    ap.add_argument("--out", default=DOCUMENT_REGISTRY_PATH, help="Path registry")
    ap.add_argument("--dry-run", action="store_true", help="Tampilkan saja, jangan tulis")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out_path = Path(args.out)

    meta_lama, registry = _load_existing(out_path)

    if pdf_dir.exists():
        pdf_files = sorted(pdf_dir.glob("*.pdf"))
    else:
        print(f"[WARN ] Folder PDF tidak ada: {pdf_dir}")
        print("        Registry tetap ditulis dari entri yang sudah ada.")
        pdf_files = []

    baru: list[tuple[str, str]] = []
    changed_sha: list[tuple[str, str, str]] = []
    kosong: list[str] = []

    for pdf in pdf_files:
        sha = file_sha256(pdf)
        entry = registry.get(pdf.name)

        if entry is None or not (entry.get("document_id") or "").strip():
            slug = slugify(pdf.name)
            if not slug:
                kosong.append(pdf.name)
                continue
            registry[pdf.name] = {
                **(entry or {}),
                "document_id": slug,
                "sha256": sha,
                "slug_rule_version": SLUG_RULE_VERSION,
                "source": "auto",
            }
            baru.append((pdf.name, slug))
            continue

        # ── Entri yang sudah punya document_id TIDAK PERNAH ditimpa ──
        lama = (entry.get("sha256") or "").strip()
        if lama and lama != sha:
            changed_sha.append((pdf.name, lama, sha))
        entry["sha256"] = sha
        registry[pdf.name] = entry

    # ── Slug kosong: gagal keras ────────────────────────────────────────────
    if kosong:
        print(f"\n[ERROR] {len(kosong)} berkas menghasilkan slug KOSONG:")
        for f in kosong:
            print(f"          {f}")
        print("        Tidak ada karakter alfanumerik yang tersisa setelah normalisasi.")
        print("        Ganti nama berkasnya, atau tambahkan entri manual di registry.")
        sys.exit(1)

    # ── Tabrakan: gagal keras, JANGAN tulis apa pun ─────────────────────────
    by_slug: dict[str, list[str]] = defaultdict(list)
    for fname, e in registry.items():
        did = (e.get("document_id") or "").strip()
        if did:
            by_slug[did].append(fname)
    bentrok = {s: fs for s, fs in by_slug.items() if len(fs) > 1}

    if bentrok:
        print(f"\n[ERROR] {len(bentrok)} document_id dipakai lebih dari satu berkas:")
        for slug, files in sorted(bentrok.items()):
            print(f"\n          {slug!r}")
            for f in sorted(files):
                print(f"            - {f}")
        print("\n        TIDAK ada sufiks otomatis: sufiks berbasis urutan tidak stabil,")
        print("        dan document_id yang bergeser membatalkan seluruh anotasi gold.")
        print("        Selesaikan manual — edit document_id salah satu berkas di")
        print(f"        {out_path}, lalu jalankan ulang.")
        print("\n        Registry TIDAK ditulis.")
        sys.exit(1)

    # ── Laporan ─────────────────────────────────────────────────────────────
    ada_di_disk = {p.name for p in pdf_files}
    missing = sorted(n for n in registry if n not in ada_di_disk)
    digit_only = sorted(
        (n, e["document_id"]) for n, e in registry.items()
        if (e.get("document_id") or "").replace("-", "").isdigit()
    )
    drifted = sorted(
        (n, e["document_id"], slugify(n), e.get("slug_rule_version", "?"))
        for n, e in registry.items()
        if (e.get("document_id") or "").strip()
        and e["document_id"] != slugify(n)
    )

    print(f"Folder PDF : {pdf_dir}")
    print(f"Registry   : {out_path}")
    print(f"PDF terbaca: {len(pdf_files)}")
    print()
    print(f"  entri baru (document_id di-generate)   : {len(baru)}")
    for n, s in baru[:20]:
        print(f"      + {n}")
        print(f"        -> {s}")
    if len(baru) > 20:
        print(f"      ... dan {len(baru) - 20} lagi")
    print(f"  entri lama (TIDAK ditimpa)             : {len(registry) - len(baru)}")

    if digit_only:
        print(f"  document_id hanya angka                : {len(digit_only)}")
        for n, d in digit_only:
            print(f"      ! {n} -> {d}")
        print("        Valid dan unik, tapi tidak terbaca manusia.")
        print(f"        Tercatat juga di {NOTES_NAME}.")

    if drifted:
        print(f"  beda dari slug aturan sekarang         : {len(drifted)}")
        print("        Sengaja tidak ditimpa (aturan lama atau suntingan manual).")

    if changed_sha:
        print(f"  ISI BERUBAH (sha256 berbeda)           : {len(changed_sha)}")
        for n, l, b in changed_sha:
            print(f"      ! {n}: {l[:16]} -> {b[:16]}")
        print("        Bila ini revisi, biasanya butuh document_id BARU.")

    if missing:
        print(f"  entri tanpa berkas (dipertahankan)     : {len(missing)}")

    if args.dry_run:
        print("\n[dry-run] tidak ada yang ditulis.")
        return

    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "_meta": {
            "generated_at": generated_at,
            "slug_rule_version": SLUG_RULE_VERSION,
            "tool": "scripts/scaffold_document_registry.py",
            "n_documents": len(registry),
            "catatan": (
                "Artefak bersama. Entri yang sudah punya document_id tidak pernah "
                "ditimpa. Tabrakan slug diselesaikan manual. Lihat "
                f"{NOTES_NAME} di direktori yang sama."
            ),
        },
        "documents": dict(sorted(registry.items())),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    notes_path = out_path.parent / NOTES_NAME
    _write_notes(notes_path, {
        "registry_path": str(out_path),
        "generated_at": generated_at,
        "slug_rule_version": SLUG_RULE_VERSION,
        "pdf_dir": str(pdf_dir),
        "n_pdf": len(pdf_files),
        "n_entries": len(registry),
        "n_new": len(baru),
        "digit_only": digit_only,
        "drifted": drifted,
        "changed_sha": changed_sha,
        "missing": missing,
    })

    print(f"\nDitulis: {out_path}  ({len(registry)} entri)")
    print(f"         {notes_path}")
    if meta_lama.get("slug_rule_version") not in (None, SLUG_RULE_VERSION):
        print(f"\n[WARN ] Registry sebelumnya dibuat dengan aturan slug versi "
              f"{meta_lama['slug_rule_version']}, sekarang {SLUG_RULE_VERSION}.")
        print("        Entri lama tetap dipertahankan apa adanya.")


if __name__ == "__main__":
    main()
