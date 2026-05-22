"""Fetch & extract PDF files dari zip archive ke folder tujuan.

Sumber bisa berupa:
  - Folder lokal yang berisi .zip file
  - GitHub repo URL (di-clone otomatis, lalu extract semua .zip di dalamnya)

Usage:
    # Dari GitHub repo (clone otomatis)
    python scripts/fetch_pdfs.py --source https://github.com/org/pdf-data-repo

    # Dari folder lokal berisi zip
    python scripts/fetch_pdfs.py --source /path/to/zips

    # Custom destination (default: data/pdfs/)
    python scripts/fetch_pdfs.py --source /path/to/zips --dest data/pdfs/

    # Force overwrite PDF yang sudah ada
    python scripts/fetch_pdfs.py --source /path/to/zips --force

    # Dry run (lihat apa yang akan di-extract tanpa eksekusi)
    python scripts/fetch_pdfs.py --source /path/to/zips --dry-run
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


# ─── ANSI colours untuk terminal output ────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def _log(level: str, msg: str):
    prefix = {
        "info":    f"{GREEN}[INFO]{RESET}",
        "skip":    f"{YELLOW}[SKIP]{RESET}",
        "warn":    f"{YELLOW}[WARN]{RESET}",
        "error":   f"{RED}[ERROR]{RESET}",
        "dry":     f"{CYAN}[DRY]{RESET}",
        "summary": f"{BOLD}[DONE]{RESET}",
    }.get(level, "[LOG]")
    print(f"{prefix}  {msg}", flush=True)


# ─── Git clone helper ───────────────────────────────────────────────────────

def _clone_repo(repo_url: str, target_dir: Path) -> Path:
    """Clone GitHub repo ke target_dir. Jika sudah ada, jalankan git pull."""
    if target_dir.exists() and (target_dir / ".git").exists():
        _log("info", f"Repo sudah ada di {target_dir}, menjalankan git pull...")
        result = subprocess.run(
            ["git", "-C", str(target_dir), "pull", "--ff-only"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            _log("warn", f"git pull gagal: {result.stderr.strip()} — pakai versi lokal")
        else:
            _log("info", result.stdout.strip() or "Already up to date.")
    else:
        _log("info", f"Cloning {repo_url} → {target_dir}")
        target_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(target_dir)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            _log("error", f"git clone gagal:\n{result.stderr.strip()}")
            sys.exit(1)
        _log("info", "Clone selesai.")
    return target_dir


# ─── Core extract logic ─────────────────────────────────────────────────────

def _extract_zips(
    zip_dir: Path,
    dest_dir: Path,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    """Extract semua .zip di zip_dir, ambil hanya file .pdf, simpan ke dest_dir.

    Returns:
        dict berisi stats: extracted, skipped, overwritten, errors
    """
    zip_files = sorted(zip_dir.rglob("*.zip"))

    if not zip_files:
        _log("warn", f"Tidak ada file .zip ditemukan di {zip_dir}")
        return {"extracted": 0, "skipped": 0, "overwritten": 0, "errors": 0}

    _log("info", f"Ditemukan {len(zip_files)} file .zip di {zip_dir}")
    _log("info", f"Destination: {dest_dir.resolve()}")

    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)

    stats = {"extracted": 0, "skipped": 0, "overwritten": 0, "errors": 0}

    for zip_path in zip_files:
        _log("info", f"Memproses: {zip_path.name}")

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Filter hanya entry yang berakhiran .pdf (case-insensitive)
                pdf_entries = [
                    entry for entry in zf.infolist()
                    if not entry.is_dir() and entry.filename.lower().endswith(".pdf")
                ]

                if not pdf_entries:
                    _log("warn", f"  Tidak ada PDF di {zip_path.name}, skip.")
                    continue

                for entry in pdf_entries:
                    # Ambil nama file saja (buang path dalam zip)
                    pdf_name = Path(entry.filename).name
                    dest_path = dest_dir / pdf_name

                    if dest_path.exists() and not force:
                        _log("skip", f"  {pdf_name} (sudah ada, gunakan --force untuk overwrite)")
                        stats["skipped"] += 1
                        continue

                    action = "overwrite" if dest_path.exists() else "extract"

                    if dry_run:
                        _log("dry", f"  [{action}] {pdf_name} → {dest_path}")
                        stats["extracted"] += 1
                        continue

                    # Extract ke temp lalu pindahkan (atomic, hindari file parsial)
                    with tempfile.NamedTemporaryFile(
                        dir=dest_dir, suffix=".pdf.tmp", delete=False
                    ) as tmp_file:
                        tmp_path = Path(tmp_file.name)

                    try:
                        with zf.open(entry) as src, open(tmp_path, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        tmp_path.rename(dest_path)

                        if action == "overwrite":
                            _log("info", f"  ✓ overwrite  {pdf_name}")
                            stats["overwritten"] += 1
                        else:
                            _log("info", f"  ✓ extracted  {pdf_name}")
                            stats["extracted"] += 1

                    except Exception as e:
                        _log("error", f"  Gagal extract {pdf_name}: {e}")
                        tmp_path.unlink(missing_ok=True)
                        stats["errors"] += 1

        except zipfile.BadZipFile:
            _log("error", f"  {zip_path.name} bukan file zip valid, skip.")
            stats["errors"] += 1
        except Exception as e:
            _log("error", f"  Gagal buka {zip_path.name}: {e}")
            stats["errors"] += 1

    return stats


# ─── Entry point ────────────────────────────────────────────────────────────

def main():
    # Default dest relatif ke root project (satu level di atas folder scripts/)
    default_dest = Path(__file__).parent.parent / "data" / "pdfs"

    parser = argparse.ArgumentParser(
        description="Fetch & extract PDF dari zip archive ke data/pdfs/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source", "-s",
        required=True,
        help=(
            "Sumber zip: folder lokal yang berisi .zip, "
            "atau GitHub repo URL (https://github.com/org/repo)"
        ),
    )
    parser.add_argument(
        "--dest", "-d",
        default=str(default_dest),
        help=f"Folder tujuan PDF hasil extract (default: {default_dest})",
    )
    parser.add_argument(
        "--clone-dir",
        default=None,
        help=(
            "Folder tempat clone repo (hanya berlaku jika --source adalah URL). "
            "Default: temp folder otomatis (dihapus setelah selesai)"
        ),
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite PDF yang sudah ada di folder tujuan",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Tampilkan apa yang akan di-extract tanpa benar-benar melakukan ekstraksi",
    )

    args = parser.parse_args()

    source = args.source
    dest_dir = Path(args.dest)

    # ── Tentukan folder zip ─────────────────────────────────────────────────
    is_url = source.startswith("http://") or source.startswith("https://")
    tmp_clone_dir = None  # akan di-cleanup setelah selesai jika auto-created

    if is_url:
        if args.clone_dir:
            zip_dir = Path(args.clone_dir)
        else:
            # Buat temp dir yang akan dihapus setelah selesai
            tmp_clone_dir = tempfile.mkdtemp(prefix="rag_pdf_repo_")
            zip_dir = Path(tmp_clone_dir)

        zip_dir = _clone_repo(source, zip_dir)
    else:
        zip_dir = Path(source)
        if not zip_dir.exists():
            _log("error", f"Folder sumber tidak ditemukan: {zip_dir}")
            sys.exit(1)
        if not zip_dir.is_dir():
            _log("error", f"--source harus berupa folder atau URL, bukan file: {zip_dir}")
            sys.exit(1)

    # ── Jalankan extraction ─────────────────────────────────────────────────
    if args.dry_run:
        print(f"\n{CYAN}{BOLD}=== DRY RUN — tidak ada file yang benar-benar di-extract ==={RESET}\n")

    stats = _extract_zips(
        zip_dir=zip_dir,
        dest_dir=dest_dir,
        force=args.force,
        dry_run=args.dry_run,
    )

    # ── Bersihkan temp clone dir ────────────────────────────────────────────
    if tmp_clone_dir:
        try:
            shutil.rmtree(tmp_clone_dir)
            _log("info", f"Temp clone dir dihapus: {tmp_clone_dir}")
        except Exception as e:
            _log("warn", f"Gagal hapus temp dir {tmp_clone_dir}: {e}")

    # ── Summary ─────────────────────────────────────────────────────────────
    print()
    _log("summary", "=" * 48)
    _log("summary", f"  Extracted  : {stats['extracted']}")
    _log("summary", f"  Overwritten: {stats['overwritten']}")
    _log("summary", f"  Skipped    : {stats['skipped']}  (sudah ada, pakai --force untuk overwrite)")
    _log("summary", f"  Errors     : {stats['errors']}")
    _log("summary", f"  Destination: {dest_dir.resolve()}")
    _log("summary", "=" * 48)

    if args.dry_run:
        print(f"\n{CYAN}Dry run selesai. Jalankan tanpa --dry-run untuk eksekusi nyata.{RESET}")

    if stats["errors"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
