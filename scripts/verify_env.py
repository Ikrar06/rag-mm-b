"""verify_env.py — periksa kesiapan lingkungan sebelum indexing.

Tidak mengubah apa pun. Dijalankan pertama kali setelah venv dibuat, dan ulang
setelah tiap tahap pemasangan paket di DEPLOY.md — terutama untuk mendeteksi
torch yang terbayangi wheel PyPI.

Usage:
    python scripts/verify_env.py

Kode keluar:
    0  semua pemeriksaan WAJIB lolos
    1  ada pemeriksaan WAJIB yang gagal
Pemeriksaan bertanda [info] tidak pernah memengaruhi kode keluar.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TARGET_PYTHON = (3, 12)

_gagal = 0


def _p(status: str, label: str, detail: str = "") -> None:
    """Satu baris per pemeriksaan, penanda di depan."""
    global _gagal
    if status == "GAGAL":
        _gagal += 1
    print(f"  [{status:^5}] {label:<34} {detail}")


def ok(label, detail=""):    _p("OK", label, detail)
def gagal(label, detail=""): _p("GAGAL", label, detail)
def warn(label, detail=""):  _p("WARN", label, detail)
def info(label, detail=""):  _p("info", label, detail)


def bagian(judul: str) -> None:
    print(f"\n{judul}")


# ─── Pemeriksaan ─────────────────────────────────────────────────────────────

def cek_python() -> None:
    bagian("Python")
    v = sys.version_info
    detail = f"{v.major}.{v.minor}.{v.micro}  ({sys.executable})"
    if (v.major, v.minor) == TARGET_PYTHON:
        ok("versi", detail)
    else:
        gagal("versi", f"{detail}  <-- target {TARGET_PYTHON[0]}.{TARGET_PYTHON[1]}.x; "
                       f"unstructured 0.16.11 butuh <3.13")
    info("venv", "aktif" if sys.prefix != sys.base_prefix else "TIDAK aktif (base interpreter)")


def _parse_sm(arch: str) -> tuple[int, int] | None:
    """'sm_120' -> (12, 0). None kalau bukan entri sm_ berangka.

    Digit terakhir adalah minor, sisanya mayor — sm_86 adalah (8, 6) sedangkan
    sm_120 adalah (12, 0). Memperlakukan tiap digit sebagai satu komponen akan
    membaca sm_120 sebagai (1, 2, 0) dan mengurutkannya di bawah sm_86.
    `get_arch_list()` juga memuat entri 'compute_120' yang di sini dilewati.
    """
    if not arch.startswith("sm_"):
        return None
    d = arch[3:]
    if not d.isdigit() or len(d) < 2:
        return None
    return int(d[:-1]), int(d[-1])


def cek_torch() -> None:
    bagian("torch / GPU")
    try:
        import torch
    except Exception as e:
        gagal("import torch", f"{type(e).__name__}: {e}")
        return

    lokasi = "site-packages venv" if str(ROOT) not in torch.__file__ and sys.prefix in torch.__file__ \
             else "sistem (--system-site-packages)"
    ok("versi", f"{torch.__version__}   [{lokasi}]")

    try:
        tersedia = torch.cuda.is_available()
    except Exception as e:
        gagal("cuda.is_available()", f"{type(e).__name__}: {e}")
        return

    if not tersedia:
        gagal("cuda.is_available()", "False — GPU tidak terlihat dari proses ini")
        return
    ok("cuda.is_available()", "True")

    try:
        cap = torch.cuda.get_device_capability()
        nama = torch.cuda.get_device_name(0)
    except Exception as e:
        gagal("get_device_capability()", f"{type(e).__name__}: {e}")
        return

    sm = f"sm_{cap[0]}{cap[1]}"
    arsitektur = []
    try:
        arsitektur = torch.cuda.get_arch_list()
    except Exception:
        pass

    if not arsitektur:
        warn("get_device_capability()", f"{sm} ({nama}) — daftar arsitektur build tidak terbaca")
    elif sm in arsitektur:
        ok("get_device_capability()", f"{sm} ({nama}) — didukung build torch ini")
    else:
        # PERINGATAN, bukan KEGAGALAN. GPU yang lebih baru daripada arsitektur
        # tertinggi di build tetap berjalan lewat PTX forward compatibility:
        # wheel resmi menyertakan PTX untuk arsitektur tertinggi yang
        # dikompilasinya, dan driver meng-JIT ulang PTX itu untuk GPU yang lebih
        # baru. Ongkosnya kompilasi JIT sekali di awal proses, bukan kegagalan.
        #
        # Kasus nyata: GB10 melaporkan sm_121 sedangkan wheel cu130 resmi hanya
        # dikompilasi sampai sm_120. Terverifikasi bekerja — matmul 4096x4096
        # fp16 selesai dalam 75 ms.
        tertinggi = max(
            (p for p in (_parse_sm(a) for a in arsitektur) if p), default=(0, 0)
        )
        lebih_baru = tuple(cap) > tertinggi
        warn("get_device_capability()",
             f"{sm} ({nama}) tidak ada di build torch: {' '.join(arsitektur)}")
        if lebih_baru:
            info("", "GPU lebih baru daripada arsitektur tertinggi di build — "
                     "dijalankan lewat PTX forward compatibility (JIT sekali di "
                     "awal proses). Verifikasi dengan matmul besar; kalau selesai "
                     "wajar, ini bukan masalah.")
        else:
            info("", "GPU lebih LAMA daripada build torch — PTX forward "
                     "compatibility TIDAK menolong ke arah ini. Pasang build "
                     "torch yang menyertakan arsitektur GPU ini.")

    try:
        info("torch.version.cuda", str(torch.version.cuda))
    except Exception:
        pass


def cek_unstructured() -> None:
    bagian("Ekstraksi PDF")
    try:
        import unstructured
        versi = getattr(unstructured, "__version__", "?")
    except Exception as e:
        gagal("import unstructured", f"{type(e).__name__}: {e}")
        return
    ok("import unstructured", versi)

    try:
        from unstructured.partition.pdf import partition_pdf  # noqa: F401
        ok("partition_pdf tersedia", "strategy='hi_res' dapat dipanggil")
    except Exception as e:
        gagal("partition_pdf", f"{type(e).__name__}: {str(e)[:70]}")


def cek_binari() -> None:
    bagian("Binari sistem")
    tess = shutil.which("tesseract")
    if not tess:
        gagal("tesseract di PATH", "tidak ditemukan — hi_res tidak dapat meng-OCR")
    else:
        try:
            out = subprocess.run([tess, "--version"], capture_output=True, text=True, timeout=10)
            baris = (out.stdout or out.stderr).splitlines()[0].strip()
        except Exception as e:
            baris = f"(gagal baca versi: {e})"
        ok("tesseract di PATH", f"{baris}   [{tess}]")

    pop = shutil.which("pdftoppm")
    if pop:
        ok("poppler (pdftoppm)", pop)
    else:
        gagal("poppler (pdftoppm)", "tidak ditemukan — pdf2image tidak dapat me-render halaman")

    # PaddleOCR: informasi saja. Riset memakai hi_res/tesseract, dan
    # _extract_text_from_page_fast mendegradasi per halaman bila Paddle absen.
    try:
        importlib.import_module("paddleocr")
        info("paddleocr", "terpasang (tidak dipakai jalur hi_res)")
    except Exception:
        info("paddleocr", "tidak terpasang — sesuai rencana, bukan kegagalan")


def cek_paket() -> None:
    bagian("Versi paket kunci")
    try:
        from importlib.metadata import version, PackageNotFoundError
    except Exception:
        warn("importlib.metadata", "tidak tersedia")
        return
    for nama in ("llama-index-core", "tiktoken", "pymupdf", "transformers",
                 "sentence-transformers", "unstructured", "qdrant-client",
                 "numpy", "prometheus-client"):
        try:
            ok(nama, version(nama))
        except PackageNotFoundError:
            gagal(nama, "tidak terpasang")
        except Exception as e:
            warn(nama, f"{type(e).__name__}")


def cek_qdrant() -> None:
    bagian("Qdrant")
    try:
        from backend.config import QDRANT_URL, QDRANT_COLLECTION
    except Exception as e:
        gagal("baca config", f"{type(e).__name__}: {e}")
        return

    try:
        import httpx
        r = httpx.get(f"{QDRANT_URL.rstrip('/')}/collections", timeout=5)
        r.raise_for_status()
        koleksi = [c["name"] for c in r.json().get("result", {}).get("collections", [])]
    except Exception as e:
        gagal("terjangkau", f"{QDRANT_URL} — {type(e).__name__}: {str(e)[:50]}")
        return

    ok("terjangkau", f"{QDRANT_URL}  ({len(koleksi)} collection)")
    if QDRANT_COLLECTION in koleksi:
        info("QDRANT_COLLECTION", f"{QDRANT_COLLECTION!r} SUDAH ADA — re-index akan menimpa")
    else:
        info("QDRANT_COLLECTION", f"{QDRANT_COLLECTION!r} belum ada (akan dibuat)")
    if koleksi:
        info("collection lain di server", ", ".join(sorted(koleksi))[:60])


def cek_sumber_daya() -> None:
    bagian("Sumber daya")
    try:
        du = shutil.disk_usage(ROOT)
        gb = 1024 ** 3
        sisa_pct = 100 * du.free / du.total
        detail = f"{du.free / gb:.0f} GB sisa dari {du.total / gb:.0f} GB ({sisa_pct:.0f}%)  [{ROOT}]"
        (ok if du.free / gb >= 20 else warn)("disk", detail)
    except Exception as e:
        warn("disk", f"{type(e).__name__}")

    # Memori terpadu: tidak ada VRAM terpisah, jadi angka sistem yang berlaku.
    try:
        with open("/proc/meminfo") as f:
            mi = {k.strip(): v for k, v in
                  (l.split(":", 1) for l in f if ":" in l)}
        total = int(mi["MemTotal"].split()[0]) / 1024 / 1024
        avail = int(mi["MemAvailable"].split()[0]) / 1024 / 1024
        detail = f"{avail:.0f} GB tersedia dari {total:.0f} GB"
        (ok if avail >= 16 else warn)("memori", detail)
        info("", "memori terpadu — angka ini juga plafon alokasi GPU, dan dibagi "
                 "dengan peneliti lain")
    except Exception:
        # macOS / non-Linux
        try:
            total = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                       capture_output=True, text=True).stdout) / 1024 ** 3
            info("memori", f"total {total:.0f} GB (MemAvailable tidak tersedia di platform ini)")
        except Exception:
            warn("memori", "tidak dapat dibaca")


def cek_prasyarat_riset() -> None:
    """Bukan kegagalan lingkungan — penanda pekerjaan yang belum ada."""
    bagian("Prasyarat riset (belum tentu ada)")
    try:
        from backend.config import DATA_DIR, DOCUMENT_REGISTRY_PATH
    except Exception:
        return

    pdfs = sorted(Path(DATA_DIR).glob("*.pdf")) if Path(DATA_DIR).exists() else []
    info("korpus PDF", f"{len(pdfs)} berkas di {DATA_DIR}"
                       + ("" if pdfs else "  <-- indexing belum bisa dijalankan"))

    reg = Path(DOCUMENT_REGISTRY_PATH)
    if not reg.exists():
        info("document_registry.json", f"belum ada di {reg}  <-- "
                                       f"python scripts/scaffold_document_registry.py")
        return
    try:
        import json
        from backend.services.document_registry import split_document_block

        # Registry berbentuk {"_meta": {...}, "documents": {...}}. Membaca akar
        # JSON langsung akan menghitung "_meta" dan "documents" sebagai dua entri
        # dokumen — laporannya jadi "0/2 entri punya document_id" padahal isinya
        # ratusan. split_document_block menangani bentuk ini DAN bentuk datar lama.
        meta, entri = split_document_block(json.loads(reg.read_text(encoding="utf-8")))
        terisi = sum(1 for e in entri.values()
                     if isinstance(e, dict) and (e.get("document_id") or "").strip())
        versi = meta.get("slug_rule_version")
        info("document_registry.json",
             f"{terisi}/{len(entri)} entri punya document_id"
             + (f", aturan slug v{versi}" if versi is not None else "")
             + ("" if terisi == len(entri) else "  <-- sisanya akan DILEWATI"))
    except Exception as e:
        warn("document_registry.json", f"tidak terbaca: {type(e).__name__}: {e}")


def main() -> int:
    print("verify_env.py — pemeriksaan lingkungan, tidak mengubah apa pun")
    print(f"repo: {ROOT}")

    cek_python()
    cek_torch()
    cek_unstructured()
    cek_binari()
    cek_paket()
    cek_qdrant()
    cek_sumber_daya()
    cek_prasyarat_riset()

    print()
    print("=" * 72)
    if _gagal:
        print(f"HASIL: {_gagal} pemeriksaan WAJIB gagal — perbaiki sebelum indexing.")
    else:
        print("HASIL: seluruh pemeriksaan wajib lolos.")
    print("Baris [info] tidak memengaruhi hasil.")
    print("=" * 72)
    return 1 if _gagal else 0


if __name__ == "__main__":
    sys.exit(main())
