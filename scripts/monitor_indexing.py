"""monitor_indexing.py — pantau progres indexing secara realtime.

Membaca log yang ditulis `tee` saat indexing berjalan. Read-only, aman
dijalankan kapan saja dan tidak menyentuh proses indexing.

Pemakaian:
    python3 scripts/monitor_indexing.py
    python3 scripts/monitor_indexing.py --log ~/rag_mm_b_data/log/index_varian_c.log
    python3 scripts/monitor_indexing.py --total 214 --interval 20
    python3 scripts/monitor_indexing.py --once     # cetak sekali lalu keluar

Ctrl+C hanya menutup monitor. Indexing tetap berjalan di tmux.
"""

import argparse
import os
import re
import sys
import time
from datetime import datetime, timedelta

TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+")
RE_CHUNKS = re.compile(r"chunks=(\d+)")
RE_COUNT = re.compile(r"count=(\d+)")
RE_FILE = re.compile(r"pdf_extract file=(.+?) strategy=")
RE_TOKENS = re.compile(r"tokens=(\d+)")


def parse_ts(line):
    m = TS.match(line)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def fmt(sec):
    if sec is None or sec < 0:
        return "?"
    return str(timedelta(seconds=int(sec)))


def scan(log_path):
    d = {
        "done": 0, "chunks": 0, "images": 0,
        "dekor": 0, "unclear": 0, "vl_err": 0, "vl_ok": 0,
        "over": 0, "over_tokens": [], "dropped_title": 0, "dropped_min": 0,
        "degraded": 0, "ocr_fail": 0, "extract_fail": 0,
        "cur": "-", "t0": None, "tlast": None, "per_doc": [],
        "last_done": None, "exists": True, "finished": False,
    }
    prev = None
    try:
        with open(log_path, errors="replace") as f:
            for line in f:
                t = parse_ts(line)
                if t:
                    if d["t0"] is None:
                        d["t0"] = t
                    d["tlast"] = t

                if "pdf_chunked" in line and "strategy=" in line:
                    d["done"] += 1
                    d["last_done"] = t
                    if prev and t:
                        d["per_doc"].append((t - prev).total_seconds())
                    prev = t
                    m = RE_CHUNKS.search(line)
                    if m:
                        d["chunks"] += int(m.group(1))
                elif "pdf_extract file=" in line:
                    m = RE_FILE.search(line)
                    if m:
                        d["cur"] = m.group(1)
                    if prev is None and t:
                        prev = t
                elif "images_persisted" in line:
                    m = RE_COUNT.search(line)
                    if m:
                        d["images"] += int(m.group(1))
                elif "verdict=decorative" in line:
                    d["dekor"] += 1
                elif "verdict=unclear" in line:
                    d["unclear"] += 1
                elif "image_describer_error" in line:
                    d["vl_err"] += 1
                elif "/api/generate" in line and "200 OK" in line:
                    d["vl_ok"] += 1
                elif "chunk_over_budget_kept_whole" in line:
                    d["over"] += 1
                    m = RE_TOKENS.search(line)
                    if m:
                        d["over_tokens"].append(int(m.group(1)))
                elif "chunk_dropped" in line:
                    if "title_only" in line:
                        d["dropped_title"] += 1
                    else:
                        d["dropped_min"] += 1
                elif "degraded_documents=" in line or "run_degraded" in line:
                    d["degraded"] += 1
                elif "page_ocr_failed" in line:
                    d["ocr_fail"] += 1
                elif "pdf_extract_failed" in line:
                    d["extract_fail"] += 1
                elif "indexing_complete" in line:
                    d["finished"] = True
    except FileNotFoundError:
        d["exists"] = False
    return d


def render(d, total, log_path):
    now = datetime.now()
    done = d["done"]
    pct = done / total if total else 0
    filled = int(pct * 40)
    bar = "#" * filled + "." * (40 - filled)

    elapsed = (d["tlast"] - d["t0"]).total_seconds() if d["t0"] and d["tlast"] else 0
    recent = d["per_doc"][-15:] or d["per_doc"]
    avg = sum(recent) / len(recent) if recent else None
    avg_all = sum(d["per_doc"]) / len(d["per_doc"]) if d["per_doc"] else None
    sisa = (total - done) * avg if avg and done < total else None
    eta = (now + timedelta(seconds=sisa)).strftime("%H:%M %d-%b") if sisa else "-"

    idle = (now - d["tlast"]).total_seconds() if d["tlast"] else None
    ot = d["over_tokens"]
    ot_max = max(ot) if ot else 0
    ot_avg = sum(ot) / len(ot) if ot else 0
    err_total = d["vl_err"] + d["degraded"] + d["extract_fail"]

    print("\033[2J\033[H", end="")
    print("=" * 64)
    print(f"  MONITOR INDEXING            {now.strftime('%H:%M:%S  %d-%b-%Y')}")
    print(f"  log: {os.path.basename(log_path)}")
    print("=" * 64)

    if not d["exists"]:
        print("\n  Log belum ada. Indexing belum dimulai?")
        print(f"  Dicari di: {log_path}\n")
        print("=" * 64)
        return

    print(f"  [{bar}] {pct * 100:5.1f}%")
    print(f"  dokumen          : {done} / {total}      sisa {total - done}")
    print()
    print("  -- WAKTU " + "-" * 52)
    print(f"  sudah berjalan   : {fmt(elapsed)}")
    print(f"  rata2 per dokumen: {fmt(avg)}"
          + (f"   ({avg:.0f} dtk, 15 terakhir)" if avg else ""))
    print(f"  rata2 keseluruhan: {fmt(avg_all)}"
          + (f"   ({avg_all:.0f} dtk)" if avg_all else ""))
    print(f"  perkiraan sisa   : {fmt(sisa)}")
    print(f"  perkiraan selesai: {eta}")
    print()
    print("  -- HASIL " + "-" * 52)
    print(f"  chunk terbentuk  : {d['chunks']:,}"
          + (f"   (~{d['chunks'] / done:.0f} per dokumen)" if done else ""))
    print(f"  gambar disimpan  : {d['images']:,}")
    print(f"    dideskripsikan : {max(0, d['images'] - d['dekor'] - d['unclear']):,}")
    print(f"    dekoratif      : {d['dekor']:,}")
    print(f"    tidak jelas    : {d['unclear']:,}")
    print(f"  panggilan vision : {d['vl_ok']:,} berhasil")
    print()
    print(f"  chunk dibuang    : {d['dropped_title'] + d['dropped_min']:,}"
          f"   (judul {d['dropped_title']:,} / pendek {d['dropped_min']:,})")
    print(f"  over budget      : {d['over']:,}"
          + (f"   maks {ot_max} tok, rata2 {ot_avg:.0f} tok" if ot else ""))
    print("                     (tabel sengaja tidak dipecah)")
    print()
    print("  -- KESEHATAN " + "-" * 48)
    print(f"  error vision     : {d['vl_err']}")
    print(f"  OCR gagal        : {d['ocr_fail']}   (halaman, bukan dokumen)")
    print(f"  ekstraksi gagal  : {d['extract_fail']}")
    print(f"  degraded         : {d['degraded']}", end="")
    print("   <-- HARUS NOL" if d["degraded"] else "   OK")
    if err_total == 0:
        print("  status           : semua bersih")
    else:
        print(f"  status           : {err_total} masalah, periksa log")
    print()
    print("  -- SEKARANG " + "-" * 49)
    print(f"  memproses        : {d['cur'][:44]}")
    print(f"  log terakhir     : "
          f"{d['tlast'].strftime('%H:%M:%S') if d['tlast'] else '-'}"
          + (f"   ({fmt(idle)} lalu)" if idle and idle > 60 else ""))

    if d["finished"]:
        print()
        print("  *** INDEXING SELESAI ***")
    elif idle is not None and idle > 600:
        print()
        print(f"  !! TIDAK ADA LOG BARU {fmt(idle)}")
        print("     cek: ps aux | grep index_documents | grep -v grep")

    print("=" * 64)
    print("  Ctrl+C keluar dari monitor (indexing tetap berjalan)")


def main():
    p = argparse.ArgumentParser(description="Pantau progres indexing")
    p.add_argument("--log",
                   default=os.path.expanduser(
                       "~/rag_mm_b_data/log/index_varian_b.log"))
    p.add_argument("--total", type=int, default=214)
    p.add_argument("--interval", type=int, default=20)
    p.add_argument("--once", action="store_true")
    a = p.parse_args()

    log_path = os.path.expanduser(a.log)
    while True:
        try:
            render(scan(log_path), a.total, log_path)
            if a.once:
                return
            time.sleep(a.interval)
        except KeyboardInterrupt:
            print("\nkeluar. indexing tetap berjalan.")
            return


if __name__ == "__main__":
    main()
