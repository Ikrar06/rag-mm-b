"""sebaran_setara.py — sebaran containment pasangan (chunk teks, chunk tabel) yang
lolos syarat LETAK relevan_setara, untuk menetapkan ambang isi dari data.

Seluruh chunk teks di dump dihitung, bukan hanya yang dirujuk gold, supaya
celah terlihat dari populasi. Pasangan yang sudah ditinjau manusia diberikan
lewat --tolak / --terima (chunk_teks~chunk_tabel, dipisah koma) dan dicetak
posisinya di sebaran. Read-only.

Usage:
    python scripts/sebaran_setara.py --chunks <dump v5>/chunks.jsonl \\
        --gold <gold v4>.jsonl \\
        --tolak "manual-dosen_p17_c02~manual-dosen_p17_c01,..." --out ~/sebaran_setara.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.relevan_setara import kandidat_letak  # noqa: E402
from lib.transkripsi_ukur import celah_terbesar  # noqa: E402

EMBER = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)


def baca(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _pasangan(teks: str) -> set[tuple[str, str]]:
    return {tuple(x.split("~", 1)) for x in teks.split(",") if "~" in x}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunks", type=Path, required=True)
    ap.add_argument("--gold", type=Path)
    ap.add_argument("--tolak", default="")
    ap.add_argument("--terima", default="")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    rows = baca(a.chunks)
    if not any(r.get("render_bbox") for r in rows):
        print("GAGAL: dump tanpa render_bbox (bukan dump v5 bertranskripsi)")
        return 2
    per_id = {r["chunk_id"]: r for r in rows}
    gold = set()
    if a.gold:
        gold = {c for g in baca(a.gold) for c in g.get("relevant_text_chunks") or []}
    tolak, terima = _pasangan(a.tolak), _pasangan(a.terima)
    kand = sorted(kandidat_letak(None, rows), key=lambda x: -x[2])

    print(f"pasangan lolos syarat letak: {len(kand)}  (chunk teks dirujuk gold: "
          f"{sum(1 for t, _, _ in kand if t in gold)})")
    print("sebaran containment:")
    for lo, hi in zip(EMBER, EMBER[1:]):
        n = sum(1 for _, _, c in kand if lo <= c < hi)
        print(f"  [{lo:.1f}, {min(hi, 1):.1f}{']' if hi > 1 else ')'} {n:5d} {'#' * min(n, 60)}")
    celah = celah_terbesar([c for _, _, c in kand])
    if celah:
        print(f"celah terbesar: {celah[0]:.3f} -> {celah[1]:.3f}")
    print("\npasangan teratas (periksa: duplikat sungguhan?):")
    for t, b, c in kand[:25]:
        tanda = (" [TOLAK manusia]" if (t, b) in tolak else " [TERIMA manusia]" if (t, b) in terima
                 else "") + (" [gold]" if t in gold else "")
        teks = " ".join((per_id[t].get("text_content") or "").split())[:90]
        print(f"  {c:.3f}  {t}  ~  {b}{tanda}\n         {teks!r}")
    for nama, kumpulan in (("DITOLAK", tolak), ("DITERIMA", terima)):
        for t, b in sorted(kumpulan):
            c = next((x for tt, bb, x in kand if (tt, bb) == (t, b)), None)
            print(f"{nama} manusia: {t} ~ {b}  containment="
                  f"{'tidak lolos syarat letak' if c is None else f'{c:.3f}'}")
    if a.out:
        a.out.write_text(json.dumps([{"chunk_teks": t, "chunk_tabel": b, "containment": round(c, 4),
                                      "gold": t in gold} for t, b, c in kand],
                                    ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nrincian: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
