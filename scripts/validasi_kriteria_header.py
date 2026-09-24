"""validasi_kriteria_header.py — uji kriteria header terhadap DATA NYATA.

READ-ONLY. Tidak mengindeks, tidak menulis ke Qdrant, tidak mengubah pipeline.

Latar
-----
Kriteria header pernah diuji terhadap dua belas contoh yang baris TUBUHNYA
dikarang, padahal aturan (d) — kontras kolom — bergantung penuh pada tubuh
tabel sungguhan. "Nol salah terima" karenanya belum terbukti. Skrip ini
menjalankan kedua varian atas seluruh pasangan yang DISETUJUI di berkas
keputusan, memakai baris nyata dari dump.

Kenapa membaca `text_as_html`, bukan `text_content`
---------------------------------------------------
Teks chunk di v3 SUDAH tergabung: potongan yang merupakan lanjutan sudah
diawali header (yang mungkin salah pilih). Menilai kriteria di atas teks itu
berarti menilai hasil keputusan lama, bukan tabel aslinya. `text_as_html` tidak
pernah disentuh penggabungan — ia catatan ekstraksi yang setia — jadi ia yang
dipakai. Ia juga satu-satunya sumber penanda `<th>`/`<thead>`.

Diagnosis butir 6
-----------------
Untuk pasangan yang teks B-nya TIDAK berubah, dicek kondisi mana yang berlaku:
`baris_header_markdown` mengembalikan kosong (A tidak berbentuk tabel Markdown),
atau B memang sudah diawali header itu. Perbandingan v2<->v3 memakai `text_sha`,
bukan `chunk_id`: penomoran halaman diperbaiki di antara kedua run sehingga id
bisa bergeser, sedangkan sha ikut isinya.

Usage:
    python scripts/validasi_kriteria_header.py \
        --keputusan ~/rag_mm_b_shared/table_continuation.json \
        --chunks-v3 .../20260921T095218Z-589ddf85/chunks.jsonl \
        --chunks-v2 .../20260826T100956Z-6dd92574/chunks.jsonl
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

from lib.header_tabel import (  # noqa: E402
    STATUS_DIKETAHUI, STATUS_MATI, deteksi_header, lanjutkan_rantai,
    panjang_sel_terpanjang,
)
from lib.tabel_html import _Pengurai, urai  # noqa: E402

PEMISAH_MD = re.compile(r"\|[\s:|-]+\|")


def baca_jsonl(path: Path) -> list[dict]:
    out = []
    for i, ln in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not ln.strip():
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} baris {i}: {e}") from e
    return out


def baris_header_markdown(teks: str) -> str:
    """Salinan persis logika produksi, untuk mendiagnosis butir 6."""
    baris = (teks or "").splitlines()
    mulai = next((i for i, b in enumerate(baris) if b.lstrip().startswith("|")), None)
    if mulai is None or mulai + 1 >= len(baris):
        return ""
    if not PEMISAH_MD.fullmatch(baris[mulai + 1].strip()):
        return ""
    return "\n".join(baris[mulai:mulai + 2])


def _potong(sel, n=56) -> str:
    s = " | ".join(str(x).strip() for x in sel)
    return s if len(s) <= n else s[: n - 1] + "…"


def sebab_tak_terurai(html) -> str:
    """Kenapa text_as_html tidak menghasilkan tabel. Tiga sebab dibedakan
    karena tindak lanjutnya berbeda."""
    if not isinstance(html, str) or not html.strip():
        return "html_kosong"
    rendah = html.lower()
    if "<table" not in rendah:
        return "tanpa_table"
    p = _Pengurai()
    try:
        p.feed(html)
        p.close()
    except Exception as e:
        return f"pengurai_gagal ({type(e).__name__})"
    if not p.baris:
        return "table_tanpa_baris" if "<tr" not in rendah else "baris_tanpa_sel"
    return "tidak_diketahui"


def _diagnosa_tak_berubah(a: dict, b: dict) -> str:
    """Kenapa teks B tidak berubah walau pasangannya disetujui."""
    teks_a = a.get("text_content") or ""
    header = baris_header_markdown(teks_a)
    if not header:
        fmt = a.get("table_format")
        return (f"A tidak berbentuk tabel Markdown (table_format={fmt!r}) — "
                f"tidak ada header untuk diulang")
    if (b.get("text_content") or "").strip().startswith(header.strip()):
        return "B SUDAH diawali header itu di dokumen asli — tidak digandakan"
    return "tidak diketahui — header ada dan B tidak diawali header itu"


def main() -> int:
    ap = argparse.ArgumentParser(description="Validasi kriteria header di data nyata")
    ap.add_argument("--keputusan", required=True)
    ap.add_argument("--chunks-v3", required=True)
    ap.add_argument("--chunks-v2", default=None,
                    help="Dump sebelum penggabungan, untuk diagnosis butir 6")
    ap.add_argument("--hanya-beda", action="store_true",
                    help="Cetak hanya pasangan yang kedua varian berbeda putusan")
    ap.add_argument("--json", default=None)
    ap.add_argument("--maks-panjang-sel", type=int, default=None,
                    help="Terapkan aturan (e) dengan batas ini. Ambil dari celah terukur.")
    ap.add_argument("--modal-markup", action="store_true",
                    help="Ukur juga aturan jumlah-sel=kolom-modal pada markup")
    args = ap.parse_args()

    try:
        raw = json.loads(Path(args.keputusan).expanduser().read_text(encoding="utf-8"))
        pasangan_keputusan = (raw or {}).get("pasangan") or {}
        v3 = baca_jsonl(Path(args.chunks_v3).expanduser())
        v2 = baca_jsonl(Path(args.chunks_v2).expanduser()) if args.chunks_v2 else []
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"GAGAL membaca masukan: {e}")
        return 2

    per_id = {c["chunk_id"]: c for c in v3 if c.get("chunk_id")}
    sha_v2 = {c.get("text_sha") for c in v2 if c.get("text_sha")}

    disetujui = [k for k, v in pasangan_keputusan.items()
                 if isinstance(v, dict)
                 and str(v.get("keputusan", "")).strip().lower() == "terima"]

    print(f"Keputusan : {args.keputusan}")
    print(f"Dump v3   : {args.chunks_v3}  ({len(v3)} chunk)")
    if v2:
        print(f"Dump v2   : {args.chunks_v2}  ({len(v2)} chunk)")
    print(f"Disetujui : {len(disetujui)} pasangan\n")

    pasangan = []
    hilang = []
    for kunci in sorted(disetujui):
        id_a, _, id_b = kunci.partition("__")
        if id_a not in per_id or id_b not in per_id:
            hilang.append((kunci, "A" if id_a not in per_id else "B"))
            continue
        pasangan.append((id_a, id_b, kunci))
    # Kunci yang tidak menunjuk dua chunk TABEL berarti berkas keputusan dibuat
    # dengan penomoran lain. Dianalisis tetap, tapi ditandai mencolok.
    bukan_tabel = {k for a_, b_, k in pasangan
                   if per_id[a_].get("element_type") != "Table"
                   or per_id[b_].get("element_type") != "Table"}

    tabel_cache: dict[str, object] = {}

    def tabel(cid):
        if cid not in tabel_cache:
            tabel_cache[cid] = urai(per_id[cid].get("text_as_html"))
        return tabel_cache[cid]

    kw = {"maks_panjang_sel": args.maks_panjang_sel}

    def nilai(cid, **ekstra):
        t = tabel(cid)
        if t is None:
            return None, None
        return t, deteksi_header(t.baris, t.ada_th, t.ada_thead, **{**kw, **ekstra})

    # Keadaan rantai SETELAH tiap potongan, berjalan dari kepala ke ujung.
    sesudah = {a_: b_ for a_, b_, _ in pasangan}
    punya_sebelum = set(sesudah.values())
    keadaan: dict[str, object] = {}
    for kepala in (a_ for a_, _, _ in pasangan if a_ not in punya_sebelum):
        k, cid, terlihat = None, kepala, set()
        while cid is not None and cid not in terlihat:
            terlihat.add(cid)
            t, p = nilai(cid)
            k = lanjutkan_rantai(k, cid, t, p)
            keadaan[cid] = k
            cid = sesudah.get(cid)

    hasil = []
    for id_a, id_b, kunci in pasangan:
        a, b = per_id[id_a], per_id[id_b]
        t, pa = nilai(id_a, pakai_kontras=True)
        _, pb = nilai(id_a, pakai_kontras=False)
        _, pm = nilai(id_a, pakai_kontras=True, modal_untuk_markup=True)

        k = keadaan.get(id_a)
        if k is not None and k.status == STATUS_DIKETAHUI:
            diulang = list(k.header)
            sumber = "sendiri" if k.sumber == id_a else f"warisan dari {k.sumber}"
            tb = tabel(id_b)
            if tb and tb.baris and [c.strip() for c in tb.baris[0]] == [c.strip() for c in diulang]:
                sumber += " — B sudah diawali header itu, tidak digandakan"
                diulang = None
        elif k is not None and k.status == STATUS_MATI:
            diulang, sumber = None, f"tidak ada — rantai mati di {k.sumber} (baris data)"
        else:
            diulang, sumber = None, "tidak ada — belum ada potongan terurai sejauh ini"

        tak_berubah = None
        if v2 and b.get("text_sha") in sha_v2:
            tak_berubah = _diagnosa_tak_berubah(a, b)

        def ringkas(p):
            if p is None:
                return {"header": False, "aturan": "tak-terurai",
                        "alasan": sebab_tak_terurai(a.get("text_as_html"))}
            return {"header": p.header, "aturan": p.aturan, "alasan": p.alasan}

        hasil.append({
            "kunci": kunci, "document_id": id_a.rsplit("_p", 1)[0],
            "halaman": [a.get("page_number"), b.get("page_number")],
            "kandidat": list(t.baris[0]) if t else None,
            "tubuh": [list(r) for r in t.baris[1:3]] if t else [],
            "markup": bool(t and (t.ada_th or t.ada_thead)),
            "bukan_tabel": kunci in bukan_tabel,
            "jenis": [a.get("element_type"), b.get("element_type")],
            "A": ringkas(pa), "B": ringkas(pb), "modal": ringkas(pm),
            "diulang": diulang, "sumber_header": sumber,
            "panjang_sel": panjang_sel_terpanjang(t.baris[0]) if t and t.baris else 0,
            "tak_berubah": tak_berubah,
        })

    beda = [r for r in hasil if r["A"]["header"] != r["B"]["header"]]
    cetak = beda if args.hanya_beda else hasil

    for r in cetak:
        tanda = "  <-- BEDA" if r in beda else ""
        if r["bukan_tabel"]:
            tanda += f"  <-- KUNCI TIDAK MENUNJUK TABEL {r['jenis']}"
        print(f"{r['document_id']}  hal {r['halaman'][0]}->{r['halaman'][1]}{tanda}")
        print(f"    kandidat : {_potong(r['kandidat'] or ['<tidak terurai>'])}"
              f"{'   [markup]' if r['markup'] else ''}")
        for i, baris in enumerate(r["tubuh"], 1):
            print(f"    tubuh {i}  : {_potong(baris)}")
        print(f"    Varian A : {'HEADER' if r['A']['header'] else 'tolak ':<7} "
              f"[{r['A']['aturan']}] {r['A']['alasan']}")
        print(f"    Varian B : {'HEADER' if r['B']['header'] else 'tolak ':<7} "
              f"[{r['B']['aturan']}] {r['B']['alasan']}")
        print(f"    DIULANG  : {_potong(r['diulang']) if r['diulang'] else '-'}"
              f"   ({r['sumber_header']})")
        if r["tak_berubah"]:
            print(f"    butir 6  : {r['tak_berubah']}")
        print()

    print("=" * 78)
    print("RINGKASAN")
    print("=" * 78)
    n = len(hasil) or 1
    for v in ("A", "B"):
        ya = sum(1 for r in hasil if r[v]["header"])
        label = "dengan kontras kolom" if v == "A" else "tanpa kontras"
        print(f"  Varian {v} ({label:<21}) header={ya:>4}  tolak={len(hasil)-ya:>4}"
              f"  ({100*ya/n:.0f}% diulang)")
    print(f"\n  Putusan BERBEDA antar varian: {len(beda)}"
          f"   <-- ini yang perlu ditinjau manual")

    print("\n  Aturan penentu (Varian A):")
    for aturan, jml in Counter(r["A"]["aturan"] for r in hasil).most_common():
        contoh = next(r for r in hasil if r["A"]["aturan"] == aturan)
        print(f"    {aturan:<12}{jml:>4}   mis. {_potong(contoh['kandidat'] or ['-'], 40)}")

    per_dok = defaultdict(lambda: [0, 0])
    for r in hasil:
        per_dok[r["document_id"]][0 if r["A"]["header"] else 1] += 1
    print("\n  Per dokumen (header diulang / ditolak), Varian A:")
    for dok, (ya, tidak) in sorted(per_dok.items(), key=lambda kv: -sum(kv[1])):
        print(f"    {dok[:52]:<54}{ya:>4} / {tidak:<4}")

    print("\n  Jalur markup (Varian A):")
    mk = [r for r in hasil if r["markup"]]
    print(f"    ber-<th>/<thead>        : {len(mk)}")
    print(f"    lolos (a1)-(c)          : {sum(1 for r in mk if r['A']['header'])}")
    for aturan, jml in Counter(r["A"]["aturan"] for r in mk
                               if not r["A"]["header"]).most_common():
        print(f"    ditolak {aturan:<18}: {jml}")
    lolos_mk = [r for r in mk if r["A"]["header"]]
    if lolos_mk:
        print("\n  SELURUH header markup yang LOLOS — periksa sisa sampah:")
        for r in lolos_mk:
            print(f"    {r['document_id'][:34]:<36}{r['halaman'][0]}->{r['halaman'][1]:<5}"
                  f"{_potong(r['kandidat'], 60)}")

    if args.modal_markup:
        ubah = [r for r in mk if r["A"]["header"] and not r["modal"]["header"]]
        print(f"\n  Aturan modal pada markup AKAN menolak {len(ubah)} header tambahan:")
        for r in ubah:
            print(f"    {r['document_id'][:34]:<36}{r['halaman'][0]}->{r['halaman'][1]:<5}"
                  f"{_potong(r['kandidat'], 44)}  [{r['modal']['alasan']}]")

    # ── Butir 2: panjang sel terpanjang, header lolos vs sisanya ───────────
    if lolos_mk:
        print("\n  PANJANG SEL TERPANJANG — header markup yang lolos (menurun):")
        urut = sorted(lolos_mk, key=lambda r: -r["panjang_sel"])
        for r in urut:
            print(f"    {r['panjang_sel']:>4}  {r['document_id'][:30]:<32}"
                  f"{r['halaman'][0]}->{r['halaman'][1]:<5}{_potong(r['kandidat'], 50)}")
        pj = [r["panjang_sel"] for r in urut]
        celah = sorted(((pj[i] - pj[i + 1], pj[i + 1], pj[i]) for i in range(len(pj) - 1)),
                       reverse=True)[:3]
        print("  Celah terbesar antar-panjang berurutan (lebar, bawah, atas):")
        for lebar, bawah, atas in celah:
            print(f"    {lebar:>4}  antara {bawah} dan {atas}")
        print("  Celah dianggap BERSIH hanya bila semua baris di atasnya baris data dan")
        print("  semua di bawahnya header sungguhan — periksa baris di sekitar celah.")

    bt = [r for r in hasil if r["bukan_tabel"]]
    if bt:
        print(f"\n  PERINGATAN: {len(bt)} kunci tidak menunjuk dua chunk TABEL di dump ini.")
        print("  Berkas keputusan kemungkinan dibuat dengan penomoran lain — jalankan")
        print("  scripts/migrasi_keputusan.py dulu, lalu ulangi validasi dengan berkasnya.")
        for jenis, jml in Counter(tuple(r["jenis"]) for r in bt).most_common():
            print(f"    {jml:>4}  {jenis}")

    warisan = [r for r in hasil if r["sumber_header"].startswith("warisan")]
    putus = [r for r in hasil if r["sumber_header"].startswith("tidak ada")]
    mati = [r for r in putus if "mati" in r["sumber_header"]]
    print(f"\n  Pewarisan rantai:")
    print(f"    header sendiri     : {sum(1 for r in hasil if r['sumber_header']=='sendiri')}")
    print(f"    header warisan     : {len(warisan)}")
    print(f"    tanpa header       : {len(putus)}  (rantai mati: {len(mati)}, "
          f"belum diketahui: {len(putus) - len(mati)})")
    print(f"    TOTAL akan diulang : {sum(1 for r in hasil if r['diulang'])}")
    for r in warisan[:12]:
        print(f"      {r['document_id'][:30]:<32}{r['halaman'][0]}->{r['halaman'][1]:<5}"
              f"{r['sumber_header'][:40]}: {_potong(r['diulang'], 36)}")

    tt = [r for r in hasil if r["A"]["aturan"] == "tak-terurai"]
    if tt:
        print(f"\n  TIDAK TERURAI: {len(tt)}")
        for sebab, jml in Counter(r["A"]["alasan"] for r in tt).most_common():
            print(f"    {jml:>4}  {sebab}")

    tb = [r for r in hasil if r["tak_berubah"]]
    if tb:
        print(f"\n  BUTIR 6 — teks B tidak berubah antara v2 dan v3: {len(tb)}")
        for alasan, jml in Counter(r["tak_berubah"].split(" —")[0] for r in tb).most_common():
            print(f"    {jml:>4}  {alasan}")

    if hilang:
        print(f"\n  PERINGATAN: {len(hilang)} pasangan tidak ditemukan di dump v3")
        for kunci, sisi in hilang[:5]:
            print(f"    potongan {sisi} hilang: {kunci}")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"n_disetujui": len(disetujui), "hasil": hasil,
                        "beda": [r["kunci"] for r in beda]},
                       indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  Hasil lengkap: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
