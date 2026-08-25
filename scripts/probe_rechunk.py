"""probe_rechunk.py — Tahap 0: buktikan apakah LlamaIndex memecah ulang Document.

Konteks
-------
`indexing.py:160` memanggil `VectorStoreIndex.from_documents(...)` TANPA argumen
`transformations=` maupun `node_parser=`. LlamaIndex karenanya menerapkan node
parser default (SentenceSplitter) yang dikonfigurasi dari `Settings.chunk_size`
dan `Settings.chunk_overlap` — dua nilai yang disetel di `rag_pipeline.py:624-625`.
Node anak mewarisi metadata induk, termasuk `chunk_index`.

Mekanismenya sudah pasti dari dokumentasi. Yang diukur di sini adalah SEBERAPA
SERING itu terpicu pada bentuk data yang benar-benar diproduksi repo ini, plus
dua pertanyaan turunan: kalibrasi satuan (token vs karakter) dan sisa anggaran
metadata.

Probe ini TIDAK menyentuh Qdrant dan TIDAK memuat model embedding.

Jalankan
--------
    python scripts/probe_rechunk.py

Butuh: llama-index-core==0.14.13, tiktoken, markdownify==0.13.1, pymupdf
(pymupdf hanya karena `preprocessing.py:21` mengimpor `fitz` di level modul).

Catatan kejujuran
-----------------
Korpus PDF belum tersedia, jadi teks uji di bawah disusun tangan agar menyerupai
dokumen yang dideskripsikan di deck riset (SOP izin ujian, tabel persyaratan
sidang). Tabel Markdown-nya BUKAN tiruan: ia dihasilkan dengan memanggil
`_html_table_to_markdown` yang asli dari `backend/services/preprocessing.py`.
Angka yang keluar dari probe ini berlaku untuk bentuk-bentuk ini; distribusi
sebenarnya baru bisa diukur setelah korpus ada (lihat bagian akhir output).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llama_index.core import Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.utils import get_tokenizer

from backend.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PDF_TABLE_MAX_CHARS,
    INDEX_DISABLE_NODE_PARSER,
    INDEX_EXCLUDE_METADATA_FROM_EMBED,
    INDEX_MAX_CHUNK_TOKENS,
    INDEX_MIN_CHUNK_TOKENS,
    INDEX_STRUCTURAL_METADATA,
    INDEX_TABLES_AS_OWN_CHUNKS,
    EMBED_EXCLUDED_METADATA_KEYS,
)
from backend.services.preprocessing import (
    _chunk_elements,
    _html_table_to_markdown,
)

TOKENIZER = get_tokenizer()

# Mirror indexing.py:262-266 — pengecualian hanya aktif saat flag menyala.
EXCLUDED_KEYS = (
    list(EMBED_EXCLUDED_METADATA_KEYS) if INDEX_EXCLUDE_METADATA_FROM_EMBED else []
)


def ntok(text: str) -> int:
    return len(TOKENIZER(text))


def rule(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


# ─── Bahan uji ───────────────────────────────────────────────────────────────
# Meniru bentuk element yang diproduksi _chunk_elements (preprocessing.py:308-407).

SOP_KALIMAT = (
    "Mahasiswa mengajukan permohonan izin ujian akhir melalui sistem akademik "
    "daring paling lambat empat belas hari kerja sebelum tanggal pelaksanaan. "
    "Berkas permohonan diverifikasi oleh admin program studi, kemudian diteruskan "
    "kepada Wakil Dekan Bidang Akademik untuk memperoleh persetujuan. "
)


def teks_naratif(target_chars: int) -> str:
    """Paragraf Bahasa Indonesia dengan batas kalimat nyata, dipanjangkan
    sampai mendekati target_chars. SentenceSplitter memotong di batas kalimat,
    jadi teks tanpa titik tidak representatif."""
    out = ""
    i = 0
    while len(out) < target_chars:
        out += SOP_KALIMAT.replace("empat belas", f"{14 + i}") if i else SOP_KALIMAT
        i += 1
    return out.strip()


def tabel_persyaratan_html(n_baris: int) -> str:
    """Tabel persyaratan sidang bergaya `persyaratan_izin_ujian_akhir_online_1.pdf`
    yang dijelaskan di deck: kolom jenjang S1/S2/S3, baris syarat."""
    baris_syarat = [
        ("Indeks Prestasi Kumulatif minimum", "2,75", "3,00", "3,25"),
        ("Jumlah SKS lulus minimum", "144", "36", "42"),
        ("Masa studi maksimum (semester)", "14", "8", "10"),
        ("Nilai TOEFL / setara minimum", "450", "500", "550"),
        ("Jumlah publikasi wajib", "0", "1", "2"),
        ("Masa bimbingan minimum (bulan)", "6", "9", "12"),
    ]
    html = ["<table><thead><tr><th>Persyaratan</th><th>S1</th><th>S2</th><th>S3</th></tr></thead><tbody>"]
    for i in range(n_baris):
        label, s1, s2, s3 = baris_syarat[i % len(baris_syarat)]
        suffix = f" (kategori {i // len(baris_syarat) + 1})" if i >= len(baris_syarat) else ""
        html.append(f"<tr><td>{label}{suffix}</td><td>{s1}</td><td>{s2}</td><td>{s3}</td></tr>")
    html.append("</tbody></table>")
    return "".join(html)


def tabel_markdown_mendekati(target_chars: int) -> str:
    """Perbesar tabel sampai hasil Markdown-nya mendekati target_chars.
    Konversi memakai _html_table_to_markdown yang ASLI dari repo."""
    n, terakhir = 1, ""
    while n < 200:
        md, _ = _html_table_to_markdown(tabel_persyaratan_html(n))
        if len(md) > target_chars:
            return terakhir or md
        terakhir, n = md, n + 1
    return terakhir


DESKRIPSI_FLOWCHART = (
    "Diagram alir prosedur pengurusan izin ujian akhir online. Alur dimulai dari "
    "simpul Mahasiswa mengisi formulir permohonan, lalu menuju simpul keputusan "
    "berlabel TOEFL tersedia? dengan dua cabang. Cabang Ya menuju Verifikasi berkas "
    "oleh admin prodi. Cabang Tidak menuju Lampirkan surat keterangan sedang "
    "mengikuti tes, kemudian bergabung kembali ke Verifikasi berkas oleh admin prodi. "
    "Dari verifikasi, panah menuju Persetujuan Wakil Dekan Bidang Akademik, lalu ke "
    "Penerbitan surat izin ujian. Terdapat panah balik berlabel Berkas tidak lengkap "
    "dari simpul verifikasi kembali ke simpul pengisian formulir."
)

# Metadata persis seperti indexing.py:251-260 (delapan field).
META_SEKARANG = {
    "file_name": "sop_pengurusan_izin_ujian_akhir_online_2.pdf",
    "file_hash": "9f2b1c4e8a7d3f60b5e2c9a1d8f4b7e3c6a0d5f9b2e8c1a7d4f0b6e3c9a2d5f8",
    "page": 3,
    "chunk_index": 7,
    "element_type": "Table",
    "section": "Persyaratan Pengajuan Izin Ujian Akhir",
    "extraction_strategy": "hi_res",
    "source_type": "pdf",
}

# Field yang direncanakan ditambah (laporan 2, butir #2/#3/#10/#12).
META_RENCANA_TAMBAHAN = {
    "chunk_id": "sop-izin-ujian-online-v2_p3_c02",
    "document_id": "sop-izin-ujian-online-v2",
    "image_id": "sop-izin-ujian-online-v2_p3_img01",
    "visual_type": "tabel_sebagai_gambar",
    "embed_model": "Qwen/Qwen3-Embedding-0.6B",
    "vision_model": "Qwen/Qwen3-VL-8B-Instruct",
    "prompt_version": "desc-v1",
    "indexed_at": "2026-08-24T11:26:00+08:00",
}


def doc(text: str, **override) -> Document:
    """Bangun Document persis seperti indexing.py:248-270, termasuk exclusion."""
    meta = dict(META_SEKARANG)
    meta.update(override)
    return Document(
        text=text,
        metadata=meta,
        excluded_embed_metadata_keys=list(EXCLUDED_KEYS),
        excluded_llm_metadata_keys=list(EXCLUDED_KEYS),
    )


# ─── Lingkungan eksekusi ─────────────────────────────────────────────────────

# Target repo, dari requirements.txt baris 3 ("Python 3.12.x | CUDA 12.6.3").
TARGET_PYTHON = (3, 12)
KEY_PACKAGES = ("llama-index-core", "tiktoken", "pymupdf", "markdownify", "unstructured")


def bagian_env() -> bool:
    """Cetak lingkungan eksekusi. Return True bila cocok dengan target repo.

    Seluruh angka Tahap 1 & 2 pertama kali diukur di Python 3.14 karena venv
    proyek belum ada. Perilaku SentenceSplitter bergantung pada versi
    llama-index-core dan tiktoken, jadi angka-angka itu harus diverifikasi
    ulang di venv target sebelum indexing pertama.
    """
    import platform
    try:
        from importlib.metadata import version, PackageNotFoundError
    except ImportError:  # pragma: no cover
        version = None

    aktual = sys.version_info[:2]
    cocok = aktual == TARGET_PYTHON

    print("LINGKUNGAN EKSEKUSI")
    tanda = "" if cocok else f"   <-- TARGET REPO {TARGET_PYTHON[0]}.{TARGET_PYTHON[1]}.x"
    print(f"  python  : {platform.python_version()}{tanda}")
    for name in KEY_PACKAGES:
        try:
            v = version(name) if version else "?"
        except Exception:
            v = "(tidak terpasang)"
        print(f"  {name:<18}: {v}")

    if not cocok:
        print()
        print("  PERINGATAN: versi Python berbeda dari target repo.")
        print("  Angka di bawah belum tentu berlaku untuk lingkungan indexing yang")
        print("  sebenarnya. Jalankan ulang probe ini di venv target sebelum indexing")
        print("  pertama, lalu bandingkan seluruh angkanya.")
    return cocok


# ─── A. Kalibrasi satuan: token vs karakter ──────────────────────────────────

def bagian_a() -> float:
    rule("A. KALIBRASI SATUAN — chunk_size LlamaIndex (TOKEN) vs pemeriksaan repo (KARAKTER)")
    print(f"Tokenizer default LlamaIndex : {TOKENIZER}")
    print(f"CHUNK_SIZE (config.py:90)    : {CHUNK_SIZE}")
    print(f"CHUNK_OVERLAP (config.py:91) : {CHUNK_OVERLAP}")
    print(f"PDF_TABLE_MAX_CHARS (:111)   : {PDF_TABLE_MAX_CHARS}")
    print()
    print("Rasio karakter per token, teks akademik Bahasa Indonesia:")
    print(f"  {'sampel':<34}{'chars':>8}{'token':>8}{'char/tok':>10}")

    sampel = {
        "paragraf SOP (naratif)": teks_naratif(1200),
        "tabel Markdown (hasil repo)": tabel_markdown_mendekati(PDF_TABLE_MAX_CHARS),
        "deskripsi flowchart (VL)": DESKRIPSI_FLOWCHART,
        "judul section pendek": META_SEKARANG["section"],
    }
    rasio = []
    for nama, teks in sampel.items():
        c, t = len(teks), ntok(teks)
        r = c / t
        rasio.append((nama, r))
        print(f"  {nama:<34}{c:>8}{t:>8}{r:>10.2f}")

    r_naratif = dict(rasio)["paragraf SOP (naratif)"]
    r_tabel = dict(rasio)["tabel Markdown (hasil repo)"]

    print()
    print(f"→ {CHUNK_SIZE} token teks naratif  ≈ {int(CHUNK_SIZE * r_naratif):,} karakter")
    print(f"→ {CHUNK_SIZE} token tabel Markdown ≈ {int(CHUNK_SIZE * r_tabel):,} karakter")
    print()
    print("Konsekuensi ambang di preprocessing.py:393")
    print("  `buffer_size + len(text) > CHUNK_SIZE` membandingkan KARAKTER dengan 512,")
    print("  sedangkan SentenceSplitter membandingkan TOKEN dengan 512.")
    print(f"  Chunk teks yang di-flush pada ~{CHUNK_SIZE} karakter hanya berisi "
          f"~{int(CHUNK_SIZE / r_naratif)} token —")
    print(f"  yaitu ~{CHUNK_SIZE / r_naratif / CHUNK_SIZE:.0%} dari kuota SentenceSplitter.")
    return r_naratif


# ─── B. Probe pemecahan ulang ────────────────────────────────────────────────

def bagian_b(splitter: SentenceSplitter) -> list[tuple]:
    rule("B. PEMECAHAN ULANG — SentenceSplitter.get_nodes_from_documents([doc])")

    kasus = [
        ("teks pendek (1 element)", teks_naratif(200)),
        ("teks ~CHUNK_SIZE char (batas flush :393)", teks_naratif(CHUNK_SIZE)),
        ("teks 1 element panjang (lolos :393 utuh)", teks_naratif(1500)),
        ("teks 1 element sangat panjang", teks_naratif(4000)),
        (f"tabel kecil (< {PDF_TABLE_MAX_CHARS} char, digabung :371)",
         tabel_markdown_mendekati(600)),
        (f"tabel besar (chunk sendiri :361-368)",
         tabel_markdown_mendekati(PDF_TABLE_MAX_CHARS)),
        ("deskripsi gambar (chunk sendiri :381-388)",
         f"## {META_SEKARANG['section']}\n\n[Deskripsi Gambar] {DESKRIPSI_FLOWCHART}"),
        # Jalur fast: _extract_fast (:161-170) membuat SATU element per HALAMAN,
        # lalu :393 mem-flush buffer sebelum append sehingga tiap halaman jadi
        # satu chunk utuh. Halaman A4 teks Indonesia lazimnya 2.000-3.500 karakter.
        ("fast path: 1 halaman A4 penuh (:161-170)", teks_naratif(2500)),
        ("fast path: 1 halaman padat (:161-170)", teks_naratif(3500)),
    ]

    print("Tiap kasus dilewatkan _chunk_elements (jalur nyata indexing),")
    print("lalu SETIAP chunk dijalankan melalui SentenceSplitter.")
    print("Kolom 'chunk' = keluaran _chunk_elements; 'node' = total setelah splitter.")
    print()
    print(f"{'kasus':<42}{'token':>7}{'chunk':>6}{'node':>6}  {'chunk_index':<12}{'chunk_id':<10}")
    print("-" * 78)

    hasil = []
    for nama, teks in kasus:
        et = "Table" if "tabel" in nama else (
            "ImageDescription" if "gambar" in nama else "NarrativeText")

        element = {
            "category": et,
            "text": teks,
            "page": 3,
            "metadata": {
                "raw_html": "<table><tr><td>x</td></tr></table>" if et == "Table" else None,
                "table_format": "markdown" if et == "Table" else None,
                "bbox": [0.1, 0.2, 0.9, 0.8],
            },
        }
        chunks = _chunk_elements([element], "probe.pdf", document_id="probe-doc")

        total_nodes, idxs = 0, []
        for c in chunks:
            if INDEX_DISABLE_NODE_PARSER:
                # transformations=[] di indexing.py: 1 Document -> 1 node.
                total_nodes += 1
                idxs.append(c["chunk_index"])
            else:
                d = doc(c["text"], element_type=c["element_type"],
                        chunk_index=c["chunk_index"])
                nodes = splitter.get_nodes_from_documents([d])
                total_nodes += len(nodes)
                idxs.extend(n.metadata.get("chunk_index") for n in nodes)

        ids = [c.get("chunk_id") for c in chunks if c.get("chunk_id")]
        if not ids:
            status_id = "-"
        elif len(set(ids)) == len(ids):
            status_id = "unik"
        else:
            status_id = "TABRAKAN"

        tanda = "  <-- MASIH DIPECAH" if total_nodes > len(chunks) else ""
        unik = "unik" if len(set(idxs)) == len(idxs) else "TABRAKAN"
        print(f"{nama:<42}{ntok(teks):>7}{len(chunks):>6}{total_nodes:>6}"
              f"  {unik:<12}{status_id:<10}{tanda}")
        hasil.append((nama, teks, total_nodes, idxs, len(chunks), ids))
    return hasil


# ─── C. Jalur produksi: apa yang sebenarnya dipakai from_documents ───────────

def bagian_c() -> None:
    rule("C. JALUR PRODUKSI — apa yang di-resolve VectorStoreIndex.from_documents")

    print("indexing.py:160 tidak mengirim transformations= / node_parser=,")
    print("jadi LlamaIndex memakai Settings.node_parser. Dua entry point, dua nilai:")
    print()

    # Entry point HTTP: POST /api/index (main.py:265) TIDAK memanggil _configure_settings.
    np_default = Settings.node_parser
    print(f"  [1] POST /api/index (main.py:265) — Settings tidak pernah dikonfigurasi")
    print(f"      node_parser  : {type(np_default).__name__}")
    print(f"      chunk_size   : {getattr(np_default, 'chunk_size', '?')} token")
    print(f"      chunk_overlap: {getattr(np_default, 'chunk_overlap', '?')} token")

    # Entry point CLI: scripts/index_documents.py:54 memanggil _configure_settings,
    # yang menyetel Settings.chunk_size/chunk_overlap (rag_pipeline.py:624-625).
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP
    np_cli = Settings.node_parser
    print()
    print(f"  [2] CLI scripts/index_documents.py:54 → _configure_settings (rag_pipeline.py:624-625)")
    print(f"      node_parser  : {type(np_cli).__name__}")
    print(f"      chunk_size   : {getattr(np_cli, 'chunk_size', '?')} token")
    print(f"      chunk_overlap: {getattr(np_cli, 'chunk_overlap', '?')} token")

    transforms = Settings.transformations
    print()
    print(f"  Settings.transformations = {[type(t).__name__ for t in transforms]}")
    print("  → inilah yang dijalankan from_documents atas setiap Document.")


# ─── D. Anggaran metadata ────────────────────────────────────────────────────

def bagian_d() -> None:
    rule("D. ANGGARAN METADATA — effective_chunk_size = chunk_size - metadata_len")

    print("Mekanisme (llama_index/core/node_parser/):")
    print("  interface.py:248-259  metadata_str = yang TERPANJANG antara mode EMBED dan LLM")
    print("  sentence.py:157       metadata_len = len(tokenizer(metadata_str))")
    print("  sentence.py:158       effective_chunk_size = chunk_size - metadata_len")
    print("  sentence.py:159-163   ValueError bila effective <= 0")
    print("  sentence.py:165-168   warning bila effective < 50")
    print()
    print("indexing.py:251-260 tidak menyetel excluded_embed_metadata_keys maupun")
    print("excluded_llm_metadata_keys, jadi SELURUH field masuk hitungan.")
    print()

    skenario = [
        ("sekarang (8 field, indexing.py:251-260)", META_SEKARANG),
        ("+ chunk_id, document_id", {**META_SEKARANG,
                                     "chunk_id": META_RENCANA_TAMBAHAN["chunk_id"],
                                     "document_id": META_RENCANA_TAMBAHAN["document_id"]}),
        ("+ image_id, visual_type", {**META_SEKARANG,
                                     **{k: META_RENCANA_TAMBAHAN[k] for k in
                                        ("chunk_id", "document_id", "image_id", "visual_type")}}),
        ("+ manifest model & prompt (semua)", {**META_SEKARANG, **META_RENCANA_TAMBAHAN}),
    ]

    print(f"{'skenario':<42}{'field':>6}{'meta tok':>10}{'efektif':>9}{'status':>11}")
    print("-" * 78)
    for nama, meta in skenario:
        d = Document(
            text="x",
            metadata=meta,
            excluded_embed_metadata_keys=list(EXCLUDED_KEYS),
            excluded_llm_metadata_keys=list(EXCLUDED_KEYS),
        )
        # Replikasi interface.py:248-259
        from llama_index.core.schema import MetadataMode
        s_embed = d.get_metadata_str(mode=MetadataMode.EMBED)
        s_llm = d.get_metadata_str(mode=MetadataMode.LLM)
        s = s_embed if len(s_embed) > len(s_llm) else s_llm
        mlen = ntok(s)
        eff = CHUNK_SIZE - mlen
        if eff <= 0:
            status = "ValueError"
        elif eff < 50:
            status = "warning"
        else:
            status = "ok"
        print(f"{nama:<42}{len(meta):>6}{mlen:>10}{eff:>9}{status:>11}")

    print()
    print(f"Ambang: ValueError bila metadata >= {CHUNK_SIZE} token; "
          f"warning bila metadata > {CHUNK_SIZE - 50} token.")


# ─── E. Ambang pecah empiris per skenario metadata ───────────────────────────

def bagian_e(r_naratif: float) -> None:
    rule("E. AMBANG PECAH EMPIRIS — metadata memakan kuota, jadi ambang < CHUNK_SIZE")

    print("Konsekuensi bagian D: chunk TIDAK pecah di 512 token, tapi di")
    print("(512 - metadata_len). Makin banyak field metadata, makin cepat pecah.")
    print()

    skenario = [
        ("sekarang (8 field)", META_SEKARANG),
        ("+ chunk_id, document_id", {**META_SEKARANG,
                                     "chunk_id": META_RENCANA_TAMBAHAN["chunk_id"],
                                     "document_id": META_RENCANA_TAMBAHAN["document_id"]}),
        ("+ image_id, visual_type", {**META_SEKARANG,
                                     **{k: META_RENCANA_TAMBAHAN[k] for k in
                                        ("chunk_id", "document_id", "image_id", "visual_type")}}),
        ("+ manifest lengkap (16 field)", {**META_SEKARANG, **META_RENCANA_TAMBAHAN}),
    ]

    from llama_index.core.schema import MetadataMode

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    print(f"{'skenario':<34}{'ambang tok':>11}{'~char naratif':>15}{'~char tabel':>13}")
    print("-" * 78)
    for nama, meta in skenario:
        def _mk(text: str) -> Document:
            return Document(
                text=text,
                metadata=meta,
                excluded_embed_metadata_keys=list(EXCLUDED_KEYS),
                excluded_llm_metadata_keys=list(EXCLUDED_KEYS),
            )

        d = _mk("x")
        s_e = d.get_metadata_str(mode=MetadataMode.EMBED)
        s_l = d.get_metadata_str(mode=MetadataMode.LLM)
        mlen = ntok(s_e if len(s_e) > len(s_l) else s_l)
        ambang_tok = CHUNK_SIZE - mlen  # sentence.py:158

        # Verifikasi empiris: cari panjang teks terbesar yang masih 1 node.
        # teks_naratif membulat ke kalimat utuh, jadi resolusinya ~1 kalimat.
        tok_utuh_max, tok_pecah_min = 0, None
        for target in range(200, 5000, 60):
            t = teks_naratif(target)
            n = len(splitter.get_nodes_from_documents([_mk(t)]))
            if n == 1:
                tok_utuh_max = max(tok_utuh_max, ntok(t))
            elif tok_pecah_min is None:
                tok_pecah_min = ntok(t)
        empiris = f"{tok_utuh_max}..{tok_pecah_min}"

        print(f"{nama:<34}{ambang_tok:>11}{int(ambang_tok * r_naratif):>15,}"
              f"{int(ambang_tok * 2.44):>13,}  {empiris:>12}")

    print()
    print("ambang tok   = effective_chunk_size dari sentence.py:158 (eksak, dari rumus).")
    print("~char        = konversi memakai rasio terukur bagian A.")
    print("empiris tok  = token teks terpanjang yang masih 1 node .. terpendek yang pecah.")
    print("               Rentang ini harus mengapit kolom 'ambang tok'.")
    print()
    print("Bandingkan dengan batas yang dipakai preprocessing.py:")
    print(f"  :393  flush chunk teks pada  {CHUNK_SIZE} KARAKTER "
          f"(~{int(CHUNK_SIZE / r_naratif)} token) — jauh di bawah ambang, aman")
    print(f"  :357  tabel jadi chunk sendiri sampai {PDF_TABLE_MAX_CHARS} KARAKTER "
          f"— di ATAS ambang, pecah")
    print("  Element tunggal yang lolos :393 utuh (karena cek terjadi SEBELUM append)")
    print("  tidak punya batas atas sama sekali — inilah sumber pecahan pada teks biasa.")


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    print("probe_rechunk.py")
    print(f"repo: {ROOT}")
    print()
    bagian_env()
    print()
    print("KONFIGURASI AKTIF (jalankan ulang dengan env berbeda untuk membandingkan)")
    print(f"  INDEX_EXCLUDE_METADATA_FROM_EMBED = {INDEX_EXCLUDE_METADATA_FROM_EMBED}")
    print(f"  INDEX_MAX_CHUNK_TOKENS            = {INDEX_MAX_CHUNK_TOKENS}"
          f"{'  (0 = pemecahan 1B mati)' if INDEX_MAX_CHUNK_TOKENS <= 0 else ''}")
    print(f"  INDEX_MIN_CHUNK_TOKENS            = {INDEX_MIN_CHUNK_TOKENS}"
          f"{'  (0 = penyaring chunk pendek mati)' if INDEX_MIN_CHUNK_TOKENS <= 0 else ''}")
    print(f"  INDEX_STRUCTURAL_METADATA         = {INDEX_STRUCTURAL_METADATA}")
    print(f"  INDEX_TABLES_AS_OWN_CHUNKS        = {INDEX_TABLES_AS_OWN_CHUNKS}")
    print(f"  INDEX_DISABLE_NODE_PARSER         = {INDEX_DISABLE_NODE_PARSER}"
          f"{'  (transformations=[], 1 Document = 1 node)' if INDEX_DISABLE_NODE_PARSER else ''}")
    if EXCLUDED_KEYS:
        print(f"    dikecualikan: {', '.join(EXCLUDED_KEYS)}")
        tersisa = [k for k in META_SEKARANG if k not in EXCLUDED_KEYS]
        print(f"    divektorkan : {', '.join(tersisa)}")
    else:
        print("    dikecualikan: (tidak ada — seluruh field divektorkan)")

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    r_naratif = bagian_a()
    hasil = bagian_b(splitter)
    bagian_c()
    bagian_d()
    bagian_e(r_naratif)

    rule("RINGKASAN")
    gagal = [(n, t, k, c) for n, t, k, _, c, _ in hasil if k > c]
    print(f"Kasus diuji                     : {len(hasil)}")
    print(f"Masih dipecah splitter kedua    : {len(gagal)}")
    for n, t, k, c in gagal:
        print(f"    {n}  ({len(t)} char / {ntok(t)} tok) -> {c} chunk -> {k} node")

    tabrakan = [n for n, _, _, i, _, _ in hasil if len(set(i)) != len(i)]
    print(f"chunk_index bertabrakan         : {len(tabrakan)}")
    for n in tabrakan:
        print(f"    {n}")

    id_tabrakan = [n for n, _, _, _, _, ids in hasil if ids and len(set(ids)) != len(ids)]
    total_ids = sum(len(ids) for *_, ids in hasil)
    if INDEX_STRUCTURAL_METADATA:
        print(f"chunk_id bertabrakan            : {len(id_tabrakan)}  ({total_ids} id diperiksa)")
        for n in id_tabrakan:
            print(f"    {n}")
    else:
        print("chunk_id                        : tidak dihasilkan "
              "(INDEX_STRUCTURAL_METADATA mati)")

    print()
    print("-" * 78)
    print("Kriteria lulus: nol tabrakan chunk_index, dan chunk_id unik.")
    if tabrakan or id_tabrakan:
        print(f"VERDICT: GAGAL — chunk_index bertabrakan di {len(tabrakan)} kasus, "
              f"chunk_id di {len(id_tabrakan)} kasus.")
    elif INDEX_STRUCTURAL_METADATA and total_ids == 0:
        print("VERDICT: GAGAL — INDEX_STRUCTURAL_METADATA aktif tapi tidak ada chunk_id "
              "yang dihasilkan.")
    else:
        print(f"VERDICT: LULUS — {len(hasil)}/{len(hasil)} kasus, nol tabrakan chunk_index"
              + (f", {total_ids} chunk_id unik." if total_ids else "."))
        if gagal:
            print(f"         ({len(gagal)} kasus masih dipecah splitter kedua tapi tidak "
                  f"bertabrakan — periksa apakah itu disengaja.)")
    print("-" * 78)

    print()
    print("BELUM TERJAWAB TANPA KORPUS")
    print("  Probe ini memakai teks susunan tangan yang meniru bentuk dokumen di deck.")
    print("  Yang belum diketahui: berapa PROPORSI chunk nyata yang jatuh di atas ambang.")
    print("  Setelah data/pdfs terisi, jalankan V9 di INSPECTION_REPORT_2.md untuk")
    print("  menghitung tabrakan (file_name, chunk_index) pada index sungguhan.")


if __name__ == "__main__":
    main()
