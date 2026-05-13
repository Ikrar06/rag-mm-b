"""
preprocess-template.py
======================
Konversi semua file JSON di folder data/json menjadi file teks narasi
yang siap di-index ke vector DB (Qdrant) oleh pipeline RAG UNHAS.

Struktur output:
  data/
  └── narratives/
      ├── fakultas/
      │   ├── FEB.txt
      │   ├── FH.txt
      │   └── ...
      ├── prodi/
      │   ├── Informatika.txt
      │   └── ...
      ├── jenjang/
      ├── kurikulum/
      ├── mata-kuliah/
      ├── prasyarat/
      ├── rps/
      ├── kelas/
      ├── jadwal/
      ├── fasilitas/
      ├── pmb/
      └── pengumuman/

Setiap file .txt = 1 item = 1 chunk siap di-embed.

Usage:
    # Dari root project (C:\\Users\\ikrar\\Documents\\rag-prototype)
    python scripts/preprocess-template.py

    # Atau dengan path custom:
    python scripts/preprocess-template.py --input data/json --output data/narratives
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path


# ─────────────────────────────────────────────────────────────────
# Pastikan folder scripts bisa import dari root project
# ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent  # rag-prototype/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────
# Copy semua fungsi narasi di sini supaya script ini self-contained
# (tidak perlu import dari tempat lain)
# ─────────────────────────────────────────────────────────────────

def fmt_date(dt_str: str) -> str:
    try:
        dt = datetime.fromisoformat(str(dt_str).replace("Z", "+00:00"))
        return dt.strftime("%d %B %Y")
    except Exception:
        return str(dt_str)


def join_list(items: list, separator: str = ", ", last: str = " dan ") -> str:
    if not items:
        return "-"
    if len(items) == 1:
        return str(items[0])
    return separator.join(str(i) for i in items[:-1]) + last + str(items[-1])


def _safe(d: dict, key: str, fallback: str = "-") -> str:
    v = d.get(key)
    return str(v) if v is not None else fallback


# ── 1. Fakultas ──────────────────────────────────────────────────
def narrate_fakultas(d: dict) -> str:
    alamat = f", berlokasi di {d['alamat_fakultas']}" if d.get("alamat_fakultas") else ""
    kontak_parts = []
    if d.get("telepon_fakultas"):
        kontak_parts.append(f"telepon {d['telepon_fakultas']}")
    if d.get("email_fakultas"):
        kontak_parts.append(f"email {d['email_fakultas']}")
    if d.get("website_fakultas"):
        kontak_parts.append(f"website {d['website_fakultas']}")
    kontak = f"Dapat dihubungi melalui {join_list(kontak_parts)}." if kontak_parts else "Informasi kontak belum tersedia."

    return (
        f"{d['nama_fakultas']} (singkatan: {d['singkatan_fakultas']}, kode: {d['kode_fakultas']}) "
        f"adalah salah satu fakultas di Universitas Hasanuddin{alamat}.\n"
        f"{kontak}\n"
        f"Saat ini fakultas ini memiliki {d['jumlah_prodi_fakultas']} program studi aktif, "
        f"{d['jumlah_gedung_fakultas']} gedung, {d['jumlah_dosen_fakultas']} dosen, "
        f"{d['jumlah_mahasiswa_fakultas']} mahasiswa, {d['jumlah_laboratorium_fakultas']} laboratorium, "
        f"dan {d['jumlah_kelas_fakultas']} kelas.\n"
        f"Data terakhir diperbarui pada {fmt_date(d['updated_at'])}."
    )


# ── 2. Prodi ─────────────────────────────────────────────────────
def narrate_prodi(d: dict) -> str:
    kontak_parts = []
    if d.get("telepon_prodi"):
        kontak_parts.append(f"telepon {d['telepon_prodi']}")
    if d.get("email_prodi"):
        kontak_parts.append(f"email {d['email_prodi']}")
    if d.get("website_prodi"):
        kontak_parts.append(f"website {d['website_prodi']}")
    kontak = f"Informasi lebih lanjut dapat dihubungi melalui {join_list(kontak_parts)}." if kontak_parts else ""

    return (
        f"{_safe(d,'nama_prodi')} ({_safe(d,'singkatan_prodi')}) adalah program studi jenjang "
        f"{_safe(d,'nama_jenjang')} di bawah naungan Fakultas {_safe(d,'nama_fakultas')} "
        f"Universitas Hasanuddin, beralamat di {_safe(d,'alamat_prodi')}.\n"
        f"Kode DIKTI: {_safe(d,'kode_dikti_prodi')} | Kode internal UNHAS: {_safe(d,'kode_unhas_prodi')}.\n"
        f"Lulusan berhak menyandang gelar {_safe(d,'gelar_kelulusan_prodi')}. "
        f"Ketua Program Studi saat ini adalah {_safe(d,'nama_kaprodi')}.\n"
        f"Saat ini program studi ini memiliki {_safe(d,'jumlah_dosen_prodi')} dosen, "
        f"{_safe(d,'jumlah_mahasiswa_prodi')} mahasiswa aktif, "
        f"{_safe(d,'jumlah_kelas_prodi')} kelas, dan {_safe(d,'jumlah_mata_kuliah_prodi')} mata kuliah.\n"
        + (kontak + "\n" if kontak else "")
        + f"Data terakhir diperbarui pada {fmt_date(d['updated_at'])}."
    )


# ── 3. Jenjang ───────────────────────────────────────────────────
def narrate_jenjang(d: dict) -> str:
    return (
        f"{d['nama_jenjang']} ({d['kode_jenjang']}) adalah jenjang pendidikan yang tersedia "
        f"di Universitas Hasanuddin.\n"
        f"Mahasiswa pada jenjang ini dapat menempuh maksimal {d['maksimal_sks']} SKS "
        f"dalam masa studi maksimal {d['maksimal_semester']} semester.\n"
        f"Pada semester-semester awal, beban SKS dibatasi maksimal {d['maksimal_sks_semester_awal']} SKS, "
        f"dengan evaluasi akademik awal dilakukan pada akhir semester ke-{d['batas_evaluasi_semester_awal']}.\n"
        f"Saat ini terdapat {d['jumlah_prodi_jenjang']} program studi dan "
        f"{d['jumlah_mahasiswa_jenjang']} mahasiswa di jenjang ini.\n"
        f"Data terakhir diperbarui pada {fmt_date(d['updated_at'])}."
    )


# ── 4. Kurikulum ─────────────────────────────────────────────────
def narrate_kurikulum(d: dict) -> str:
    return (
        f"{d['nama_kurikulum']} adalah kurikulum yang berlaku sejak tahun {d['tahun_berlaku']} "
        f"untuk Program Studi {d['nama_program_studi']} di Universitas Hasanuddin, "
        f"ditetapkan berdasarkan SK Rektor nomor {d['nomor_sk_rektor']} "
        f"tertanggal {fmt_date(d['tanggal_sk_rektor'])}.\n"
        f"Untuk dapat lulus, mahasiswa wajib menyelesaikan total {d['total_sks_lulus']} SKS, "
        f"terdiri dari {d['total_sks_wajib']} SKS mata kuliah wajib dan "
        f"{d['total_sks_pilihan']} SKS mata kuliah pilihan.\n"
        f"Masa studi ideal adalah {d['masa_studi_ideal']} semester dengan batas maksimal "
        f"{d['masa_studi_maksimal']} semester.\n"
        f"Kurikulum ini memuat {d['jumlah_mata_kuliah']} mata kuliah: "
        f"{d['jumlah_mk_wajib']} mata kuliah wajib dan {d['jumlah_mk_pilihan']} mata kuliah pilihan.\n"
        f"Data terakhir diperbarui pada {fmt_date(d['updated_at'])}."
    )


# ── 5. Mata Kuliah ───────────────────────────────────────────────
def narrate_mata_kuliah(d: dict) -> str:
    status_umum = (
        "Mata kuliah ini termasuk dalam kategori Mata Kuliah Umum (MKU)."
        if d.get("status_mata_kuliah_umum")
        else "Mata kuliah ini bukan termasuk Mata Kuliah Umum."
    )
    return (
        f"{d['nama_mk']} (kode: {d['kode_mk']}) adalah mata kuliah "
        f"{str(d['mk_wajib_atau_pilihan']).lower()} bertipe {str(d['tipe_kelas_mk']).lower()} "
        f"yang ditawarkan pada paket semester ke-{d['paket_mk_setiap_semester']} "
        f"di Universitas Hasanuddin.\n"
        f"{status_umum}\n"
        f"Saat ini mata kuliah ini memiliki {d['jumlah_kelas']} kelas aktif "
        f"dengan total {d['jumlah_mahasiswa']} mahasiswa terdaftar.\n"
        f"Data terakhir diperbarui pada {fmt_date(d['updated_at'])}."
    )


# ── 6. Prasyarat ─────────────────────────────────────────────────
def narrate_prasyarat(d: dict) -> str:
    if d.get("daftar_mk_prasyarat"):
        nama_mk_list = [mk["nama_mk"] for mk in d["daftar_mk_prasyarat"]]
        prasyarat_clause = (
            f"Selain itu, mahasiswa juga wajib telah lulus mata kuliah prasyarat berikut: "
            f"{join_list(nama_mk_list)}."
        )
    else:
        prasyarat_clause = "Tidak ada mata kuliah prasyarat khusus yang dipersyaratkan."

    return (
        f"Untuk dapat mengambil mata kuliah {d['nama_mk']} (kode: {d['kode_mk']}), "
        f"mahasiswa harus memenuhi persyaratan berikut: "
        f"minimal telah menyelesaikan {d['min_sks']} SKS dan memiliki IPK minimal {d['min_ipk']}.\n"
        f"{prasyarat_clause}\n"
        f"Data terakhir diperbarui pada {fmt_date(d['updated_at'])}."
    )


# ── 7. RPS ───────────────────────────────────────────────────────
def narrate_rps(d: dict) -> str:
    evaluasi_parts = [
        f"{e['nama_evaluasi']} ({e['bobot_evaluasi']}%)"
        for e in d.get("daftar_evaluasi", [])
    ]
    evaluasi_list = join_list(evaluasi_parts) if evaluasi_parts else "-"

    pertemuan = d.get("daftar_pertemuan", [])
    if pertemuan:
        sample_indices = sorted({0, len(pertemuan) // 2, len(pertemuan) - 1})
        sample = [pertemuan[i]["materi_indonesia"] for i in sample_indices]
        materi_ringkas = (
            f"Dimulai dengan {sample[0]}"
            + (f", kemudian {sample[1]}" if len(sample) > 2 else "")
            + (f", dan diakhiri dengan {sample[-1]}" if len(sample) > 1 else "")
            + "."
        )
        # Juga sertakan semua topik pertemuan secara lengkap agar bisa di-retrieve
        semua_materi = "\n".join(
            f"Pertemuan {p['pertemuan']}: {p['materi_indonesia']}"
            for p in pertemuan
        )
    else:
        materi_ringkas = "Informasi materi belum tersedia."
        semua_materi = ""

    return (
        f"Rencana Pembelajaran Semester (RPS) mata kuliah {d['nama_mk']} (kode: {d['kode_mk']}) "
        f"terdiri dari {len(pertemuan)} pertemuan.\n"
        f"Sistem evaluasi menggunakan {d['komponen_evaluasi']} komponen penilaian "
        f"dengan {d['evaluasi_dominan']} sebagai komponen berbobot terbesar.\n"
        f"Rincian komponen evaluasi: {evaluasi_list}.\n"
        f"Gambaran materi perkuliahan: {materi_ringkas}\n"
        + (f"\nRincian topik per pertemuan:\n{semua_materi}\n" if semua_materi else "")
        + f"Data terakhir diperbarui pada {fmt_date(d['updated_at'])}."
    )


# ── 8. Kelas ─────────────────────────────────────────────────────
def narrate_kelas(d: dict) -> str:
    flags = []
    if d.get("is_batal"):
        flags.append("Perhatian: kelas ini telah dibatalkan.")
    if d.get("is_blok"):
        flags.append("Kelas ini diselenggarakan dalam format blok.")
    if d.get("is_antara_semester"):
        flags.append("Kelas ini merupakan kelas antara semester.")
    status = " ".join(flags) if flags else "Kelas berjalan normal."

    return (
        f"{d['nama_kelas']} adalah kelas {str(d['jenis_kelas']).lower()} "
        f"untuk mata kuliah {d['nama_mk']} (kode: {d['kode_mk']}) "
        f"pada tahun ajaran {d['tahun_ajaran']} semester {d['jenis_semester']} "
        f"di Universitas Hasanuddin.\n"
        f"Kelas ini berkapasitas {d['kapasitas_kelas']} mahasiswa dan saat ini diikuti oleh "
        f"{d['jumlah_peserta']} peserta, dengan sisa kuota {d['sisa_kuota']} tempat.\n"
        f"{status}\n"
        f"Diampu oleh {d['jumlah_dosen_pengampu']} dosen pengampu.\n"
        f"Data terakhir diperbarui pada {fmt_date(d['updated_at'])}."
    )


# ── 9. Jadwal ────────────────────────────────────────────────────
def narrate_jadwal(d: dict) -> str:
    nama_dosen = d.get("nama_dosen", [])
    nama_dosen_list = join_list(nama_dosen) if nama_dosen else "belum ditentukan"

    return (
        f"Mata kuliah {d['nama_mk']} (kode: {d['kode_mk']}) untuk {d['nama_kelas']} "
        f"dijadwalkan pada hari {d['hari']} pukul {d['jam_mulai']}–{d['jam_selesai']} "
        f"di {d['nama_ruang']}, {d['nama_gedung']}.\n"
        f"Jadwal ini berlaku untuk tahun ajaran {d['tahun_ajaran']} semester {d['jenis_semester']} "
        f"dengan jenis pertemuan {str(d['jenis_pertemuan']).lower()}.\n"
        f"Kapasitas ruang: {d['kapasitas_ruang']} orang. "
        f"Dosen pengampu: {nama_dosen_list}.\n"
        f"Data terakhir diperbarui pada {fmt_date(d['updated_at'])}."
    )


# ── 10. Fasilitas ────────────────────────────────────────────────
def narrate_fasilitas(d: dict) -> str:
    mku = (
        "Gedung ini juga melayani perkuliahan Mata Kuliah Umum (MKU)."
        if d.get("status_mku") else ""
    )
    ruangan_list = d.get("daftar_ruangan", [])
    if ruangan_list:
        r = ruangan_list[0]
        fasilitas = join_list(r.get("daftar_fasilitas", []))
        ruangan_clause = (
            f"Contoh ruangan: {r['nama_ruangan']} (kode {r['kode_ruangan']}) "
            f"di lantai {r['lantai']}, kapasitas {r['daya_tampung']} orang "
            f"(ujian: {r['daya_tampung_ujian']} orang), dilengkapi {fasilitas}."
        )
    else:
        ruangan_clause = ""

    return (
        f"{d['nama_gedung']} adalah gedung milik Fakultas {d['nama_fakultas']} "
        f"di Universitas Hasanuddin.\n"
        f"Gedung ini memiliki {d['jumlah_ruangan_gedung']} ruangan dengan total kapasitas "
        f"{d['total_kapasitas_gedung']} orang, terdiri dari "
        f"{d['jumlah_ruangan_kuliah_gedung']} ruang perkuliahan "
        f"({d['jumlah_ruangan_layak_gedung']} di antaranya dalam kondisi layak pakai).\n"
        + (mku + "\n" if mku else "")
        + (ruangan_clause + "\n" if ruangan_clause else "")
        + f"Data terakhir diperbarui pada {fmt_date(d['updated_at'])}."
    )


# ── 11. PMB ──────────────────────────────────────────────────────
def narrate_pmb(d: dict) -> str:
    nama_prodi = d.get("nama_prodi", [])
    kuota_list = d.get("jumlah_per_prodi_per_jalur_per_tahun", [])
    prodi_kuota = [
        f"{prodi} ({kuota} kursi)"
        for prodi, kuota in zip(nama_prodi, kuota_list)
    ]
    return (
        f"Jalur penerimaan {d['jalur_masuk']} ({d['keterangan_jalur']}) tersedia "
        f"untuk tahun masuk {d['tahun_masuk']} jenjang {d['jenjang']} "
        f"di Universitas Hasanuddin.\n"
        f"Melalui jalur ini, total {d['jumlah_diterima_per_jalur_per_tahun']} mahasiswa diterima "
        f"ke {len(nama_prodi)} program studi.\n"
        f"Rincian kuota per program studi: {join_list(prodi_kuota)}.\n"
        f"Data terakhir diperbarui pada {fmt_date(d['updated_at'])}."
    )


# ── 12. Pengumuman ───────────────────────────────────────────────
def narrate_pengumuman(d: dict) -> str:
    return (
        f"[Pengumuman] {d['judul']}\n"
        f"Tanggal: {fmt_date(d['tanggal'])}\n\n"
        f"{d['isi']}\n\n"
        f"Data terakhir diperbarui pada {fmt_date(d['updated_at'])}."
    )


# ── 13. Mahasiswa (info publik: NIM, nama, prodi, status) ────────
def narrate_mahasiswa(d: dict) -> str:
    return (
        f"Mahasiswa dengan NIM {d['nim']} bernama {d['nama']}, angkatan {d['angkatan']}, "
        f"terdaftar di Program Studi {d['prodi']} Fakultas {d['fakultas']} "
        f"Universitas Hasanuddin.\n"
        f"Status akademik saat ini: {d['status']}.\n"
        f"Data terakhir diperbarui pada {fmt_date(d['updated_at'])}."
    )


# ─────────────────────────────────────────────────────────────────
# SUMMARY NARRATORS — satu file ringkasan per endpoint
# Dipanggil dengan data.summary dari masing-masing JSON
# ─────────────────────────────────────────────────────────────────

def _narrate_summary_fakultas(s: dict) -> str:
    return (
        f"Ringkasan data Universitas Hasanuddin: terdapat {s['jumlah_fakultas_unhas']} fakultas "
        f"dengan total {s['jumlah_prodi_unhas']} program studi, {s['jumlah_gedung_unhas']} gedung, "
        f"dan {s['jumlah_dosen_unhas']} dosen di seluruh kampus."
    )

def _narrate_summary_jenjang(s: dict) -> str:
    return (
        f"Universitas Hasanuddin menyelenggarakan {s['jumlah_jenjang_unhas']} jenjang pendidikan, "
        f"mulai dari Diploma (D3/D4) hingga Doktoral (S3)."
    )

def _narrate_summary_kurikulum(s: dict) -> str:
    return (
        f"Saat ini terdapat {s['jumlah_kurikulum_aktif']} kurikulum aktif "
        f"yang berlaku di Universitas Hasanuddin."
    )

def _narrate_summary_mata_kuliah(s: dict) -> str:
    return (
        f"Total mata kuliah yang tersedia di Universitas Hasanuddin "
        f"saat ini adalah {s['jumlah_mata_kuliah']} mata kuliah."
    )

def _narrate_summary_prasyarat(s: dict) -> str:
    return (
        f"Terdapat {s['jumlah_mata_kuliah_prasyarat']} mata kuliah yang memiliki "
        f"ketentuan prasyarat akademik di Universitas Hasanuddin."
    )

def _narrate_summary_kelas(s: dict) -> str:
    return (
        f"Jumlah kelas aktif yang sedang berjalan di Universitas Hasanuddin "
        f"saat ini adalah {s['jumlah_kelas_aktif']} kelas."
    )

def _narrate_summary_jadwal(s: dict) -> str:
    n = s.get("jumlah_jadwal_per_hari") or s.get("jumlah_jadwal")
    if not n:
        return ""
    return f"Terdapat {n} jadwal perkuliahan yang terjadwal per hari di Universitas Hasanuddin."

def _narrate_summary_fasilitas(s: dict) -> str:
    return (
        f"Universitas Hasanuddin memiliki {s['jumlah_gedung_unhas']} gedung "
        f"yang tercatat dalam sistem fasilitas kampus."
    )

def _narrate_summary_pmb(s: dict) -> str:
    return (
        f"Universitas Hasanuddin membuka {s['jumlah_jalur_masuk_unhas']} jalur "
        f"penerimaan mahasiswa baru (PMB), yaitu SNBP, SNBT, dan Mandiri."
    )

def _narrate_summary_mahasiswa(s: dict) -> str:
    n = s.get("jumlah_mahasiswa_aktif") or s.get("jumlah_mahasiswa")
    if not n:
        return ""
    return f"Jumlah mahasiswa aktif Universitas Hasanuddin saat ini adalah {n:,} mahasiswa.".replace(",", ".")

SUMMARY_NARRATORS = {
    "fakultas":    _narrate_summary_fakultas,
    "jenjang":     _narrate_summary_jenjang,
    "kurikulum":   _narrate_summary_kurikulum,
    "mata-kuliah": _narrate_summary_mata_kuliah,
    "prasyarat":   _narrate_summary_prasyarat,
    "kelas":       _narrate_summary_kelas,
    "jadwal":      _narrate_summary_jadwal,
    "fasilitas":   _narrate_summary_fasilitas,
    "pmb":         _narrate_summary_pmb,
    "mahasiswa":   _narrate_summary_mahasiswa,
}


# ─────────────────────────────────────────────────────────────────
# DISPATCHER
# ─────────────────────────────────────────────────────────────────

NARRATORS = {
    "fakultas":    narrate_fakultas,
    "prodi":       narrate_prodi,
    "jenjang":     narrate_jenjang,
    "kurikulum":   narrate_kurikulum,
    "mata-kuliah": narrate_mata_kuliah,
    "prasyarat":   narrate_prasyarat,
    "rps":         narrate_rps,
    "kelas":       narrate_kelas,
    "jadwal":      narrate_jadwal,
    "fasilitas":   narrate_fasilitas,
    "pmb":         narrate_pmb,
    "pengumuman":  narrate_pengumuman,
    "mahasiswa":   narrate_mahasiswa,
}

# Mapping filename → endpoint key
FILE_MAP = {
    "fakultas.json":    "fakultas",
    "prodi.json":       "prodi",
    "jenjang.json":     "jenjang",
    "kurikulum.json":   "kurikulum",
    "mata-kuliah.json": "mata-kuliah",
    "prasyarat.json":   "prasyarat",
    "rps.json":         "rps",
    "kelas.json":       "kelas",
    "jadwal.json":      "jadwal",
    "fasilitas.json":   "fasilitas",
    "pmb.json":         "pmb",
    "pengumuman.json":  "pengumuman",
    "mahasiswa.json":   "mahasiswa",
}


def slugify(text: str) -> str:
    """Buat nama file aman dari string apapun."""
    text = str(text).strip()
    text = re.sub(r"[^\w\s\-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[\s]+", "_", text)
    return text[:80]  # batasi panjang nama file


def get_item_label(endpoint: str, item: dict, index: int) -> str:
    """Buat nama file yang deskriptif per item."""
    label_keys = {
        "fakultas":    "singkatan_fakultas",
        "prodi":       "nama_prodi",
        "jenjang":     "kode_jenjang",
        "kurikulum":   "nama_program_studi",
        "mata-kuliah": "kode_mk",
        "prasyarat":   "kode_mk",
        "rps":         "kode_mk",
        "kelas":       "nama_kelas",
        "jadwal":      "kode_mk",
        "fasilitas":   "nama_gedung",
        "pmb":         None,  # pakai kombinasi
        "pengumuman":  "judul",
        "mahasiswa":   "nim",
    }

    key = label_keys.get(endpoint)
    if endpoint == "pmb":
        label = f"{item.get('jalur_masuk','')}_{item.get('tahun_masuk','')}"
    elif key and item.get(key):
        label = str(item[key])
    else:
        label = str(index)

    return f"{index:04d}_{slugify(label)}"


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess JSON → narasi .txt untuk RAG UNHAS"
    )
    parser.add_argument(
        "--input", "-i",
        default="data/json",
        help="Folder berisi file JSON (default: data/json)"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/narratives",
        help="Folder output narasi .txt (default: data/narratives)"
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Encoding file output (default: utf-8)"
    )
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"[ERROR] Folder input tidak ditemukan: {input_dir.resolve()}")
        sys.exit(1)

    print("=" * 60)
    print("  RAG Preprocess — UNHAS Narrative Generator")
    print(f"  Input  : {input_dir.resolve()}")
    print(f"  Output : {output_dir.resolve()}")
    print("=" * 60)

    total_written = 0
    total_skipped = 0
    total_errors  = 0

    for filename, endpoint in FILE_MAP.items():
        filepath = input_dir / filename
        if not filepath.exists():
            print(f"\n[SKIP] {filename} tidak ditemukan — lewati.")
            total_skipped += 1
            continue

        # Buat subfolder output per endpoint
        ep_dir = output_dir / endpoint
        ep_dir.mkdir(parents=True, exist_ok=True)

        try:
            raw  = filepath.read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception as e:
            print(f"\n[ERROR] Gagal baca {filename}: {e}")
            total_errors += 1
            continue

        items = data.get("data", {}).get("items", [])
        if not items:
            print(f"\n[SKIP] {filename} kosong (0 item).")
            total_skipped += 1
            continue

        narrator = NARRATORS[endpoint]
        written  = 0
        errors   = 0

        print(f"\n── {endpoint.upper()} ({len(items)} item) ──")

        for idx, item in enumerate(items):
            try:
                narrative = narrator(item)
                label     = get_item_label(endpoint, item, idx)
                out_path  = ep_dir / f"{label}.txt"
                out_path.write_text(narrative, encoding=args.encoding)
                written += 1
            except Exception as e:
                print(f"   [!] Item #{idx} error: {e}")
                errors += 1

        # Tulis _ringkasan.txt dari data.summary jika tersedia
        summary_data = data.get("data", {}).get("summary", {})
        if summary_data and endpoint in SUMMARY_NARRATORS:
            try:
                summary_text = SUMMARY_NARRATORS[endpoint](summary_data)
                if summary_text:
                    summary_path = ep_dir / "_ringkasan.txt"
                    summary_path.write_text(summary_text, encoding=args.encoding)
                    written += 1
                    print(f"   + _ringkasan.txt")
            except Exception as e:
                print(f"   [!] Summary error: {e}")
                errors += 1

        total_written += written
        total_errors  += errors
        status = f"✓ {written} file ditulis"
        if errors:
            status += f", {errors} error"
        print(f"   {status}  →  {ep_dir}")

    print("\n" + "=" * 60)
    print(f"  ✅ Selesai!")
    print(f"     Total file  : {total_written}")
    print(f"     Dilewati    : {total_skipped} endpoint")
    print(f"     Error       : {total_errors}")
    print(f"     Output dir  : {output_dir.resolve()}")
    print("=" * 60)

    if total_written > 0:
        print(f"\nLangkah berikutnya:")
        print(f"  Jalankan indexing pipeline untuk embed file di {output_dir.resolve()}")
        print(f"  Contoh: python scripts/indexing.py --input {args.output}")


if __name__ == "__main__":
    main()