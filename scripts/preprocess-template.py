"""
preprocess-template.py
======================
Konversi semua file JSON di folder data/json menjadi file teks narasi
yang siap di-index ke vector DB (Qdrant) oleh pipeline RAG UNHAS.

Struktur JSON input (per file):
  {
    "success": true,
    "error": null,
    "data": {
      "items": [...],
      "pagination": { "total": X, "page": X, "page_size": X, "total_pages": X },
      "summary": { ... }
    }
  }

Struktur output:
  data/
  └── narratives/
      ├── fakultas/
      │   ├── 0000_EKO.txt
      │   └── _ringkasan.txt
      ├── prodi/
      ├── jenjang/
      ├── kurikulum/
      ├── mata-kuliah/
      ├── prasyarat/
      ├── kelas/
      ├── jadwal/
      ├── fasilitas/
      ├── pmb/
      └── pengumuman/

Usage:
    python scripts/preprocess-template.py
    python scripts/preprocess-template.py --input data/json --output data/narratives
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────
# UTIL
# ─────────────────────────────────────────────────────────────────

def fmt_date(dt_str) -> str:
    if not dt_str:
        return "-"
    try:
        s = str(dt_str).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt.strftime("%d %B %Y")
    except Exception:
        return str(dt_str)


def fmt_time(t: str) -> str:
    """Kembalikan '-' jika 00:00:00 (belum diisi), lain-lain tampilkan apa adanya."""
    if not t or t.strip() == "00:00:00":
        return "belum ditentukan"
    return t.strip()


def safe(d: dict, key: str, fallback: str = "-") -> str:
    """Ambil nilai dari dict, return fallback jika None/tidak ada."""
    v = d.get(key)
    if v is None or v == "":
        return fallback
    return str(v)


def num(d: dict, key: str, fallback: str = "0") -> str:
    """Ambil nilai numerik, return fallback jika None."""
    v = d.get(key)
    if v is None:
        return fallback
    return str(v)


def join_list(items: list, sep: str = ", ", last: str = " dan ") -> str:
    items = [str(i) for i in items if i]
    if not items:
        return "-"
    if len(items) == 1:
        return items[0]
    return sep.join(items[:-1]) + last + items[-1]


def expand_mk_type(kode: str) -> str:
    """Expand kode P/W menjadi kata lengkap."""
    mapping = {
        "P": "pilihan",
        "W": "wajib",
        "p": "pilihan",
        "w": "wajib",
    }
    return mapping.get(str(kode).strip(), str(kode).lower())


def format_semester_kode(kode) -> str:
    """
    Semester kode seperti 20261 → 'Tahun 2026, Semester 1 (Ganjil)'
    Jika tidak bisa diparse, kembalikan apa adanya.
    """
    try:
        s = str(kode)
        if len(s) == 5:
            tahun = s[:4]
            sem   = s[4]
            label = "Ganjil" if sem == "1" else "Genap" if sem == "2" else sem
            return f"{tahun} Semester {label}"
        return s
    except Exception:
        return str(kode)


# ─────────────────────────────────────────────────────────────────
# NARRATORS
# ─────────────────────────────────────────────────────────────────

# ── 1. Fakultas ──────────────────────────────────────────────────
def narrate_fakultas(d: dict) -> str:
    nama   = safe(d, "nama_fakultas")
    singk  = safe(d, "singkatan_fakultas")
    kode   = safe(d, "kode_fakultas")
    alamat = safe(d, "alamat_fakultas", "")

    lokasi = f", berlokasi di {alamat}" if alamat and alamat != "-" else ""

    kontak_parts = []
    if d.get("telepon_fakultas"):
        kontak_parts.append(f"telepon {d['telepon_fakultas']}")
    if d.get("email_fakultas"):
        kontak_parts.append(f"email {d['email_fakultas']}")
    if d.get("website_fakultas"):
        kontak_parts.append(f"website {d['website_fakultas']}")
    kontak = (
        f"Dapat dihubungi melalui {join_list(kontak_parts)}."
        if kontak_parts else
        "Informasi kontak belum tersedia."
    )

    return (
        f"{nama} (singkatan: {singk}, kode: {kode}) adalah salah satu fakultas "
        f"di Universitas Hasanuddin{lokasi}.\n"
        f"{kontak}\n"
        f"Saat ini fakultas ini memiliki {num(d,'jumlah_prodi_fakultas')} program studi aktif, "
        f"{num(d,'jumlah_gedung_fakultas')} gedung, "
        f"{num(d,'jumlah_dosen_fakultas')} dosen, "
        f"{num(d,'jumlah_mahasiswa_fakultas')} mahasiswa, "
        f"{num(d,'jumlah_laboratorium_fakultas')} laboratorium, "
        f"dan {num(d,'jumlah_kelas_fakultas')} kelas.\n"
        f"Data terakhir diperbarui pada {fmt_date(d.get('updated_at'))}."
    )


# ── 2. Prodi ─────────────────────────────────────────────────────
def narrate_prodi(d: dict) -> str:
    nama     = safe(d, "nama_prodi")
    singk    = safe(d, "singkatan_prodi")
    jenjang  = safe(d, "nama_jenjang")
    fakultas = safe(d, "nama_fakultas")
    alamat   = safe(d, "alamat_prodi", "")
    kode_dikti  = safe(d, "kode_dikti_prodi")
    kode_unhas  = safe(d, "kode_unhas_prodi")
    gelar       = safe(d, "gelar_kelulusan_prodi")
    kaprodi     = safe(d, "nama_kaprodi")
    status_aktif = int(d.get("status_aktif_prodi") or 0)

    lokasi = f", beralamat di {alamat}" if alamat and alamat != "-" else ""
    status = "Aktif" if status_aktif else "Tidak Aktif"

    kontak_parts = []
    if d.get("telepon_prodi"):
        kontak_parts.append(f"telepon {d['telepon_prodi']}")
    if d.get("email_prodi"):
        kontak_parts.append(f"email {d['email_prodi']}")
    if d.get("website_prodi"):
        kontak_parts.append(f"website {d['website_prodi']}")
    kontak = (
        f"Informasi lebih lanjut dapat dihubungi melalui {join_list(kontak_parts)}.\n"
        if kontak_parts else ""
    )

    kode_info = []
    if kode_dikti != "-":
        kode_info.append(f"Kode DIKTI: {kode_dikti}")
    if kode_unhas != "-":
        kode_info.append(f"Kode internal UNHAS: {kode_unhas}")
    kode_line = " | ".join(kode_info) + ".\n" if kode_info else ""

    gelar_line = f"Lulusan berhak menyandang gelar {gelar}.\n" if gelar != "-" else ""
    kaprodi_line = f"Ketua Program Studi saat ini adalah {kaprodi}.\n" if kaprodi != "-" else ""

    return (
        f"{nama} ({singk}) adalah program studi jenjang {jenjang} "
        f"di bawah naungan Fakultas {fakultas} Universitas Hasanuddin{lokasi}.\n"
        f"Status: {status}.\n"
        f"{kode_line}"
        f"{gelar_line}"
        f"{kaprodi_line}"
        f"Saat ini program studi ini memiliki {num(d,'jumlah_dosen_prodi')} dosen, "
        f"{num(d,'jumlah_mahasiswa_prodi')} mahasiswa aktif, "
        f"{num(d,'jumlah_kelas_prodi')} kelas, dan "
        f"{num(d,'jumlah_mata_kuliah_prodi')} mata kuliah.\n"
        f"{kontak}"
        f"Data terakhir diperbarui pada {fmt_date(d.get('updated_at'))}."
    )


# ── 3. Jenjang ───────────────────────────────────────────────────
def narrate_jenjang(d: dict) -> str:
    kode   = safe(d, "kode_jenjang")
    nama   = safe(d, "nama_jenjang")
    maks_sks = num(d, "maksimal_sks")
    maks_sem = d.get("maksimal_semester")
    maks_sks_awal = num(d, "maksimal_sks_semester_awal")
    batas_eval    = num(d, "batas_evaluasi_semester_awal")

    sem_line = (
        f"dalam masa studi maksimal {maks_sem} semester"
        if maks_sem is not None else
        "dengan masa studi maksimal yang belum ditetapkan"
    )

    return (
        f"{nama} ({kode}) adalah jenjang pendidikan yang tersedia "
        f"di Universitas Hasanuddin.\n"
        f"Mahasiswa pada jenjang ini dapat menempuh maksimal {maks_sks} SKS per semester, "
        f"{sem_line}.\n"
        f"Pada semester-semester awal, beban SKS dibatasi maksimal {maks_sks_awal} SKS, "
        f"dengan evaluasi akademik awal dilakukan pada akhir semester ke-{batas_eval}.\n"
        f"Saat ini terdapat {num(d,'jumlah_prodi_jenjang')} program studi dan "
        f"{num(d,'jumlah_mahasiswa_jenjang')} mahasiswa di jenjang ini.\n"
        f"Data terakhir diperbarui pada {fmt_date(d.get('updated_at'))}."
    )


# ── 4. Kurikulum ─────────────────────────────────────────────────
def narrate_kurikulum(d: dict) -> str:
    nama     = safe(d, "nama_kurikulum")
    tahun    = num(d, "tahun_berlaku")
    prodi    = safe(d, "nama_program_studi")
    sk_no    = safe(d, "nomor_sk_rektor")
    sk_tgl   = fmt_date(d.get("tanggal_sk_rektor"))
    aktif    = int(d.get("status_aktif_kurikulum") or 0)
    status   = "aktif" if aktif else "tidak aktif"

    sks_lulus  = int(d.get("total_sks_lulus") or 0)
    sks_wajib  = int(d.get("total_sks_wajib") or 0)
    sks_pilihan = int(d.get("total_sks_pilihan") or 0)
    jml_mk      = int(d.get("jumlah_mata_kuliah") or 0)
    jml_wajib   = int(d.get("jumlah_mk_wajib") or 0)
    jml_pilihan = int(d.get("jumlah_mk_pilihan") or 0)

    # SK info — sembunyikan kalau "0000"
    sk_info = (
        f"ditetapkan berdasarkan SK Rektor nomor {sk_no} tertanggal {sk_tgl}"
        if sk_no not in ("-", "0000") else
        f"(SK Rektor belum tersedia)"
    )

    sks_info = (
        f"Untuk dapat lulus, mahasiswa wajib menyelesaikan total {sks_lulus} SKS, "
        f"terdiri dari {sks_wajib} SKS mata kuliah wajib dan {sks_pilihan} SKS mata kuliah pilihan.\n"
        if sks_lulus > 0 else
        "Informasi total SKS kelulusan belum tersedia.\n"
    )

    mk_info = (
        f"Kurikulum ini memuat {jml_mk} mata kuliah: "
        f"{jml_wajib} mata kuliah wajib dan {jml_pilihan} mata kuliah pilihan.\n"
        if jml_mk > 0 else
        "Daftar mata kuliah pada kurikulum ini belum tersedia.\n"
    )

    return (
        f"{nama} adalah kurikulum {status} yang berlaku sejak tahun {tahun} "
        f"untuk Program Studi {prodi} di Universitas Hasanuddin, {sk_info}.\n"
        f"{sks_info}"
        f"Masa studi ideal adalah {num(d,'masa_studi_ideal')} semester "
        f"dengan batas maksimal {num(d,'masa_studi_maksimal')} semester.\n"
        f"{mk_info}"
        f"Data terakhir diperbarui pada {fmt_date(d.get('updated_at'))}."
    )


# ── 5. Mata Kuliah ───────────────────────────────────────────────
def narrate_mata_kuliah(d: dict) -> str:
    nama   = safe(d, "nama_mk")
    kode   = safe(d, "kode_mk")
    singk  = safe(d, "singkatan_mk", "")
    singk_str = f" ({singk})" if singk and singk != "-" else ""

    tipe_raw  = safe(d, "mk_wajib_atau_pilihan")
    tipe      = expand_mk_type(tipe_raw)          # W→wajib, P→pilihan
    jenis_kls = safe(d, "tipe_kelas_mk", "tatap muka").lower()
    paket_sem = num(d, "paket_mk_setiap_semester")

    umum      = int(d.get("status_mata_kuliah_umum") or 0)
    aktif     = int(d.get("status_mata_kuliah_aktif") or 0)
    status_umum = (
        "Mata kuliah ini termasuk dalam kategori Mata Kuliah Umum (MKU)."
        if umum else
        "Mata kuliah ini bukan termasuk Mata Kuliah Umum."
    )
    status_aktif = "aktif" if aktif else "tidak aktif"

    return (
        f"{nama}{singk_str} (kode: {kode}) adalah mata kuliah {tipe} "
        f"bertipe kelas {jenis_kls} yang ditawarkan pada paket semester ke-{paket_sem} "
        f"di Universitas Hasanuddin.\n"
        f"Status: {status_aktif}. {status_umum}\n"
        f"Saat ini mata kuliah ini memiliki {num(d,'jumlah_kelas')} kelas aktif "
        f"dengan total {num(d,'jumlah_mahasiswa')} mahasiswa terdaftar.\n"
        f"Data terakhir diperbarui pada {fmt_date(d.get('updated_at'))}."
    )


# ── 6. Prasyarat ─────────────────────────────────────────────────
def narrate_prasyarat(d: dict) -> str:
    nama = safe(d, "nama_mk")
    kode = safe(d, "kode_mk")

    min_sks = float(d.get("min_sks") or 0)
    min_ipk = float(d.get("min_ipk") or 0)

    syarat_parts = []
    if min_sks > 0:
        syarat_parts.append(f"minimal telah menyelesaikan {min_sks:.0f} SKS")
    if min_ipk > 0:
        syarat_parts.append(f"memiliki IPK minimal {min_ipk:.2f}")

    if syarat_parts:
        syarat_clause = f"Persyaratan akademik: {join_list(syarat_parts)}."
    else:
        syarat_clause = "Tidak ada persyaratan SKS atau IPK minimum yang ditetapkan."

    mk_prasyarat = d.get("daftar_mk_prasyarat", [])
    if mk_prasyarat:
        nama_list = [mk.get("nama_mk", "") for mk in mk_prasyarat if mk.get("nama_mk")]
        prasyarat_clause = (
            f"Mahasiswa juga wajib telah lulus mata kuliah prasyarat berikut: "
            f"{join_list(nama_list)}."
        )
    else:
        prasyarat_clause = "Tidak ada mata kuliah prasyarat yang dipersyaratkan."

    return (
        f"Informasi prasyarat untuk mata kuliah {nama} (kode: {kode}):\n"
        f"{syarat_clause}\n"
        f"{prasyarat_clause}\n"
        f"Total jumlah mata kuliah prasyarat: {num(d,'jumlah_mk_prasyarat')}.\n"
        f"Data terakhir diperbarui pada {fmt_date(d.get('updated_at'))}."
    )


# ── 7. Kelas ─────────────────────────────────────────────────────
def narrate_kelas(d: dict) -> str:
    flags = []
    if int(d.get("is_batal") or 0):
        flags.append("Perhatian: kelas ini telah dibatalkan.")
    if int(d.get("is_blok") or 0):
        flags.append("Kelas ini diselenggarakan dalam format blok.")
    if int(d.get("is_antara_semester") or 0):
        flags.append("Kelas ini merupakan kelas antara semester.")
    status = " ".join(flags) if flags else "Kelas berjalan normal."

    nama_kelas  = safe(d, "nama_kelas")
    jenis_kelas = safe(d, "jenis_kelas", "reguler").lower()
    nama_mk     = safe(d, "nama_mk")
    kode_mk     = safe(d, "kode_mk")
    ta          = safe(d, "tahun_ajaran")
    semester    = safe(d, "jenis_semester")
    prodi       = safe(d, "nama_prodi")

    return (
        f"{nama_kelas} adalah kelas {jenis_kelas} untuk mata kuliah "
        f"{nama_mk} (kode: {kode_mk}) "
        f"pada tahun ajaran {ta} semester {semester} "
        f"di Program Studi {prodi}, Universitas Hasanuddin.\n"
        f"Kapasitas: {num(d,'kapasitas_kelas')} mahasiswa. "
        f"Peserta saat ini: {num(d,'jumlah_peserta')}. "
        f"Sisa kuota: {num(d,'sisa_kuota')} tempat.\n"
        f"{status}\n"
        f"Diampu oleh {num(d,'jumlah_dosen_pengampu')} dosen pengampu.\n"
        f"Data terakhir diperbarui pada {fmt_date(d.get('updated_at'))}."
    )


# ── 8. Jadwal ────────────────────────────────────────────────────
def narrate_jadwal(d: dict) -> str:
    nama_dosen = d.get("nama_dosen", [])
    dosen_str  = join_list(nama_dosen) if nama_dosen else "belum ditentukan"

    jam_mulai   = fmt_time(d.get("jam_mulai", ""))
    jam_selesai = fmt_time(d.get("jam_selesai", ""))
    waktu       = (
        f"pukul {jam_mulai}–{jam_selesai}"
        if jam_mulai != "belum ditentukan" else
        "waktu belum ditentukan"
    )

    return (
        f"Mata kuliah {safe(d,'nama_mk')} (kode: {safe(d,'kode_mk')}) "
        f"untuk kelas {safe(d,'nama_kelas')} "
        f"dijadwalkan pada hari {safe(d,'hari')} {waktu} "
        f"di {safe(d,'nama_ruang')}, Gedung {safe(d,'nama_gedung')}.\n"
        f"Tahun ajaran {safe(d,'tahun_ajaran')} semester {safe(d,'jenis_semester')}, "
        f"jenis pertemuan: {safe(d,'jenis_pertemuan').lower()}.\n"
        f"Kapasitas slot: {num(d,'kapasitas_slot')} orang. "
        f"Kapasitas ruang: {num(d,'kapasitas_ruang')} orang.\n"
        f"Dosen pengampu: {dosen_str}.\n"
        f"Data terakhir diperbarui pada {fmt_date(d.get('updated_at'))}."
    )


# ── 9. Fasilitas ─────────────────────────────────────────────────
def narrate_fasilitas(d: dict) -> str:
    mku = (
        "Gedung ini juga melayani perkuliahan Mata Kuliah Umum (MKU)."
        if int(d.get("status_mku") or 0) else ""
    )

    ruangan_list = d.get("daftar_ruangan", [])
    ruangan_detail = ""
    if ruangan_list:
        # Tampilkan semua ruangan (atau ringkasan jika banyak)
        if len(ruangan_list) <= 5:
            lines = []
            for r in ruangan_list:
                fasil = join_list(r.get("daftar_fasilitas", []))
                fasil_str = f", fasilitas: {fasil}" if fasil != "-" else ""
                dt_ujian = r.get("daya_tampung_ujian") or 0
                ujian_str = f" (ujian: {dt_ujian} orang)" if dt_ujian > 0 else ""
                lines.append(
                    f"  - {r.get('nama_ruangan','-')} (kode {r.get('kode_ruangan','-')}, "
                    f"lantai {r.get('lantai','-')}, kapasitas {r.get('daya_tampung',0)} orang{ujian_str}{fasil_str})"
                )
            ruangan_detail = "Daftar ruangan:\n" + "\n".join(lines) + "\n"
        else:
            # Ringkasan saja — detail ada di chunk terpisah jika diperlukan
            r0 = ruangan_list[0]
            fasil0 = join_list(r0.get("daftar_fasilitas", []))
            fasil_str0 = f", dilengkapi {fasil0}" if fasil0 != "-" else ""
            ruangan_detail = (
                f"Contoh ruangan: {r0.get('nama_ruangan','-')} "
                f"(kode {r0.get('kode_ruangan','-')}, lantai {r0.get('lantai','-')}, "
                f"kapasitas {r0.get('daya_tampung',0)} orang{fasil_str0}).\n"
            )

    return (
        f"{safe(d,'nama_gedung')} adalah gedung milik Fakultas {safe(d,'nama_fakultas')} "
        f"di Universitas Hasanuddin.\n"
        f"Gedung ini memiliki {num(d,'jumlah_ruangan_gedung')} ruangan "
        f"dengan total kapasitas {num(d,'total_kapasitas_gedung')} orang, "
        f"terdiri dari {num(d,'jumlah_ruangan_kuliah_gedung')} ruang perkuliahan "
        f"({num(d,'jumlah_ruangan_layak_gedung')} di antaranya dalam kondisi layak pakai).\n"
        + (mku + "\n" if mku else "")
        + ruangan_detail
        + f"Data terakhir diperbarui pada {fmt_date(d.get('updated_at'))}."
    )


# ── 10. PMB ──────────────────────────────────────────────────────
def narrate_pmb(d: dict) -> str:
    jalur      = safe(d, "jalur_masuk")
    keterangan = safe(d, "keterangan_jalur")
    tahun_raw  = d.get("tahun_masuk")
    tahun      = format_semester_kode(tahun_raw) if tahun_raw else "-"
    jenjang    = safe(d, "jenjang")
    total      = num(d, "jumlah_diterima_per_jalur_per_tahun")

    kuota_list = d.get("jumlah_per_prodi_per_jalur_per_tahun", [])
    if kuota_list and isinstance(kuota_list, list):
        if isinstance(kuota_list[0], dict):
            # Format baru: list of {nama_prodi, jumlah_diterima}
            prodi_kuota = [
                f"{item.get('nama_prodi','-')} ({item.get('jumlah_diterima',0)} kursi)"
                for item in kuota_list
            ]
        else:
            prodi_kuota = [str(k) for k in kuota_list]
        detail_prodi = f"Rincian kuota per program studi: {join_list(prodi_kuota)}.\n"
    else:
        detail_prodi = "Rincian kuota per program studi belum tersedia.\n"

    keterangan_str = (
        f" ({keterangan})" if keterangan != jalur and keterangan != "-" else ""
    )

    return (
        f"Jalur penerimaan mahasiswa baru (PMB) {jalur}{keterangan_str} "
        f"tersedia untuk {tahun} jenjang {jenjang} di Universitas Hasanuddin.\n"
        f"Melalui jalur ini, total {total} mahasiswa diterima "
        f"ke {len(kuota_list)} program studi.\n"
        f"{detail_prodi}"
        f"Data terakhir diperbarui pada {fmt_date(d.get('updated_at'))}."
    )


# ── 11. Pengumuman ───────────────────────────────────────────────
def narrate_pengumuman(d: dict) -> str:
    return (
        f"[Pengumuman Universitas Hasanuddin]\n"
        f"Judul  : {safe(d,'judul')}\n"
        f"Tanggal: {fmt_date(d.get('tanggal'))}\n\n"
        f"{safe(d,'isi','-')}\n\n"
        f"Data terakhir diperbarui pada {fmt_date(d.get('updated_at'))}."
    )


# ── 12. Metadata (khusus, tidak ada items) ───────────────────────
def narrate_metadata(d: dict) -> str:
    return (
        f"[Metadata API Satu Data UNHAS]\n"
        f"Versi: {safe(d,'version')}\n"
        f"Terakhir diperbarui: {fmt_date(d.get('last_updated'))}\n"
        f"Deskripsi: {safe(d,'description','-')}\n"
    )


# ─────────────────────────────────────────────────────────────────
# SUMMARY NARRATORS
# ─────────────────────────────────────────────────────────────────

def _summary_fakultas(s: dict) -> str:
    return (
        f"Ringkasan data Universitas Hasanuddin: terdapat "
        f"{s.get('jumlah_fakultas_unhas','?')} fakultas "
        f"dengan total {s.get('jumlah_prodi_unhas','?')} program studi, "
        f"{s.get('jumlah_gedung_unhas','?')} gedung, "
        f"dan {s.get('jumlah_dosen_unhas','?')} dosen di seluruh kampus."
    )

def _summary_prodi(s: dict) -> str:
    return (
        f"Universitas Hasanuddin memiliki {s.get('jumlah_prodi_total','?')} program studi "
        f"dengan total {s.get('jumlah_mahasiswa_total','?')} mahasiswa "
        f"dan {s.get('jumlah_dosen_total','?')} dosen."
    )

def _summary_jenjang(s: dict) -> str:
    return (
        f"Universitas Hasanuddin menyelenggarakan {s.get('jumlah_jenjang_unhas','?')} "
        f"jenjang pendidikan, mulai dari Diploma hingga Doktoral."
    )

def _summary_kurikulum(s: dict) -> str:
    return (
        f"Universitas Hasanuddin saat ini memiliki {s.get('jumlah_kurikulum','?')} kurikulum, "
        f"dengan {s.get('jumlah_kurikulum_aktif','?')} di antaranya berstatus aktif."
    )

def _summary_mata_kuliah(s: dict) -> str:
    return (
        f"Total mata kuliah di Universitas Hasanuddin: {s.get('jumlah_mata_kuliah','?')}, "
        f"terdiri dari {s.get('jumlah_mk_wajib','?')} mata kuliah wajib "
        f"dan {s.get('jumlah_mk_pilihan','?')} mata kuliah pilihan."
    )

def _summary_prasyarat(s: dict) -> str:
    return (
        f"Terdapat {s.get('jumlah_mata_kuliah_prasyarat','?')} mata kuliah yang memiliki "
        f"ketentuan prasyarat akademik di Universitas Hasanuddin "
        f"({s.get('jumlah_mk_tanpa_prasyarat','?')} mata kuliah tanpa prasyarat)."
    )

def _summary_pmb(s: dict) -> str:
    return (
        f"Universitas Hasanuddin membuka {s.get('jumlah_jalur_masuk_unhas','?')} jalur "
        f"penerimaan mahasiswa baru (PMB)."
    )

def _summary_pengumuman(s: dict) -> str:
    n = s.get("jumlah_pengumuman")
    if not n:
        return ""
    return f"Terdapat {n} pengumuman yang telah diterbitkan oleh Universitas Hasanuddin."

def _summary_fasilitas(s: dict) -> str:
    return (
        f"Universitas Hasanuddin memiliki {s.get('jumlah_gedung_unhas','?')} gedung "
        f"dengan total {s.get('jumlah_ruangan_unhas','?')} ruangan, "
        f"kapasitas keseluruhan {s.get('total_kapasitas_unhas','?')} orang. "
        f"Terdapat {s.get('jumlah_ruangan_kuliah_unhas','?')} ruang perkuliahan, "
        f"{s.get('jumlah_ruangan_layak_unhas','?')} di antaranya layak pakai."
    )

def _summary_jadwal(s: dict) -> str:
    per_hari = s.get("jumlah_jadwal_per_hari", {})
    if isinstance(per_hari, dict) and per_hari:
        hari_str = ", ".join(f"{h}: {n} jadwal" for h, n in per_hari.items())
        jadwal_line = f"Distribusi jadwal per hari: {hari_str}. "
    else:
        jadwal_line = ""
    ruang   = s.get("jumlah_ruang_terpakai", "?")
    dosen   = s.get("jumlah_dosen_mengajar", "?")
    return (
        f"Jadwal perkuliahan Universitas Hasanuddin: {jadwal_line}"
        f"{ruang} ruang digunakan dan {dosen} dosen mengajar."
    )

def _summary_kelas(s: dict) -> str:
    aktif  = s.get("jumlah_kelas_aktif", "?")
    batal  = s.get("jumlah_kelas_batal", "?")
    per_sem = s.get("jumlah_kelas_per_semester", [])
    sem_str = ""
    if isinstance(per_sem, list) and per_sem:
        items = [
            f"{x.get('tahun_ajaran','')} {x.get('jenis_semester','')}: {x.get('jumlah_kelas',0)} kelas"
            for x in per_sem[:5]   # tampilkan maks 5 semester terakhir
        ]
        sem_str = " | ".join(items)
        sem_str = f" Distribusi per semester: {sem_str}."
    return (
        f"Universitas Hasanuddin memiliki {aktif} kelas aktif "
        f"dan {batal} kelas yang dibatalkan saat ini.{sem_str}"
    )

SUMMARY_NARRATORS = {
    "fakultas":    _summary_fakultas,
    "prodi":       _summary_prodi,
    "jenjang":     _summary_jenjang,
    "kurikulum":   _summary_kurikulum,
    "mata-kuliah": _summary_mata_kuliah,
    "prasyarat":   _summary_prasyarat,
    "pmb":         _summary_pmb,
    "pengumuman":  _summary_pengumuman,
    "fasilitas":   _summary_fasilitas,
    "jadwal":      _summary_jadwal,
    "kelas":       _summary_kelas,
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
    "kelas":       narrate_kelas,
    "jadwal":      narrate_jadwal,
    "fasilitas":   narrate_fasilitas,
    "pmb":         narrate_pmb,
    "pengumuman":  narrate_pengumuman,
}

FILE_MAP = {
    "fakultas.json":    "fakultas",
    "prodi.json":       "prodi",
    "jenjang.json":     "jenjang",
    "kurikulum.json":   "kurikulum",
    "mata-kuliah.json": "mata-kuliah",
    "prasyarat.json":   "prasyarat",
    "kelas.json":       "kelas",
    "jadwal.json":      "jadwal",
    "fasilitas.json":   "fasilitas",
    "pmb.json":         "pmb",
    "pengumuman.json":  "pengumuman",
    "metadata.json":    None,   # ditangani khusus
}


# ─────────────────────────────────────────────────────────────────
# SLUGIFY & LABEL
# ─────────────────────────────────────────────────────────────────

def slugify(text: str, max_len: int = 80) -> str:
    text = str(text).strip()
    text = re.sub(r"[^\w\s\-]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "_", text)
    return text[:max_len]


def get_item_label(endpoint: str, item: dict, index: int) -> str:
    label_keys = {
        "fakultas":    "singkatan_fakultas",
        "prodi":       "nama_prodi",
        "jenjang":     "kode_jenjang",
        "kurikulum":   "nama_kurikulum",
        "mata-kuliah": "kode_mk",
        "prasyarat":   "kode_mk",
        "kelas":       "nama_kelas",
        "jadwal":      "kode_mk",
        "fasilitas":   "nama_gedung",
        "pengumuman":  "judul",
    }

    if endpoint == "pmb":
        jalur = item.get("jalur_masuk", "")
        tahun = item.get("tahun_masuk", "")
        label = f"{jalur}_{tahun}"
    else:
        key   = label_keys.get(endpoint)
        label = str(item.get(key, index)) if key and item.get(key) else str(index)

    return f"{index:04d}_{slugify(label)}"


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess JSON → narasi .txt untuk RAG UNHAS"
    )
    parser.add_argument("--input",  "-i", default="data/json",       help="Folder JSON input")
    parser.add_argument("--output", "-o", default="data/narratives",  help="Folder output .txt")
    parser.add_argument("--encoding",     default="utf-8",            help="Encoding output")
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"[ERROR] Folder input tidak ditemukan: {input_dir.resolve()}")
        sys.exit(1)

    print("=" * 65)
    print("  RAG Preprocess — UNHAS Narrative Generator")
    print(f"  Input  : {input_dir.resolve()}")
    print(f"  Output : {output_dir.resolve()}")
    print("=" * 65)

    total_written = 0
    total_skipped = 0
    total_errors  = 0

    # ── Metadata: tangani khusus ──────────────────────────────────
    meta_path = input_dir / "metadata.json"
    if meta_path.exists():
        try:
            meta_raw  = json.loads(meta_path.read_text(encoding="utf-8"))
            meta_data = meta_raw.get("data", meta_raw)  # bisa langsung dict
            if isinstance(meta_data, list):
                meta_data = meta_data[0] if meta_data else {}
            ep_dir = output_dir / "metadata"
            ep_dir.mkdir(parents=True, exist_ok=True)
            txt = narrate_metadata(meta_data)
            (ep_dir / "metadata.txt").write_text(txt, encoding=args.encoding)
            total_written += 1
            print(f"\n── METADATA ── ✓ metadata.txt")
        except Exception as e:
            print(f"\n[ERROR] metadata.json: {e}")
            total_errors += 1

    # ── Endpoint lainnya ─────────────────────────────────────────
    for filename, endpoint in FILE_MAP.items():
        if endpoint is None:    # metadata sudah ditangani
            continue

        filepath = input_dir / filename
        if not filepath.exists():
            print(f"\n[SKIP] {filename} tidak ditemukan.")
            total_skipped += 1
            continue

        ep_dir = output_dir / endpoint
        ep_dir.mkdir(parents=True, exist_ok=True)

        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"\n[ERROR] Gagal baca {filename}: {e}")
            total_errors += 1
            continue

        items   = data.get("data", {}).get("items", [])
        summary = data.get("data", {}).get("summary", {})

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
                text  = narrator(item)
                label = get_item_label(endpoint, item, idx)
                (ep_dir / f"{label}.txt").write_text(text, encoding=args.encoding)
                written += 1
            except Exception as e:
                print(f"   [!] Item #{idx} error: {e} | data: {str(item)[:120]}")
                errors += 1

        # Ringkasan
        if summary and endpoint in SUMMARY_NARRATORS:
            try:
                txt = SUMMARY_NARRATORS[endpoint](summary)
                if txt:
                    (ep_dir / "_ringkasan.txt").write_text(txt, encoding=args.encoding)
                    written += 1
                    print(f"   + _ringkasan.txt")
            except Exception as e:
                print(f"   [!] Summary error: {e}")
                errors += 1

        total_written += written
        total_errors  += errors
        status = f"✓ {written} file"
        if errors:
            status += f"  ✗ {errors} error"
        print(f"   {status}  →  {ep_dir}")

    print("\n" + "=" * 65)
    print(f"  ✅ Selesai! Total file: {total_written} | "
          f"Lewati: {total_skipped} | Error: {total_errors}")
    print(f"  Output: {output_dir.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()