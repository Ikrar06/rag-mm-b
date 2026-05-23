# QA Evaluation Google Sheets Sync

Modul ini menyalin data evaluasi QA dari PostgreSQL ke satu Google Sheet.

Script bersifat append-only. Baris lama di Google Sheets tidak diubah, sehingga kolom manual QA seperti `labeled_intent`, `labeled_quality`, `notes_respond`, `notes_intent`, dan `labeled_by` tetap aman.

## Flow Singkat

1. Baca nomor dari kolom `A` dan `source_message_id` dari kolom `D` di Google Sheets.
2. Ambil pasangan pertanyaan user dan jawaban assistant dari tabel `messages`.
3. Skip data yang sudah pernah tersinkron berdasarkan `source_message_id`.
4. Append row baru ke Google Sheets, maksimal 500 row per request.

Detail flow ada di [docs/sync-flow.md](docs/sync-flow.md).

## Header Sheet

```text
no | pertanyaan | gambar_user | created_at | source_message_id | model | jawaban | sumber_referensi | labeled_by | intent_predicted | labeled_quality | notes_respond | labeled_intent | notes_intent
```

`intent_predicted` ditulis dengan confidence, misalnya `chitchat (0.76)`.
`gambar_user` berisi URL attachment yang user kirim (presigned MinIO link,
valid 7 hari). Kosong kalau user tidak kirim gambar — reviewer click URL
untuk lihat di browser.

Kolom manual QA sengaja dikirim kosong saat append.

**Migration dari header lama (tanpa `gambar_user`):**
1. Buka sheet QA, klik kanan kolom C → Insert 1 column right (atau before, sesuai posisi).
2. Set header baru di row 1: `gambar_user` di kolom C.
3. Backend pickup column shift otomatis di sync berikutnya — `source_message_id`
   sekarang ada di kolom E (sebelumnya D).

## Setup Cepat

1. Buat satu Google Sheet dan isi header di baris 1.
2. Buat Google Cloud service account.
3. Enable Google Sheets API.
4. Download JSON credential ke `automation_qa/secrets/google_service_account.json`.
5. Share Google Sheet ke email service account sebagai Editor.
6. Isi `.env`:

```env
QA_SYNC_GOOGLE_CREDENTIALS_FILE=automation_qa/secrets/google_service_account.json
QA_SYNC_SPREADSHEET_ID=CHANGE_ME
QA_SYNC_WORKSHEET_NAME=Sheet1
QA_SYNC_INTERVAL_SECONDS=3600
```

Panduan setup detail ada di [docs/google-sheets-setup.md](docs/google-sheets-setup.md).

Jika menjalankan sync langsung dari host/local environment, install dependency khusus sync:

```bash
python -m pip install -r automation_qa/requirements.txt
```

## Manual Run

Dry run membaca DB dan Google Sheets, tetapi tidak append:

```bash
python -m automation_qa.sync_to_sheets --once --dry-run
```

Run sync sekarang tanpa menunggu scheduler:

```bash
python -m automation_qa.sync_to_sheets --once
```

## Scheduler

Service Docker `qa-sheet-sync` memakai image ringan dari `automation_qa/Dockerfile`. Image ini hanya menginstall dependency sync Google Sheets/PostgreSQL, sehingga rebuild service ini tidak ikut menginstall dependency backend/ML seperti CUDA, Paddle, Torch, atau OCR.

Dev:

```bash
# Start semua service dev, termasuk qa-sheet-sync
docker compose -f docker-compose.dev.yml up -d --build

# Cek status dan lihat log scheduler
docker compose -f docker-compose.dev.yml ps qa-sheet-sync
docker compose -f docker-compose.dev.yml logs -f qa-sheet-sync
```

Kalau service dev lain sudah jalan dan hanya ingin fokus ke QA sync:

```bash
docker compose -f docker-compose.dev.yml up -d --build qa-sheet-sync
docker compose -f docker-compose.dev.yml logs -f qa-sheet-sync
```

POC:

```bash
# Start semua service POC, termasuk qa-sheet-sync
docker compose -f docker-compose.poc.yml up -d
docker compose -f docker-compose.poc.yml ps qa-sheet-sync
docker compose -f docker-compose.poc.yml logs -f qa-sheet-sync
```

Kalau service POC lain sudah jalan dan hanya ingin fokus ke QA sync:

```bash
docker compose -f docker-compose.poc.yml up -d --build qa-sheet-sync
docker compose -f docker-compose.poc.yml logs -f qa-sheet-sync
```

Scheduler Docker berjalan dalam mode `--watch`; intervalnya mengikuti `QA_SYNC_INTERVAL_SECONDS`.

## Maintenance

- Jika muncul duplicate, cek kolom `source_message_id` di sheet.
- Jika error 403, cek apakah sheet sudah di-share ke email service account.
- Jika error 404, cek spreadsheet ID dan worksheet/tab name.
- Jika format `messages.sources` berubah, update formatter di `automation_qa/db.py`.

## Catatan Performa

- State sheet dibaca dengan `values.batchGet` untuk range `A2:A` dan `D2:D` saja, sehingga kolom `pertanyaan` dan `created_at` tidak ikut terbaca.
- Append dilakukan per chunk 500 row supaya payload Google Sheets API tetap stabil saat ada backlog besar.
- Batas 500 row dipilih karena row QA bisa berisi teks panjang dan multi-line pada kolom `jawaban` serta `sumber_referensi`; chunk ini menjaga request tidak terlalu besar tetapi tetap hemat jumlah request.
- Jika row baru kurang dari atau sama dengan 500, sync tetap hanya memakai 1 read request dan 1 write request.
- Detail quota dan alasan teknis chunking dijelaskan di [docs/sync-flow.md](docs/sync-flow.md).

## Docker Image

Runtime Docker untuk sync dipisah dari backend utama:

- `Dockerfile` root tetap dipakai backend dan berisi dependency aplikasi utama/ML.
- `automation_qa/Dockerfile` dipakai `qa-sheet-sync` dan hanya menyalin package `automation_qa`.
- `automation_qa/requirements.txt` berisi dependency minimal sync: PostgreSQL client, Google Sheets client, Google auth, dan dotenv.

Dengan pemisahan ini, rebuild `qa-sheet-sync` tidak perlu mengunduh package besar seperti `paddlepaddle-gpu` atau `torch`.
