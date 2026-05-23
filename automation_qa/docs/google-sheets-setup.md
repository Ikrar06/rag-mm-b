# Google Sheets Setup

Dokumen ini menjelaskan setup Google Sheets dan Google Cloud agar `automation_qa` bisa append data evaluasi QA.

## 1. Buat Google Sheets

1. Buat satu spreadsheet baru.
2. Pastikan nama worksheet/tab sesuai `.env`.
   - Default: `Sheet1`
3. Isi header di baris 1.

```text
A: no
B: pertanyaan
C: created_at
D: source_message_id
E: model
F: jawaban
G: sumber_referensi
H: labeled_by
I: intent_predicted
J: labeled_quality
K: notes_respond
L: labeled_intent
M: notes_intent
```

## 2. Buat Google Cloud Project

1. Buka Google Cloud Console: <https://console.cloud.google.com/>
2. Klik project selector di kiri atas.
3. Pilih `New Project`.
4. Isi nama project, misalnya `QA-Evaluation-Sync`.
5. Klik `Create`.
6. Pastikan project baru tersebut sedang aktif.

## 3. Enable API

1. Cari `Google Sheets API`.
2. Buka hasilnya dan klik `Enable`.

## 4. Buat Service Account

1. Buka `IAM & Admin > Service Accounts`.
2. Klik `Create Service Account`.
3. Isi:
   - Service account name: `qa-sheet-sync-bot`
4. Klik `Create and Continue`.
5. Bagian role bisa dilewati.
6. Klik `Done`.

## 5. Download Credential JSON

1. Klik service account yang baru dibuat.
2. Buka tab `Keys`.
3. Klik `Add Key > Create new key`.
4. Pilih `JSON`.
5. Klik `Create`.
6. Pindahkan file JSON ke:

```text
automation_qa/secrets/google_service_account.json
```

Folder `automation_qa/secrets/` sudah di-ignore Git. Jangan commit file JSON credential.

## 6. Share Spreadsheet ke Service Account

1. Buka file JSON credential.
2. Ambil nilai `client_email`.
3. Buka Google Sheet evaluasi.
4. Klik `Share`.
5. Paste email service account.
6. Beri role `Editor`.
7. Matikan notifikasi jika ada opsi `Notify people`.
8. Klik `Share`.

## 7. Ambil Spreadsheet ID

URL Google Sheets berbentuk seperti ini:

```text
https://docs.google.com/spreadsheets/d/SPREADSHEET_ID/edit#gid=0
```

Ambil string di antara `/d/` dan `/edit`.

Isi `.env`:

```env
QA_SYNC_GOOGLE_CREDENTIALS_FILE=automation_qa/secrets/google_service_account.json

QA_SYNC_SPREADSHEET_ID=CHANGE_ME
QA_SYNC_WORKSHEET_NAME=Sheet1

QA_SYNC_INTERVAL_SECONDS=3600
```

Untuk Docker POC, gunakan path credential dalam container:

```env
QA_SYNC_GOOGLE_CREDENTIALS_FILE=/run/secrets/google_service_account.json
```

Pada `docker-compose.dev.yml` dan `docker-compose.poc.yml`, path ini juga diarahkan ke mount:

```text
./automation_qa/secrets/google_service_account.json -> /run/secrets/google_service_account.json
```

## 8. Test

Jika test dijalankan langsung dari host/local environment, install dependency khusus sync:

```bash
python -m pip install -r automation_qa/requirements.txt
```

Dry run (nge-check aja, tanpa perbarui gsheet):

```bash
python -m automation_qa.sync_to_sheets --once --dry-run
```

Jika tidak ada error 403 atau 404, koneksi Google Sheets sudah benar.

Run append manual:

```bash
python -m automation_qa.sync_to_sheets --once
```

Untuk menjalankan scheduler via Docker:

```bash
docker compose -f docker-compose.dev.yml up -d --build qa-sheet-sync
docker compose -f docker-compose.dev.yml logs -f qa-sheet-sync
```

Service `qa-sheet-sync` memakai `automation_qa/Dockerfile`, bukan Dockerfile backend utama, sehingga build-nya tidak menginstall dependency ML seperti Paddle, Torch, atau CUDA.

## Troubleshooting

- `403 Forbidden`: spreadsheet belum di-share ke `client_email`, atau role bukan Editor.
- `404 Not Found`: spreadsheet ID salah, worksheet/tab name salah, atau service account tidak punya akses.
- Duplicate row: kolom `source_message_id` di sheet pernah dihapus/diubah.
