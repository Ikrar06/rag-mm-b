"""Configuration for QA evaluation sync jobs."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


# Target Google Sheet sengaja dipisah dari SyncConfig supaya mudah diganti
# tanpa mengubah logic sync.
@dataclass(frozen=True)
class SheetTarget:
    name: str
    spreadsheet_id: str
    worksheet_name: str


@dataclass(frozen=True)
class SyncConfig:
    database_url: str
    google_credentials_file: str
    interval_seconds: int
    qa_sheet: SheetTarget
    # ISO datetime (mis. "2026-05-23T14:00:00+08:00"). Kalau di-set, hanya message
    # dengan created_at >= waktu ini yang di-sync. Berguna untuk reset evaluasi
    # tanpa harus hapus history percakapan dari DB.
    start_from: str | None


def _env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None or value.strip() == "":
        raise ValueError(f"Missing required environment variable: {name}")
    return value.strip()


def load_config() -> SyncConfig:
    # Interval hanya dipakai untuk mode --watch; mode --once mengabaikannya.
    interval = int(os.getenv("QA_SYNC_INTERVAL_SECONDS", "3600"))
    if interval <= 0:
        raise ValueError("QA_SYNC_INTERVAL_SECONDS must be greater than 0")

    start_from = os.getenv("QA_SYNC_START_FROM", "").strip() or None

    return SyncConfig(
        database_url=_env("DATABASE_URL"),
        google_credentials_file=_env("QA_SYNC_GOOGLE_CREDENTIALS_FILE"),
        interval_seconds=interval,
        qa_sheet=SheetTarget(
            name="qa_evaluation",
            spreadsheet_id=_env("QA_SYNC_SPREADSHEET_ID"),
            worksheet_name=_env("QA_SYNC_WORKSHEET_NAME", "Sheet1"),
        ),
        start_from=start_from,
    )
