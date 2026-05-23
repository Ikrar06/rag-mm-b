"""Google Sheets client for append-only QA evaluation sync."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build


SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
APPEND_CHUNK_SIZE = 500


@dataclass(frozen=True)
class SheetState:
    existing_numbers: set[int]
    existing_source_ids: set[str]

    @property
    def next_no(self) -> int:
        # Nomor baru mengikuti nomor terbesar di sheet, bukan jumlah row,
        # supaya tetap aman jika ada row kosong atau nomor pernah dilewati.
        if not self.existing_numbers:
            return 1
        return max(self.existing_numbers) + 1


def _quote_sheet_name(name: str) -> str:
    # Nama worksheet bisa mengandung spasi atau petik satu.
    return "'" + name.replace("'", "''") + "'"


def _parse_int(value: Any) -> int | None:
    try:
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except (TypeError, ValueError):
        return None


class GoogleSheetsClient:
    def __init__(self, credentials_file: str):
        credentials = Credentials.from_service_account_file(
            credentials_file,
            scopes=SCOPES,
        )
        self._service = build("sheets", "v4", credentials=credentials, cache_discovery=False)

    def get_sheet_state(
        self,
        spreadsheet_id: str,
        worksheet_name: str,
        source_message_id_column: str,
    ) -> SheetState:
        # Baca hanya kolom yang diperlukan agar pertanyaan/jawaban panjang tidak ikut terbawa.
        sheet_name = _quote_sheet_name(worksheet_name)
        number_range = f"{sheet_name}!A2:A"
        source_id_range = f"{sheet_name}!{source_message_id_column}2:{source_message_id_column}"
        response = (
            self._service.spreadsheets()
            .values()
            .batchGet(
                spreadsheetId=spreadsheet_id,
                ranges=[number_range, source_id_range],
            )
            .execute()
        )
        value_ranges = response.get("valueRanges", [])
        number_values = value_ranges[0].get("values", []) if len(value_ranges) > 0 else []
        source_id_values = value_ranges[1].get("values", []) if len(value_ranges) > 1 else []

        numbers: set[int] = set()
        for row in number_values:
            if row:
                number = _parse_int(row[0])
                if number is not None:
                    numbers.add(number)

        source_ids: set[str] = set()
        for row in source_id_values:
            if row:
                source_id = str(row[0]).strip()
                if source_id:
                    source_ids.add(source_id)

        return SheetState(existing_numbers=numbers, existing_source_ids=source_ids)

    def append_rows(
        self,
        spreadsheet_id: str,
        worksheet_name: str,
        rows: Sequence[Sequence[Any]],
    ) -> int:
        if not rows:
            return 0

        # Append dipecah agar payload tetap stabil saat ada backlog besar.
        end_column = chr(ord("A") + len(rows[0]) - 1)
        range_name = f"{_quote_sheet_name(worksheet_name)}!A:{end_column}"

        updated_rows = 0
        for start in range(0, len(rows), APPEND_CHUNK_SIZE):
            chunk = rows[start : start + APPEND_CHUNK_SIZE]
            body = {"values": [list(row) for row in chunk]}
            response = (
                self._service.spreadsheets()
                .values()
                .append(
                    spreadsheetId=spreadsheet_id,
                    range=range_name,
                    valueInputOption="RAW",
                    insertDataOption="INSERT_ROWS",
                    body=body,
                )
                .execute()
            )
            updates = response.get("updates", {})
            updated_rows += int(updates.get("updatedRows", len(chunk)))

        return updated_rows
