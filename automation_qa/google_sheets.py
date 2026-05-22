"""Google Sheets client for append-only QA evaluation sync."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build


SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


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
        # Baca A:D sekali saja: kolom A untuk nomor, kolom D untuk dedup source_message_id.
        range_name = f"{_quote_sheet_name(worksheet_name)}!A2:{source_message_id_column}"
        response = (
            self._service.spreadsheets()
            .values()
            .get(spreadsheetId=spreadsheet_id, range=range_name)
            .execute()
        )
        values = response.get("values", [])
        source_index = ord(source_message_id_column.upper()) - ord("A")

        numbers: set[int] = set()
        source_ids: set[str] = set()
        for row in values:
            if row:
                number = _parse_int(row[0])
                if number is not None:
                    numbers.add(number)
            if len(row) > source_index:
                source_id = str(row[source_index]).strip()
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

        # Append dilakukan batch supaya hanya perlu satu request Google Sheets API per siklus.
        end_column = chr(ord("A") + len(rows[0]) - 1)
        range_name = f"{_quote_sheet_name(worksheet_name)}!A:{end_column}"
        body = {"values": [list(row) for row in rows]}
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
        return int(updates.get("updatedRows", len(rows)))
