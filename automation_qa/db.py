"""Read-only PostgreSQL access for QA evaluation sync."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Iterable

import psycopg2
from psycopg2.extras import RealDictCursor


WITA = timezone(timedelta(hours=8))


def _to_sheet_value(value: Any) -> Any:
    # Google Sheets API menerima list nilai scalar; None dibuat kosong agar cell manual QA tetap rapi.
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    return str(value)


def _to_wita_timestamp(value: Any) -> str:
    # Sheet memakai waktu lokal WITA sebagai teks agar tidak bergeser karena timezone Google Sheets.
    if value is None:
        return ""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(WITA).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


def _format_sources(sources: Any) -> str:
    # Satu cell berisi semua referensi dalam format multi-line agar mudah dibaca reviewer.
    if not sources:
        return ""
    if not isinstance(sources, list):
        return str(sources)

    formatted: list[str] = []
    for index, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            formatted.append(f"{index}. {source}")
            continue

        file_name = source.get("file_name") or "-"
        score = source.get("score")
        chunk_index = source.get("chunk_index")
        element_type = source.get("element_type")
        text_preview = (source.get("text_preview") or "").strip()

        lines = [f"{index}. {file_name}"]
        if score not in (None, ""):
            lines.append(f"Score: {score}")

        meta_parts: list[str] = []
        if chunk_index not in (None, ""):
            meta_parts.append(f"Chunk #{chunk_index}")
        if element_type:
            meta_parts.append(f"[{element_type}]")
        if meta_parts:
            lines.append(" - ".join(meta_parts))

        if text_preview:
            lines.append(text_preview)

        formatted.append("\n".join(lines))

    return "\n\n".join(formatted)


def _get_debug_value(debug: Any, key: str) -> Any:
    if isinstance(debug, dict):
        return debug.get(key)
    return None


def _format_intent_prediction(debug: Any) -> str:
    # Format final yang diminta QA: contoh "chitchat (0.76)".
    intent = _get_debug_value(debug, "intent")
    if not intent:
        return ""

    confidence = _get_debug_value(debug, "intent_confidence")
    if confidence in (None, ""):
        return _to_sheet_value(intent)

    try:
        confidence_text = f"{float(confidence):.2f}"
    except (TypeError, ValueError):
        confidence_text = _to_sheet_value(confidence)

    return f"{intent} ({confidence_text})"


def _fetch_answered_messages(
    database_url: str,
    existing_source_ids: Iterable[str],
) -> list[dict[str, Any]]:
    existing = {str(source_id) for source_id in existing_source_ids if source_id}
    existing_filter = list(existing) or [""]

    # Pairing dilakukan dari setiap jawaban assistant ke user message terakhir
    # dalam session yang sama. ID assistant dipakai sebagai source_message_id
    # sekaligus filter dedup.
    sql = """
        SELECT
            a.id AS assistant_message_id,
            u.content AS pertanyaan,
            a.content AS jawaban,
            a.sources AS sources,
            a.debug AS debug,
            a.created_at AS created_at
        FROM messages a
        JOIN LATERAL (
            SELECT id, content
            FROM messages
            WHERE session_id = a.session_id
              AND role = 'user'
              AND created_at < a.created_at
            ORDER BY created_at DESC
            LIMIT 1
        ) u ON TRUE
        WHERE a.role = 'assistant'
          AND NOT (a.id::text = ANY(%s))
        ORDER BY a.created_at ASC, a.id ASC
    """

    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(sql, (existing_filter,))
            rows = cursor.fetchall()

    return [dict(row) for row in rows]


def fetch_new_qa_evaluation_rows(
    database_url: str,
    existing_source_ids: Iterable[str],
    start_no: int,
) -> list[list[Any]]:
    rows = _fetch_answered_messages(database_url, existing_source_ids)
    output: list[list[Any]] = []
    next_no = start_no

    for row in rows:
        debug = row.get("debug") or {}

        # Urutan kolom harus sama dengan header Google Sheet:
        # no, pertanyaan, created_at, source_message_id, model, jawaban,
        # sumber_referensi, labeled_by, intent_predicted, labeled_quality,
        # notes_respond, labeled_intent, notes_intent.
        output.append(
            [
                next_no,
                _to_sheet_value(row["pertanyaan"]),
                _to_wita_timestamp(row["created_at"]),
                _to_sheet_value(row["assistant_message_id"]),
                _to_sheet_value(_get_debug_value(debug, "model")),
                _to_sheet_value(row["jawaban"]),
                _format_sources(row.get("sources")),
                "",
                _format_intent_prediction(debug),
                "",
                "",
                "",
                "",
            ]
        )
        next_no += 1

    return output
