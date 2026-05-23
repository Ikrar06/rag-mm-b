"""CLI for syncing QA evaluation PostgreSQL tables to Google Sheets."""

from __future__ import annotations

import argparse
import logging
import time

from automation_qa.config import SheetTarget, SyncConfig, load_config
from automation_qa.db import fetch_new_qa_evaluation_rows


logger = logging.getLogger("automation_qa.sync_to_sheets")


def _sync_target(
    config: SyncConfig,
    client: GoogleSheetsClient,
    target: SheetTarget,
    dry_run: bool,
) -> int:
    # source_message_id pindah dari kolom D ke E setelah penambahan kolom
    # 'gambar_user' di posisi C. Layout baru:
    #   A=no, B=pertanyaan, C=gambar_user, D=created_at, E=source_message_id, ...
    sheet_state = client.get_sheet_state(
        spreadsheet_id=target.spreadsheet_id,
        worksheet_name=target.worksheet_name,
        source_message_id_column="E",
    )

    if target.name != "qa_evaluation":
        raise ValueError(f"Unsupported target: {target.name}")

    rows = fetch_new_qa_evaluation_rows(
        config.database_url,
        sheet_state.existing_source_ids,
        sheet_state.next_no,
        start_from=config.start_from,
    )

    logger.info(
        "%s: found %s existing sheet numbers, %s synced source IDs, and %s new database rows",
        target.name,
        len(sheet_state.existing_numbers),
        len(sheet_state.existing_source_ids),
        len(rows),
    )

    if dry_run:
        # Dry-run tetap membaca DB dan Sheet, tetapi tidak menulis row baru.
        logger.info("%s: dry-run enabled, skipping append", target.name)
        return 0

    appended = client.append_rows(
        spreadsheet_id=target.spreadsheet_id,
        worksheet_name=target.worksheet_name,
        rows=rows,
    )
    logger.info("%s: appended %s rows", target.name, appended)
    return appended


def run_once(config: SyncConfig, dry_run: bool) -> dict[str, int]:
    # Import lokal membuat module ini tetap ringan saat hanya parse config/tests.
    from automation_qa.google_sheets import GoogleSheetsClient

    client = GoogleSheetsClient(config.google_credentials_file)
    return {
        config.qa_sheet.name: _sync_target(config, client, config.qa_sheet, dry_run),
    }


def run_watch(config: SyncConfig, dry_run: bool) -> None:
    logger.info("Starting QA sheet sync loop every %s seconds", config.interval_seconds)
    while True:
        try:
            run_once(config, dry_run=dry_run)
        except Exception:
            # Scheduler tidak berhenti karena satu siklus gagal; error dicatat lalu coba lagi.
            logger.exception("QA sheet sync cycle failed")
        time.sleep(config.interval_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync QA evaluation tables to Google Sheets.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="Run one sync cycle and exit.")
    mode.add_argument("--watch", action="store_true", help="Run sync cycles forever.")
    parser.add_argument("--dry-run", action="store_true", help="Read DB/Sheets and report counts without appending.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = parse_args()
    config = load_config()

    if args.once:
        run_once(config, dry_run=args.dry_run)
        return

    run_watch(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
