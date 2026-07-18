"""Filesystem write helpers with uniform error handling.

Centralizes the try/except around file writes so a permission or OS error prints
a clear message instead of dumping a traceback, and so the handling can't drift
between call sites. Import and use `write_csv` / `write_json` rather than opening
files inline.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def write_csv(path: Path, header: tuple[str, str], rows: list[tuple[str, int]]) -> None:
    """Write ranked (name, count) rows to a CSV, reporting write errors."""
    try:
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)
            writer.writerows(rows)
    except PermissionError:
        print(f"Error writing CSV to {path}: permission denied.")
    except OSError as error:
        print(f"Error writing CSV to {path}: {error}.")


def write_json(path: Path, data: object) -> None:
    """Write `data` as indented JSON, reporting write errors."""
    try:
        with path.open("w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=2, default=str)
    except PermissionError:
        print(f"Error writing JSON to {path}: permission denied.")
    except OSError as error:
        print(f"Error writing JSON to {path}: {error}.")
