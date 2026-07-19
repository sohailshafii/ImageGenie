#!/usr/bin/env python3
"""Pull rendered views out of the running pipeline and build a contact sheet.

Local analysis helper (gitignored — not part of the pipeline). Copies one model's
per-view PNGs out of the `render` compose service's storage volume and montages
them into a single grid image so the multi-view renders can be eyeballed.

Usage:
    .venv/bin/python capture_renders.py [uid]   # defaults to the first model found

Outputs to render_captures/<uid>/ (raw views + <uid>_contact_sheet.png), also
gitignored. Requires the compose stack to be up and Pillow (in the venv).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

COMPOSE = ["docker", "compose", "-f", "server/docker-compose.yml"]
RENDERS_ROOT = "/data/storage/processed/renders"
OUTPUT_ROOT = Path("render_captures")
COLUMNS = 4


def _render_container_id() -> str:
    result = subprocess.run(
        [*COMPOSE, "ps", "-q", "render"], capture_output=True, text=True, check=True
    )
    container_id = result.stdout.strip()
    if not container_id:
        sys.exit("render service is not running — start it with `make compose-up`")
    return container_id


def _list_model_uids(container_id: str) -> list[str]:
    result = subprocess.run(
        ["docker", "exec", container_id, "sh", "-c", f"ls {RENDERS_ROOT}"],
        capture_output=True,
        text=True,
    )
    return sorted(name for name in result.stdout.split() if name)


def _copy_views(container_id: str, uid: str, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["docker", "cp", f"{container_id}:{RENDERS_ROOT}/{uid}/.", str(destination)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return sorted(destination.glob("view_*.png"))


def _build_contact_sheet(view_paths: list[Path], output_path: Path) -> None:
    cell = Image.open(view_paths[0]).width
    padding, label_height = 8, 16
    rows = (len(view_paths) + COLUMNS - 1) // COLUMNS
    width = COLUMNS * cell + (COLUMNS + 1) * padding
    height = rows * (cell + label_height) + (rows + 1) * padding
    sheet = Image.new("RGB", (width, height), (245, 245, 247))
    draw = ImageDraw.Draw(sheet)
    for index, view_path in enumerate(view_paths):
        row, column = divmod(index, COLUMNS)
        x = padding + column * (cell + padding)
        y = padding + row * (cell + label_height + padding)
        sheet.paste(Image.open(view_path), (x, y))
        draw.text((x + 4, y + cell + 2), view_path.stem, fill=(90, 90, 95))
    sheet.save(output_path)


def main() -> None:
    container_id = _render_container_id()
    uid = sys.argv[1] if len(sys.argv) > 1 else None
    if uid is None:
        uids = _list_model_uids(container_id)
        if not uids:
            sys.exit(f"no rendered models under {RENDERS_ROOT} yet")
        uid = uids[0]

    destination = OUTPUT_ROOT / uid
    view_paths = _copy_views(container_id, uid, destination)
    if not view_paths:
        sys.exit(f"no views found for {uid}")

    contact_sheet = OUTPUT_ROOT / f"{uid}_contact_sheet.png"
    _build_contact_sheet(view_paths, contact_sheet)
    print(f"{uid}: {len(view_paths)} views → {contact_sheet}")


if __name__ == "__main__":
    main()
