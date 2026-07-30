#!/usr/bin/env python3
"""Pull a model's rendered views and montage them into one contact sheet.

Local analysis helper — not part of the pipeline. Its *output* is gitignored
(`render_captures/`), the script itself is tracked.

Two sources, because a model's renders live in one of two places depending on
where it was processed:

    # from the running compose stack's render volume (local pipeline smoke)
    .venv/bin/python capture_renders.py [uid]

    # from the processed bucket, for models ingested in the cloud
    PYTHONPATH=server IMAGEGENIE_STORAGE_BACKEND=gcs \
      IMAGEGENIE_PROCESSED_BUCKET=imagegenie-pipeline-processed \
      .venv/bin/python capture_renders.py --gcs <uid> [uid ...] [--cell 128]

`--gcs` is what makes the tool usable for review passes over ingested data (the
M8 review queue — ml.md#the-review-queue-milestone-8): the compose volume only
ever holds what was rendered locally, so eyeballing a real corpus model needs the
bucket. `--cell` downscales each view, which matters when reading dozens of sheets
in one sitting rather than inspecting one closely.

Outputs to render_captures/<uid>_contact_sheet.png (plus the raw views, local mode
only). Requires Pillow (in the venv).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
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


def _views_from_volume(uid: str) -> tuple[str, list[tuple[str, bytes]]]:
    """The uid and its (label, png bytes) views, copied out of the render container.

    Returns the uid because an empty one means "whichever model is there", and the
    caller needs the resolved name for the output filename.
    """
    container_id = _render_container_id()
    if not uid:
        uids = _list_model_uids(container_id)
        if not uids:
            sys.exit(f"no rendered models under {RENDERS_ROOT} yet")
        uid = uids[0]
    view_paths = _copy_views(container_id, uid, OUTPUT_ROOT / uid)
    if not view_paths:
        sys.exit(f"no views found for {uid}")
    return uid, [(path.stem, path.read_bytes()) for path in view_paths]


def _views_from_storage(uid: str) -> list[tuple[str, bytes]]:
    """(label, png bytes) per view, read through the `Storage` abstraction.

    Imported lazily so the local-volume mode needs neither `PYTHONPATH=server` nor
    the server's dependencies.
    """
    from app.artifact_keys import view_keys
    from app.config import get_settings
    from app.storage import build_storage

    storage = build_storage(get_settings())
    keys = view_keys(uid)
    # Threaded because each view is a separate round trip to the bucket, and a
    # serial read of twelve is dominated by latency rather than bytes.
    with ThreadPoolExecutor(max_workers=len(keys)) as pool:
        payloads = list(pool.map(storage.get_bytes, keys))
    return [(Path(key).stem, data) for key, data in zip(keys, payloads, strict=True)]


def build_contact_sheet(
    views: list[tuple[str, bytes]], output_path: Path, cell: int | None = None
) -> None:
    """Montage views into a labelled grid, optionally downscaling each cell."""
    first = Image.open(BytesIO(views[0][1]))
    size = cell or first.width
    padding, label_height = 8, 16
    rows = (len(views) + COLUMNS - 1) // COLUMNS
    width = COLUMNS * size + (COLUMNS + 1) * padding
    height = rows * (size + label_height) + (rows + 1) * padding
    sheet = Image.new("RGB", (width, height), (245, 245, 247))
    draw = ImageDraw.Draw(sheet)
    for index, (label, data) in enumerate(views):
        row, column = divmod(index, COLUMNS)
        left = padding + column * (size + padding)
        top = padding + row * (size + label_height + padding)
        view = Image.open(BytesIO(data)).convert("RGB")
        if view.width != size:
            view = view.resize((size, size))
        sheet.paste(view, (left, top))
        draw.text((left + 4, top + size + 2), label, fill=(90, 90, 95))
    sheet.save(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Contact-sheet a model's renders.")
    parser.add_argument(
        "uids", nargs="*", help="model uids (local mode: omit for the first found)"
    )
    parser.add_argument(
        "--gcs",
        action="store_true",
        help="read views from object storage instead of the compose render volume",
    )
    parser.add_argument(
        "--cell", type=int, default=None, help="downscale each view to N pixels square"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.gcs and not args.uids:
        raise SystemExit("--gcs needs at least one uid: nothing to list against")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    # Local mode with no uid keeps its "first model found" convenience.
    for requested in args.uids or [""]:
        if args.gcs:
            uid, views = requested, _views_from_storage(requested)
        else:
            uid, views = _views_from_volume(requested)
        output_path = OUTPUT_ROOT / f"{uid}_contact_sheet.png"
        build_contact_sheet(views, output_path, args.cell)
        print(f"{uid}: {len(views)} views → {output_path}")


if __name__ == "__main__":
    main()
