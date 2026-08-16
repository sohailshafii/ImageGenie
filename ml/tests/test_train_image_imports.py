"""What the training image's entrypoints may import at module scope.

The training image installs `ml/requirements-train.txt`, a deliberate *subset*
of the server's dependencies, and gets torch/torchvision from its CUDA base.
Anything else an entrypoint imports at module scope is a `ModuleNotFoundError`
that only appears on Vertex — after a spot GPU is provisioned and a multi-GB
image is pulled, in a job nobody is watching.

That is not hypothetical. `evaluate.py` imports `build_dev_set` for
`load_dev_set`, which imported `eval_weak_labels` -> `objaverse` at module
scope, and objaverse is one of the packages requirements-train.txt names as
excluded. The first evaluation ever submitted from the UI died on the import,
*before* `start_evaluation` could claim its row — so the failure the row-claiming
was built to make visible left nothing behind at all: no row, no report, no
error text. Only the job's stderr knew.

The check is static on purpose. Importing the modules here would prove only that
the *dev* environment can import them, which was true all along and is exactly
why the test suite missed this.

Deferred imports (inside a function) are ignored, because they are the fix: a
module needed by one CLI path and not another belongs behind the call, not at the
top of the file.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent.parent
APP_DIR = ML_DIR.parent / "server" / "app"

# The entrypoints the training image can be asked to run: its ENTRYPOINT, and
# the command `app/training_jobs.py` overrides it with to score a run.
ENTRYPOINTS = ("train", "evaluate")

# What the image actually provides, by *import* name — not what the entrypoints
# happen to use today, so adding a legitimate one is not a test failure.
#
#   requirements-train.txt : sqlalchemy, psycopg, pg8000, google.*,
#                            pydantic_settings, numpy, PIL
#   the CUDA base image    : torch, torchvision (see the file's own note on why
#                            they are not pinned in requirements-train.txt)
IMAGE_PROVIDES = frozenset(
    {
        "sqlalchemy",
        "psycopg",
        "pg8000",
        "google",
        "pydantic_settings",
        "numpy",
        "PIL",
        "torch",
        "torchvision",
    }
)


def _module_level_imports(path: Path) -> set[str]:
    """Every module imported when `path` is loaded, ignoring deferred ones.

    "Deferred" means inside a function or class body, which is not executed at
    import time. An import under `if TYPE_CHECKING:` is module-level by this
    definition and would be flagged — acceptable, since the repo does not use
    that idiom and a false alarm here costs a comment, not a paid job.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    deferred = {
        node
        for parent in ast.walk(tree)
        if isinstance(parent, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        for node in ast.walk(parent)
        if isinstance(node, ast.Import | ast.ImportFrom)
    }
    imported: set[str] = set()
    for node in ast.walk(tree):
        if node in deferred:
            continue
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module)
    return imported


def _source_of(module: str) -> Path | None:
    """The file backing a first-party module, or None if it is not one of ours.

    Both trees are flat, and both are in the image: `ml/Dockerfile` copies
    `ml/*.py` next to `server/app`.
    """
    ml_path = ML_DIR / f"{module}.py"
    if ml_path.exists():
        return ml_path
    if module.startswith("app."):
        app_path = APP_DIR / f"{module.split('.')[1]}.py"
        if app_path.exists():
            return app_path
    return None


def _third_party_reachable_from(entrypoint: str) -> dict[str, set[str]]:
    """Non-stdlib roots reachable at module scope, mapped to who imports them."""
    reached: dict[str, set[str]] = {}
    visited: set[str] = set()
    pending = [entrypoint]
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        source = _source_of(module)
        if source is None:
            continue
        for imported in _module_level_imports(source):
            if _source_of(imported) is not None:
                pending.append(imported)
                continue
            root = imported.split(".")[0]
            if root != "__future__" and root not in sys.stdlib_module_names:
                reached.setdefault(root, set()).add(module)
    return reached


def test_entrypoints_import_nothing_the_training_image_lacks() -> None:
    for entrypoint in ENTRYPOINTS:
        reached = _third_party_reachable_from(entrypoint)
        missing = {
            root: sorted(importers)
            for root, importers in reached.items()
            if root not in IMAGE_PROVIDES
        }
        assert not missing, (
            f"{entrypoint}.py reaches {missing} at module scope, which the training "
            "image does not install. Import it inside the function that needs it, or "
            "add it to ml/requirements-train.txt and rebuild the image."
        )


def test_the_walk_actually_reaches_the_far_side_of_the_graph() -> None:
    """Guards the guard: a walk that silently found nothing would pass anything.

    `torchvision` is imported only by `model.py`, which both entrypoints reach
    through two hops, so seeing it proves the traversal is transitive rather
    than a check on the entrypoint's own import block.
    """
    for entrypoint in ENTRYPOINTS:
        reached = _third_party_reachable_from(entrypoint)
        assert "torchvision" in reached
        assert reached["torchvision"] == {"model"}
