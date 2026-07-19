from app import models  # noqa: F401  — imported to register Model on Base.metadata
from app.db import Base


def test_model_table_registered() -> None:
    assert "model" in Base.metadata.tables
    columns = {column.name for column in Base.metadata.tables["model"].columns}
    assert {"uid", "download_status", "content_hash", "raw_key"} <= columns
