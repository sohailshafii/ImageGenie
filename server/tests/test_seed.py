"""The download seeder — that a bulk publish is paced, not a bare loop.

Publishing thousands of jobs as fast as the loop can go is what dead-lettered 499
of 1,000 uids on the 2026-07-30 dev-set seed: the overflow is rejected at push
level, *before* a worker runs, so it carries no error text and leaves no
`dead_letter` row (server.md#queue--workers). The pacing lived in `replay_dlq`
long before it lived in the thing that caused the pile-up, so the regression worth
pinning is this module going back to publishing in a loop.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app import seed


def _write_labels(path: Path, uids: list[str]) -> Path:
    rows = "\n".join(f"{uid},chair,category" for uid in uids)
    path.write_text(f"uid,class,reason\n{rows}\n", encoding="utf-8")
    return path


def test_labeled_uids_are_capped_to_the_count(tmp_path: Path) -> None:
    """`--count` is the cost guardrail: pilot a few hundred before a full run."""
    path = _write_labels(tmp_path / "weak_labels.csv", ["a", "b", "c"])

    assert seed._labeled_uids(path, 2) == ["a", "b"]
    assert seed._labeled_uids(path, None) == ["a", "b", "c"]


def test_the_seed_publishes_paced_and_reports_what_landed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The count printed is what `publish_paced` says landed, not `len(uids)` —
    a seeder whose only symptom of trouble is the word "seeded" is how the last
    two pile-ups stayed invisible until someone counted the bucket."""
    path = _write_labels(tmp_path / "weak_labels.csv", ["a", "b", "c"])
    calls: list[tuple[str, list[dict], int, float]] = []

    monkeypatch.setattr(seed.pubsub_v1, "PublisherClient", lambda: object())
    monkeypatch.setattr(seed.pubsub_v1, "SubscriberClient", lambda: object())
    monkeypatch.setattr(seed, "ensure_subscription", lambda *args: None)
    monkeypatch.setattr(
        seed,
        "publish_paced",
        lambda topic_id, payloads, batch_size, pause_seconds: (
            calls.append((topic_id, list(payloads), batch_size, pause_seconds))
            or len(payloads)
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["seed", "--from-labels", str(path), "--batch-size", "2", "--batch-pause", "0"],
    )

    seed.main()

    topic_id, payloads, batch_size, pause_seconds = calls[0]
    assert topic_id == seed.get_settings().download_topic
    # Byte-identical to the pre-variant payload: the default arm carries no
    # `variant` field, so messages in flight across a deploy stay valid.
    assert payloads == [{"uid": "a"}, {"uid": "b"}, {"uid": "c"}]
    assert (batch_size, pause_seconds) == (2, 0.0)
    assert "seeded 3 download jobs" in capsys.readouterr().out
