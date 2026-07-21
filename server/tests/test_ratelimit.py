"""Unit tests for the rate-limit primitives (server.md#rate-limiting).

`now` is injected throughout rather than sleeping, so these are deterministic and
fast — a test suite that sleeps through a 15-minute lockout ladder is untestable.
"""

import pytest

from app.ratelimit import BackoffRule, FixedWindowRateLimiter, LoginBackoff, RateLimitRule

RULE = RateLimitRule(max_hits=3, window_seconds=10.0)


def test_allows_up_to_the_cap_then_blocks() -> None:
    limiter = FixedWindowRateLimiter()
    assert [limiter.check("key", RULE, now=0.0) for _ in range(3)] == [True, True, True]
    assert limiter.check("key", RULE, now=0.0) is False


def test_keys_are_independent() -> None:
    limiter = FixedWindowRateLimiter()
    for _ in range(3):
        limiter.check("first", RULE, now=0.0)
    assert limiter.check("first", RULE, now=0.0) is False
    assert limiter.check("second", RULE, now=0.0) is True  # unaffected


def test_window_refreshes_after_it_elapses() -> None:
    limiter = FixedWindowRateLimiter()
    for _ in range(3):
        limiter.check("key", RULE, now=0.0)
    assert limiter.check("key", RULE, now=9.9) is False  # still inside
    assert limiter.check("key", RULE, now=10.0) is True  # window rolled over


def test_retry_after_counts_down_and_floors_at_zero() -> None:
    limiter = FixedWindowRateLimiter()
    limiter.check("key", RULE, now=0.0)
    assert limiter.retry_after("key", now=4.0) == pytest.approx(6.0)
    assert limiter.retry_after("key", now=99.0) == 0.0
    assert limiter.retry_after("never-seen") == 0.0


BACKOFF = BackoffRule(free_retries=2, base_seconds=1.0, max_seconds=8.0)


def test_backoff_ladder_doubles_and_caps() -> None:
    backoff = LoginBackoff(BACKOFF)
    # Two free failures — typos shouldn't lock anyone out.
    backoff.record_failure("account", now=0.0)
    assert backoff.retry_after("account", now=0.0) == 0.0
    backoff.record_failure("account", now=0.0)
    assert backoff.retry_after("account", now=0.0) == 0.0
    # Then 1s, 2s, 4s, 8s, and capped at 8s.
    for expected in (1.0, 2.0, 4.0, 8.0, 8.0):
        backoff.record_failure("account", now=0.0)
        assert backoff.retry_after("account", now=0.0) == pytest.approx(expected)


def test_backoff_expires_with_time() -> None:
    backoff = LoginBackoff(BACKOFF)
    for _ in range(3):
        backoff.record_failure("account", now=0.0)
    assert backoff.retry_after("account", now=0.5) == pytest.approx(0.5)
    assert backoff.retry_after("account", now=1.0) == 0.0  # lockout served


def test_success_clears_the_streak() -> None:
    """An honest user who mistypes then logs in starts clean — the whole point
    of keying on failures rather than a volumetric cap."""
    backoff = LoginBackoff(BACKOFF)
    for _ in range(4):
        backoff.record_failure("account", now=0.0)
    assert backoff.retry_after("account", now=0.0) > 0
    backoff.record_success("account")
    assert backoff.retry_after("account", now=0.0) == 0.0
    # And the ladder restarts from the grace window, not where it left off.
    backoff.record_failure("account", now=0.0)
    assert backoff.retry_after("account", now=0.0) == 0.0


def test_accounts_are_locked_independently() -> None:
    backoff = LoginBackoff(BACKOFF)
    for _ in range(4):
        backoff.record_failure("victim", now=0.0)
    assert backoff.retry_after("victim", now=0.0) > 0
    assert backoff.retry_after("bystander", now=0.0) == 0.0
