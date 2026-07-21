"""Rate limiting for the REST API (server.md#rate-limiting).

Two primitives, deliberately kept separate because they answer different threats:

- :class:`FixedWindowRateLimiter` — a volumetric cap ("no more than N hits per
  window"). Bounds bulk abuse and runaway clients.
- :class:`LoginBackoff` — failure-driven exponential backoff, keyed per account.
  This, not a volumetric cap, is what makes credential grinding expensive: an
  honest user who mistypes resets their streak by logging in, while an attacker
  grinding one account is locked out for geometrically longer.

**Both are per-process, in-memory.** That is correct while the API runs as a
single instance (it is not yet in Terraform — see server.md#api-layer). The day it
is deployed with more than one instance, these caps become per-instance and the
effective limit multiplies by the instance count; that is the point to move the
counters into a shared store, or to pin the service to one instance.

Clocks are ``time.monotonic`` rather than wall-clock so an NTP step can't widen a
window or cancel a lockout; tests inject ``now`` instead of sleeping.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RateLimitRule:
    """At most ``max_hits`` per ``window_seconds`` for a given key."""

    max_hits: int
    window_seconds: float


@dataclass
class _Window:
    hits: int
    resets_at: float


class FixedWindowRateLimiter:
    """Fixed-window counter, keyed by an arbitrary string.

    Fixed rather than sliding: it is a handful of lines and cheap, and its known
    weakness (a burst straddling a window boundary can briefly reach 2x the cap)
    doesn't matter for the volumetric caps here — the credential-grinding threat,
    where precision would matter, is handled by :class:`LoginBackoff` instead.
    """

    def __init__(self) -> None:
        self._key_to_window: dict[str, _Window] = {}

    def check(self, key: str, rule: RateLimitRule, now: float | None = None) -> bool:
        """Consume one hit for `key`. Returns True if the request is allowed."""
        moment = time.monotonic() if now is None else now
        window = self._key_to_window.get(key)
        if window is None or moment >= window.resets_at:
            self._key_to_window[key] = _Window(hits=1, resets_at=moment + rule.window_seconds)
            return True
        window.hits += 1
        return window.hits <= rule.max_hits

    def retry_after(self, key: str, now: float | None = None) -> float:
        """Seconds until `key`'s window resets — for the `Retry-After` header."""
        window = self._key_to_window.get(key)
        if window is None:
            return 0.0
        moment = time.monotonic() if now is None else now
        return max(0.0, window.resets_at - moment)

    def reset(self) -> None:
        """Drop all state. Used by tests so windows don't leak between them."""
        self._key_to_window.clear()


@dataclass(frozen=True)
class BackoffRule:
    """Lockout ladder: ``free_retries`` typos are free, then doubling delays."""

    free_retries: int
    base_seconds: float
    max_seconds: float


@dataclass
class _Streak:
    failures: int = 0
    locked_until: float = field(default=0.0)


class LoginBackoff:
    """Exponential backoff on repeated *failed* logins for one key.

    Only failures count, so an honest user is never penalised for logging in
    often — unlike a volumetric cap, which counts successes too and doesn't
    escalate. A success clears the streak outright.
    """

    def __init__(self, rule: BackoffRule) -> None:
        self._rule = rule
        self._key_to_streak: dict[str, _Streak] = {}

    def retry_after(self, key: str, now: float | None = None) -> float:
        """Seconds the caller must wait; 0.0 when the key isn't locked out."""
        streak = self._key_to_streak.get(key)
        if streak is None:
            return 0.0
        moment = time.monotonic() if now is None else now
        return max(0.0, streak.locked_until - moment)

    def record_failure(self, key: str, now: float | None = None) -> None:
        moment = time.monotonic() if now is None else now
        streak = self._key_to_streak.setdefault(key, _Streak())
        streak.failures += 1
        penalized = streak.failures - self._rule.free_retries
        if penalized <= 0:
            return  # still inside the grace window — no lockout yet
        delay = min(
            self._rule.base_seconds * (2 ** (penalized - 1)), self._rule.max_seconds
        )
        streak.locked_until = moment + delay

    def record_success(self, key: str) -> None:
        self._key_to_streak.pop(key, None)

    def reset(self) -> None:
        self._key_to_streak.clear()
