from __future__ import annotations

import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import monotonic

_enabled = False
_started = monotonic()
_lock = threading.Lock()


def configure(*, enabled: bool = True) -> None:
    """Enable or disable CLI progress messages.

    Progress is disabled for library callers and enabled by the CLI. Messages
    are written to stderr so stdout remains valid machine-readable JSON.
    """
    global _enabled, _started
    _enabled = enabled
    _started = monotonic()


def is_enabled() -> bool:
    return _enabled


def report(message: str) -> None:
    if not _enabled:
        return
    elapsed = monotonic() - _started
    minutes, seconds = divmod(int(elapsed), 60)
    with _lock:
        print(f"[for2107 {minutes:02d}:{seconds:02d}] {message}", file=sys.stderr, flush=True)


@contextmanager
def stage(label: str) -> Iterator[None]:
    """Report the start, completion, and duration of a pipeline stage."""
    started = monotonic()
    report(f"START {label}")
    try:
        yield
    except Exception:
        report(f"FAILED {label} ({monotonic() - started:.1f}s)")
        raise
    report(f"DONE  {label} ({monotonic() - started:.1f}s)")


@dataclass
class ProgressCounter:
    """Emit bounded percentage updates for a repeated operation."""

    label: str
    total: int
    updates: int = 10
    completed: int = 0
    _next: int = field(init=False)

    def __post_init__(self) -> None:
        self.total = max(0, int(self.total))
        self.updates = max(1, int(self.updates))
        self._next = max(1, (self.total + self.updates - 1) // self.updates)
        if self.total:
            report(f"{self.label}: 0/{self.total} (0%)")

    def advance(self, amount: int = 1) -> None:
        if not self.total:
            return
        self.completed = min(self.total, self.completed + amount)
        if self.completed == 1 and self._next > 1:
            report(f"{self.label}: 1/{self.total} ({round(100 / self.total)}%)")
            return
        if self.completed < self._next and self.completed < self.total:
            return
        percent = round(100 * self.completed / self.total)
        report(f"{self.label}: {self.completed}/{self.total} ({percent}%)")
        increment = max(1, (self.total + self.updates - 1) // self.updates)
        while self._next <= self.completed:
            self._next += increment
