"""Alert debouncing for the HORUS detector.

A raw flag is "this one window looks anomalous". An alarm is "we are confident
enough to trigger the SDN failover". Debouncing turns flags into alarms with one
rule: raise the alarm only after N consecutive anomalous windows.

Two forms, guaranteed to behave identically:
    - AlertDebouncer : streaming, one window at a time (for the live API)
    - debounce       : whole array at once (for offline evaluation)
"""

from __future__ import annotations
import numpy as np


def _check_n(n: int) -> int:
    if not isinstance(n, np.integer) or isinstance(n, bool) or n < 1:
        raise ValueError(f"n must be an integer >= 1, got {n!r}")
    return int(n)


class AlertDebouncer:
    """Streaming debouncer. Feed it one window verdict at a time.

    The alarm turns on once the current run of anomalous windows reaches n,
    stays on while the run continues, and turns off at the first normal window.
    """

    def __init__(self, n:int):
        self.n = _check_n(n)
        self._run_length = 0

    @property
    def run_length(self) -> int:
        """How many anomalous windows in a row we have seen so far."""
        return self._run_length

    def update(self, flag: bool | int) -> bool:
        """Consume one window's raw flag; return True if the alarm is active."""
        if flag not in (0, 1, True, False):
            raise ValueError(f"flag must be 0/1, got {flag!r}")
        self._run_length = self._run_length + 1 if flag else 0 
        return self._run_length >= self.n

    def reset(self) -> None:
        """Forget history (e.g. after a failover, or when switching towers)."""
        self._run_length = 0


def debounce(flags, n:int) -> np.ndarray:
    """Batch version: raw 0/1 flags in, 0/1 alarms out (same length).

    Implemented by replaying the flags through AlertDebouncer, so offline
    evaluation and the live service can never disagree.
    """
    arr = np.asarray(flags)
    if arr.ndim != 1:
        raise ValueError(f"flags must be 1-D, got shape {arr.shape}")
    if not np.isin(arr, (0, 1)).all():
        raise ValueError(f"flags must contain only 0/1 values")

    debouncer = AlertDebouncer(n)
    return np.array([debouncer.update(int(f)) for f in arr], dtype=int)
