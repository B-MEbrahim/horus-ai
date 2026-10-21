"""Attack-level (event) evaluation for the HORUS detector.

Window-level metrics (metrics.py) ask "was each window right?".
This file asks, per attack: "was it caught, and how fast?".

Conventions (fixed, so the numbers are comparable across experiments):
    - y_true, alarms : 1-D 0/1 arrays, one entry per window.
        alarms is what comes OUT of debounce(), i.e. what the SDN controller sees.
    - times          : the END time (seconds) of each window. A window's verdict is
                       only known once the window has finished.
    - window_seconds : length of one window, so window start = end - window_seconds.
    - An attack event is a run of consecutive windows with y_true == 1.
    - Event start time = start of its first window.
    - Detection delay  = (end time of first alarm window inside the event) - event start.
    - A missed event has delay None. It is never counted as 0.
    - Alarms before an event are false alarms, not negative delays.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .metrics import _to_binary


@dataclass(frozen=True)
class AttackEvent:
    start_idx: int  # first attack window
    end_idx: int  # last attack window (inclusive)
    start_time: float  # seconds: start of the first attack window
    end_time: float  # seconds: end of the last attack window


@dataclass(frozen=True)
class EventDetection:
    event: AttackEvent
    first_alarm_idx: int | None  # None = attack was missed
    first_alarm_time: float | None
    delay_seconds: float | None

    @property
    def detected(self) -> bool:
        return self.first_alarm_idx is not None


def _prepare(y_true, alarms, times, window_seconds):
    t = _to_binary("y_true", y_true)
    a = _to_binary("alarms", alarms)
    ts = np.asarray(times, dtype=float)
    if ts.ndim != 1:
        raise ValueError(f"times must be 1-D, got shape {ts.shape}")
    if not (t.shape == a.shape == ts.shape):
        raise ValueError(
            f"length mismatch: y_true={t.shape[0]}, alarms={a.shape[0]}, times={ts.shape[0]}"
        )
    if np.any(np.diff(ts) <= 0):
        raise ValueError("times must be strictly increasing")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be > 0")
    return t, a, ts


def _runs(binary: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end_inclusive) for every run of consecutive 1s."""
    padded = np.concatenate(([0], binary, [0]))
    change = np.diff(padded)
    starts = np.flatnonzero(change == 1)
    ends = np.flatnonzero(change == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def find_attack_events(y_true, times, window_seconds: float) -> list[AttackEvent]:
    """Split the ground-truth labels into separate attack events."""
    t = _to_binary("y_true", y_true)
    ts = np.asarray(times, dtype=float)
    if t.shape != ts.shape:
        raise ValueError(f"length mismatch: y_true={t.shape[0]}, times={ts.shape[0]}")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be > 0")

    return [
        AttackEvent(
            start_idx=s,
            end_idx=e,
            start_time=float(ts[s] - window_seconds),
            end_time=float(ts[e]),
        )
        for s, e in _runs(t)
    ]


def detection_delays(y_true, alarms, times, window_seconds: float) -> list[EventDetection]:
    """For every attack event: was it caught, and how long did it take?

    Only alarms INSIDE the event count as detecting it.
    """
    t, a, ts = _prepare(y_true, alarms, times, window_seconds)

    results = []
    for ev in find_attack_events(t, ts, window_seconds):
        inside = a[ev.start_idx : ev.end_idx + 1]
        if inside.any():
            idx = ev.start_idx + int(np.argmax(inside))  # first alarm in the event
            alarm_time = float(ts[idx])
            results.append(
                EventDetection(ev, idx, alarm_time, alarm_time - ev.start_time)
            )
        else:
            results.append(EventDetection(ev, None, None, None))
    return results


def summarize_detections(detections: list[EventDetection]) -> dict:
    """One-line summary over all events. Delay stats use detected events only."""
    delays = [d.delay_seconds for d in detections if d.detected]
    n = len(detections)
    return {
        "n_events": n,
        "n_detected": len(delays),
        "event_recall": len(delays) / n if n else 0.0,
        "mean_delay_s": float(np.mean(delays)) if delays else None,
        "median_delay_s": float(np.median(delays)) if delays else None,
        "max_delay_s": float(np.max(delays)) if delays else None,
    }


def count_false_alarm_episodes(y_true, alarms) -> int:
    """Number of separate false-alarm bursts.

    A burst is a run of consecutive windows with alarm=1 while y_true=0.
    Five false-alarm windows in a row = 1 episode (one needless failover).
    """
    t = _to_binary("y_true", y_true)
    a = _to_binary("alarms", alarms)
    if t.shape != a.shape:
        raise ValueError(f"length mismatch: y_true={t.shape[0]}, alarms={a.shape[0]}")
    return len(_runs(((t == 0) & (a == 1)).astype(int)))


def false_alarm_episodes_per_hour(y_true, alarms, seconds_per_window: float) -> float:
    """False-alarm episodes per hour of NORMAL operation."""
    if seconds_per_window <= 0:
        raise ValueError("seconds_per_window must be > 0")
    t = _to_binary("y_true", y_true)
    normal_hours = int(np.sum(t == 0)) * seconds_per_window / 3600.0
    episodes = count_false_alarm_episodes(y_true, alarms)
    return episodes / normal_hours if normal_hours > 0 else 0.0
