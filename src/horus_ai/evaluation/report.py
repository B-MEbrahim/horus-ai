"""One-call evaluation of a detector run: evaluate_run(...).

Pipeline (the same one the live service will follow):

    scores --(>= threshold)--> raw flags --debounce(N)--> alarms --> metrics

Inputs, all 1-D with one entry per window, in time order:
    y_true      : 0/1 ground truth (1 = the window is under attack)
    times       : END time of each window, in seconds (strictly increasing)
    offsets_ns  : TRUE clock offset in ns at the end of each window (ground truth)
    scores      : detector output, HIGHER = MORE ANOMALOUS   (give scores+threshold)
    flags       : or give raw 0/1 flags directly              (give flags only)

Warning on scores: some libraries return the opposite sign. For example sklearn's
IsolationForest.score_samples() is LOWER for anomalies, so pass `-score_samples(X)`.

Everything about the alarms is computed on the DEBOUNCED alarms, because those are
what the SDN controller actually sees. Raw-flag metrics are reported too, so you can
see what debouncing bought you.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .debounce import debounce
from .events import (
    count_false_alarm_episodes,
    detection_delays,
    false_alarm_episodes_per_hour,
    summarize_detections,
)
from .lead_time import FAILURE_LIMIT_NS, lead_times, summarize_lead_times
from .metrics import (
    false_alarms_per_hour,
    f1_score,
    false_positive_rate,
    precision,
    recall,
)


@dataclass(frozen=True)
class EvaluationReport:
    config: dict  # settings used for this run
    raw: dict  # window-level metrics on the raw flags (before debouncing)
    alarm: dict  # window-level metrics on the debounced alarms
    false_alarms: dict  # false-alarm rates on the debounced alarms
    detection: dict  # attack-level: events caught + detection delay summary
    lead_time: dict  # attack-level: alarm before the failure limit?

    def to_dict(self) -> dict:
        """Nested dictionary (good for JSON)."""
        return asdict(self)

    def to_flat_dict(self) -> dict:
        """Single-level dictionary like {"alarm.f1": 0.8, ...} (one row of a table)."""
        flat = {}
        for section, values in self.to_dict().items():
            for key, value in values.items():
                flat[f"{section}.{key}"] = value
        return flat


def _window_metrics(y_true, y_pred) -> dict:
    return {
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "fpr": false_positive_rate(y_true, y_pred),
    }


def _flags_from_scores(scores, threshold, n_windows: int) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1:
        raise ValueError(f"scores must be 1-D, got shape {s.shape}")
    if s.shape[0] != n_windows:
        raise ValueError(f"length mismatch: y_true={n_windows}, scores={s.shape[0]}")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite (no NaN or inf)")
    return (s >= threshold).astype(int)


def evaluate_run(
    y_true,
    times,
    offsets_ns,
    window_seconds: float,
    *,
    scores=None,
    threshold: float | None = None,
    flags=None,
    n_consecutive: int = 1,
    stride_seconds: float | None = None,
    limit_ns: float = FAILURE_LIMIT_NS,
) -> EvaluationReport:
    """Evaluate one detector run end to end.

    Give EITHER (scores and threshold) OR flags, never both.

    n_consecutive  : debounce N (alarm after N consecutive anomalous windows).
    stride_seconds : time between consecutive windows. Defaults to window_seconds
                     (non-overlapping windows). Use a smaller value if windows overlap.
                     It is used only for the "per hour" false-alarm rates.
    """
    # ---- choose the input mode -------------------------------------------------
    if (scores is None) == (flags is None):
        raise ValueError("give exactly one of `scores` (with `threshold`) or `flags`")
    if scores is not None and threshold is None:
        raise ValueError("`threshold` is required when `scores` is given")
    if flags is not None and threshold is not None:
        raise ValueError("`threshold` only applies to `scores`, not to `flags`")

    stride = window_seconds if stride_seconds is None else stride_seconds
    if stride <= 0:
        raise ValueError("stride_seconds must be > 0")

    y = np.asarray(y_true)
    if y.ndim != 1:
        raise ValueError(f"y_true must be 1-D, got shape {y.shape}")

    # ---- scores -> raw flags -> alarms ----------------------------------------
    if scores is not None:
        raw_flags = _flags_from_scores(scores, threshold, y.shape[0])
    else:
        raw_flags = np.asarray(flags)
        if raw_flags.ndim == 1 and raw_flags.shape[0] != y.shape[0]:
            raise ValueError(
                f"length mismatch: y_true={y.shape[0]}, flags={raw_flags.shape[0]}"
            )
    alarms = debounce(raw_flags, n_consecutive)  # also validates flags are 0/1

    # ---- compute every part ---------------------------------------------------
    detections = detection_delays(y, alarms, times, window_seconds)
    leads = lead_times(y, alarms, times, offsets_ns, window_seconds, limit_ns)

    return EvaluationReport(
        config={
            "n_windows": int(y.shape[0]),
            "n_attack_windows": int(np.sum(y == 1)),
            "window_seconds": float(window_seconds),
            "stride_seconds": float(stride),
            "n_consecutive": int(n_consecutive),
            "threshold": None if threshold is None else float(threshold),
            "limit_ns": float(limit_ns),
        },
        raw=_window_metrics(y, raw_flags),
        alarm=_window_metrics(y, alarms),
        false_alarms={
            "episodes": count_false_alarm_episodes(y, alarms),
            "episodes_per_hour": false_alarm_episodes_per_hour(y, alarms, stride),
            "windows_per_hour": false_alarms_per_hour(y, alarms, stride),
        },
        detection=summarize_detections(detections),
        lead_time=summarize_lead_times(leads),
    )
