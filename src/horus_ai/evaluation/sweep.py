"""Threshold x debounce sweep: the trade-off table for choosing both knobs.

The detector has two settings that pull in opposite directions:

    threshold      : lower  -> more windows flagged -> more recall, more false alarms
    n_consecutive  : higher -> fewer false alarms   -> later first alarm, less lead time

sweep() runs evaluate_run() once for every (threshold, N) pair and returns one flat
row per pair. Nothing new is computed here: every number comes from evaluate_run().

    rows = sweep(y_true, times, offsets_ns, 10.0, scores, n_values=(1, 2, 3, 5))
    table = pandas.DataFrame(rows)          # if you want a table
    best = select_operating_point(rows, max_false_episodes_per_hour=1.0)

Run the sweep on VALIDATION data, then evaluate the chosen (threshold, N) once on
held-out TEST data. Choosing the best row on the same data you report would give
optimistic numbers.
"""

from __future__ import annotations

import numpy as np

from .lead_time import FAILURE_LIMIT_NS
from .report import evaluate_run

# These two are already the first two columns of every row, under shorter names.
_DUPLICATE_KEYS = {"config.threshold", "config.n_consecutive"}


def threshold_grid(scores, num: int = 25) -> np.ndarray:
    """Candidate thresholds taken from the score distribution itself.

    Score scales differ a lot between models (Isolation Forest, LSTM error, EKF
    residual), so instead of hard-coding thresholds we use evenly spaced quantiles
    of the observed scores, from the minimum (flags everything) up to the maximum
    (flags only the top window). Duplicates are removed.
    """
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1 or s.size == 0:
        raise ValueError("scores must be a non-empty 1-D array")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite (no NaN or inf)")
    if num < 2:
        raise ValueError("num must be >= 2")
    return np.unique(np.quantile(s, np.linspace(0.0, 1.0, num)))


def sweep(
    y_true,
    times,
    offsets_ns,
    window_seconds: float,
    scores,
    *,
    thresholds=None,
    n_values=(1, 2, 3, 5),
    stride_seconds: float | None = None,
    limit_ns: float = FAILURE_LIMIT_NS,
) -> list[dict]:
    """Evaluate every (threshold, N) pair. Returns one flat dict per pair.

    Row order: for each threshold (ascending as given), every N in n_values.
    Columns: "threshold", "n_consecutive", then the keys of
    EvaluationReport.to_flat_dict() (e.g. "alarm.f1", "lead_time.protection_rate").
    """
    if thresholds is None:
        thresholds = threshold_grid(scores)
    thresholds = [float(t) for t in np.atleast_1d(thresholds)]
    n_values = list(n_values)
    if not thresholds:
        raise ValueError("thresholds must not be empty")
    if not n_values:
        raise ValueError("n_values must not be empty")

    rows = []
    for threshold in thresholds:
        for n in n_values:
            report = evaluate_run(
                y_true,
                times,
                offsets_ns,
                window_seconds,
                scores=scores,
                threshold=threshold,
                n_consecutive=n,
                stride_seconds=stride_seconds,
                limit_ns=limit_ns,
            )
            row = {"threshold": threshold, "n_consecutive": int(n)}
            row.update(
                {k: v for k, v in report.to_flat_dict().items() if k not in _DUPLICATE_KEYS}
            )
            rows.append(row)
    return rows


def select_operating_point(rows: list[dict], max_false_episodes_per_hour: float) -> dict | None:
    """Pick one row: respect a false-alarm budget, then protect as many attacks as possible.

    Rule, in order:
      1. Keep only rows with false_alarms.episodes_per_hour <= the budget.
      2. Highest lead_time.protection_rate (None counts as lowest).
      3. Tie -> higher alarm.f1.
      4. Tie -> fewer false-alarm episodes per hour.
    Returns None when no row fits the budget.
    """
    if max_false_episodes_per_hour < 0:
        raise ValueError("max_false_episodes_per_hour must be >= 0")

    allowed = [
        r for r in rows if r["false_alarms.episodes_per_hour"] <= max_false_episodes_per_hour
    ]
    if not allowed:
        return None

    def key(row):
        rate = row["lead_time.protection_rate"]
        return (
            -1.0 if rate is None else rate,
            row["alarm.f1"],
            -row["false_alarms.episodes_per_hour"],
        )

    return max(allowed, key=key)
