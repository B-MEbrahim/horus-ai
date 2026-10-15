"""Window-level classification metrics for the detector.

Convention: 1 = attack (positive), 0 = normal (negative).
Every function takes two 1-D arrays with one entry per window:
    y_true : ground truth (was this window really under attack?)
    y_pred : detector output (did it flag this window?)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ConfusionCounts:
    tp: int # attack window, flagged as attack
    fp: int # normal window, flagged  
    tn: int # normal window, not flagged
    fn: int # attack window, not flagged 


def _to_binary(name: str, values) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {arr.shape}")
    if not np.isin(arr, (0, 1)).all():
        raise ValueError(f"{name} must contain only 0/1 values")
    return arr.astype(int)

def _prepare(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    t = _to_binary("y_true", y_true)
    p = _to_binary("y_pred", y_pred)
    if t.shape != p.shape:
        raise ValueError(f"length mismatch: y_true={t.shape[0]}, y_pred={p.shape[0]}")
    return t, p

def _safe_divide(numerator: float, denominator: float) -> float:
    """Divide numerator by denominator, returning 0 if denominator is 0."""
    return numerator / denominator if denominator != 0 else 0.0

def confusion_counts(y_true, y_pred) -> ConfusionCounts:
    """Count true positives, false positives, true negatives, and false negatives."""
    t, p = _prepare(y_true, y_pred)
    return ConfusionCounts(
        tp = int(np.sum((t == 1) & (p == 1))),
        fp = int(np.sum((t == 0) & (p == 1))),
        tn = int(np.sum((t == 0) & (p == 0))),
        fn = int(np.sum((t == 1) & (p == 0)))
    )

def precision(y_true, y_pred) -> float:
    """Of all windows we flagged, what fraction were real attacks? TP / (TP + FP)"""
    counts = confusion_counts(y_true, y_pred)
    return _safe_divide(counts.tp, counts.tp + counts.fp)

def recall(y_true, y_pred) -> float:
    """Of all real attack windows, what fraction did we flag? TP / (TP + FN)"""
    counts = confusion_counts(y_true, y_pred)
    return _safe_divide(counts.tp, counts.tp + counts.fn)

def f1_score(y_true, y_pred) -> float:
    """Harmonic mean of precision and recall: 2PR / (P + R)"""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return _safe_divide(2 * p * r, p + r)

def false_positive_rate(y_true, y_pred) -> float:
    """Of all normal windows, what fraction raised a false alarm? FP / (FP + TN)"""
    counts = confusion_counts(y_true, y_pred)
    return _safe_divide(counts.fp, counts.fp + counts.tn)

def false_alarms_per_hour(y_true, y_pred, seconds_per_window: float) -> float:
    """False-alarm windows per hour of NORMAL operation.

    seconds_per_window is the time between consecutive windows (the stride).
    Note: consecutive false-alarm windows are counted separately here.
    """
    if seconds_per_window <= 0:
        raise ValueError(f"seconds_per_window must be > 0")
    counts = confusion_counts(y_true, y_pred)
    normal_hours = (counts.fp + counts.tn) * seconds_per_window / 3600.0
    return _safe_divide(counts.fp, normal_hours)