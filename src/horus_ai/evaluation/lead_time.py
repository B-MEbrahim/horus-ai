"""Lead time before failure: was the alarm raised BEFORE the network broke?

This is the metric closest to the goal of HORUS. A detector can have a great F1
and still be useless if it only fires after the clocks have drifted past the
1.5 µs limit, because by then the towers are already desynchronized.

Extra input compared to events.py:
    offsets_ns : 1-D array, one entry per window. The TRUE clock offset (in
                 nanoseconds) at the END of that window. This is ground truth
                 from the data generator, not something the detector sees.

Conventions:
    - Failure time of an attack = end time of the first window INSIDE the attack
      where |offset| >= limit_ns (default 1500 ns = 1.5 µs).
    - Alarm time = end time of the first alarm window inside the attack
      (exactly the alarm that detection_delays() finds).
    - lead_seconds = failure_time - alarm_time.
        positive -> alarm came before the failure (good)
        zero     -> alarm and failure in the same window (counted as too late)
        negative -> alarm came after the failure
    - Outcome per attack:
        "no_failure" : the offset never reached the limit (nothing to protect against)
        "missed"     : it reached the limit and we never raised an alarm
        "too_late"   : alarm raised, but at or after the failure time
        "in_time"    : alarm raised strictly before the failure time
"""


from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .events import AttackEvent, detection_delays

FAILURE_LIMIT_NS = 1500  # 1.5 µs


@dataclass(frozen=True)
class LeadTimeResult:
    event: AttackEvent
    outcome: str  # "no_failure" | "missed" | "too_late" | "in_time"
    failure_idx: int | None
    failure_time: float | None # seconds
    alarm_time: float | None # seconds
    lead_seconds: float | None # set only when both a failure and an alarm exist


def lead_times(
        y_true,
        alarms,
        times,
        offsets_ns,
        window_seconds: float,
        limit_ns: float = FAILURE_LIMIT_NS
) -> list[LeadTimeResult]:
    """For every attack event: did the alarm come before the 1.5 µs failure?"""
    # Validates y_true / alarms / times / window_seconds and finds first alarms.
    detections = detection_delays(y_true, alarms, times, window_seconds)

    ts = np.asarray(times, dtype=float)
    off = np.asarray(offsets_ns, dtype=float)
    if off.ndim != 1:
        raise ValueError(f"offsets_ns must be 1-D, got shape {off.shape}")
    if off.shape != ts.shape:
        raise ValueError(f"length mismatch: offsets_ns={off.shape[0]}, times={ts.shape[0]}")
    if not np.all(np.isfinite(off)):
        raise ValueError("offsets_ns must be finite (no NaN or inf)")
    if limit_ns <= 0:
        raise ValueError("limit_ns must be > 0")

    results = []
    for det in detections:
        ev = det.event
        segment = np.abs(off[ev.start_idx : ev.end_idx + 1]) >= limit_ns

        if not segment.any():
            results.append(
                LeadTimeResult(ev, "no_failure", None, None, 
                det.first_alarm_time, None)
            )
            continue

        failure_idx = ev.start_idx + int(np.argmax(segment)) # first window over the limit
        failure_time = float(ts[failure_idx]) 

        if not det.detected:
            results.append(
                LeadTimeResult(ev, "missed", failure_idx, failure_time, None, None)
            )
            continue

        lead = failure_time - det.first_alarm_time
        outcome = "in_time" if lead > 0 else "too_late"
        results.append(
            LeadTimeResult(ev, outcome, failure_idx, failure_time, 
                           det.first_alarm_time, lead)
        )
    return results


def summarize_lead_times(results: list[LeadTimeResult]) -> dict:
     """Summary over all attacks.

    protection_rate = in_time / (attacks that actually reached the limit).
    It is None when no attack reached the limit (nothing to measure).
    Lead statistics cover every attack that has a lead value (in_time and too_late).
    """
     n_failure = sum(r.outcome != "no_failure" for r in results)
     n_in_time = sum(r.outcome == "in_time" for r in results)
     leads = [r.lead_seconds for r in results if r.lead_seconds is not None]
     return {
         "n_events": len(results),
         "n_no_failure": sum(r.outcome == "no_failure" for r in results),
         "n_failure_events": n_failure,
         "n_in_time": n_in_time,
         "n_too_late": sum(r.outcome == "too_late" for r in results),
         "n_missed": sum(r.outcome == "missed" for r in results),
         "protection_rate": n_in_time / n_failure if n_failure > 0 else None,
         "mean_lead_s": float(np.mean(leads)) if leads else None,
         "min_lead_s": float(np.min(leads)) if leads else None,
     }
