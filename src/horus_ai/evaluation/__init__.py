"""HORUS evaluation module: judge a spoofing detector the way the network needs it judged.

Main entry point:

    from horus_ai.evaluation import evaluate_run

    report = evaluate_run(
        y_true=y_true,            # 0/1 ground truth per window
        times=times,              # END time of each window, in seconds
        offsets_ns=offsets_ns,    # true clock offset (ns) at the end of each window
        window_seconds=10.0,
        scores=scores,            # higher = more anomalous
        threshold=0.5,
        n_consecutive=3,          # alarm after 3 anomalous windows in a row
    )
    report.alarm["f1"]
    report.lead_time["protection_rate"]
    report.to_flat_dict()         # one row for a comparison table

Building blocks (also importable from here):
    metrics   : window-level precision, recall, F1, FPR
    debounce  : N-consecutive-windows rule (batch + streaming)
    events    : attack events, detection delay, false-alarm episodes
    lead_time : was the alarm raised before the 1.5 µs failure limit?
    report    : evaluate_run(), which chains all of the above
"""

from .debounce import AlertDebouncer, debounce
from .events import (
    AttackEvent,
    EventDetection,
    count_false_alarm_episodes,
    detection_delays,
    false_alarm_episodes_per_hour,
    find_attack_events,
    summarize_detections,
)
from .lead_time import (
    FAILURE_LIMIT_NS,
    LeadTimeResult,
    lead_times,
    summarize_lead_times,
)
from .metrics import (
    ConfusionCounts,
    confusion_counts,
    f1_score,
    false_alarms_per_hour,
    false_positive_rate,
    precision,
    recall,
)
from .report import EvaluationReport, evaluate_run

__all__ = [
    # one-call evaluation
    "evaluate_run",
    "EvaluationReport",
    # window-level metrics
    "ConfusionCounts",
    "confusion_counts",
    "precision",
    "recall",
    "f1_score",
    "false_positive_rate",
    "false_alarms_per_hour",
    # debouncing
    "AlertDebouncer",
    "debounce",
    # attack-level metrics
    "AttackEvent",
    "EventDetection",
    "find_attack_events",
    "detection_delays",
    "summarize_detections",
    "count_false_alarm_episodes",
    "false_alarm_episodes_per_hour",
    # lead time
    "FAILURE_LIMIT_NS",
    "LeadTimeResult",
    "lead_times",
    "summarize_lead_times",
]