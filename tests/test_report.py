"""Tests for horus_ai.evaluation.report, using one small hand-made run.

The real logic lives in metrics.py, debounce.py, events.py and lead_time.py (already
tested). These tests check the WIRING: that evaluate_run() chains the pieces in the
right order and reports the right numbers.

Shared run (12 windows, window_seconds = 1, window i ends at t = i + 1):

    idx        : 0    1    2    3    4     5     6     7     8    9    10   11
    y_true     : 0    0    0    1    1     1     1     1     0    0    0    0
    offset ns  : 0    0    0    300  700   1100  1600  2000  0    0    0    0
    score      : 0.1  0.9  0.2  0.4  0.5   0.8   0.9   0.7   0.3  0.1  0.6  0.2
    raw flag   : 0    1    0    0    1     1     1     1     0    0    1    0   (threshold 0.5)

    - the attack is idx 3..7 and starts at t = 3.0 s
    - the offset first reaches 1500 ns at idx 6, so the failure time is t = 7.0 s
    - idx 1 and idx 10 are isolated false alarms; idx 3 is missed by the raw flags
    - idx 4 has a score of exactly 0.5, to test that the threshold rule is ">="
"""

import json

import numpy as np
import pytest

from horus_ai.evaluation.report import EvaluationReport, evaluate_run

Y_TRUE = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
OFFSETS = [0, 0, 0, 300, 700, 1100, 1600, 2000, 0, 0, 0, 0]
SCORES = [0.1, 0.9, 0.2, 0.4, 0.5, 0.8, 0.9, 0.7, 0.3, 0.1, 0.6, 0.2]
RAW_FLAGS = [0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0]
TIMES = np.arange(1, 13)


def run(n=1, **overrides):
    """evaluate_run on the shared run; override any argument by keyword."""
    kwargs = dict(
        y_true=Y_TRUE,
        times=TIMES,
        offsets_ns=OFFSETS,
        window_seconds=1.0,
        scores=SCORES,
        threshold=0.5,
        n_consecutive=n,
    )
    kwargs.update(overrides)
    return evaluate_run(**kwargs)


# ---------------------------------------------------------------- the worked example (N = 2)


def test_full_report_with_debounce_n2():
    r = run(n=2)

    # raw flags : 0 1 0 0 1 1 1 1 0 0 1 0
    # alarms    : 0 0 0 0 0 1 1 1 0 0 0 0   (both isolated flags and idx 4 are removed)
    assert r.config == {
        "n_windows": 12,
        "n_attack_windows": 5,
        "window_seconds": 1.0,
        "stride_seconds": 1.0,
        "n_consecutive": 2,
        "threshold": 0.5,
        "limit_ns": 1500.0,
    }

    # Raw: TP=4 (idx 4-7)  FP=2 (idx 1, 10)  FN=1 (idx 3)  TN=5
    assert r.raw["precision"] == pytest.approx(2 / 3)  # 4 / 6
    assert r.raw["recall"] == pytest.approx(0.8)  # 4 / 5
    assert r.raw["f1"] == pytest.approx(8 / 11)
    assert r.raw["fpr"] == pytest.approx(2 / 7)  # 2 / (2 + 5)

    # Alarms: TP=3 (idx 5-7)  FP=0  FN=2 (idx 3, 4)  TN=7
    assert r.alarm["precision"] == pytest.approx(1.0)
    assert r.alarm["recall"] == pytest.approx(0.6)  # 3 / 5
    assert r.alarm["f1"] == pytest.approx(0.75)  # 2 * 1 * 0.6 / 1.6
    assert r.alarm["fpr"] == 0.0

    assert r.false_alarms == {
        "episodes": 0,
        "episodes_per_hour": 0.0,
        "windows_per_hour": 0.0,
    }

    # Event starts at t = 3.0; first alarm is idx 5, whose window ends at t = 6.0.
    assert r.detection["n_events"] == 1
    assert r.detection["n_detected"] == 1
    assert r.detection["event_recall"] == pytest.approx(1.0)
    assert r.detection["mean_delay_s"] == pytest.approx(3.0)
    assert r.detection["median_delay_s"] == pytest.approx(3.0)
    assert r.detection["max_delay_s"] == pytest.approx(3.0)

    # Failure at t = 7.0 s, alarm at t = 6.0 s -> lead +1.0 s.
    assert r.lead_time["n_events"] == 1
    assert r.lead_time["n_failure_events"] == 1
    assert r.lead_time["n_in_time"] == 1
    assert r.lead_time["n_too_late"] == 0
    assert r.lead_time["n_missed"] == 0
    assert r.lead_time["protection_rate"] == pytest.approx(1.0)
    assert r.lead_time["mean_lead_s"] == pytest.approx(1.0)
    assert r.lead_time["min_lead_s"] == pytest.approx(1.0)


def test_n1_means_alarms_are_the_raw_flags():
    r = run(n=1)
    assert r.alarm == r.raw

    # Two separate isolated false alarms (idx 1 and idx 10) = 2 episodes.
    # 7 normal windows x 1 s = 7/3600 hours -> 2 / (7/3600) = 7200/7 per hour.
    assert r.false_alarms["episodes"] == 2
    assert r.false_alarms["episodes_per_hour"] == pytest.approx(7200 / 7)
    assert r.false_alarms["windows_per_hour"] == pytest.approx(7200 / 7)  # no burst here

    # First alarm inside the attack: idx 4 (t = 5.0) -> delay 2.0; lead = 7 - 5 = 2.0.
    assert r.detection["mean_delay_s"] == pytest.approx(2.0)
    assert r.lead_time["mean_lead_s"] == pytest.approx(2.0)


# ---------------------------------------------------------------- the debounce sweep


@pytest.mark.parametrize(
    "n, delay, lead, outcome",
    [
        (1, 2.0, 2.0, "in_time"),
        (2, 3.0, 1.0, "in_time"),
        (3, 4.0, 0.0, "too_late"),  # alarm and failure in the same window
        (4, 5.0, -1.0, "too_late"),
        (5, None, None, "missed"),  # the longest run of raw flags is only 4
    ],
)
def test_each_extra_debounce_window_costs_one_second(n, delay, lead, outcome):
    r = run(n=n)

    assert r.detection["mean_delay_s"] == (None if delay is None else pytest.approx(delay))
    assert r.detection["n_detected"] == (0 if delay is None else 1)

    assert r.lead_time["mean_lead_s"] == (None if lead is None else pytest.approx(lead))
    assert r.lead_time["n_in_time"] == (1 if outcome == "in_time" else 0)
    assert r.lead_time["n_too_late"] == (1 if outcome == "too_late" else 0)
    assert r.lead_time["n_missed"] == (1 if outcome == "missed" else 0)
    assert r.lead_time["protection_rate"] == (1.0 if outcome == "in_time" else 0.0)


def test_debouncing_trades_false_alarms_for_recall():
    r1, r2 = run(n=1), run(n=2)
    assert r1.false_alarms["episodes"] == 2
    assert r2.false_alarms["episodes"] == 0
    assert r1.alarm["recall"] > r2.alarm["recall"]  # 0.8 -> 0.6


def test_raw_section_does_not_depend_on_n():
    assert run(n=1).raw == run(n=2).raw == run(n=5).raw


# ---------------------------------------------------------------- threshold


def test_threshold_rule_is_greater_or_equal():
    # idx 4 has a score of exactly 0.5.
    flagged = run(threshold=0.5)  # idx 4 flagged     -> raw TP = 4
    not_flagged = run(threshold=0.51)  # idx 4 not flagged -> raw TP = 3, FP = 2, FN = 2
    assert flagged.raw["recall"] == pytest.approx(0.8)
    assert not_flagged.raw["precision"] == pytest.approx(0.6)  # 3 / 5
    assert not_flagged.raw["recall"] == pytest.approx(0.6)  # 3 / 5
    assert not_flagged.config["threshold"] == 0.51


def test_threshold_zero_flags_everything():
    r = run(threshold=0.0)
    assert r.raw["recall"] == pytest.approx(1.0)
    assert r.raw["fpr"] == pytest.approx(1.0)


def test_detector_that_never_fires():
    r = run(scores=[0.0] * 12)
    assert r.raw["precision"] == 0.0
    assert r.raw["recall"] == 0.0
    assert r.alarm["f1"] == 0.0
    assert r.alarm["fpr"] == 0.0
    assert r.false_alarms["episodes"] == 0
    assert r.detection["n_detected"] == 0
    assert r.detection["mean_delay_s"] is None
    assert r.lead_time["n_missed"] == 1
    assert r.lead_time["protection_rate"] == 0.0


# ---------------------------------------------------------------- scores and flags agree


def test_scores_and_flags_give_the_same_report():
    from_scores = run(n=2)
    from_flags = run(n=2, scores=None, threshold=None, flags=RAW_FLAGS)
    assert from_flags.config["threshold"] is None  # no threshold was used
    for section in ("raw", "alarm", "false_alarms", "detection", "lead_time"):
        assert getattr(from_scores, section) == getattr(from_flags, section)


def test_perfect_detector_given_as_flags():
    r = run(scores=None, threshold=None, flags=Y_TRUE)
    assert r.alarm == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "fpr": 0.0}
    assert r.detection["mean_delay_s"] == pytest.approx(1.0)  # alarm in the first window
    assert r.lead_time["mean_lead_s"] == pytest.approx(3.0)  # fails at 7.0, alarm at 4.0
    assert r.lead_time["protection_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------- a run with no attacks


def test_run_with_no_attacks_reports_only_false_alarm_information():
    r = evaluate_run(
        y_true=[0, 0, 0, 0],
        times=[1, 2, 3, 4],
        offsets_ns=[0, 0, 0, 0],
        window_seconds=1.0,
        flags=[0, 1, 0, 0],
    )
    assert r.config["n_attack_windows"] == 0
    assert r.alarm["fpr"] == pytest.approx(0.25)
    assert r.alarm["recall"] == 0.0  # 0/0 defined as 0.0
    assert r.false_alarms["episodes"] == 1
    # 4 normal windows x 1 s = 4/3600 h -> 1 / (4/3600) = 900 per hour
    assert r.false_alarms["episodes_per_hour"] == pytest.approx(900.0)
    assert r.detection["n_events"] == 0
    assert r.detection["event_recall"] == 0.0
    assert r.detection["mean_delay_s"] is None
    assert r.lead_time["n_events"] == 0
    assert r.lead_time["protection_rate"] is None  # nothing to protect against


# ---------------------------------------------------------------- stride and limit


def test_stride_only_changes_the_per_hour_rates():
    default = run(n=1)
    overlapping = run(n=1, stride_seconds=0.5)  # windows start every 0.5 s

    assert overlapping.config["stride_seconds"] == 0.5
    # Half the stride -> half the normal time -> double the per-hour rate.
    assert overlapping.false_alarms["episodes_per_hour"] == pytest.approx(
        2 * default.false_alarms["episodes_per_hour"]
    )
    # Everything else is unchanged.
    assert overlapping.false_alarms["episodes"] == default.false_alarms["episodes"]
    assert overlapping.raw == default.raw
    assert overlapping.alarm == default.alarm
    assert overlapping.detection == default.detection
    assert overlapping.lead_time == default.lead_time


def test_stricter_limit_moves_the_failure_earlier():
    # Limit 1000 ns -> failure at idx 5 (offset 1100), t = 6.0.
    # With N = 2 the alarm is also at t = 6.0 -> lead 0.0 -> too late.
    r = run(n=2, limit_ns=1000)
    assert r.config["limit_ns"] == 1000.0
    assert r.lead_time["n_in_time"] == 0
    assert r.lead_time["n_too_late"] == 1
    assert r.lead_time["mean_lead_s"] == pytest.approx(0.0)


# ---------------------------------------------------------------- output formats


def test_to_dict_has_the_six_sections():
    d = run(n=2).to_dict()
    assert list(d) == ["config", "raw", "alarm", "false_alarms", "detection", "lead_time"]


def test_to_dict_returns_a_copy():
    r = run(n=2)
    d = r.to_dict()
    d["alarm"]["f1"] = 99.0
    assert r.alarm["f1"] == pytest.approx(0.75)


def test_to_dict_survives_a_json_round_trip():
    d = run(n=2).to_dict()
    assert json.loads(json.dumps(d)) == d


def test_to_flat_dict_matches_the_nested_values():
    r = run(n=2)
    nested, flat = r.to_dict(), r.to_flat_dict()

    assert len(flat) == sum(len(section) for section in nested.values())
    for section, values in nested.items():
        for key, value in values.items():
            assert flat[f"{section}.{key}"] == value

    assert flat["alarm.f1"] == pytest.approx(0.75)
    assert flat["config.n_consecutive"] == 2
    assert flat["lead_time.protection_rate"] == pytest.approx(1.0)
    assert not any(isinstance(v, dict) for v in flat.values())


def test_report_type():
    assert isinstance(run(n=2), EvaluationReport)


# ---------------------------------------------------------------- input-mode errors


def test_giving_both_scores_and_flags_raises():
    with pytest.raises(ValueError, match="exactly one"):
        run(flags=RAW_FLAGS)


def test_giving_neither_scores_nor_flags_raises():
    with pytest.raises(ValueError, match="exactly one"):
        run(scores=None, threshold=None)


def test_scores_without_threshold_raises():
    with pytest.raises(ValueError, match="required when"):
        run(threshold=None)


def test_flags_with_threshold_raises():
    with pytest.raises(ValueError, match="only applies"):
        run(scores=None, flags=RAW_FLAGS)  # threshold=0.5 is still set


# ---------------------------------------------------------------- input validation


def test_scores_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        run(scores=SCORES[:-1])


def test_flags_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        run(scores=None, threshold=None, flags=RAW_FLAGS[:-1])


def test_scores_must_be_finite():
    bad = list(SCORES)
    bad[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        run(scores=bad)


def test_scores_must_be_1d():
    with pytest.raises(ValueError, match="1-D"):
        run(scores=[[0.1, 0.2], [0.3, 0.4]])


def test_flags_must_be_1d():
    with pytest.raises(ValueError, match="1-D"):
        run(scores=None, threshold=None, flags=[[0, 1], [1, 0]])


def test_y_true_must_be_1d():
    with pytest.raises(ValueError, match="1-D"):
        run(y_true=[[0, 1], [1, 0]])


def test_non_binary_flags_raise():
    bad = list(RAW_FLAGS)
    bad[0] = 2
    with pytest.raises(ValueError, match="0/1"):
        run(scores=None, threshold=None, flags=bad)


@pytest.mark.parametrize("bad_n", [0, -1, 2.5])
def test_bad_n_consecutive_raises(bad_n):
    with pytest.raises(ValueError, match="n must be"):
        run(n=bad_n)


@pytest.mark.parametrize("bad_stride", [0, -1.0])
def test_bad_stride_raises(bad_stride):
    with pytest.raises(ValueError, match="stride_seconds"):
        run(stride_seconds=bad_stride)


def test_errors_from_the_lower_modules_still_apply():
    with pytest.raises(ValueError, match="strictly increasing"):
        run(times=[1, 2, 3, 4, 5, 6, 7, 7, 9, 10, 11, 12])
    with pytest.raises(ValueError, match="length mismatch"):
        run(offsets_ns=OFFSETS[:-1])
    with pytest.raises(ValueError, match="finite"):
        run(offsets_ns=[0, 0, 0, 0, 0, np.nan, 0, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="limit_ns"):
        run(limit_ns=0)
    with pytest.raises(ValueError, match="window_seconds"):
        run(window_seconds=0, stride_seconds=1.0)