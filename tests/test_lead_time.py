"""Tests for horus_ai.evaluation.lead_time, using tiny hand-made arrays.

Unless stated otherwise: window_seconds = 1 and times = [1, 2, 3, ...],
so window i covers (i, i+1] and its END time is i + 1.

Most tests share one drift ramp (idx 2..6 is the attack):

    idx      : 0    1    2    3    4     5     6     7
    y_true   : 0    0    1    1    1     1     1     0
    offset ns: 0    0    300  700  1100  1600  2000  0

The offset first reaches 1500 ns at idx 5, so the failure time is 6.0 s.
"""

import numpy as np
import pytest

from horus_ai.evaluation.debounce import debounce
from horus_ai.evaluation.events import AttackEvent
from horus_ai.evaluation.lead_time import (
    FAILURE_LIMIT_NS,
    LeadTimeResult,
    lead_times,
    summarize_lead_times,
)


def make_times(n, window_seconds=1.0):
    """End time of each window: window_seconds, 2*window_seconds, ..."""
    return np.arange(1, n + 1) * window_seconds


RAMP_TRUE = [0, 0, 1, 1, 1, 1, 1, 0]
RAMP_OFFSETS = [0, 0, 300, 700, 1100, 1600, 2000, 0]
RAMP_EVENT = AttackEvent(start_idx=2, end_idx=6, start_time=2.0, end_time=7.0)


def run_ramp(alarms):
    (result,) = lead_times(RAMP_TRUE, alarms, make_times(8), RAMP_OFFSETS, 1.0)
    return result


def test_default_limit_is_1500_ns():
    assert FAILURE_LIMIT_NS == 1500.0


# ---------------------------------------------------------------- the four outcomes


def test_in_time_alarm_before_the_failure():
    # first alarm idx 4 -> window ends at 5.0 s; failure at 6.0 s -> lead = +1.0 s
    result = run_ramp([0, 0, 0, 0, 1, 1, 1, 0])
    assert result == LeadTimeResult(
        event=RAMP_EVENT,
        outcome="in_time",
        failure_idx=5,
        failure_time=6.0,
        alarm_time=5.0,
        lead_seconds=1.0,
    )


def test_earliest_possible_alarm_gives_the_largest_lead():
    # alarm in the very first attack window (idx 2, ends at 3.0 s) -> lead = 6 - 3 = 3.0 s
    result = run_ramp([0, 0, 1, 1, 1, 1, 1, 0])
    assert result.outcome == "in_time"
    assert result.lead_seconds == pytest.approx(3.0)


def test_too_late_alarm_after_the_failure():
    # first alarm idx 6 -> ends at 7.0 s; failure at 6.0 s -> lead = -1.0 s
    result = run_ramp([0, 0, 0, 0, 0, 0, 1, 0])
    assert result.outcome == "too_late"
    assert result.failure_time == 6.0
    assert result.alarm_time == 7.0
    assert result.lead_seconds == pytest.approx(-1.0)


def test_alarm_in_the_same_window_as_the_failure_counts_as_too_late():
    # alarm idx 5 -> ends at 6.0 s, same as the failure -> lead = 0 -> too late
    result = run_ramp([0, 0, 0, 0, 0, 1, 1, 0])
    assert result.outcome == "too_late"
    assert result.lead_seconds == pytest.approx(0.0)


def test_missed_when_the_limit_is_reached_and_there_is_no_alarm():
    result = run_ramp([0, 0, 0, 0, 0, 0, 0, 0])
    assert result == LeadTimeResult(
        event=RAMP_EVENT,
        outcome="missed",
        failure_idx=5,
        failure_time=6.0,
        alarm_time=None,
        lead_seconds=None,
    )


def test_no_failure_when_the_offset_never_reaches_the_limit():
    offsets = [0, 0, 200, 400, 600, 800, 1000, 0]  # max 1000 < 1500
    alarms = [0, 0, 0, 0, 1, 1, 1, 0]
    (result,) = lead_times(RAMP_TRUE, alarms, make_times(8), offsets, 1.0)
    assert result == LeadTimeResult(
        event=RAMP_EVENT,
        outcome="no_failure",
        failure_idx=None,
        failure_time=None,
        alarm_time=5.0,  # we still report when the alarm came
        lead_seconds=None,  # but there is no failure to measure it against
    )


def test_no_failure_and_no_alarm():
    offsets = [0, 0, 100, 100, 100, 100, 100, 0]
    (result,) = lead_times(RAMP_TRUE, [0] * 8, make_times(8), offsets, 1.0)
    assert result.outcome == "no_failure"
    assert result.alarm_time is None


# ---------------------------------------------------------------- how the failure is found


def test_negative_offsets_count_by_absolute_value():
    offsets = [0, 0, -300, -700, -1100, -1600, -2000, 0]
    alarms = [0, 0, 0, 0, 1, 1, 1, 0]
    (result,) = lead_times(RAMP_TRUE, alarms, make_times(8), offsets, 1.0)
    assert result.outcome == "in_time"
    assert result.failure_idx == 5
    assert result.lead_seconds == pytest.approx(1.0)


def test_offset_exactly_at_the_limit_counts_as_failure():
    y_true = [0, 1, 1, 0]
    offsets = [0, 100, 1500, 0]
    (result,) = lead_times(y_true, [0, 0, 0, 0], make_times(4), offsets, 1.0)
    assert result.outcome == "missed"
    assert result.failure_idx == 2


def test_offset_just_below_the_limit_is_not_a_failure():
    y_true = [0, 1, 1, 0]
    offsets = [0, 100, 1499.9, 0]
    (result,) = lead_times(y_true, [0, 0, 0, 0], make_times(4), offsets, 1.0)
    assert result.outcome == "no_failure"


def test_large_offsets_outside_the_attack_are_ignored():
    # Huge offsets before and after the attack must not create a failure.
    y_true = [0, 1, 1, 0, 0]
    offsets = [3000, 100, 200, 5000, 5000]
    (result,) = lead_times(y_true, [0, 0, 0, 0, 0], make_times(5), offsets, 1.0)
    assert result.outcome == "no_failure"


@pytest.mark.parametrize(
    "limit_ns, failure_idx, lead, outcome",
    [
        (1500, 5, 2.0, "in_time"),  # first alarm ends at 4.0 s, failure at 6.0 s
        (1000, 4, 1.0, "in_time"),  # offset 1100 at idx 4 -> failure at 5.0 s
        (500, 3, 0.0, "too_late"),  # offset 700 at idx 3  -> failure at 4.0 s
    ],
)
def test_stricter_limit_moves_the_failure_earlier(limit_ns, failure_idx, lead, outcome):
    alarms = [0, 0, 0, 1, 1, 1, 1, 0]  # first alarm idx 3 -> ends at 4.0 s
    (result,) = lead_times(
        RAMP_TRUE, alarms, make_times(8), RAMP_OFFSETS, 1.0, limit_ns=limit_ns
    )
    assert result.failure_idx == failure_idx
    assert result.outcome == outcome
    assert result.lead_seconds == pytest.approx(lead)


def test_lead_uses_real_seconds_not_window_counts():
    # w = 10 s: window end times 10..60. Failure idx 3 -> 40 s, alarm idx 2 -> 30 s.
    y_true = [0, 0, 1, 1, 1, 0]
    offsets = [0, 0, 500, 1600, 2000, 0]
    alarms = [0, 0, 1, 1, 1, 0]
    (result,) = lead_times(y_true, alarms, make_times(6, 10.0), offsets, 10.0)
    assert result.outcome == "in_time"
    assert result.failure_time == 40.0
    assert result.alarm_time == 30.0
    assert result.lead_seconds == pytest.approx(10.0)


# ---------------------------------------------------------------- sudden jumps


def test_sudden_jump_can_never_be_in_time():
    # The offset jumps past 1.5 µs inside the very first attack window.
    # Even a perfect detector alarms in that same window -> lead = 0 -> too late.
    y_true = [0, 0, 1, 1, 1, 0]
    offsets = [0, 0, 5000, 5000, 5000, 0]
    (result,) = lead_times(y_true, y_true, make_times(6), offsets, 1.0)
    assert result.failure_idx == 2
    assert result.outcome == "too_late"
    assert result.lead_seconds == pytest.approx(0.0)


def test_sudden_jump_with_debouncing_is_even_later():
    y_true = [0, 0, 1, 1, 1, 0]
    offsets = [0, 0, 5000, 5000, 5000, 0]
    alarms = debounce(y_true, n=2)  # first alarm idx 3 -> ends at 4.0 s
    (result,) = lead_times(y_true, alarms, make_times(6), offsets, 1.0)
    assert result.lead_seconds == pytest.approx(-1.0)


# ---------------------------------------------------------------- debounce <-> lead time


@pytest.mark.parametrize(
    "n, outcome, lead",
    [
        (1, "in_time", 3.0),
        (2, "in_time", 2.0),
        (3, "in_time", 1.0),
        (4, "too_late", 0.0),
        (5, "too_late", -1.0),
        (6, "missed", None),  # the attack is only 5 windows long
    ],
)
def test_each_extra_debounce_window_costs_one_second_of_lead(n, outcome, lead):
    # Perfect raw detector on the ramp. Lead = 4 - N seconds, until the run runs out.
    alarms = debounce(RAMP_TRUE, n)
    result = run_ramp(alarms)
    assert result.outcome == outcome
    if lead is None:
        assert result.lead_seconds is None
    else:
        assert result.lead_seconds == pytest.approx(lead)


# ---------------------------------------------------------------- several attacks and edge cases


def _four_event_run():
    # idx    : 0  1    2     3     4  5    6    7  8     9     10 11   12    13    14
    # y_true : 0  1    1     1     0  1    1    0  1     1     0  1    1     1     0
    # offset : 0  500  1600  2000  0  100  200  0  1600  2000  0  200  1600  2000  0
    # event A = idx 1..3   fails at idx 2 (t=3)   alarm idx 1 (t=2)   -> in_time, lead +1
    # event B = idx 5..6   never fails            alarm idx 6         -> no_failure
    # event C = idx 8..9   fails at idx 8 (t=9)   no alarm            -> missed
    # event D = idx 11..13 fails at idx 12 (t=13) alarm idx 13 (t=14) -> too_late, lead -1
    y_true = [0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0]
    offsets = [0, 500, 1600, 2000, 0, 100, 200, 0, 1600, 2000, 0, 200, 1600, 2000, 0]
    alarms = np.zeros(15, dtype=int)
    alarms[[1, 6, 13]] = 1
    return lead_times(y_true, alarms, make_times(15), offsets, 1.0)


def test_four_events_get_four_independent_outcomes():
    results = _four_event_run()
    assert [r.outcome for r in results] == ["in_time", "no_failure", "missed", "too_late"]
    assert results[0].lead_seconds == pytest.approx(1.0)
    assert results[1].lead_seconds is None
    assert results[2].lead_seconds is None
    assert results[3].lead_seconds == pytest.approx(-1.0)


def test_no_attacks_gives_empty_list():
    assert lead_times([0, 0, 0], [0, 1, 0], make_times(3), [0, 0, 0], 1.0) == []


def test_empty_input_gives_empty_list():
    assert lead_times([], [], [], [], 1.0) == []


# ---------------------------------------------------------------- summarize_lead_times


def test_summary_of_the_four_event_run():
    summary = summarize_lead_times(_four_event_run())
    assert summary["n_events"] == 4
    assert summary["n_no_failure"] == 1
    assert summary["n_failure_events"] == 3  # A, C, D reached the limit
    assert summary["n_in_time"] == 1
    assert summary["n_too_late"] == 1
    assert summary["n_missed"] == 1
    assert summary["protection_rate"] == pytest.approx(1 / 3)
    assert summary["mean_lead_s"] == pytest.approx(0.0)  # mean of +1 and -1
    assert summary["min_lead_s"] == pytest.approx(-1.0)


def test_summary_when_every_attack_is_caught_in_time():
    result = run_ramp([0, 0, 0, 0, 1, 1, 1, 0])
    summary = summarize_lead_times([result])
    assert summary["protection_rate"] == pytest.approx(1.0)
    assert summary["mean_lead_s"] == pytest.approx(1.0)
    assert summary["min_lead_s"] == pytest.approx(1.0)


def test_summary_when_no_attack_reached_the_limit():
    # Nothing to protect against: the rate is None, not a misleading 0 or 1.
    offsets = [0, 0, 100, 100, 100, 100, 100, 0]
    results = lead_times(RAMP_TRUE, [0] * 8, make_times(8), offsets, 1.0)
    summary = summarize_lead_times(results)
    assert summary["n_events"] == 1
    assert summary["n_no_failure"] == 1
    assert summary["n_failure_events"] == 0
    assert summary["protection_rate"] is None
    assert summary["mean_lead_s"] is None
    assert summary["min_lead_s"] is None


def test_summary_when_everything_is_missed():
    summary = summarize_lead_times([run_ramp([0] * 8)])
    assert summary["n_missed"] == 1
    assert summary["protection_rate"] == 0.0
    assert summary["mean_lead_s"] is None


def test_summary_with_no_events():
    summary = summarize_lead_times([])
    assert summary["n_events"] == 0
    assert summary["protection_rate"] is None
    assert summary["mean_lead_s"] is None


# ---------------------------------------------------------------- input validation


def test_offsets_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        lead_times([0, 1, 1], [0, 1, 1], [1, 2, 3], [0, 100], 1.0)


def test_offsets_must_be_1d():
    with pytest.raises(ValueError, match="1-D"):
        lead_times([0, 1], [0, 1], [1, 2], [[0, 1], [2, 3]], 1.0)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_offsets_must_be_finite(bad_value):
    with pytest.raises(ValueError, match="finite"):
        lead_times([0, 1, 1], [0, 1, 1], [1, 2, 3], [0, bad_value, 100], 1.0)


@pytest.mark.parametrize("bad_limit", [0, -1500])
def test_bad_limit_raises(bad_limit):
    with pytest.raises(ValueError, match="limit_ns"):
        lead_times([0, 1], [0, 1], [1, 2], [0, 100], 1.0, limit_ns=bad_limit)


def test_errors_from_the_events_module_still_apply():
    with pytest.raises(ValueError, match="length mismatch"):
        lead_times([0, 1, 1], [0, 1], [1, 2, 3], [0, 100, 200], 1.0)
    with pytest.raises(ValueError, match="0/1"):
        lead_times([0, 1, 1], [0, 2, 1], [1, 2, 3], [0, 100, 200], 1.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        lead_times([0, 1, 1], [0, 1, 1], [1, 1, 3], [0, 100, 200], 1.0)
    with pytest.raises(ValueError, match="window_seconds"):
        lead_times([0, 1, 1], [0, 1, 1], [1, 2, 3], [0, 100, 200], 0)