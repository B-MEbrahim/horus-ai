"""Tests for horus_ai.evaluation.events, using tiny hand-made arrays.

Unless stated otherwise: window_seconds = 1 and times = [1, 2, 3, ...],
so window i covers (i, i+1] and its END time is i + 1.
"""

import numpy as np
import pytest

from horus_ai.evaluation.debounce import debounce
from horus_ai.evaluation.events import (
    AttackEvent,
    count_false_alarm_episodes,
    detection_delays,
    false_alarm_episodes_per_hour,
    find_attack_events,
    summarize_detections,
)
from horus_ai.evaluation.metrics import false_alarms_per_hour


def make_times(n, window_seconds=1.0):
    """End time of each window: window_seconds, 2*window_seconds, ..."""
    return np.arange(1, n + 1) * window_seconds


# ---------------------------------------------------------------- find_attack_events


def test_find_two_events():
    # idx    : 0 1 2 3 4 5 6 7
    # y_true : 0 0 1 1 1 0 1 0
    # event A = idx 2..4, event B = idx 6..6
    y_true = [0, 0, 1, 1, 1, 0, 1, 0]
    events = find_attack_events(y_true, make_times(8), window_seconds=1.0)
    assert events == [
        AttackEvent(start_idx=2, end_idx=4, start_time=2.0, end_time=5.0),
        AttackEvent(start_idx=6, end_idx=6, start_time=6.0, end_time=7.0),
    ]


def test_no_attack_gives_no_events():
    assert find_attack_events([0, 0, 0], make_times(3), 1.0) == []


def test_empty_input_gives_no_events():
    assert find_attack_events([], [], 1.0) == []


def test_all_attack_is_one_event():
    events = find_attack_events([1, 1, 1], make_times(3), 1.0)
    assert events == [AttackEvent(0, 2, 0.0, 3.0)]


def test_attack_at_start_and_at_end_of_the_array():
    start = find_attack_events([1, 1, 0], make_times(3), 1.0)
    assert start == [AttackEvent(0, 1, 0.0, 2.0)]
    end = find_attack_events([0, 1, 1], make_times(3), 1.0)
    assert end == [AttackEvent(1, 2, 1.0, 3.0)]


def test_event_times_scale_with_window_seconds():
    # w = 10 s, window end times 10, 20, 30, 40
    y_true = [0, 1, 1, 0]
    events = find_attack_events(y_true, [10, 20, 30, 40], window_seconds=10.0)
    # first attack window is idx 1 -> it starts at 20 - 10 = 10 s
    assert events == [AttackEvent(1, 2, 10.0, 30.0)]


# ---------------------------------------------------------------- detection_delays


def test_delay_when_alarm_comes_one_window_late():
    # idx    : 0 1 2 3 4 5 6
    # y_true : 0 0 1 1 1 0 0     event idx 2..4, starts at t = 2
    # alarms : 0 0 0 1 1 0 0     first alarm idx 3, window ends at t = 4
    y_true = [0, 0, 1, 1, 1, 0, 0]
    alarms = [0, 0, 0, 1, 1, 0, 0]
    (d,) = detection_delays(y_true, alarms, make_times(7), 1.0)
    assert d.detected
    assert d.first_alarm_idx == 3
    assert d.first_alarm_time == 4.0
    assert d.delay_seconds == pytest.approx(2.0)  # 4 - 2
    assert d.event == AttackEvent(2, 4, 2.0, 5.0)


def test_alarm_in_the_very_first_attack_window_has_delay_of_one_window():
    # We only learn the verdict when the window ends, so the best possible
    # delay is exactly one window length.
    y_true = [0, 0, 1, 1, 1, 0, 0]
    alarms = [0, 0, 1, 1, 1, 0, 0]
    (d,) = detection_delays(y_true, alarms, make_times(7), 1.0)
    assert d.delay_seconds == pytest.approx(1.0)


def test_delay_uses_real_seconds_not_window_counts():
    # w = 10 s. Event starts at 20 s; first alarm window ends at 40 s -> 20 s.
    y_true = [0, 0, 1, 1, 0]
    alarms = [0, 0, 0, 1, 0]
    (d,) = detection_delays(y_true, alarms, [10, 20, 30, 40, 50], 10.0)
    assert d.delay_seconds == pytest.approx(20.0)


def test_missed_attack_has_no_delay_and_is_not_zero():
    y_true = [0, 1, 1, 0]
    alarms = [0, 0, 0, 0]
    (d,) = detection_delays(y_true, alarms, make_times(4), 1.0)
    assert not d.detected
    assert d.first_alarm_idx is None
    assert d.first_alarm_time is None
    assert d.delay_seconds is None


def test_alarm_before_the_attack_is_not_a_negative_delay():
    # idx    : 0 1 2 3 4 5 6
    # y_true : 0 1 1 1 0 0 0     event idx 1..3, starts at t = 1
    # alarms : 1 0 0 1 0 0 0     idx 0 is a false alarm (before the event)
    y_true = [0, 1, 1, 1, 0, 0, 0]
    alarms = [1, 0, 0, 1, 0, 0, 0]
    (d,) = detection_delays(y_true, alarms, make_times(7), 1.0)
    assert d.first_alarm_idx == 3  # idx 0 is ignored
    assert d.delay_seconds == pytest.approx(3.0)  # 4 - 1


def test_alarm_after_the_attack_ended_does_not_count_as_detection():
    y_true = [0, 1, 1, 0, 0]
    alarms = [0, 0, 0, 1, 0]
    (d,) = detection_delays(y_true, alarms, make_times(5), 1.0)
    assert not d.detected


def test_alarm_already_on_when_attack_starts_gives_one_window_delay():
    # Documented corner case: a false-alarm burst that runs into the attack.
    # y_true : 0 0 1 1 0
    # alarms : 0 1 1 1 0     alarm is on before the attack AND in its first window
    y_true = [0, 0, 1, 1, 0]
    alarms = [0, 1, 1, 1, 0]
    (d,) = detection_delays(y_true, alarms, make_times(5), 1.0)
    assert d.first_alarm_idx == 2
    assert d.delay_seconds == pytest.approx(1.0)


def test_two_events_one_caught_one_missed():
    # idx    : 0 1 2 3 4 5 6 7
    # y_true : 0 1 1 0 0 1 1 0
    # alarms : 0 0 1 0 0 0 0 0
    y_true = [0, 1, 1, 0, 0, 1, 1, 0]
    alarms = [0, 0, 1, 0, 0, 0, 0, 0]
    first, second = detection_delays(y_true, alarms, make_times(8), 1.0)
    assert first.detected
    assert first.delay_seconds == pytest.approx(2.0)  # starts at 1, alarm window ends at 3
    assert not second.detected


def test_no_events_gives_empty_list():
    assert detection_delays([0, 0, 0], [0, 1, 0], make_times(3), 1.0) == []


def test_empty_input_gives_empty_list():
    assert detection_delays([], [], [], 1.0) == []


# ---------------------------------------------------------------- debounce <-> delay link


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_debounce_n_adds_exactly_n_window_lengths_of_delay(n):
    # Perfect raw detector on an attack of 4 windows (idx 2..5), w = 2 s.
    # With debounce N, the first alarm window is idx 2 + (N-1),
    # so delay = N windows = N * 2 seconds.
    w = 2.0
    y_true = [0, 0, 1, 1, 1, 1, 0]
    alarms = debounce(y_true, n)
    (d,) = detection_delays(y_true, alarms, make_times(7, w), w)
    assert d.delay_seconds == pytest.approx(n * w)


def test_debounce_longer_than_the_attack_misses_it():
    y_true = [0, 0, 1, 1, 1, 1, 0]  # attack is 4 windows long
    alarms = debounce(y_true, n=5)
    (d,) = detection_delays(y_true, alarms, make_times(7), 1.0)
    assert not d.detected


# ---------------------------------------------------------------- summarize_detections


def test_summary_with_two_detected_events():
    # idx    : 0 1 2 3 4 5 6 7 8
    # y_true : 0 1 1 0 0 1 1 1 0     event A idx 1..2 (start 1), event B idx 5..7 (start 5)
    # alarms : 0 0 1 0 0 0 0 1 0     A: alarm ends at 3 -> 2 s, B: alarm ends at 8 -> 3 s
    y_true = [0, 1, 1, 0, 0, 1, 1, 1, 0]
    alarms = [0, 0, 1, 0, 0, 0, 0, 1, 0]
    summary = summarize_detections(detection_delays(y_true, alarms, make_times(9), 1.0))
    assert summary["n_events"] == 2
    assert summary["n_detected"] == 2
    assert summary["event_recall"] == pytest.approx(1.0)
    assert summary["mean_delay_s"] == pytest.approx(2.5)
    assert summary["median_delay_s"] == pytest.approx(2.5)
    assert summary["max_delay_s"] == pytest.approx(3.0)


def test_summary_ignores_missed_events_in_delay_stats():
    y_true = [0, 1, 1, 0, 0, 1, 1, 0]
    alarms = [0, 0, 1, 0, 0, 0, 0, 0]  # A caught with delay 2, B missed
    summary = summarize_detections(detection_delays(y_true, alarms, make_times(8), 1.0))
    assert summary["n_events"] == 2
    assert summary["n_detected"] == 1
    assert summary["event_recall"] == pytest.approx(0.5)
    assert summary["mean_delay_s"] == pytest.approx(2.0)  # NOT (2 + 0) / 2


def test_summary_when_everything_is_missed():
    y_true = [0, 1, 1, 0]
    alarms = [0, 0, 0, 0]
    summary = summarize_detections(detection_delays(y_true, alarms, make_times(4), 1.0))
    assert summary["n_events"] == 1
    assert summary["n_detected"] == 0
    assert summary["event_recall"] == 0.0
    assert summary["mean_delay_s"] is None
    assert summary["median_delay_s"] is None
    assert summary["max_delay_s"] is None


def test_summary_with_no_events():
    summary = summarize_detections([])
    assert summary["n_events"] == 0
    assert summary["n_detected"] == 0
    assert summary["event_recall"] == 0.0
    assert summary["mean_delay_s"] is None


# ---------------------------------------------------------------- false alarm episodes


def test_burst_of_false_alarms_is_one_episode():
    # alarms : 0 1 1 1 0 1 0 0  -> a burst of 3, then a single one = 2 episodes
    y_true = [0] * 8
    alarms = [0, 1, 1, 1, 0, 1, 0, 0]
    assert count_false_alarm_episodes(y_true, alarms) == 2


def test_no_false_alarms_gives_zero_episodes():
    assert count_false_alarm_episodes([0, 0, 1, 1], [0, 0, 1, 1]) == 0


def test_alarms_inside_an_attack_are_not_false_alarms():
    # y_true : 0 1 1 0
    # alarms : 1 1 1 1     false-alarm mask = [1, 0, 0, 1] -> two separate episodes
    assert count_false_alarm_episodes([0, 1, 1, 0], [1, 1, 1, 1]) == 2


def test_false_alarm_burst_stops_where_the_attack_begins():
    # y_true : 0 0 1 1
    # alarms : 0 1 1 1     only idx 1 is false
    assert count_false_alarm_episodes([0, 0, 1, 1], [0, 1, 1, 1]) == 1


def test_episodes_per_hour_counts_a_burst_once():
    # 4 normal windows x 900 s = 1 hour. Two false windows in a row = 1 episode.
    y_true = [0, 0, 0, 0]
    alarms = [1, 1, 0, 0]
    assert false_alarm_episodes_per_hour(y_true, alarms, 900) == pytest.approx(1.0)
    # The per-window version from metrics.py counts the same burst twice:
    assert false_alarms_per_hour(y_true, alarms, 900) == pytest.approx(2.0)


def test_episodes_per_hour_with_two_separate_episodes():
    assert false_alarm_episodes_per_hour([0, 0, 0, 0], [1, 0, 1, 0], 900) == pytest.approx(2.0)


def test_episodes_per_hour_excludes_attack_time():
    # 2 normal windows x 1800 s = 1 hour of normal time; 1 false episode (idx 0).
    y_true = [0, 0, 1, 1]
    alarms = [1, 0, 1, 1]
    assert false_alarm_episodes_per_hour(y_true, alarms, 1800) == pytest.approx(1.0)


def test_episodes_per_hour_with_no_normal_windows():
    assert false_alarm_episodes_per_hour([1, 1], [1, 1], 60) == 0.0


# ---------------------------------------------------------------- input validation


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        detection_delays([0, 1, 1], [0, 1], [1, 2, 3], 1.0)
    with pytest.raises(ValueError, match="length mismatch"):
        detection_delays([0, 1, 1], [0, 1, 1], [1, 2], 1.0)
    with pytest.raises(ValueError, match="length mismatch"):
        find_attack_events([0, 1, 1], [1, 2], 1.0)
    with pytest.raises(ValueError, match="length mismatch"):
        count_false_alarm_episodes([0, 1], [0, 1, 1])


def test_non_binary_alarms_raise():
    with pytest.raises(ValueError, match="0/1"):
        detection_delays([0, 1, 1], [0, 2, 1], [1, 2, 3], 1.0)


def test_times_must_be_strictly_increasing():
    with pytest.raises(ValueError, match="strictly increasing"):
        detection_delays([0, 1, 1, 0], [0, 0, 1, 0], [1, 2, 2, 4], 1.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        detection_delays([0, 1, 1, 0], [0, 0, 1, 0], [4, 3, 2, 1], 1.0)


def test_times_must_be_1d():
    with pytest.raises(ValueError, match="1-D"):
        detection_delays([0, 1], [0, 1], [[1, 2], [3, 4]], 1.0)


@pytest.mark.parametrize("bad_w", [0, -1.0])
def test_bad_window_seconds_raises(bad_w):
    with pytest.raises(ValueError, match="window_seconds"):
        find_attack_events([0, 1], [1, 2], bad_w)
    with pytest.raises(ValueError, match="window_seconds"):
        detection_delays([0, 1], [0, 1], [1, 2], bad_w)


@pytest.mark.parametrize("bad_s", [0, -900])
def test_bad_seconds_per_window_raises(bad_s):
    with pytest.raises(ValueError, match="seconds_per_window"):
        false_alarm_episodes_per_hour([0, 1], [0, 1], bad_s)