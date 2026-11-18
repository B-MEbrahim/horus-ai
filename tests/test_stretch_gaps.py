import numpy as np
import pytest

from horus_ai.features.stretches import find_stretches, split_runs_at_gaps

PERIOD = 0.125  # 8 Hz


def _ts(n):
    return np.arange(n) * PERIOD


def test_no_gaps_and_no_mask_gives_one_run():
    assert find_stretches(_ts(100), min_length=30) == [(0, 100)]


def test_gap_over_the_limit_splits():
    ts = np.concatenate([_ts(50), _ts(50) + 50 * PERIOD + 0.1])  # jump = 0.225 s
    assert find_stretches(ts, min_length=30, max_gap_s=0.2) == [(0, 50), (50, 100)]


def test_gap_under_the_limit_does_not_split():
    ts = np.concatenate([_ts(50), _ts(50) + 50 * PERIOD + 0.05])  # jump = 0.175 s
    assert find_stretches(ts, min_length=30, max_gap_s=0.2) == [(0, 100)]


def test_short_pieces_are_dropped_after_a_split():
    ts = np.concatenate([_ts(50), _ts(10) + 50 * PERIOD + 1.0])
    assert find_stretches(ts, min_length=30) == [(0, 50)]


def test_lock_mask_and_gap_combine():
    ts = _ts(200)
    ts[150:] += 1.0  # time hole before row 150
    locked = np.ones((200, 4), dtype=bool)
    locked[100, 2] = False  # tower 2 unlocked at row 100
    got = find_stretches(ts, min_length=30, locked=locked)
    assert got == [(0, 100), (101, 150), (150, 200)]


def test_split_runs_only_cuts_inside_the_given_runs():
    ts = np.concatenate([_ts(40), _ts(40) + 40 * PERIOD + 1.0])
    assert split_runs_at_gaps([(0, 40)], ts, 0.2, 30) == [(0, 40)]


def test_unsorted_timestamps_raise():
    ts = _ts(100)
    ts[40], ts[41] = ts[41], ts[40]
    with pytest.raises(ValueError):
        find_stretches(ts, min_length=30)


def test_mask_length_must_match_timestamps():
    with pytest.raises(ValueError):
        find_stretches(_ts(10), min_length=5, locked=np.ones(9, dtype=bool))


def test_empty_input_gives_no_stretches():
    assert find_stretches(np.array([]), min_length=30) == []
