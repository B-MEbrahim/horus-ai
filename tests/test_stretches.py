import numpy as np

from horus_ai.features.stretches import find_locked_stretches


def test_gap_splits_into_two_stretches():
    # 10 samples, 4 towers. Tower B (column 1) loses lock at samples 4 and 5.
    locked = np.ones((10, 4), dtype=bool)
    locked[4:6, 1] = False
    assert find_locked_stretches(locked, min_length=1) == [(0, 4), (6, 10)]


def test_always_locked_is_one_stretch():
    assert find_locked_stretches(np.ones((10, 4), dtype=bool), 1) == [(0, 10)]


def test_never_locked_is_empty():
    assert find_locked_stretches(np.zeros((10, 4), dtype=bool), 1) == []


def test_short_stretches_are_dropped():
    locked = np.ones((10, 4), dtype=bool)
    locked[4:6, 1] = False
    # Both stretches are 4 samples long, so min_length=5 drops both.
    assert find_locked_stretches(locked, min_length=5) == []


def test_two_towers_losing_lock_at_different_times():
    locked = np.ones((10, 3), dtype=bool)
    locked[2, 0] = False
    locked[5, 2] = False
    assert find_locked_stretches(locked, 1) == [(0, 2), (3, 5), (6, 10)]