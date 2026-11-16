import numpy as np
import pytest

from horus_ai.features.training_set import build_training_set
from horus_ai.features.window import N_ALL_FEATURES

N, T, W, S = 60, 4, 10, 5


def _fake(seed=0):
    rng = np.random.default_rng(seed)
    offset, freq, delay = (rng.normal(size=(N, T)) for _ in range(3))
    locked = np.ones((N, T), dtype=bool)
    attacked = np.zeros((N, T), dtype=bool)
    return offset, freq, delay, locked, attacked


def test_shapes_for_one_stretch():
    ts = build_training_set(*_fake(), window_size=W, stride=S)
    per_tower = (N - W) // S + 1                      # 11 windows
    assert ts.X.shape == (T * per_tower, N_ALL_FEATURES)
    assert len(ts.y) == len(ts.target) == len(ts.stretch) == len(ts.X)
    assert set(ts.stretch) == {0}


def test_no_window_spans_a_lock_loss():
    offset, freq, delay, locked, attacked = _fake()
    locked[30, 1] = False                             # tower 1 drops out at sample 30
    ts = build_training_set(offset, freq, delay, locked, attacked, window_size=W, stride=S)
    # stretches are [0, 30) and [31, 60): 5 + 4 windows per tower
    assert len(ts.X) == T * 9
    assert set(ts.stretch) == {0, 1}


def test_attack_label_only_on_attacked_target():
    offset, freq, delay, locked, attacked = _fake()
    attacked[20:30, 2] = True
    ts = build_training_set(offset, freq, delay, locked, attacked, window_size=W, stride=S)
    assert ts.y[ts.target == 2].any()
    assert not ts.y[ts.target != 2].any()


def test_no_usable_stretch_gives_empty_set():
    offset, freq, delay, locked, attacked = _fake()
    locked[:] = False
    ts = build_training_set(offset, freq, delay, locked, attacked, window_size=W)
    assert ts.X.shape == (0, N_ALL_FEATURES)


def test_shape_mismatch_raises():
    offset, freq, delay, locked, attacked = _fake()
    with pytest.raises(ValueError):
        build_training_set(offset, freq[:-1], delay, locked, attacked, window_size=W)