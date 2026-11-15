import numpy as np
import pytest

from horus_ai.features.dataset import build_feature_matrix
from horus_ai.features.window import (
    ALL_FEATURE_NAMES,
    CROSS_FEATURE_NAMES,
    FEATURE_NAMES,
    N_ALL_FEATURES,
    cross_tower_features,
    window_to_all_features,
    window_to_features,
)


# ---------- helpers ----------

def make_window(offset, freq=5.0, delay=200.0):
    """Build an (n, 3) window from an offset array and constant freq/delay."""
    offset = np.asarray(offset, dtype=float)
    n = len(offset)
    return np.column_stack([offset, np.full(n, freq), np.full(n, delay)])


def as_dict(vec, names):
    return dict(zip(names, vec))


# ---------- window_to_features (per-tower) ----------

def test_per_tower_output_length_and_order():
    vec = window_to_features(make_window(np.arange(20)))
    assert vec.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(vec).all()


def test_flat_window():
    f = as_dict(window_to_features(make_window(np.full(20, 100.0))), FEATURE_NAMES)
    assert f["offset_mean"] == pytest.approx(100.0)
    assert f["offset_std"] == pytest.approx(0.0)
    assert f["offset_slope"] == pytest.approx(0.0)
    assert f["offset_max_step"] == pytest.approx(0.0)
    assert f["freq_mean"] == pytest.approx(5.0)
    assert f["delay_mean"] == pytest.approx(200.0)


def test_perfect_ramp():
    # offsets 0, 10, 20, ..., 190
    f = as_dict(window_to_features(make_window(10.0 * np.arange(20))), FEATURE_NAMES)
    assert f["offset_slope"] == pytest.approx(10.0)
    assert f["offset_max_step"] == pytest.approx(10.0)
    assert f["offset_diff_std"] == pytest.approx(0.0)   # every step is identical
    assert f["offset_min"] == pytest.approx(0.0)
    assert f["offset_max"] == pytest.approx(190.0)
    assert f["offset_mean"] == pytest.approx(95.0)


def test_single_step_jump():
    offset = np.concatenate([np.zeros(10), np.full(10, 500.0)])
    f = as_dict(window_to_features(make_window(offset)), FEATURE_NAMES)
    assert f["offset_max_step"] == pytest.approx(500.0)


def test_negative_jump_counts_as_jump():
    offset = np.concatenate([np.zeros(10), np.full(10, -500.0)])
    f = as_dict(window_to_features(make_window(offset)), FEATURE_NAMES)
    assert f["offset_max_step"] == pytest.approx(500.0)   # absolute value


def test_per_tower_rejects_bad_input():
    with pytest.raises(ValueError):
        window_to_features(np.zeros((10, 2)))             # wrong number of columns
    with pytest.raises(ValueError):
        window_to_features(np.zeros(10))                  # not 2D
    with pytest.raises(ValueError):
        window_to_features(make_window([1.0, 2.0]))       # only 2 samples
    bad = make_window(np.arange(10))
    bad[3, 0] = np.nan
    with pytest.raises(ValueError):
        window_to_features(bad)


# ---------- cross_tower_features ----------

def test_cross_names_and_length():
    vec = cross_tower_features(np.zeros(20), np.zeros((20, 3)))
    assert vec.shape == (len(CROSS_FEATURE_NAMES),)


def test_cross_flat_same_level_gives_zeros():
    # Own tower and all peers sit at the same level: no difference at all.
    own = np.full(20, 50.0)
    peers = np.full((20, 3), 50.0)
    assert np.allclose(cross_tower_features(own, peers), 0.0)


def test_cross_ramp_against_flat_peers():
    own = 10.0 * np.arange(20)
    peers = np.zeros((20, 3))
    f = as_dict(cross_tower_features(own, peers), CROSS_FEATURE_NAMES)
    assert f["peer_delta_slope"] == pytest.approx(10.0)
    assert f["peer_delta_mean"] == pytest.approx(95.0)
    assert f["peer_delta_max_step"] == pytest.approx(10.0)
    assert f["peer_delta_std"] == pytest.approx(own.std())


def test_cross_ignores_one_bad_peer():
    # Own tower is flat. One of three peers drifts away; the median ignores it.
    own = np.zeros(20)
    peers = np.zeros((20, 3))
    peers[:, 0] = 10.0 * np.arange(20)
    assert np.allclose(cross_tower_features(own, peers), 0.0)


def test_cross_rejects_bad_input():
    with pytest.raises(ValueError):
        cross_tower_features(np.zeros(20), np.zeros((19, 3)))    # length mismatch
    with pytest.raises(ValueError):
        cross_tower_features(np.zeros(20), np.zeros(20))         # peers not 2D
    with pytest.raises(ValueError):
        cross_tower_features(np.zeros(20), np.zeros((20, 0)))    # no peers
    with pytest.raises(ValueError):
        cross_tower_features(np.zeros(2), np.zeros((2, 3)))      # too short
    own = np.zeros(20)
    own[5] = np.nan
    with pytest.raises(ValueError):
        cross_tower_features(own, np.zeros((20, 3)))


# ---------- combiner ----------

def test_combined_is_per_tower_then_cross():
    window = make_window(10.0 * np.arange(20))
    peers = np.zeros((20, 3))
    combined = window_to_all_features(window, peers)

    assert combined.shape == (N_ALL_FEATURES,)
    assert np.allclose(combined[: len(FEATURE_NAMES)], window_to_features(window))
    assert np.allclose(combined[len(FEATURE_NAMES):], cross_tower_features(window[:, 0], peers))


def test_feature_names_are_unique():
    assert len(set(ALL_FEATURE_NAMES)) == len(ALL_FEATURE_NAMES)
    assert N_ALL_FEATURES == len(FEATURE_NAMES) + len(CROSS_FEATURE_NAMES)


# ---------- build_feature_matrix (offline builder) ----------

def make_fleet(n=100, n_towers=4, attack_start=60):
    """Four clean towers; tower 0 gets a ramp from sample `attack_start` on."""
    offset = np.zeros((n, n_towers))
    freq = np.zeros((n, n_towers))
    delay = np.zeros((n, n_towers))
    offset[attack_start:, 0] = 10.0 * np.arange(n - attack_start)
    labels = np.arange(n) >= attack_start
    return offset, freq, delay, labels


def test_builder_shapes_and_labels():
    offset, freq, delay, labels = make_fleet()
    X, y = build_feature_matrix(offset, freq, delay, target=0, labels=labels,
                                window_size=10, stride=5)
    # window starts: 0, 5, ..., 90 -> 19 windows
    assert X.shape == (19, N_ALL_FEATURES)
    assert y.shape == (19,)
    # a window is attack if ANY sample is attacked: windows starting at 55..90
    assert y.sum() == 8
    assert not y[:11].any()


def test_builder_attacked_tower_sees_it_neighbour_does_not():
    offset, freq, delay, labels = make_fleet()
    no_attack = np.zeros(len(labels), dtype=bool)
    slope_col = ALL_FEATURE_NAMES.index("peer_delta_slope")

    X0, _ = build_feature_matrix(offset, freq, delay, 0, labels, window_size=10, stride=5)
    X1, _ = build_feature_matrix(offset, freq, delay, 1, no_attack, window_size=10, stride=5)

    # attacked tower: last window (samples 90..99) is a clean ramp of slope 10
    assert X0[-1, slope_col] == pytest.approx(10.0)
    # neighbour: the median of its peers ignores tower 0, so it sees nothing
    n_per_tower = len(FEATURE_NAMES)
    assert np.allclose(X1[:, n_per_tower:], 0.0)


def test_builder_too_short_returns_empty():
    offset, freq, delay, labels = make_fleet(n=5, attack_start=3)
    X, y = build_feature_matrix(offset, freq, delay, 0, labels, window_size=10, stride=1)
    assert X.shape == (0, N_ALL_FEATURES)
    assert y.shape == (0,)