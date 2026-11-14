import numpy as np

from horus_ai.data.attacks import (sudden_jump, gradual_drift,
                                   inject_attack, inject_attack_towers)
from horus_ai.data.config import AttackConfig
import pytest

T = np.arange(6, dtype=float)  # 0, 1, 2, 3, 4, 5


def make_cfg(**kwargs):
    return AttackConfig(attack_type="sudden_jump", start_s=2, jump_ns=100, **kwargs)


def test_jump_applies_from_start():
    added, labels = sudden_jump(T, make_cfg())
    assert added.tolist() == [0, 0, 100, 100, 100, 100]
    assert labels.tolist() == [0, 0, 1, 1, 1, 1]


def test_attack_stops_after_duration():
    added, labels = sudden_jump(T, make_cfg(duration_s=2))
    assert added.tolist() == [0, 0, 100, 100, 0, 0]
    assert labels.tolist() == [0, 0, 1, 1, 0, 0]


def test_start_after_end_of_data_means_no_attack():
    cfg = AttackConfig(attack_type="sudden_jump", start_s=100, jump_ns=100)
    added, labels = sudden_jump(T, cfg)
    assert not added.any()
    assert not labels.any()


def test_uses_time_not_row_position():
    t = np.array([0.0, 1.0, 4.0, 5.0])  # uneven spacing
    added, labels = sudden_jump(t, make_cfg())
    assert added.tolist() == [0, 0, 100, 100]
    assert labels.tolist() == [0, 0, 1, 1]


def make_drift_cfg(**kwargs):
    return AttackConfig(
    attack_type="gradual_drift", start_s=2, push_rate_ns_per_s=2, **kwargs
    )


def test_drift_ramps_from_start():
    added, labels = gradual_drift(T, make_drift_cfg())
    assert added.tolist() == [0, 0, 0, 2, 4, 6]
    assert labels.tolist() == [0, 0, 1, 1, 1, 1]


def test_drift_stops_after_duration():
    added, labels = gradual_drift(T, make_drift_cfg(duration_s=2))
    assert added.tolist() == [0, 0, 0, 2, 0, 0]
    assert labels.tolist() == [0, 0, 1, 1, 0, 0]


def test_drift_follows_time_not_row_position():
    t = np.array([0.0, 1.0, 4.0, 5.0])  # uneven spacing
    added, labels = gradual_drift(t, make_drift_cfg())
    assert added.tolist() == [0, 0, 4, 6]  # 2 ns/s * 2 s, 2 ns/s * 3 s
    assert labels.tolist() == [0, 0, 1, 1]


def test_default_drift_reaches_failure_threshold_in_750_s():
    cfg = AttackConfig(attack_type="gradual_drift")  # 2 ns/s, start at 3600 s
    t = np.array([3600.0, 3600.0 + 750.0])
    added, _ = gradual_drift(t, cfg)
    assert added.tolist() == [0, 1500]


def test_inject_adds_attack_and_leaves_input_untouched():
    clean = np.full(6, 10.0)
    r = inject_attack(clean, T, make_cfg())
    assert r.offsets_ns.tolist() == [10, 10, 110, 110, 110, 110]
    assert r.labels.tolist() == [0, 0, 1, 1, 1, 1]
    assert clean.tolist() == [10.0] * 6


def test_inject_picks_attack_from_attack_type():
    r = inject_attack(np.zeros(6), T, make_drift_cfg())
    assert r.added_ns.tolist() == [0, 0, 0, 2, 4, 6]


def test_disabled_attack_returns_clean_copy():
    clean = np.full(6, 10.0)
    r = inject_attack(clean, T, make_cfg(enabled=False))
    assert r.offsets_ns.tolist() == clean.tolist()
    assert not r.labels.any()


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        inject_attack(np.zeros(5), T, make_cfg())


def test_only_affected_towers_are_attacked():
    series = {0: (T, np.zeros(6)), 1: (T, np.zeros(6))}
    res = inject_attack_towers(series, make_cfg(affected_towers=[1]))
    assert not res[0].labels.any() and not res[0].added_ns.any()
    assert res[1].labels.tolist() == [0, 0, 1, 1, 1, 1]


def test_unknown_tower_raises():
    series = {0: (T, np.zeros(6))}
    with pytest.raises(ValueError):
        inject_attack_towers(series, make_cfg(affected_towers=[3]))