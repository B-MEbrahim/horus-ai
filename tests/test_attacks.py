import numpy as np

from horus_ai.data.attacks import sudden_jump
from horus_ai.data.config import AttackConfig

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