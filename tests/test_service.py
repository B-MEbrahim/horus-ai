import numpy as np

from horus_ai.api.schemas import TimingSample
from horus_ai.api.service import DetectionService
from horus_ai.features.window import N_ALL_FEATURES
from horus_ai.models.base import BaseDetector  # adjust import

DT = 0.125                      # one sample every 0.125 s (8 Hz)
T0 = 1000.0
TOWERS = ["OVS1", "OVS2", "OVS3"]


class DummyDetector(BaseDetector):
    name = "dummy"

    def _fit(self, X):
        pass

    def _score(self, X):
        return np.abs(X[:, 0])  # score = |offset_mean of the window| (feature 0)


def make_service(window_size=3, **kwargs):
    X = np.zeros((100, N_ALL_FEATURES))
    X[:, 0] = np.arange(100)  # "clean" scores 0..99, so threshold is about 98
    return DetectionService(DummyDetector().fit(X), window_size=window_size, **kwargs)


def make_sample(tower="OVS1", offset=0.0, state="s2", tick=0):
    return TimingSample(
        tower_id=tower, ts_wall=T0 + tick * DT, master_offset_ns=offset,
        freq_ppb=0.0, path_delay_ns=0.0, servo_state=state,
    )


def feed(svc, ticks, towers=TOWERS, offset_fn=lambda tower: 0.0):
    """Feed `ticks` ticks; within a tick the towers report one after another.

    Returns the last verdict of each tower.
    """
    last = {}
    for tick in range(ticks):
        for tower in towers:
            last[tower] = svc.process(make_sample(tower, offset_fn(tower), tick=tick))
    return last


def test_not_locked_at_start_is_warming_up():
    v = make_service().process(make_sample(state="s0"))
    assert v.status == "warming_up" and v.anomaly_score is None


def test_window_fills_then_scores():
    # 2 ticks: every window still has only 2 of 3 samples
    last = feed(make_service(window_size=3), ticks=2)
    assert all(v.status == "warming_up" and v.anomaly_score is None for v in last.values())

    # 4 ticks: all windows are full and aligned, so every tower is scored
    last = feed(make_service(window_size=3), ticks=4)
    assert all(v.status == "scoring" and v.anomaly_score is not None for v in last.values())


def test_normal_vs_attack():
    normal = feed(make_service(window_size=3), ticks=4)
    assert normal["OVS1"].is_spoofed is False

    attack = feed(
        make_service(window_size=3), ticks=4,
        offset_fn=lambda tower: 1000.0 if tower == "OVS1" else 0.0,
    )
    assert attack["OVS1"].is_spoofed is True
    assert attack["OVS2"].is_spoofed is False   # neighbours stay quiet


def test_lock_lost_then_window_restarts():
    svc = make_service(window_size=3)
    feed(svc, ticks=4)
    assert svc.process(make_sample("OVS1", state="s0", tick=4)).status == "lock_lost"
    assert svc.process(make_sample("OVS1", tick=5)).status == "warming_up"  # window was cleared


def test_towers_are_independent():
    svc = make_service(window_size=3)
    for tick in range(3):
        svc.process(make_sample(tower="OVS1", tick=tick))
    assert svc.process(make_sample(tower="OVS2", tick=3)).status == "warming_up"