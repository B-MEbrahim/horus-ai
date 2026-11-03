import numpy as np

from horus_ai.api.schemas import TimingSample
from horus_ai.api.service import DetectionService
from horus_ai.models.base import BaseDetector  # adjust import


class DummyDetector(BaseDetector):
    name = "dummy"

    def _fit(self, X):
        pass

    def _score(self, X):
        return np.abs(X[:, 0])  # score = |mean offset of the window|


def make_service(window_size=3):
    X = np.zeros((100, 6))
    X[:, 0] = np.arange(100)  # "clean" scores 0..99, so threshold is about 98
    return DetectionService(DummyDetector().fit(X), window_size=window_size)


def make_sample(tower="OVS1", offset=0.0, state="s2"):
    return TimingSample(
        tower_id=tower, ts_wall=1.0, master_offset_ns=offset,
        freq_ppb=0.0, path_delay_ns=0.0, servo_state=state,
    )


def test_not_locked_at_start_is_warming_up():
    v = make_service().process(make_sample(state="s0"))
    assert v.status == "warming_up" and v.anomaly_score is None


def test_window_fills_then_scores():
    svc = make_service(window_size=3)
    assert svc.process(make_sample()).status == "warming_up"
    assert svc.process(make_sample()).status == "warming_up"
    v = svc.process(make_sample())
    assert v.status == "scoring" and v.anomaly_score is not None


def test_normal_vs_attack():
    svc = make_service(window_size=3)
    for _ in range(3):
        normal = svc.process(make_sample(offset=0))
    assert normal.is_spoofed is False

    svc = make_service(window_size=3)
    for _ in range(3):
        attack = svc.process(make_sample(offset=1000))
    assert attack.is_spoofed is True


def test_lock_lost_then_window_restarts():
    svc = make_service(window_size=3)
    for _ in range(3):
        svc.process(make_sample())
    assert svc.process(make_sample(state="s0")).status == "lock_lost"
    assert svc.process(make_sample()).status == "warming_up"  # window was cleared


def test_towers_are_independent():
    svc = make_service(window_size=3)
    for _ in range(3):
        svc.process(make_sample(tower="OVS1"))
    assert svc.process(make_sample(tower="OVS2")).status == "warming_up"