import numpy as np

from horus_ai.api.schemas import TimingSample
from horus_ai.api.service import DetectionService
from horus_ai.features.window import ALL_FEATURE_NAMES, N_ALL_FEATURES

DT = 0.125            # one sample every 0.125 s (8 Hz)
WINDOW = 10           # small window keeps the tests short
T0 = 1000.0
SLOPE_COL = ALL_FEATURE_NAMES.index("peer_delta_slope")


class StubDetector:
    """Scores a window by |peer_delta_slope|; flags it above 5 ns/sample."""

    threshold_ = 5.0

    def __init__(self):
        self.last_X = None

    def score(self, X):
        self.last_X = X
        return np.abs(X[:, SLOPE_COL])


def sample(tower, tick, offset=0.0, state="s2", ts_shift=0.0):
    return TimingSample(
        tower_id=tower,
        ts_wall=T0 + tick * DT + ts_shift,
        master_offset_ns=offset,
        freq_ppb=0.0,
        path_delay_ns=200.0,
        servo_state=state,
    )


def run(service, towers, ticks, offset_fn=lambda tower, tick: 0.0, ts_shift=None):
    """Feed `ticks` ticks; within a tick, towers report one after another."""
    ts_shift = ts_shift or {}
    last = {}
    for tick in range(ticks):
        for tower in towers:
            s = sample(tower, tick, offset_fn(tower, tick), ts_shift=ts_shift.get(tower, 0.0))
            last[tower] = service.process(s)
    return last


def make_service(**kwargs):
    return DetectionService(StubDetector(), window_size=WINDOW, **kwargs)


# ---------- status flow ----------

def test_warming_up_until_window_is_full():
    svc = make_service()
    last = run(svc, ["t0", "t1", "t2"], ticks=WINDOW - 1)
    assert all(v.status == "warming_up" for v in last.values())


def test_single_tower_has_no_peers():
    svc = make_service()
    last = run(svc, ["t0"], ticks=WINDOW + 2)
    assert last["t0"].status == "no_peers"
    assert last["t0"].anomaly_score is None


def test_aligned_towers_are_scored_with_all_features():
    svc = make_service()
    last = run(svc, ["t0", "t1", "t2"], ticks=WINDOW + 2)
    assert all(v.status == "scoring" for v in last.values())
    assert svc.detector.last_X.shape == (1, N_ALL_FEATURES)


# ---------- peers dropping out ----------

def test_peer_that_loses_lock_stops_counting():
    svc = make_service()
    towers = ["t0", "t1", "t2", "t3"]
    run(svc, towers, ticks=WINDOW + 2)

    # t3 loses lock first in this tick: its buffer is cleared.
    assert svc.process(sample("t3", 12, state="s0")).status == "lock_lost"
    # t0 still has two good peers (t1, t2): scored normally.
    assert svc.process(sample("t0", 12)).status == "scoring"
    svc.process(sample("t1", 12))

    # t2 loses lock too: only t1 is left, below min_peers=2.
    assert svc.process(sample("t2", 13, state="s0")).status == "lock_lost"
    assert svc.process(sample("t0", 13)).status == "no_peers"


def test_peer_with_misaligned_timestamps_is_ignored():
    # t2 reports timestamps 5 s off, so it is not covering the same moments.
    svc = make_service()
    last = run(svc, ["t0", "t1", "t2"], ticks=WINDOW + 2, ts_shift={"t2": 5.0})
    assert last["t0"].status == "no_peers"        # only t1 qualifies, need 2


def test_min_peers_is_configurable():
    svc = make_service(min_peers=1)
    last = run(svc, ["t0", "t1", "t2"], ticks=WINDOW + 2, ts_shift={"t2": 5.0})
    assert last["t0"].status == "scoring"         # t1 alone is now enough


def test_peer_one_tick_behind_is_accepted():
    # Requests arrive one after another, so peers are often one sample behind.
    svc = make_service()
    last = run(svc, ["t0", "t1", "t2"], ticks=WINDOW + 2)
    assert last["t0"].status == "scoring"


# ---------- the main goal: find WHICH tower is attacked ----------

def test_only_the_attacked_tower_is_flagged():
    svc = make_service()
    ramp = lambda tower, tick: 10.0 * tick if tower == "t0" else 0.0   # t0 drifts 10 ns/sample
    last = run(svc, ["t0", "t1", "t2", "t3"], ticks=WINDOW + 5, offset_fn=ramp)

    assert last["t0"].status == "scoring" and last["t0"].is_spoofed
    for healthy in ("t1", "t2", "t3"):
        assert last[healthy].status == "scoring"
        assert not last[healthy].is_spoofed       # the median ignores the one bad tower