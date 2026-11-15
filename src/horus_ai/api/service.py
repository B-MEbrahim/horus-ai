from collections import deque

import numpy as np

from horus_ai.api.schemas import TimingSample, Verdict
from horus_ai.features.window import window_to_all_features
from horus_ai.models.base import BaseDetector


class DetectionService:
    def __init__(
        self,
        detector: BaseDetector,
        window_size: int = 30,
        min_peers: int = 2,
        max_skew_s: float = 0.15,   # a bit more than one sample period (0.125 s)
    ):
        self.detector = detector
        self.window_size = window_size
        self.min_peers = min_peers
        self.max_skew_s = max_skew_s
        self._windows: dict[str, deque] = {}
        self._was_locked: dict[str, bool] = {}

    def _peer_offsets(self, tid: str, own: deque) -> np.ndarray | None:
        """Offsets of the other towers over the same moments as `own`.

        Returns shape (window_size, k), or None if fewer than min_peers qualify.
        """
        columns = []
        for other_id, other in self._windows.items():
            if other_id == tid or len(other) < self.window_size:
                continue  # unknown, warming up, or lost lock: leave it out
            starts_together = abs(other[0][0] - own[0][0]) <= self.max_skew_s
            ends_together = abs(other[-1][0] - own[-1][0]) <= self.max_skew_s
            if starts_together and ends_together:
                columns.append([row[1] for row in other])  # row = (ts, offset, freq, delay)
        if len(columns) < self.min_peers:
            return None
        return np.array(columns).T

    def process(self, sample: TimingSample) -> Verdict:
        tid = sample.tower_id
        window = self._windows.setdefault(tid, deque(maxlen=self.window_size))

        # not locked: never let warm-up data reach the buffer
        if sample.servo_state != "s2":
            window.clear()
            status = "lock_lost" if self._was_locked.get(tid, False) else "warming_up"
            return Verdict(tower_id=tid, status=status)

        self._was_locked[tid] = True
        window.append(
            (sample.ts_wall, sample.master_offset_ns, sample.freq_ppb, sample.path_delay_ns)
        )

        if len(window) < self.window_size:
            return Verdict(tower_id=tid, status="warming_up")

        peers = self._peer_offsets(tid, window)
        if peers is None:
            return Verdict(tower_id=tid, status="no_peers")

        own = np.array([row[1:] for row in window])   # drop ts -> shape (n, 3)
        features = window_to_all_features(own, peers)
        score = float(self.detector.score(features.reshape(1, -1))[0])
        return Verdict(
            tower_id=tid,
            status="scoring",
            anomaly_score=score,
            is_spoofed=score > self.detector.threshold_,
        )