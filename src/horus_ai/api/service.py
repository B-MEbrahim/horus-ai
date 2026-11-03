from collections import deque

import numpy as np

from horus_ai.api.schemas import TimingSample, Verdict
from horus_ai.features.window import window_to_features
from horus_ai.models.base import BaseDetector  


class DetectionService:
    def __init__(self, detector: BaseDetector, window_size: int = 30):
        self.detector = detector
        self.window_size = window_size
        self._windows: dict[str, deque] = {}
        self._was_locked: dict[str, bool] = {}

    def process(self, sample: TimingSample) -> Verdict:
        tid = sample.tower_id
        window = self._windows.setdefault(tid, deque(maxlen=self.window_size))

        # not locked: never let warm-up data reach the buffer
        if sample.servo_state != "s2":
            window.clear()
            status = "lock_lost" if self._was_locked.get(tid, False) else "warming_up"
            return Verdict(tower_id=tid, status=status)

        self._was_locked[tid] = True
        window.append((sample.master_offset_ns, sample.freq_ppb, sample.path_delay_ns))

        if len(window) < self.window_size:
            return Verdict(tower_id=tid, status="warming_up")

        features = window_to_features(np.array(window))
        score = float(self.detector.score(features.reshape(1, -1))[0])
        return Verdict(
            tower_id=tid,
            status="scoring",
            anomaly_score=score,
            is_spoofed=score > self.detector.threshold_,
        )