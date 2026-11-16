from typing import NamedTuple

import numpy as np

from horus_ai.features.dataset import build_feature_matrix
from horus_ai.features.stretches import find_locked_stretches
from horus_ai.features.window import N_ALL_FEATURES


class TrainingSet(NamedTuple):
    X: np.ndarray        # (n_rows, N_ALL_FEATURES)
    y: np.ndarray        # (n_rows,) bool, True = window contains an attacked sample
    target: np.ndarray   # (n_rows,) int, tower column the row was built for
    stretch: np.ndarray  # (n_rows,) int, which locked stretch it came from


def build_training_set(
    offset: np.ndarray,    # (n_samples, n_towers), aligned in time
    freq: np.ndarray,      # (n_samples, n_towers)
    delay: np.ndarray,     # (n_samples, n_towers)
    locked: np.ndarray,    # (n_samples, n_towers) bool, True where servo is s2
    attacked: np.ndarray,  # (n_samples, n_towers) bool, True where that tower is attacked
    window_size: int,
    stride: int = 1,
) -> TrainingSet:
    """Build the full feature table from one aligned recording."""
    shape = offset.shape
    for name, arr in [("freq", freq), ("delay", delay), ("locked", locked), ("attacked", attacked)]:
        if arr.shape != shape:
            raise ValueError(f"{name} has shape {arr.shape}, expected {shape}")
    n_towers = shape[1]

    X_parts, y_parts, target_parts, stretch_parts = [], [], [], []
    for stretch_id, (start, end) in enumerate(find_locked_stretches(locked, window_size)):
        for target in range(n_towers):
            X, y = build_feature_matrix(
                offset[start:end],
                freq[start:end],
                delay[start:end],
                target=target,
                labels=attacked[start:end, target],
                window_size=window_size,
                stride=stride,
            )
            X_parts.append(X)
            y_parts.append(y)
            target_parts.append(np.full(len(y), target))
            stretch_parts.append(np.full(len(y), stretch_id))

    if not X_parts:  # no stretch was long enough
        return TrainingSet(
            np.empty((0, N_ALL_FEATURES)), np.empty(0, dtype=bool),
            np.empty(0, dtype=int), np.empty(0, dtype=int),
        )
    return TrainingSet(
        np.vstack(X_parts), np.concatenate(y_parts),
        np.concatenate(target_parts), np.concatenate(stretch_parts),
    )