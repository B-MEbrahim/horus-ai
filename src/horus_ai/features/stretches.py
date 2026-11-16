import numpy as np


def find_locked_stretches(locked: np.ndarray, min_length: int) -> list[tuple[int, int]]:
    """Find continuous runs of samples where ALL towers are locked.

    locked: (n_samples, n_towers) bool, True where that tower is locked (s2).
            A 1D (n_samples,) array is also accepted (already combined).
    min_length: drop runs shorter than this (use window_size).
    Returns [(start, end), ...] with end exclusive, in time order.
    """
    locked = np.asarray(locked, dtype=bool)
    all_locked = locked.all(axis=1) if locked.ndim == 2 else locked

    # Pad with False on both sides so runs touching the edges are detected.
    padded = np.concatenate([[False], all_locked, [False]]).astype(int)
    edges = np.diff(padded)             # +1 = a run starts, -1 = a run ends
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_length]