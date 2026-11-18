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


def split_runs_at_gaps(
    runs: list[tuple[int, int]],
    ts: np.ndarray,
    max_gap_s: float,
    min_length: int,
) -> list[tuple[int, int]]:
    """Cut each run wherever two consecutive timestamps are too far apart.

    runs: [(start, end), ...] with end exclusive (e.g. from find_locked_stretches).
    ts: (n_samples,) timestamps in seconds, sorted. A jump bigger than
        max_gap_s between row i-1 and row i starts a new piece at row i.
    min_length: drop pieces shorter than this (use window_size).
    Returns [(start, end), ...] with end exclusive, in time order.
    """
    ts = np.asarray(ts, dtype=float)
    if np.any(np.diff(ts) < 0):
        raise ValueError("timestamps must be sorted in increasing order")

    pieces = []
    for start, end in runs:
        # diff index j is the jump between rows start+j and start+j+1,
        # so a cut belongs at row start+j+1.
        cuts = np.flatnonzero(np.diff(ts[start:end]) > max_gap_s) + 1 + start
        bounds = [int(start), *(int(c) for c in cuts), int(end)]
        pieces += [
            (a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= min_length
        ]
    return pieces


def find_stretches(
    ts: np.ndarray,
    min_length: int,
    max_gap_s: float = 0.2,
    locked: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Continuous, all-locked, gap-free runs of rows: the ones windows may use.

    ts: (n_samples,) timestamps in seconds, sorted.
    min_length: drop runs shorter than this (use window_size).
    max_gap_s: a time jump bigger than this between rows cuts the run.
    locked: optional (n_samples,) or (n_samples, n_towers) bool mask, True where
            locked. None means every row counts as locked (the loader already
            keeps only s2 samples).
    Returns [(start, end), ...] with end exclusive, in time order.
    """
    ts = np.asarray(ts, dtype=float)
    if locked is None:
        locked = np.ones(len(ts), dtype=bool)
    locked = np.asarray(locked, dtype=bool)
    if len(locked) != len(ts):
        raise ValueError(
            f"locked has {len(locked)} rows but ts has {len(ts)} timestamps"
        )

    runs = find_locked_stretches(locked, min_length)
    return split_runs_at_gaps(runs, ts, max_gap_s, min_length)
