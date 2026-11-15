import numpy as np

# Order matters: the model learns "column 4 means offset_slope".
# Never reorder this without retraining.
FEATURE_NAMES: tuple[str, ...] = (
    "offset_mean",
    "offset_std",
    "offset_min",
    "offset_max",
    "offset_slope",      # ns per sample: steady drift
    "offset_max_step",   # biggest jump between neighbours: sudden jump
    "offset_diff_std",   # jitter of the step-to-step changes
    "freq_mean",
    "freq_std",
    "delay_mean",
    "delay_std",
)
N_FEATURES = len(FEATURE_NAMES)
N_COLUMNS = 3          # offset, freq, path_delay
MIN_WINDOW = 3         # need at least 2 steps to take a std of the steps


CROSS_FEATURE_NAMES: tuple[str, ...] = (
    "peer_delta_mean",      # how far am I from my peers, on average?
    "peer_delta_slope",     # is the gap growing steadily? (gradual drift)
    "peer_delta_max_step",  # did the gap suddenly jump? (sudden jump)
    "peer_delta_std",       # how noisy is the gap?
)
ALL_FEATURE_NAMES: tuple[str, ...] = FEATURE_NAMES + CROSS_FEATURE_NAMES
N_ALL_FEATURES = len(ALL_FEATURE_NAMES)


def _slope(y: np.ndarray) -> float:
    """Slope of the best straight line through y (units of y per sample)."""
    x = np.arange(len(y), dtype=float)
    x -= x.mean()
    return float((x * (y - y.mean())).sum() / (x * x).sum())


def window_to_features(window: np.ndarray) -> np.ndarray:
    """Turn one window into one feature vector.

    window: shape (n_samples, 3), columns = offset_ns, freq_ppb, path_delay_ns
    returns: 1D float vector of length N_FEATURES, ordered as FEATURE_NAMES
    """
    window = np.asarray(window, dtype=float)

    if window.ndim != 2 or window.shape[1] != N_COLUMNS:
        raise ValueError(f"window must have shape (n, {N_COLUMNS}), got {window.shape}")
    if window.shape[0] < MIN_WINDOW:
        raise ValueError(f"window needs at least {MIN_WINDOW} samples, got {window.shape[0]}")
    if not np.isfinite(window).all():
        raise ValueError("window contains NaN or inf")

    offset, freq, delay = window.T
    steps = np.diff(offset)

    return np.array(
        [
            offset.mean(),
            offset.std(),
            offset.min(),
            offset.max(),
            _slope(offset),
            np.abs(steps).max(),
            steps.std(),
            freq.mean(),
            freq.std(),
            delay.mean(),
            delay.std(),
        ]
    )


def cross_tower_features(own_offset: np.ndarray, peer_offsets: np.ndarray) -> np.ndarray:
    """Compare one tower's offset with the other towers at the same moments.

    own_offset:   shape (n,)    this tower's master_offset_ns
    peer_offsets: shape (n, k)  the other k towers' offsets, same moments
    returns: 1D vector of length len(CROSS_FEATURE_NAMES)
    """
    own_offset = np.asarray(own_offset, dtype=float)
    peer_offsets = np.asarray(peer_offsets, dtype=float)

    if own_offset.ndim != 1 or peer_offsets.ndim != 2:
        raise ValueError("own_offset must be (n,) and peer_offsets must be (n, k)")
    if peer_offsets.shape[0] != own_offset.shape[0] or peer_offsets.shape[1] < 1:
        raise ValueError(f"shape mismatch: {own_offset.shape} vs {peer_offsets.shape}")
    if own_offset.shape[0] < MIN_WINDOW:
        raise ValueError(f"need at least {MIN_WINDOW} samples")
    if not (np.isfinite(own_offset).all() and np.isfinite(peer_offsets).all()):
        raise ValueError("inputs contain NaN or inf")

    # Median of the peers, so one bad peer can't drag the reference with it.
    reference = np.median(peer_offsets, axis=1)
    delta = own_offset - reference
    steps = np.diff(delta)

    return np.array(
        [delta.mean(), _slope(delta), np.abs(steps).max(), delta.std()]
    )


def window_to_all_features(window: np.ndarray, peer_offsets: np.ndarray) -> np.ndarray:
    """Per-tower features (11) + cross-tower features (4) = 15 values."""
    window = np.asarray(window, dtype=float)
    own = window_to_features(window)                       # validates the window
    cross = cross_tower_features(window[:, 0], peer_offsets)
    return np.concatenate([own, cross])