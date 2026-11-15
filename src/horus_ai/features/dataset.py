import numpy as np

from horus_ai.features.window import N_ALL_FEATURES, window_to_all_features


def build_feature_matrix(
    offset: np.ndarray,    # (n_samples, n_towers), aligned in time
    freq: np.ndarray,      # (n_samples, n_towers)
    delay: np.ndarray,     # (n_samples, n_towers)
    target: int,           # column index of the tower we build features for
    labels: np.ndarray,    # (n_samples,) bool, True where the TARGET tower is attacked
    window_size: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window over one continuous stretch where ALL towers are locked.

    Returns X (n_windows, N_ALL_FEATURES) and y (n_windows,) using the rule
    'a window is attack if any sample inside it is attacked'.
    """
    n, n_towers = offset.shape
    peers = [c for c in range(n_towers) if c != target]
    own_all = np.column_stack([offset[:, target], freq[:, target], delay[:, target]])

    X, y = [], []
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        X.append(window_to_all_features(own_all[start:end], offset[start:end][:, peers]))
        y.append(bool(labels[start:end].any()))
    return np.array(X).reshape(-1, N_ALL_FEATURES), np.array(y)