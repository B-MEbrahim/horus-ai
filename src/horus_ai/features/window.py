import numpy as np


def window_to_features(window: np.ndarray) -> np.ndarray:
    """PLACEHOLDER. Turns one window into a single feature vector.

    window: shape (n_samples, 3), columns = offset, freq, path_delay
    returns: 1D vector of 6 values (mean and std of each column)
    """
    return np.concatenate([window.mean(axis=0), window.std(axis=0)])