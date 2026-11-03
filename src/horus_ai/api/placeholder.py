import numpy as np

from horus_ai.models.base import BaseDetector  


class PlaceholderDetector(BaseDetector):
    """Stand-in used until a trained model exists. Never flags anything."""

    name = "placeholder"

    def _fit(self, X: np.ndarray) -> None:
        pass

    def _score(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X))