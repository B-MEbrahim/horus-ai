from abc import ABC, abstractmethod
from pathlib import Path
import joblib
import numpy as np


class BaseDetector(ABC):
    """Common interface for every anomaly detector."""

    name: str = "base"

    def __init__(self, threshold_quantile: float=0.99):
        # 0.99 -> at most ~1% of clean data gets flagged (target false positive rate)
        self.threshold_quantile = threshold_quantile
        self.threshold_: float | None = None
        self.is_fitted_: bool = False

    # ---- what each concrete model must implement ----
    @abstractmethod
    def _fit(self, X: np.ndarray) -> None:
        """Train the underlying model on clean data."""

    @abstractmethod
    def _score(self, X: np.ndarray) -> np.ndarray:
        """Return one anomaly score per row. HIGHER = MORE ANOMALOUS."""

    # ---- shared behaviour (written once, inherited by all models) ----
    def fit(self, X: np.ndarray) -> "BaseDetector":
        X = self._check_input(X)
        self._fit(X)
        train_scores = self._score(X)
        self.threshold_ = float(np.quantile(train_scores, self.threshold_quantile))
        self.is_fitted_ = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        return self._score(self._check_input(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """0 = authentic, 1 = spoofed."""
        return (self.score(X) > self.threshold_).astype(int)

    def save(self, path: str | Path) -> None:
        self._require_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "BaseDetector":
        return joblib.load(path)

    # ---- helpers ----
    def _check_input(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array (n_samples, n_features), got shape {X.shape}")
        return X

    def _require_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(f"{self.name} is not fitted yet. Call fit() first.")