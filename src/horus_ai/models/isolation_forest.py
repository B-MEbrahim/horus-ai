from dataclasses import asdict

import numpy as np
from sklearn.ensemble import IsolationForest

from horus_ai.configs import IsolationForestConfig
from horus_ai.models.base import BaseDetector


class IsolationForestDetector(BaseDetector):
    name = "isolation_forest"

    def __init__(
            self,
            n_estimators: int = 200,
            max_samples: int | float | str = "auto",
            max_features: float = 1.0,
            threshold_quantile: float = 0.99,
            random_state: int = 42,
    ):
        super().__init__(threshold_quantile=threshold_quantile)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state
        self.model_: IsolationForest | None = None

    def _fit(self, X: np.ndarray) -> None:
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            random_state=self.random_state
        )
        self.model_.fit(X)

    def _score(self, X: np.ndarray) -> np.ndarray:
        # scikit-learn: higher score_samples = more normal.
        # We negate it so that higher = more anomalous (our rule).
        return -self.model_.score_samples(X)

    @classmethod
    def from_config(cls, cfg: IsolationForestConfig) -> "IsolationForestDetector":
        return cls(**asdict(cfg))
