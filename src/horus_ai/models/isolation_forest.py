from dataclasses import asdict
from typing import Literal, cast

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
        model = IsolationForest(
            n_estimators=self.n_estimators,
            # The runtime accepts integer sample counts; the installed
            # scikit-learn typing currently omits that accepted option.
            max_samples=cast(float | Literal["auto"], self.max_samples),
            max_features=self.max_features,
            random_state=self.random_state
        )
        model.fit(X)
        self.model_ = model

    def _score(self, X: np.ndarray) -> np.ndarray:
        # scikit-learn: higher score_samples = more normal.
        # We negate it so that higher = more anomalous (our rule).
        if self.model_ is None:
            raise RuntimeError(f"{self.name} is not fitted yet. Call fit() first.")
        return -self.model_.score_samples(X)

    @classmethod
    def from_config(cls, cfg: IsolationForestConfig) -> "IsolationForestDetector":
        return cls(**asdict(cfg))
