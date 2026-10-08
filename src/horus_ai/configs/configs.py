from dataclasses import dataclass, asdict


@dataclass
class IsolationForestConfig:
    n_estimators: int = 200
    max_samples: int | float | str = "auto"
    max_features: float = 1.0
    threshold_quantile: float = 0.99
    random_state: int = 42
    