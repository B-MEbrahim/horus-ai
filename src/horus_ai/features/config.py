from importlib import resources
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class FeatureConfig(BaseModel):
    """Windowing settings shared by training and serving."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # At least 5 samples: std and trend are meaningless on 1-2 points.
    window_size: int = Field(default=30, ge=5)
    sample_rate_hz: float = Field(default=8.0, gt=0)

    @property
    def window_seconds(self) -> float:
        """How much real time one window covers."""
        return self.window_size / self.sample_rate_hz


def load_feature_config(path: Path | None = None) -> FeatureConfig:
    """Load from `path`, or from the packaged default YAML if None."""
    if path is None:
        resource = resources.files("horus_ai").joinpath(
            "configs", "features_default.yaml"
        )
        text = resource.read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    return FeatureConfig(**(yaml.safe_load(text) or {}))