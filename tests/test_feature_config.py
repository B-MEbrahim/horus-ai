import pytest
from pydantic import ValidationError

from horus_ai.features.config import FeatureConfig, load_feature_config


def test_default_yaml_loads():
    cfg = load_feature_config()
    assert cfg.window_size == 30
    assert cfg.sample_rate_hz == 8.0


def test_window_seconds():
    assert FeatureConfig(window_size=30, sample_rate_hz=8.0).window_seconds == 3.75


def test_rejects_too_small_window():
    with pytest.raises(ValidationError):
        FeatureConfig(window_size=2)


def test_rejects_unknown_key():
    with pytest.raises(ValidationError):
        FeatureConfig(windw_size=30)


def test_is_immutable():
    cfg = FeatureConfig()
    with pytest.raises(ValidationError):
        cfg.window_size = 80