import copy
from importlib.resources import files

import pytest
import yaml
from pydantic import ValidationError

from horus_ai.data.config import (
    AttackConfig,
    BaselineConfig,
    DataGenConfig,
    NetworkConfig,
    OutputConfig,
    load_config,
)

CONFIG_DIR = files("horus_ai") / "configs"
PACKAGED_YAMLS = sorted(
    p.name for p in CONFIG_DIR.iterdir() if p.name.endswith((".yaml", ".yml"))
)


@pytest.fixture
def default_dict():
    """The packaged default YAML as a plain dictionary, fresh for every test."""
    text = (CONFIG_DIR / "data_default.yaml").read_text(encoding="utf-8")
    return copy.deepcopy(yaml.safe_load(text))


# ---------- defaults and packaging ----------

def test_default_config_loads():
    cfg = load_config()
    assert cfg.n_samples == int(cfg.duration_s // cfg.network.sample_interval_s)
    assert cfg.attack.affected_towers  # not empty


@pytest.mark.parametrize("name", PACKAGED_YAMLS)
def test_every_packaged_yaml_loads(name):
    # Fails if a YAML is missing from the installed package or is invalid.
    load_config(name)


def test_default_yaml_is_among_packaged_files():
    assert "data_default.yaml" in PACKAGED_YAMLS


# ---------- single-section validation ----------

@pytest.mark.parametrize(
    "cls, kwargs",
    [
        (NetworkConfig, {"n_towers": 0}),
        (NetworkConfig, {"protocol": "gps"}),
        (NetworkConfig, {"protocoll": "ptp"}),            # typo in key
        (BaselineConfig, {"jitter_std_ns": -5}),
        (BaselineConfig, {"drift_ppb_range": (10, -10)}),  # low > high
        (AttackConfig, {"attack_type": "spoof"}),
        (AttackConfig, {"affected_towers": []}),
        (AttackConfig, {"affected_towers": [1, 1]}),
        (AttackConfig, {"affected_towers": [-1]}),
        (AttackConfig, {"push_rate_ns_per_s": 0}),
        (OutputConfig, {"format": "parquet"}),
        (OutputConfig, {"layout": "tall"}),
        (OutputConfig, {"file_stem": ""}),
        (OutputConfig, {"columns": {"timestamp": "t", "tower_id": "t"}}),  # duplicate
        (OutputConfig, {"columns": {"time": "t"}}),                         # unknown key
    ],
)
def test_invalid_section_values_are_rejected(cls, kwargs):
    with pytest.raises(ValidationError):
        cls(**kwargs)


def test_section_defaults_are_valid():
    for cls in (NetworkConfig, BaselineConfig, AttackConfig, OutputConfig):
        cls()


# ---------- cross-section rules ----------

def test_tower_index_must_exist(default_dict):
    default_dict["attack"]["affected_towers"] = [default_dict["network"]["n_towers"]]
    with pytest.raises(ValidationError):
        DataGenConfig(**default_dict)


def test_attack_must_start_inside_recording(default_dict):
    default_dict["duration_s"] = default_dict["attack"]["start_s"]
    with pytest.raises(ValidationError):
        DataGenConfig(**default_dict)


def test_disabled_attack_may_start_after_recording(default_dict):
    default_dict["attack"]["enabled"] = False
    default_dict["duration_s"] = default_dict["attack"]["start_s"]
    DataGenConfig(**default_dict)  # must not raise


def test_interval_longer_than_recording_is_rejected(default_dict):
    default_dict["network"]["sample_interval_s"] = default_dict["duration_s"] + 1
    with pytest.raises(ValidationError):
        DataGenConfig(**default_dict)


def test_top_level_typo_is_rejected(default_dict):
    default_dict["sede"] = 1
    with pytest.raises(ValidationError):
        DataGenConfig(**default_dict)


# ---------- loader behavior ----------

def test_missing_config_raises_and_lists_available():
    with pytest.raises(FileNotFoundError, match="data_default.yaml"):
        load_config("does_not_exist.yaml")


def test_load_from_disk_path(tmp_path):
    f = tmp_path / "custom.yaml"
    f.write_text("seed: 7\n", encoding="utf-8")
    cfg = load_config(f)
    assert cfg.seed == 7
    assert cfg.network.n_towers == NetworkConfig().n_towers  # defaults fill the rest


def test_empty_yaml_gives_all_defaults(tmp_path):
    f = tmp_path / "empty.yaml"
    f.write_text("", encoding="utf-8")
    assert load_config(f).seed == DataGenConfig().seed


def test_yaml_that_is_not_a_mapping_is_rejected(tmp_path):
    f = tmp_path / "list.yaml"
    f.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(f)