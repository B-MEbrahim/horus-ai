from typing import Literal
from importlib.resources import files
from pathlib import Path
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_CONFIG_NAME = "data_default.yaml"


class NetworkConfig(BaseModel):
    """Network-level assumptions. Values here depend on network team decisions."""

    model_config = ConfigDict(extra="forbid")

    protocol: Literal["ptp", "ntp"] = "ptp"
    n_towers: int = Field(default=4, ge=1)
    sample_interval_s : float = Field(default=1.0, gt=0)
    threshold_ns: float = Field(default=1500.0, gt=0)

class BaselineConfig(BaseModel):
    """What a healthy (un-attacked) tower clock looks like."""

    model_config = ConfigDict(extra="forbid")

    initial_offset_std_ns: float = Field(default=100.0, ge=0)
    jitter_std_ns: float = Field(default=50.0, ge=0)
    drift_ppb_range: tuple[float, float] = (-10.0, 10.0)
    wander_std_ns: float = Field(default=1.0, ge=0)

    @field_validator("drift_ppb_range")
    @classmethod
    def check_drift_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        if v[0] > v[1]:
            raise ValueError("drift_ppb_range must be [low, high] with low <= high")
        return v

class AttackConfig(BaseModel):
    """The time-push attack injected into the baseline."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    attack_type: Literal["sudden_jump", "gradual_drift"] = "gradual_drift"
    start_s: float = Field(default=3600.0, ge=0)
    duration_s: float | None = Field(default=None, gt=0)
    affected_towers: list[int] = Field(default_factory=lambda: [0])
    jump_ns: float = Field(default=2000.0, gt=0)
    push_rate_ns_per_s: float = Field(default=2.0, gt=0)

    @field_validator("affected_towers")
    @classmethod
    def check_towers(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("affected_towers must not be empty")
        if any(t < 0 for t in v):
            raise ValueError("tower indices must be 0 or more")
        if len(set(v)) != len(v):
            raise ValueError("tower indices must be unique")
        return v

class ColumnNames(BaseModel):
    """Column names in the output file. Swap these to match the network team's format."""

    model_config = ConfigDict(extra="forbid")

    timestamp: str = "timestamp"
    tower_id: str = "tower_id"
    offset_ns: str = "offset_ns"
    label: str = "label"

    @model_validator(mode="after")
    def chechk_names(self) -> "ColumnNames":
        names = [self.timestamp, self.tower_id, self.offset_ns, self.label]
        if any(not n.strip() for n in names):
            raise ValueError("column names must not be empty")
        if len(set(names)) != len(names):
            raise ValueError("Column names must be unique")
        return self

class OutputConfig(BaseModel):
    """How the generated data is written to disk."""

    model_config = ConfigDict(extra="forbid")

    format: Literal["csv", "json"] = "csv"
    layout: Literal["long", "wide"] = "long"
    output_dir: str = "data/synthetic"
    file_stem: str = Field(default="synthetic_run", min_length=1)
    include_label: bool = True
    columns: ColumnNames = Field(default_factory=ColumnNames)

class DataGenConfig(BaseModel):
    """Top-level config: everything the data generator needs for one run."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    duration_s: float = Field(default=7200.0, gt=0)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    attack: AttackConfig = Field(default_factory=AttackConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @property
    def n_samples(self) -> int:
        """Number of reports per tower in one run."""
        return int(self.duration_s // self.network.sample_interval_s)

    @model_validator(mode="after")
    def check_cross_section_rules(self) -> "DataGenConfig":
        n = self.network.n_towers
        bad = [t for t in self.attack.affected_towers if t >= n]
        if bad:
            raise ValueError(
                f"affected_towers {bad} do not exist: only towers 0 to {n - 1} "
                f"are available (n_towers={n})"
            )
        if self.attack.enabled and self.attack.start_s >= self.duration_s:
            raise ValueError(
                "attack.start_s must be smaller than duration_s, "
                "otherwise the attack never happens"
            )
        if self.network.sample_interval_s > self.duration_s:
            raise ValueError("sample_interval_s must not be longer than duration_s")
        return self


def load_config(source: str | Path = DEFAULT_CONFIG_NAME) -> DataGenConfig:
    """Load and validate a generator config.

    `source` is either a path to a YAML file on disk, or the name of a YAML
    file packaged inside horus_ai/configs/.
    """
    path = Path(source)
    if path.is_file():
        text = path.read_text(encoding="utf-8")
    else:
        resource = files("horus_ai") / "configs" / str(source)
        if not resource.is_file():
            available = sorted(
                p.name for p in (files("horus_ai") / "configs").iterdir()
                if p.name.endswith((".yaml", ".yml"))
            )
            raise FileNotFoundError(
                f"Config '{source}' not found as a file or inside horus_ai/configs/. "
                f"Packaged configs: {available}"
            )
        text = resource.read_text(encoding="utf-8")

    data = yaml.safe_load(text)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"Config '{source}' must contain key: value pairs at the top level")

    return DataGenConfig(**data)