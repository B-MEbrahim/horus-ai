import json
from pathlib import Path

import numpy as np

from horus_ai.configs import IsolationForestConfig
from horus_ai.data.config import AttackConfig
from horus_ai.features.config import FeatureConfig
from horus_ai.training.train import (
    TrainingConfig,
    default_tower_paths,
    load_training_artifact,
    resolve_tower_paths,
    train,
)


def _write_tower(path, tower_id, n=120):
    rows = []
    for index in range(n):
        rows.append(
            {
                "kind": "sample",
                "ts_mono": 100.0 + index * 0.125,
                "ts_wall": 1_000_000.0 + index * 0.125,
                "tower_id": f"ovs{tower_id + 1}",
                "master_offset_ns": float(tower_id * 10 + np.sin(index / 10)),
                "freq_ppb": float(tower_id),
                "path_delay_ns": 500.0 + tower_id,
                "servo_state": "s2",
            }
        )
    path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )
    return path


def test_train_saves_detector_and_metadata_sidecar(tmp_path):
    paths = {
        tower: _write_tower(tmp_path / f"tower-{tower}.jsonl", tower)
        for tower in range(3)
    }
    config = TrainingConfig(
        feature=FeatureConfig(window_size=5, sample_rate_hz=8.0),
        detector=IsolationForestConfig(
            n_estimators=10,
            threshold_quantile=0.95,
            random_state=0,
        ),
        threshold_grid_size=5,
        debounce_values=(1,),
        attack=AttackConfig(push_rate_ns_per_s=20.0),
        attack_types=("gradual_drift",),
        attack_towers=(0,),
        # The tiny fixture is only 15 seconds long, so one episode scales to
        # thousands per hour; use a budget that is meaningful for this fixture.
        false_alarm_budget_per_hour=10_000.0,
    )
    artifact_path = tmp_path / "model.joblib"
    phases = []

    result = train(paths, artifact_path, config=config, on_progress=phases.append)

    assert result.artifact_path == artifact_path
    assert result.metadata_path == tmp_path / "model.joblib.json"
    assert result.metadata["feature_names"]
    assert result.metadata["window_size"] == 5
    assert result.metadata["n_consecutive"] == 1
    assert result.test_reports
    assert len(phases) == 7

    detector, metadata = load_training_artifact(artifact_path)
    assert detector.name == "isolation_forest"
    assert detector.threshold_ == result.detector.threshold_
    assert metadata == result.metadata


def test_default_and_overridden_tower_paths():
    defaults = default_tower_paths()

    assert set(defaults) == {0, 1, 2, 3}
    assert defaults[0].name.startswith("ovs1_")
    assert defaults[3].name.startswith("ovs4_")

    override = resolve_tower_paths([(2, Path("custom-ovs3.jsonl"))])
    assert override[0] == defaults[0]
    assert override[2] == Path("custom-ovs3.jsonl")
