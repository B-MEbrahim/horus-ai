"""Train and evaluate the first HORUS detector artifact.

The pipeline keeps the time order of the recording:

1. load and align the tower recordings;
2. split every continuous stretch into train, validation, and test pieces;
3. fit Isolation Forest on clean train windows only;
4. choose one threshold/debounce operating point on validation attacks;
5. evaluate that operating point on held-out synthetic attacks;
6. save the detector and JSON metadata side by side.

Example:

    python -m horus_ai.training.train \
        --tower 0=datasets/clean/ovs1_20261115T133840Z_onehour.jsonl \
        --tower 1=datasets/clean/ovs2_20261115T133840Z_onehour.jsonl \
        --tower 2=datasets/clean/ovs3_20261115T133840Z_onehour.jsonl \
        --tower 3=datasets/clean/ovs4_20261115T133840Z_onehour.jsonl \
        --output artifacts/isolation_forest.joblib
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from horus_ai.data.align import align_towers
from horus_ai.data.attacks import inject_attack
from horus_ai.data.config import AttackConfig
from horus_ai.data.loader import load_tower_frames
from horus_ai.evaluation import (
    evaluate_run,
    select_operating_point,
    sweep,
    threshold_grid,
)
from horus_ai.features.config import FeatureConfig
from horus_ai.features.dataset import build_feature_matrix
from horus_ai.features.stretches import find_stretches
from horus_ai.features.training_set import TrainingSet, build_training_set
from horus_ai.features.window import ALL_FEATURE_NAMES, N_ALL_FEATURES
from horus_ai.models.base import BaseDetector
from horus_ai.models.isolation_forest import IsolationForestDetector
from horus_ai.configs import IsolationForestConfig


AttackType = Literal["sudden_jump", "gradual_drift"]


class TrainingConfig(BaseModel):
    """Configuration for one reproducible offline training run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature: FeatureConfig = Field(default_factory=FeatureConfig)
    detector: IsolationForestConfig = Field(default_factory=IsolationForestConfig)
    stride: int = Field(default=1, ge=1)
    train_fraction: float = Field(default=0.5, gt=0, lt=1)
    validation_fraction: float = Field(default=0.2, gt=0, lt=1)
    align_tolerance_s: float = Field(default=0.15, gt=0)
    max_reused_frac: float = Field(default=0.01, ge=0, lt=1)
    max_gap_s: float = Field(default=0.2, gt=0)
    false_alarm_budget_per_hour: float = Field(default=1.0, ge=0)
    threshold_grid_size: int = Field(default=25, ge=2)
    debounce_values: tuple[int, ...] = (1, 2, 3, 5)
    attack: AttackConfig = Field(default_factory=AttackConfig)
    attack_types: tuple[AttackType, ...] = ("sudden_jump", "gradual_drift")
    attack_towers: tuple[int, ...] | None = None
    validation_attack_fraction: float = Field(default=0.5, ge=0, lt=1)

    @model_validator(mode="after")
    def validate_split_and_debounce(self) -> "TrainingConfig":
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("train_fraction + validation_fraction must be < 1")
        if not self.debounce_values:
            raise ValueError("debounce_values must not be empty")
        if any(value < 1 for value in self.debounce_values):
            raise ValueError("debounce_values must contain only positive integers")
        if len(set(self.debounce_values)) != len(self.debounce_values):
            raise ValueError("debounce_values must not contain duplicates")
        if not self.attack_types:
            raise ValueError("attack_types must not be empty")
        return self


@dataclass(frozen=True)
class AlignedRecording:
    """Numeric arrays extracted from one aligned recording."""

    timestamps_s: np.ndarray
    offsets_ns: np.ndarray
    freq_ppb: np.ndarray
    path_delay_ns: np.ndarray
    tower_ids: tuple[int, ...]


@dataclass(frozen=True)
class WindowRun:
    """Feature windows and ground truth for one evaluation scenario."""

    X: np.ndarray
    y: np.ndarray
    times_s: np.ndarray
    offsets_ns: np.ndarray


@dataclass(frozen=True)
class TrainingResult:
    """Outputs from a completed training run."""

    artifact_path: Path
    metadata_path: Path
    detector: BaseDetector
    metadata: dict[str, Any]
    validation_report: dict[str, Any]
    test_reports: dict[str, dict[str, Any]]


def load_aligned_recording(
    paths: Mapping[int, str | Path],
    *,
    tolerance_s: float = 0.15,
    max_reused_frac: float = 0.01,
) -> AlignedRecording:
    """Load tower files and convert their aligned frame to numeric arrays."""
    if len(paths) < 2:
        raise ValueError("at least two tower recordings are required")

    normalized_paths = {int(tower): Path(path) for tower, path in paths.items()}
    frames = load_tower_frames(normalized_paths)
    aligned = align_towers(
        frames,
        tolerance_s=tolerance_s,
        max_reused_frac=max_reused_frac,
    )
    tower_ids = tuple(sorted(frames))

    def columns(name: str) -> np.ndarray:
        return np.column_stack(
            [aligned[f"{name}_{tower}"].to_numpy(dtype=float) for tower in tower_ids]
        )

    timestamps_s = aligned["ts_mono"].to_numpy(dtype=float)
    timestamps_s = timestamps_s - timestamps_s[0]
    return AlignedRecording(
        timestamps_s=timestamps_s,
        offsets_ns=columns("offset_ns"),
        freq_ppb=columns("freq_ppb"),
        path_delay_ns=columns("path_delay_ns"),
        tower_ids=tower_ids,
    )


def _split_stretches(
    stretches: Sequence[tuple[int, int]],
    *,
    train_fraction: float,
    validation_fraction: float,
    window_size: int,
) -> dict[str, list[tuple[int, int]]]:
    """Split each continuous stretch in time without crossing its boundary."""
    result = {"train": [], "validation": [], "test": []}
    for start, end in stretches:
        length = end - start
        train_end = start + int(length * train_fraction)
        validation_end = start + int(length * (train_fraction + validation_fraction))
        pieces = {
            "train": (start, train_end),
            "validation": (train_end, validation_end),
            "test": (validation_end, end),
        }
        for name, piece in pieces.items():
            if piece[1] - piece[0] >= window_size:
                result[name].append(piece)

    missing = [name for name, pieces in result.items() if not pieces]
    if missing:
        raise ValueError(
            "recording does not contain enough continuous samples for: "
            + ", ".join(missing)
        )
    return result


def _build_training_set_for_segments(
    recording: AlignedRecording,
    segments: Sequence[tuple[int, int]],
    *,
    attacked: np.ndarray,
    window_size: int,
    stride: int,
) -> TrainingSet:
    """Build windows independently for each segment."""
    parts = []
    for segment_id, (start, end) in enumerate(segments):
        part = build_training_set(
            recording.offsets_ns[start:end],
            recording.freq_ppb[start:end],
            recording.path_delay_ns[start:end],
            np.ones_like(recording.offsets_ns[start:end], dtype=bool),
            attacked[start:end],
            window_size=window_size,
            stride=stride,
        )
        if len(part.X) == 0:
            continue
        parts.append(
            TrainingSet(
                part.X,
                part.y,
                part.target,
                np.full(len(part.X), segment_id, dtype=int),
            )
        )

    if not parts:
        return TrainingSet(
            np.empty((0, N_ALL_FEATURES)),
            np.empty(0, dtype=bool),
            np.empty(0, dtype=int),
            np.empty(0, dtype=int),
        )
    return TrainingSet(
        np.vstack([part.X for part in parts]),
        np.concatenate([part.y for part in parts]),
        np.concatenate([part.target for part in parts]),
        np.concatenate([part.stretch for part in parts]),
    )


def _build_window_run(
    recording: AlignedRecording,
    segments: Sequence[tuple[int, int]],
    *,
    offsets_ns: np.ndarray,
    attacked: np.ndarray,
    target: int,
    window_size: int,
    stride: int,
) -> WindowRun:
    """Build chronological windows for one target tower."""
    X_parts, y_parts, times_parts, offset_parts = [], [], [], []
    for start, end in segments:
        X, y = build_feature_matrix(
            offsets_ns[start:end],
            recording.freq_ppb[start:end],
            recording.path_delay_ns[start:end],
            target=target,
            labels=attacked[start:end, target],
            window_size=window_size,
            stride=stride,
        )
        if len(X) == 0:
            continue
        starts = np.arange(0, end - start - window_size + 1, stride)
        final_rows = starts + window_size - 1
        X_parts.append(X)
        y_parts.append(y)
        times_parts.append(recording.timestamps_s[start:end][final_rows])
        offset_parts.append(offsets_ns[start:end, target][final_rows])

    if not X_parts:
        raise ValueError(f"no usable windows for tower {target}")
    return WindowRun(
        X=np.vstack(X_parts),
        y=np.concatenate(y_parts),
        times_s=np.concatenate(times_parts),
        offsets_ns=np.concatenate(offset_parts),
    )


def _scenario_runs(
    recording: AlignedRecording,
    segments: Sequence[tuple[int, int]],
    *,
    config: TrainingConfig,
    attack_start_s: float,
) -> dict[str, WindowRun]:
    """Create one synthetic attack scenario for every type and target tower."""
    targets = (
        recording.tower_ids
        if config.attack_towers is None
        else tuple(config.attack_towers)
    )
    unknown = sorted(set(targets) - set(recording.tower_ids))
    if unknown:
        raise ValueError(f"attack_towers not found in recordings: {unknown}")

    runs = {}
    for attack_type in config.attack_types:
        for tower_id in targets:
            target = recording.tower_ids.index(tower_id)
            attack_config = config.attack.model_copy(
                update={
                    "enabled": True,
                    "attack_type": attack_type,
                    "start_s": float(attack_start_s),
                    "affected_towers": [tower_id],
                }
            )
            result = inject_attack(
                recording.offsets_ns[:, target],
                recording.timestamps_s,
                attack_config,
            )
            offsets = recording.offsets_ns.copy()
            offsets[:, target] = result.offsets_ns
            attacked = np.zeros_like(offsets, dtype=bool)
            attacked[:, target] = result.labels.astype(bool)
            key = f"{attack_type}:tower_{tower_id}"
            runs[key] = _build_window_run(
                recording,
                segments,
                offsets_ns=offsets,
                attacked=attacked,
                target=target,
                window_size=config.feature.window_size,
                stride=config.stride,
            )
    return runs


def _combine_runs(
    runs: Sequence[WindowRun],
    *,
    stride_seconds: float,
) -> WindowRun:
    """Concatenate runs while keeping their timestamps strictly increasing."""
    if not runs:
        raise ValueError("at least one window run is required")

    times_parts = []
    previous_end = None
    for run in runs:
        times = run.times_s.copy()
        if previous_end is not None:
            times += previous_end + stride_seconds - times[0]
        previous_end = float(times[-1])
        times_parts.append(times)
    return WindowRun(
        X=np.vstack([run.X for run in runs]),
        y=np.concatenate([run.y for run in runs]),
        times_s=np.concatenate(times_parts),
        offsets_ns=np.concatenate([run.offsets_ns for run in runs]),
    )


def metadata_path_for(artifact_path: str | Path) -> Path:
    """Return the JSON sidecar path for a detector artifact."""
    path = Path(artifact_path)
    return path.with_name(path.name + ".json")


def load_training_artifact(
    artifact_path: str | Path,
) -> tuple[BaseDetector, dict[str, Any]]:
    """Load a detector and its metadata sidecar."""
    path = Path(artifact_path)
    detector = BaseDetector.load(path)
    metadata = json.loads(metadata_path_for(path).read_text(encoding="utf-8"))
    return detector, metadata


def _write_artifact(
    detector: BaseDetector,
    artifact_path: Path,
    metadata: dict[str, Any],
) -> Path:
    detector.save(artifact_path)
    metadata_path = metadata_path_for(artifact_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata_path


def _split_bounds(
    recording: AlignedRecording,
    segments: Sequence[tuple[int, int]],
) -> tuple[float, float]:
    return (
        float(recording.timestamps_s[segments[0][0]]),
        float(recording.timestamps_s[segments[-1][1] - 1]),
    )


def train(
    paths: Mapping[int, str | Path],
    output_path: str | Path,
    *,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Train, evaluate, and save one detector artifact."""
    config = TrainingConfig() if config is None else config
    recording = load_aligned_recording(
        paths,
        tolerance_s=config.align_tolerance_s,
        max_reused_frac=config.max_reused_frac,
    )
    stretches = find_stretches(
        recording.timestamps_s,
        min_length=config.feature.window_size,
        max_gap_s=config.max_gap_s,
    )
    splits = _split_stretches(
        stretches,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
        window_size=config.feature.window_size,
    )

    clean_attacked = np.zeros_like(recording.offsets_ns, dtype=bool)
    training_set = _build_training_set_for_segments(
        recording,
        splits["train"],
        attacked=clean_attacked,
        window_size=config.feature.window_size,
        stride=config.stride,
    )
    if len(training_set.X) == 0:
        raise ValueError("training split produced no usable feature windows")

    detector = IsolationForestDetector.from_config(config.detector)
    detector.fit(training_set.X)

    validation_start, validation_end = _split_bounds(
        recording, splits["validation"]
    )
    validation_attack_start = validation_start + (
        validation_end - validation_start
    ) * config.validation_attack_fraction
    validation_runs = _scenario_runs(
        recording,
        splits["validation"],
        config=config,
        attack_start_s=validation_attack_start,
    )
    validation_combined = _combine_runs(
        list(validation_runs.values()),
        stride_seconds=config.stride / config.feature.sample_rate_hz,
    )
    validation_scores = detector.score(validation_combined.X)
    threshold_candidates = threshold_grid(
        validation_scores,
        num=config.threshold_grid_size,
    )
    operating_rows = sweep(
        validation_combined.y,
        validation_combined.times_s,
        validation_combined.offsets_ns,
        config.feature.window_seconds,
        validation_scores,
        thresholds=threshold_candidates,
        n_values=config.debounce_values,
        stride_seconds=config.stride / config.feature.sample_rate_hz,
    )
    operating_point = select_operating_point(
        operating_rows,
        max_false_episodes_per_hour=config.false_alarm_budget_per_hour,
    )
    if operating_point is None:
        raise ValueError(
            "no validation operating point satisfies the false-alarm budget "
            f"of {config.false_alarm_budget_per_hour} episodes/hour"
        )

    detector.threshold_ = float(operating_point["threshold"])
    selected_n = int(operating_point["n_consecutive"])
    validation_report = evaluate_run(
        validation_combined.y,
        validation_combined.times_s,
        validation_combined.offsets_ns,
        config.feature.window_seconds,
        scores=validation_scores,
        threshold=detector.threshold_,
        n_consecutive=selected_n,
        stride_seconds=config.stride / config.feature.sample_rate_hz,
    )

    test_start, _ = _split_bounds(recording, splits["test"])
    test_runs = _scenario_runs(
        recording,
        splits["test"],
        config=config,
        attack_start_s=test_start,
    )
    test_reports: dict[str, dict[str, Any]] = {}
    for key, run in test_runs.items():
        scores = detector.score(run.X)
        test_reports[key] = evaluate_run(
            run.y,
            run.times_s,
            run.offsets_ns,
            config.feature.window_seconds,
            scores=scores,
            threshold=detector.threshold_,
            n_consecutive=selected_n,
            stride_seconds=config.stride / config.feature.sample_rate_hz,
        ).to_dict()

    metadata = {
        "schema_version": 1,
        "detector": detector.name,
        "feature_names": list(ALL_FEATURE_NAMES),
        "n_features": N_ALL_FEATURES,
        "window_size": config.feature.window_size,
        "sample_rate_hz": config.feature.sample_rate_hz,
        "stride": config.stride,
        "tower_ids": list(recording.tower_ids),
        "threshold": detector.threshold_,
        "threshold_quantile": detector.threshold_quantile,
        "n_consecutive": selected_n,
        "operating_point": {
            "threshold": detector.threshold_,
            "n_consecutive": selected_n,
            "false_alarm_budget_per_hour": config.false_alarm_budget_per_hour,
        },
        "split": {
            "train_fraction": config.train_fraction,
            "validation_fraction": config.validation_fraction,
            "train_windows": int(len(training_set.X)),
            "validation_windows": int(len(validation_combined.X)),
            "test_windows": int(sum(len(run.X) for run in test_runs.values())),
        },
        "config": config.model_dump(mode="json"),
        "validation_report": validation_report.to_dict(),
        "test_reports": test_reports,
    }

    artifact_path = Path(output_path)
    metadata_path = _write_artifact(detector, artifact_path, metadata)
    return TrainingResult(
        artifact_path=artifact_path,
        metadata_path=metadata_path,
        detector=detector,
        metadata=metadata,
        validation_report=validation_report.to_dict(),
        test_reports=test_reports,
    )


def _parse_tower_argument(value: str) -> tuple[int, Path]:
    try:
        tower_text, path_text = value.split("=", 1)
        tower = int(tower_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "tower must use the form TOWER=PATH"
        ) from exc
    return tower, Path(path_text)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tower",
        action="append",
        required=True,
        type=_parse_tower_argument,
        metavar="ID=PATH",
        help="tower JSONL recording; repeat once per tower",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    paths = dict(args.tower)
    result = train(paths, args.output)
    print(json.dumps(result.metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
