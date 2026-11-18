"""Load tower timing recordings (one JSONL file per tower)."""
from pathlib import Path

import numpy as np
import pandas as pd

FRAME_COLUMNS = ["ts_mono", "ts_wall", "offset_ns", "freq_ppb", "path_delay_ns"]


def load_tower_frame(path) -> pd.DataFrame:
    """Read one tower's JSONL file -> DataFrame of its locked (s2) samples.

    Columns: ts_mono, ts_wall, offset_ns, freq_ppb, path_delay_ns.
    Rows are sorted by ts_mono. A row with a missing value is dropped; the
    hole it leaves is caught later by the gap rule.
    """
    df = pd.read_json(path, lines=True, convert_dates=False)
    df = df[(df["kind"] == "sample") & (df["servo_state"] == "s2")]
    df = df.rename(columns={"master_offset_ns": "offset_ns"})
    df = df[FRAME_COLUMNS].dropna().astype(float).sort_values("ts_mono")

    if df.empty:
        raise ValueError(f"no locked (s2) samples in {path}")
    return df.reset_index(drop=True)


def load_tower_frames(paths: dict[int, Path]) -> dict[int, pd.DataFrame]:
    """{tower_index: file} -> {tower_index: DataFrame from load_tower_frame}."""
    return {tower: load_tower_frame(p) for tower, p in paths.items()}


def load_tower_series(path) -> tuple[np.ndarray, np.ndarray]:
    """Read one tower's JSONL file -> (t_s, offsets_ns) of its locked samples.

    t_s counts seconds from this tower's own first locked sample, so it
    cannot be used to line up different towers (use load_tower_frame).
    """
    df = pd.read_json(path, lines=True, convert_dates=False)
    df = df[(df["kind"] == "sample") & (df["servo_state"] == "s2")]
    df = (
        df[["ts_mono", "master_offset_ns"]]
        .dropna()
        .astype(float)
        .sort_values("ts_mono")
    )

    if df.empty:
        raise ValueError(f"no locked (s2) samples in {path}")

    t_s = (df["ts_mono"] - df["ts_mono"].iloc[0]).to_numpy(dtype=float)
    return t_s, df["master_offset_ns"].to_numpy(dtype=float)


def load_towers(paths: dict[int, Path]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """{tower_index: file} -> {tower_index: (t_s, offsets_ns)}."""
    return {tower: load_tower_series(p) for tower, p in paths.items()}
