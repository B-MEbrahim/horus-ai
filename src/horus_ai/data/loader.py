from pathlib import Path
import numpy as np
import pandas as pd


def load_tower_series(path) -> tuple[np.ndarray, np.ndarray]:
    """Read one tower's JSONL file -> (t_s, offsets_ns) of its locked samples."""
    df = pd.read_json(path, lines=True)
    df = df[(df['kind'] == 'sample') & (df['servo_state'] == 's2')]

    if df.empty:
        raise ValueError(f"no locked (s2) samples in {path}")

    df = df.sort_values('ts_mono')
    t_s = (df['ts_mono'] - df['ts_mono'].iloc[0].to_numpy(dtype=float))
    offsets_ns = df['master_offset_ns'].to_numpy(dtype=float)
    return t_s, offsets_ns


def load_towers(paths: dict[int, Path]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """{tower_index: file} -> {tower_index: (t_s, offsets_ns)}."""
    return {tower: load_tower_series(p) for tower, p in paths.items()}