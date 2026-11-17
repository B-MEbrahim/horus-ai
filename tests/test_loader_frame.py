import json

import pytest

from horus_ai.data.loader import (
    FRAME_COLUMNS,
    load_tower_frame,
    load_tower_series,
)


def _row(t, state="s2", kind="sample", offset=10.0, freq=1.0):
    return {
        "kind": kind, "ts_mono": t, "ts_wall": 1e9 + t, "servo_state": state,
        "master_offset_ns": offset, "freq_ppb": freq, "path_delay_ns": 500.0,
    }


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def test_frame_keeps_only_locked_samples_sorted(tmp_path):
    p = _write(tmp_path / "t.jsonl", [
        _row(2.0), _row(1.0), _row(3.0, state="s1"), _row(4.0, kind="event"),
    ])
    df = load_tower_frame(p)
    assert list(df.columns) == FRAME_COLUMNS
    assert list(df["ts_mono"]) == [1.0, 2.0]


def test_frame_drops_rows_with_missing_values(tmp_path):
    p = _write(tmp_path / "t.jsonl", [_row(1.0), _row(2.0, freq=None), _row(3.0)])
    assert list(load_tower_frame(p)["ts_mono"]) == [1.0, 3.0]


def test_frame_raises_when_nothing_is_locked(tmp_path):
    p = _write(tmp_path / "t.jsonl", [_row(1.0, state="s0"), _row(2.0, state="s1")])
    with pytest.raises(ValueError):
        load_tower_frame(p)


def test_series_starts_at_zero_and_keeps_offsets(tmp_path):
    p = _write(tmp_path / "t.jsonl", [_row(5.0, offset=7.0), _row(5.125, offset=9.0)])
    t_s, offsets = load_tower_series(p)
    assert list(t_s) == [0.0, 0.125]
    assert list(offsets) == [7.0, 9.0]
