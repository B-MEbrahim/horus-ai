import numpy as np
import pandas as pd
import pytest

from horus_ai.data.align import align_towers
from horus_ai.data.loader import FRAME_COLUMNS

REF = np.arange(10) * 0.125  # 10 samples at 8 Hz


def _frame(times):
    times = np.asarray(times, dtype=float)
    return pd.DataFrame({
        "ts_mono": times,
        "ts_wall": 1e9 + times,
        "offset_ns": np.arange(len(times), dtype=float),  # = row number
        "freq_ppb": 1.0,
        "path_delay_ns": 500.0,
    })[FRAME_COLUMNS]


def test_phase_shift_is_matched_row_for_row():
    frames = {1: _frame(REF), 2: _frame(REF + 0.03)}
    out = align_towers(frames)
    assert len(out) == 9  # row 0 of tower 1 starts before tower 2 does
    assert (out["offset_ns_1"] == out["offset_ns_2"]).all()
    assert ((out["ts_mono_2"] - out["ts_mono"]).abs() <= 0.0625).all()


def test_output_columns():
    out = align_towers({1: _frame(REF), 2: _frame(REF)})
    assert list(out.columns) == [
        "ts_mono",
        "ts_mono_1", "offset_ns_1", "freq_ppb_1", "path_delay_ns_1",
        "ts_mono_2", "offset_ns_2", "freq_ppb_2", "path_delay_ns_2",
    ]


def test_moment_without_match_is_dropped():
    hole = np.delete(REF, 4)  # tower 2 lost the sample at 0.5 s
    out = align_towers({1: _frame(REF), 2: _frame(hole)}, tolerance_s=0.1)
    assert len(out) == 9
    assert out.attrs["dropped"] == 1
    assert 0.5 not in set(out["ts_mono"])


def test_reused_sample_is_counted():
    hole = np.delete(REF, 4)
    out = align_towers({1: _frame(REF), 2: _frame(hole)}, max_reused_frac=1.0)
    assert len(out) == 10
    assert out.attrs["reused"] == {1: 0, 2: 1}


def test_too_many_reused_samples_raises():
    hole = np.delete(REF, 4)
    with pytest.raises(ValueError):
        align_towers({1: _frame(REF), 2: _frame(hole)})  # 1 of 10 > 1%


def test_no_common_time_range_raises():
    with pytest.raises(ValueError):
        align_towers({1: _frame(REF), 2: _frame(REF + 100.0)})
