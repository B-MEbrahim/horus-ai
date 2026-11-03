import pytest
from pydantic import ValidationError

from horus_ai.api.schemas import TimingSample

VALID = dict(
    tower_id="OVS1",
    ts_wall=1.793321e9,
    master_offset_ns=-85046,
    freq_ppb=120.5,
    path_delay_ns=45000,
    servo_state="s2",
)


def test_valid_sample():
    assert TimingSample(**VALID).tower_id == "OVS1"


def test_optional_fields_default_to_none():
    s = TimingSample(**VALID)
    assert s.ts_mono is None and s.seq is None


def test_missing_required_field():
    bad = {k: v for k, v in VALID.items() if k != "master_offset_ns"}
    with pytest.raises(ValidationError):
        TimingSample(**bad)


def test_bad_servo_state():
    with pytest.raises(ValidationError):
        TimingSample(**{**VALID, "servo_state": "s9"})


def test_nan_rejected():
    with pytest.raises(ValidationError):
        TimingSample(**{**VALID, "master_offset_ns": float("nan")})


def test_extreme_value_accepted():
    s = TimingSample(**{**VALID, "master_offset_ns": 5e9})
    assert s.master_offset_ns == 5e9