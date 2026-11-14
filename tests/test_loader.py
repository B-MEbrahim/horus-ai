import json

from horus_ai.data.loader import load_tower_series


def test_loader_keeps_only_locked_samples(tmp_path):
    rows = [
        {"kind": "sample", "ts_mono": 100.0, "servo_state": "s0", "master_offset_ns": 9999},
        {"kind": "event", "ts_mono": 100.5, "servo_state": None, "master_offset_ns": None},
        {"kind": "sample", "ts_mono": 101.0, "servo_state": "s2", "master_offset_ns": 5},
        {"kind": "sample", "ts_mono": 103.0, "servo_state": "s2", "master_offset_ns": -7},
    ]
    f = tmp_path / "tower.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows))

    t_s, offsets = load_tower_series(f)

    assert t_s.tolist() == [0.0, 2.0]
    assert offsets.tolist() == [5.0, -7.0]