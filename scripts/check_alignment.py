from pathlib import Path
from horus_ai.data.loader import load_tower_frames
from horus_ai.data.align import align_towers

files = sorted(Path("datasets/clean").glob("ovs*_onehour.jsonl"))
frames = load_tower_frames({i + 1: p for i, p in enumerate(files)})
out = align_towers(frames)
print(len(out), out.attrs["reused"], out.attrs["dropped"])
print((out["ts_mono_4"] - out["ts_mono"]).abs().describe())