"""Line up the towers' samples so every row is one moment for all towers."""
import numpy as np
import pandas as pd

VALUE_COLUMNS = ["offset_ns", "freq_ppb", "path_delay_ns"]


def align_towers(
    frames: dict[int, pd.DataFrame],
    reference: int | None = None,
    tolerance_s: float = 0.15,
    max_reused_frac: float = 0.01,
) -> pd.DataFrame:
    """Match every tower's samples to the reference tower's timeline.

    All towers sample at the same rate but start at their own phase, so for
    each reference timestamp we take the NEAREST sample of every tower
    (within tolerance_s). Moments where any tower has no match are dropped.

    Columns: ts_mono (reference time) and, for each tower k: ts_mono_k,
    offset_ns_k, freq_ppb_k, path_delay_ns_k.
    out.attrs["reused"]  = {tower: samples matched more than once}
    out.attrs["dropped"] = moments dropped for lack of a match
    """
    towers = sorted(frames)
    ref = towers[0] if reference is None else reference
    if ref not in frames:
        raise ValueError(f"reference tower {ref} is not in frames")

    # only the time range that every tower covers
    start = max(f["ts_mono"].iloc[0] for f in frames.values())
    end = min(f["ts_mono"].iloc[-1] for f in frames.values())
    if start >= end:
        raise ValueError("towers have no common time range")

    ref_t = frames[ref]["ts_mono"]
    out = pd.DataFrame({"ts_mono": ref_t[(ref_t >= start) & (ref_t <= end)].to_numpy()})

    reused = {}
    for k in towers:
        cols = ["ts_mono", *VALUE_COLUMNS]
        other = frames[k][cols].rename(columns={c: f"{c}_{k}" for c in cols})
        other[f"_row_{k}"] = np.arange(len(other))
        out = pd.merge_asof(
            out, other, left_on="ts_mono", right_on=f"ts_mono_{k}",
            direction="nearest", tolerance=tolerance_s,
        )
        reused[k] = int(out[f"_row_{k}"].dropna().duplicated().sum())
        if reused[k] > max_reused_frac * len(out):
            raise ValueError(
                f"tower {k}: {reused[k]} samples matched more than once "
                f"(limit {max_reused_frac:.1%} of {len(out)} moments); "
                "the towers may not share a sampling rate"
            )

    n_before = len(out)
    out = out.drop(columns=[f"_row_{k}" for k in towers]).dropna().reset_index(drop=True)
    out.attrs["reused"] = reused
    out.attrs["dropped"] = n_before - len(out)
    return out
