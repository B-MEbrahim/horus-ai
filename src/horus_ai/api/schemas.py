from typing import Literal

from pydantic import BaseModel, Field

ServoState = Literal["s0", "s1", "s2"]
Status = Literal["warming_up", "scoring", "lock_lost"]


class TimingSample(BaseModel):
    """One measurement sample sent by a tower's data agent."""

    tower_id: str = Field(min_length=1)
    ts_wall: float
    master_offset_ns: float = Field(allow_inf_nan=False)
    freq_ppb: float = Field(allow_inf_nan=False)
    path_delay_ns: float = Field(allow_inf_nan=False)
    servo_state: ServoState

    # optional
    ts_mono: float | None = None
    seq: int | None = None


class Verdict(BaseModel):
    """The model's answer for one sample."""

    tower_id: str
    status: Status
    anomaly_score: float | None = None  # empty until the window is ready
    is_spoofed: bool = False