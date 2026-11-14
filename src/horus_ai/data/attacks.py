import numpy as np
from horus_ai.data.config import AttackConfig
from dataclasses import dataclass


def _active_mask(t_s: np.ndarray, cfg: AttackConfig) -> np.ndarray:
    """True for every sample where the attack is switched on."""
    active = t_s >= cfg.start_s
    if cfg.duration_s is not None:
        active &= t_s < cfg.start_s + cfg.duration_s
    return active


def sudden_jump(t_s: np.ndarray, cfg: AttackConfig) -> tuple[np.ndarray, np.ndarray]:
    """Offset added by a sudden jump, plus 0/1 labels (1 = attack active)."""
    t_s = np.asarray(t_s, dtype=float)
    active = _active_mask(t_s, cfg)
    added_ns = np.where(active, cfg.jump_ns, 0.0)
    return added_ns, active.astype(int)


def gradual_drift(t_s: np.ndarray, cfg: AttackConfig) -> tuple[np.ndarray, np.ndarray]:
    """Offset added by a slow ramp, plus 0/1 labels (1 = attack active)."""
    t_s = np.asarray(t_s, dtype=float)
    active = _active_mask(t_s, cfg)
    ramp_ns = cfg.push_rate_ns_per_s * (t_s - cfg.start_s)
    added_ns = np.where(active, ramp_ns, 0.0)
    return added_ns, active.astype(int)


@dataclass(frozen=True)
class AttackResult:
    offsets_ns: np.ndarray  # clean offsets + the attack
    labels: np.ndarray      # 0 = clean, 1 = attack active
    added_ns: np.ndarray    # what the attack added (handy for plots/debugging)


_ATTACKS = {"sudden_jump": sudden_jump, "gradual_drift": gradual_drift}


