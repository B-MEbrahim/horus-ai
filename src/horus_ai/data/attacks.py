import numpy as np
from horus_ai.data.config import AttackConfig

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