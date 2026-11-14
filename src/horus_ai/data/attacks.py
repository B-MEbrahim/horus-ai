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


def inject_attack(offsets_ns, t_s, cfg: AttackConfig) -> AttackResult:
    """Apply the configured attack to ONE tower's offset series."""
    offsets_ns = np.asarray(offsets_ns, dtype=float)
    t_s = np.asarray(t_s, dtype=float)
    if offsets_ns.ndim != 1 or offsets_ns.shape != t_s.shape:
        raise ValueError("offsets_ns and t_s must be 1-D arrays of the same length")

    if not cfg.enabled:
        zeros = np.zeros_like(offsets_ns)
        return AttackResult(offsets_ns.copy(), zeros.astype(int), zeros)

    added_ns, labels = _ATTACKS[cfg.attack_type](t_s, cfg)
    return AttackResult(offsets_ns + added_ns, labels, added_ns)


def inject_attack_towers(series: dict, cfg: AttackConfig) -> dict:
    """series: {tower_index: (t_s, offsets_ns)} -> {tower_index: AttackResult}.

    Towers not in cfg.affected_towers come back unchanged with all-zero labels.
    """
    unknown = set(cfg.affected_towers) - set(series)
    if unknown:
        raise ValueError(f"affected_towers not found in data: {sorted(unknown)}")

    clean_cfg = cfg.model_copy(update={"enabled": False})
    return {
        tower: inject_attack(
            offsets, t_s, cfg if tower in cfg.affected_towers else clean_cfg
        )
        for tower, (t_s, offsets) in series.items()
    }