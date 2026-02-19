import numpy as np
from phase_weaver.core.grid import Grid


def trapz_uniform(dt: float, y: np.ndarray) -> float:
    return float(dt * (y.sum() - 0.5 * (y[0] + y[-1])))


def normalize_center_nonneg_density(p: np.ndarray, g: Grid, eps: float = 1e-30) -> np.ndarray:
    """
    Interpret p(t) as a density on g.t and enforce:
      - p >= 0
      - ∫ p dt = 1
      - centroid (first moment) at 0 via circular shift
    """
    p = np.asarray(p, dtype=float)

    # non-negativity
    p = np.maximum(p, 0.0)

    # normalize to unit integral
    area = trapz_uniform(g.dt, p)
    if not np.isfinite(area) or area <= eps:
        return np.zeros_like(p)
    p = p / area

    # center by first moment (approx, to nearest sample)
    mean_t = trapz_uniform(g.dt, p * g.t)  # since ∫p dt = 1
    shift = int(np.round(mean_t / g.dt))
    p = np.roll(p, -shift)

    # renormalize (numerical)
    area2 = trapz_uniform(g.dt, p)
    if np.isfinite(area2) and area2 > eps:
        p = p / area2

    return p