import numpy as np

from phase_weaver.core.grid import Grid


def trapz_uniform(dt: float, y: np.ndarray) -> float:
    return float(dt * (y.sum() - 0.5 * (y[0] + y[-1])))


def _center_by_first_moment(p: np.ndarray, g: Grid) -> np.ndarray:
    """
    Center by first moment so that ∫ t p(t) dt = 0 (approximately, to nearest sample).
    Implemented as a circular shift (np.roll).
    """
    dt = g.dt
    t = g.t

    area = trapz_uniform(dt, p)
    if not np.isfinite(area) or area <= 0:
        return p

    mean_t = float(trapz_uniform(dt, p * t) / area)  # since area ~ 1, but keep robust
    shift = int(np.round(mean_t / dt))
    # If mean_t > 0 (centroid to the right), shift left => roll by -shift
    return np.roll(p, -shift)


def _apply_time_constraints(
    x: np.ndarray,
    g: Grid,
    *,
    enforce_nonneg: bool = True,
    normalize_area_to_one: bool = True,
    center: bool = True,
    eps_area: float = 1e-30,
) -> np.ndarray:
    """
    Apply constraints to a time-domain array interpreted as a density p(t).
    Output is p(t) with:
      - p>=0
      - ∫p dt = 1
      - centered (first moment 0)
    """
    p = np.asarray(x, dtype=float)

    if enforce_nonneg:
        p = np.maximum(p, 0.0)

    if normalize_area_to_one:
        area = trapz_uniform(g.dt, p)
        if not np.isfinite(area) or area <= eps_area:
            # If we can't normalize, return zeros to avoid blowing up.
            # (Alternatively: raise an error.)
            return np.zeros_like(p)
        p = p / area

    if center:
        p = _center_by_first_moment(p, g)
        # Re-normalize after roll (should be unchanged, but helps numerical drift)
        if normalize_area_to_one:
            area = trapz_uniform(g.dt, p)
            if np.isfinite(area) and area > eps_area:
                p = p / area

    return p


def _min_phase_from_mag_pos(
    mag_pos: np.ndarray, N: int, eps: float = 1e-30
) -> np.ndarray:
    """
    Minimum-phase estimate from magnitude using real cepstrum (fast).
    Returns unwrapped phase for bins [0..Nyquist], length N/2+1.
    """
    mag_pos = np.asarray(mag_pos, dtype=float)
    if mag_pos.shape != (N // 2 + 1,):
        raise ValueError(f"mag_pos must have length N/2+1 = {N // 2 + 1}")

    logmag_pos = np.log(np.maximum(mag_pos, eps))

    logmag_full = np.empty(N, dtype=float)
    logmag_full[: N // 2 + 1] = logmag_pos
    logmag_full[N // 2 + 1 :] = logmag_pos[1:-1][::-1]  # mirror interior

    c = np.fft.ifft(logmag_full).real

    cmin = np.zeros_like(c)
    cmin[0] = c[0]
    cmin[1 : N // 2] = 2.0 * c[1 : N // 2]
    cmin[N // 2] = c[N // 2]  # Nyquist for even N

    spec = np.exp(np.fft.fft(cmin))
    return np.unwrap(np.angle(spec[: N // 2 + 1]))


def _rfft_pos_from_centered_time(p_centered: np.ndarray, g: Grid) -> np.ndarray:
    """
    Forward (centered time grid):
      P_pos = rfft(ifftshift(p)) * dt
    """
    p_shift = np.fft.ifftshift(p_centered)
    return np.fft.rfft(p_shift) * g.dt


def _irfft_centered_from_pos(P_pos: np.ndarray, g: Grid) -> np.ndarray:
    """
    Inverse (centered time grid):
      p = fftshift(irfft(P_pos / dt, n=N))
    """
    p_shift = np.fft.irfft(P_pos / g.dt, n=g.N)
    return np.fft.fftshift(p_shift)


def reconstruct_current_from_mag(
    g: Grid,
    mag_pos: np.ndarray,
    *,
    charge_C: float = 250e-12,  # 250 pC
    n_iter: int = 60,
    init_phase: str = "minphase",  # "minphase" | "zero" | "random"
    enforce_dc_one: bool = True,
    enforce_nonneg: bool = True,
    normalize_area_to_one: bool = True,
    center: bool = True,
    eps_mag: float = 1e-30,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Reduced GS-style reconstruction from measured magnitude only.

    Returns:
      t: seconds (centered)
      current: current in A (scaled so integral is charge_C)
      info: includes final phase, convergence metrics

    Constraints enforced *also on the initial time profile*.
    """
    mag_pos = np.asarray(mag_pos, dtype=float)
    if mag_pos.shape != (g.N // 2 + 1,):
        raise ValueError(f"mag_pos must have length N/2+1 = {g.N // 2 + 1}")

    mag_pos = np.maximum(mag_pos, eps_mag)

    # Optional: enforce F(0)=1 for a normalized density (common for form factor)
    if enforce_dc_one:
        if mag_pos[0] <= 0:
            mag_pos[0] = 1.0
        else:
            mag_pos = mag_pos / mag_pos[0]

    # --- initial phase ---
    if init_phase == "minphase":
        phase = _min_phase_from_mag_pos(mag_pos, g.N, eps=eps_mag)
    elif init_phase == "zero":
        phase = np.zeros_like(mag_pos)
    elif init_phase == "random":
        rng = np.random.default_rng()
        phase = np.unwrap(rng.uniform(0.0, 2.0 * np.pi, size=mag_pos.shape))
    else:
        raise ValueError("init_phase must be 'minphase', 'zero', or 'random'")

    # initial spectrum
    P_pos = mag_pos * np.exp(1j * phase)

    # --- apply constraints to INITIAL time profile too ---
    p0 = _irfft_centered_from_pos(P_pos, g)
    p0 = _apply_time_constraints(
        p0,
        g,
        enforce_nonneg=enforce_nonneg,
        normalize_area_to_one=normalize_area_to_one,
        center=center,
    )
    P_pos = _rfft_pos_from_centered_time(p0, g)
    phase = np.unwrap(np.angle(P_pos))
    P_pos = mag_pos * np.exp(1j * phase)

    phase_change_mse = []
    neg_mass = []

    # --- iterations ---
    for _ in range(n_iter):
        prev_phase = phase

        # to time domain
        p = _irfft_centered_from_pos(P_pos, g)

        # measure how much negativity we had before clipping (diagnostic)
        if enforce_nonneg:
            neg_mass.append(float(trapz_uniform(g.dt, np.maximum(-p, 0.0))))
        else:
            neg_mass.append(0.0)

        # enforce constraints
        p = _apply_time_constraints(
            p,
            g,
            enforce_nonneg=enforce_nonneg,
            normalize_area_to_one=normalize_area_to_one,
            center=center,
        )

        # back to frequency domain
        P_new = _rfft_pos_from_centered_time(p, g)

        # enforce measured magnitude, keep phase
        phase = np.unwrap(np.angle(P_new))
        P_pos = mag_pos * np.exp(1j * phase)

        phase_change_mse.append(float(np.mean((phase - prev_phase) ** 2)))

    # Final time density + scale to current with requested charge
    p_final = _irfft_centered_from_pos(P_pos, g)
    p_final = _apply_time_constraints(
        p_final,
        g,
        enforce_nonneg=enforce_nonneg,
        normalize_area_to_one=normalize_area_to_one,
        center=center,
    )

    current = charge_C * p_final  # A, since p has units 1/s when ∫p dt = 1

    info = {
        "phase_pos": phase,
        "phase_change_mse": np.array(phase_change_mse),
        "neg_mass": np.array(neg_mass),
        "mag_pos_used": mag_pos,
    }
    return g.t, current, info
