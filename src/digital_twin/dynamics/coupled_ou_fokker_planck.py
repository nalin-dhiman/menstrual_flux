from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CoupledOUFokkerPlanckResult:
    """Stored marginals and fluxes of the joint progress--log-speed PDE."""

    time: np.ndarray
    progress_grid: np.ndarray
    log_speed_grid: np.ndarray
    follicular_mass: np.ndarray
    luteal_mass: np.ndarray
    total_mass: np.ndarray
    ovulation_flux: np.ndarray
    menses_flux: np.ndarray
    follicular_progress_marginal: np.ndarray
    luteal_progress_marginal: np.ndarray
    follicular_speed_marginal: np.ndarray
    luteal_speed_marginal: np.ndarray
    final_follicular_density: np.ndarray
    final_luteal_density: np.ndarray


def _bernoulli(value: np.ndarray) -> np.ndarray:
    result = np.empty_like(value, dtype=float)
    small = np.abs(value) < 1e-6
    result[small] = (
        1.0 - value[small] / 2.0 + value[small] ** 2 / 12.0
    )
    result[~small] = value[~small] / np.expm1(value[~small])
    return result


def _internal_flux(
    left: np.ndarray,
    right: np.ndarray,
    drift: np.ndarray,
    diffusion: float,
    spacing: float,
) -> np.ndarray:
    """Scharfetter--Gummel flux, vectorized over arbitrary face arrays."""

    if diffusion == 0:
        return np.where(drift >= 0, drift * left, drift * right)
    peclet = drift * spacing / diffusion
    return diffusion / spacing * (
        _bernoulli(-peclet) * left - _bernoulli(peclet) * right
    )


def _stage_step(
    density: np.ndarray,
    *,
    log_speed_grid: np.ndarray,
    progress_diffusion: float,
    kappa: float,
    mean_log_speed: float,
    log_speed_sigma: float,
    dz: float,
    dell: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance one stage and return log-speed-resolved exit flux."""

    n_ell, n_z = density.shape
    progress_faces = np.zeros((n_ell, n_z + 1))
    speed = np.exp(log_speed_grid)[:, None]
    progress_faces[:, 1:-1] = _internal_flux(
        density[:, :-1],
        density[:, 1:],
        speed,
        progress_diffusion,
        dz,
    )
    if progress_diffusion == 0:
        progress_faces[:, -1] = speed[:, 0] * density[:, -1]
    else:
        half_peclet = speed[:, 0] * (dz / 2.0) / progress_diffusion
        progress_faces[:, -1] = (
            progress_diffusion
            / (dz / 2.0)
            * _bernoulli(-half_peclet)
            * density[:, -1]
        )

    ell_faces = np.zeros((n_ell + 1, n_z))
    face_locations = (
        log_speed_grid[:-1] + log_speed_grid[1:]
    ) / 2.0
    ell_drift = kappa * (mean_log_speed - face_locations)[:, None]
    ell_faces[1:-1] = _internal_flux(
        density[:-1],
        density[1:],
        ell_drift,
        log_speed_sigma**2 / 2.0,
        dell,
    )

    updated = density - dt / dz * (
        progress_faces[:, 1:] - progress_faces[:, :-1]
    )
    updated -= dt / dell * (ell_faces[1:] - ell_faces[:-1])
    minimum = float(np.min(updated))
    if minimum < -1e-10:
        raise RuntimeError(
            f"joint Fokker--Planck density became negative ({minimum})"
        )
    return np.maximum(updated, 0.0), progress_faces[:, -1]


def solve_coupled_ou_cycle(
    *,
    follicular_mean_speed: float = 1.0 / 14.0,
    luteal_mean_speed: float = 1.0 / 14.0,
    follicular_progress_diffusion: float = 0.00015,
    luteal_progress_diffusion: float = 0.00010,
    follicular_kappa: float = 0.35,
    luteal_kappa: float = 0.45,
    follicular_log_speed_sigma: float = 0.035,
    luteal_log_speed_sigma: float = 0.025,
    dz: float = 0.02,
    log_speed_min: float = -3.8,
    log_speed_max: float = -1.8,
    dell: float = 0.04,
    dt: float = 0.01,
    max_time: float = 60.0,
    store_every: int = 10,
) -> CoupledOUFokkerPlanckResult:
    """Solve the coupled joint ``(progress, log-speed)`` conservation law.

    Boundary transfer preserves log speed: follicular exit flux at a given
    log speed enters luteal progress zero at the same log speed, and vice
    versa. Log-speed domain boundaries are zero-flux. The progress origin is
    reflecting except for flux injected by the other stage.
    """

    positive = (
        follicular_mean_speed,
        luteal_mean_speed,
        dz,
        dell,
        dt,
        max_time,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("speeds, spacings, dt, and max_time must be positive")
    if log_speed_max <= log_speed_min or store_every < 1:
        raise ValueError("invalid log-speed bounds or storage interval")
    if any(
        value < 0
        for value in (
            follicular_progress_diffusion,
            luteal_progress_diffusion,
            follicular_kappa,
            luteal_kappa,
            follicular_log_speed_sigma,
            luteal_log_speed_sigma,
        )
    ):
        raise ValueError("diffusion, kappa, and sigma must be nonnegative")

    n_z = int(round(1.0 / dz))
    n_ell = int(round((log_speed_max - log_speed_min) / dell))
    if not np.isclose(n_z * dz, 1.0) or not np.isclose(
        log_speed_min + n_ell * dell, log_speed_max
    ):
        raise ValueError("dz and dell must divide their domains")
    z = (np.arange(n_z) + 0.5) * dz
    ell = log_speed_min + (np.arange(n_ell) + 0.5) * dell

    max_progress_rate = np.exp(log_speed_max) / dz + 2.0 * max(
        follicular_progress_diffusion, luteal_progress_diffusion
    ) / dz**2
    max_ell_drift = max(
        follicular_kappa
        * np.max(np.abs(np.log(follicular_mean_speed) - ell)),
        luteal_kappa
        * np.max(np.abs(np.log(luteal_mean_speed) - ell)),
    )
    max_ell_rate = max_ell_drift / dell + max(
        follicular_log_speed_sigma**2,
        luteal_log_speed_sigma**2,
    ) / dell**2
    if dt * (max_progress_rate + max_ell_rate) > 0.8:
        raise ValueError("unstable joint grid: reduce dt")

    f = np.zeros((n_ell, n_z))
    l = np.zeros_like(f)
    stationary_sd = (
        follicular_log_speed_sigma
        / np.sqrt(2.0 * follicular_kappa)
        if follicular_kappa > 0
        else dell
    )
    initial = np.exp(
        -0.5
        * (
            (ell - np.log(follicular_mean_speed))
            / max(stationary_sd, dell / 2.0)
        )
        ** 2
    )
    initial /= np.sum(initial) * dell
    f[:, 0] = initial / dz

    steps = int(round(max_time / dt))
    stored_steps = np.unique(
        np.r_[np.arange(0, steps + 1, store_every), steps]
    ).astype(int)
    time = stored_steps * dt
    f_z = np.empty((len(stored_steps), n_z))
    l_z = np.empty_like(f_z)
    f_ell = np.empty((len(stored_steps), n_ell))
    l_ell = np.empty_like(f_ell)
    f_mass = np.empty(len(stored_steps))
    l_mass = np.empty(len(stored_steps))
    ovulation_flux = np.zeros(len(stored_steps))
    menses_flux = np.zeros(len(stored_steps))

    def record(index: int) -> None:
        f_z[index] = np.sum(f, axis=0) * dell
        l_z[index] = np.sum(l, axis=0) * dell
        f_ell[index] = np.sum(f, axis=1) * dz
        l_ell[index] = np.sum(l, axis=1) * dz
        f_mass[index] = np.sum(f) * dz * dell
        l_mass[index] = np.sum(l) * dz * dell

    record(0)
    stored_index = 1
    for step in range(1, steps + 1):
        next_f, f_out = _stage_step(
            f,
            log_speed_grid=ell,
            progress_diffusion=follicular_progress_diffusion,
            kappa=follicular_kappa,
            mean_log_speed=np.log(follicular_mean_speed),
            log_speed_sigma=follicular_log_speed_sigma,
            dz=dz,
            dell=dell,
            dt=dt,
        )
        next_l, l_out = _stage_step(
            l,
            log_speed_grid=ell,
            progress_diffusion=luteal_progress_diffusion,
            kappa=luteal_kappa,
            mean_log_speed=np.log(luteal_mean_speed),
            log_speed_sigma=luteal_log_speed_sigma,
            dz=dz,
            dell=dell,
            dt=dt,
        )
        next_f[:, 0] += dt * l_out / dz
        next_l[:, 0] += dt * f_out / dz
        f, l = next_f, next_l
        if step == stored_steps[stored_index]:
            ovulation_flux[stored_index] = np.sum(f_out) * dell
            menses_flux[stored_index] = np.sum(l_out) * dell
            record(stored_index)
            stored_index += 1
            if stored_index == len(stored_steps):
                break

    return CoupledOUFokkerPlanckResult(
        time=time,
        progress_grid=z,
        log_speed_grid=ell,
        follicular_mass=f_mass,
        luteal_mass=l_mass,
        total_mass=f_mass + l_mass,
        ovulation_flux=ovulation_flux,
        menses_flux=menses_flux,
        follicular_progress_marginal=f_z,
        luteal_progress_marginal=l_z,
        follicular_speed_marginal=f_ell,
        luteal_speed_marginal=l_ell,
        final_follicular_density=f,
        final_luteal_density=l,
    )
