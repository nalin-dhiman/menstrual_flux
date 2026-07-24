from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CoupledFokkerPlanckResult:
    """Probability densities and boundary fluxes for a two-stage cycle."""

    time: np.ndarray
    grid: np.ndarray
    follicular_density: np.ndarray
    luteal_density: np.ndarray
    follicular_mass: np.ndarray
    luteal_mass: np.ndarray
    total_mass: np.ndarray
    ovulation_flux: np.ndarray
    menses_flux: np.ndarray
    cumulative_ovulations: np.ndarray
    cumulative_menses: np.ndarray
    first_ovulation_density: np.ndarray
    first_menses_density: np.ndarray
    first_cycle_survival: np.ndarray


def _check_stage(drift: float, diffusion: float, name: str) -> None:
    if drift <= 0:
        raise ValueError(f"{name} drift must be positive")
    if diffusion < 0:
        raise ValueError(f"{name} diffusion must be nonnegative")


def _absorbing_step(
    density: np.ndarray,
    drift: float,
    diffusion: float,
    dz: float,
    dt: float,
) -> tuple[np.ndarray, float]:
    """One conservative exponentially fitted finite-volume step.

    Scharfetter--Gummel face fluxes avoid the artificial diffusion of a coarse
    first-order upwind discretization. The absorbing face is located half a
    cell beyond the final cell center and has zero boundary density.
    """

    faces = np.empty(len(density) + 1)
    faces[0] = 0.0
    if diffusion == 0:
        faces[1:-1] = drift * density[:-1]
        faces[-1] = drift * density[-1]
    else:
        cell_peclet = drift * dz / diffusion

        def bernoulli(value: float) -> float:
            if abs(value) < 1e-6:
                return 1.0 - value / 2.0 + value**2 / 12.0
            return float(value / np.expm1(value))

        left_coefficient = (
            diffusion / dz * bernoulli(-cell_peclet)
        )
        right_coefficient = (
            diffusion / dz * bernoulli(cell_peclet)
        )
        faces[1:-1] = (
            left_coefficient * density[:-1]
            - right_coefficient * density[1:]
        )
        half_cell_peclet = drift * (dz / 2.0) / diffusion
        faces[-1] = (
            diffusion
            / (dz / 2.0)
            * bernoulli(-half_cell_peclet)
            * density[-1]
        )
    updated = density - (dt / dz) * (faces[1:] - faces[:-1])
    # The CFL check should make meaningful negativity impossible. Permit only
    # floating-point-scale corrections and fail loudly otherwise.
    if float(np.min(updated)) < -1e-10:
        raise RuntimeError("finite-volume density became negative")
    updated = np.maximum(updated, 0.0)
    return updated, float(faces[-1])


def solve_coupled_cycle(
    *,
    follicular_drift: float = 1.0 / 14.0,
    luteal_drift: float = 1.0 / 14.0,
    follicular_diffusion: float = 0.00015,
    luteal_diffusion: float = 0.00010,
    dz: float = 0.01,
    dt: float = 0.02,
    max_time: float = 90.0,
    store_every: int = 1,
    initial_stage: str = "follicular",
    initial_progress: float = 0.0,
) -> CoupledFokkerPlanckResult:
    """Solve coupled follicular/luteal drift-diffusion with cyclic resetting.

    Flux leaving the follicular absorbing boundary is injected at the luteal
    reflecting boundary. Luteal exit flux is injected back into follicular
    progress zero. Parallel transient densities record the first ovulation and
    first subsequent menses without contamination by later cycles.
    """

    _check_stage(follicular_drift, follicular_diffusion, "follicular")
    _check_stage(luteal_drift, luteal_diffusion, "luteal")
    if dz <= 0 or dt <= 0 or max_time <= 0 or store_every < 1:
        raise ValueError("dz, dt, and max_time must be positive")
    if initial_stage not in {"follicular", "luteal"}:
        raise ValueError("initial_stage must be follicular or luteal")
    if not 0 <= initial_progress < 1:
        raise ValueError("initial_progress must lie in [0, 1)")
    for drift, diffusion in (
        (follicular_drift, follicular_diffusion),
        (luteal_drift, luteal_diffusion),
    ):
        if drift * dt / dz + 2.0 * diffusion * dt / dz**2 > 0.95:
            raise ValueError("unstable grid: reduce dt or increase dz")

    cells = int(round(1.0 / dz))
    if not np.isclose(cells * dz, 1.0):
        raise ValueError("dz must divide the unit stage interval")
    grid = (np.arange(cells) + 0.5) * dz
    integration_steps = int(round(max_time / dt))
    stored_steps = np.unique(
        np.r_[
            np.arange(0, integration_steps + 1, store_every),
            integration_steps,
        ]
    ).astype(int)
    time = stored_steps * dt
    f = np.zeros(cells)
    l = np.zeros(cells)
    target = f if initial_stage == "follicular" else l
    target[int(np.clip(np.floor(initial_progress / dz), 0, cells - 1))] = 1.0 / dz

    # Transient first-cycle states. Once the first luteal exit occurs, its mass
    # is removed rather than reset, so survival is P(first menses > t).
    first_f = f.copy()
    first_l = l.copy()

    f_density = np.empty((len(stored_steps), cells))
    l_density = np.empty((len(stored_steps), cells))
    f_mass = np.empty(len(stored_steps))
    l_mass = np.empty(len(stored_steps))
    ov_flux = np.zeros(len(stored_steps))
    men_flux = np.zeros(len(stored_steps))
    cumulative_ov = np.zeros(len(stored_steps))
    cumulative_men = np.zeros(len(stored_steps))
    first_ov = np.zeros(len(stored_steps))
    first_men = np.zeros(len(stored_steps))
    first_survival = np.empty(len(stored_steps))

    def record(index: int) -> None:
        f_density[index] = f
        l_density[index] = l
        f_mass[index] = np.sum(f) * dz
        l_mass[index] = np.sum(l) * dz
        first_survival[index] = np.sum(first_f + first_l) * dz

    record(0)
    stored_index = 1
    cumulative_ovulations = 0.0
    cumulative_menses = 0.0
    for step in range(1, integration_steps + 1):
        next_f, f_out = _absorbing_step(
            f, follicular_drift, follicular_diffusion, dz, dt
        )
        next_l, l_out = _absorbing_step(
            l, luteal_drift, luteal_diffusion, dz, dt
        )
        next_f[0] += dt * l_out / dz
        next_l[0] += dt * f_out / dz
        f, l = next_f, next_l
        cumulative_ovulations += dt * f_out
        cumulative_menses += dt * l_out

        next_first_f, first_f_out = _absorbing_step(
            first_f, follicular_drift, follicular_diffusion, dz, dt
        )
        next_first_l, first_l_out = _absorbing_step(
            first_l, luteal_drift, luteal_diffusion, dz, dt
        )
        next_first_l[0] += dt * first_f_out / dz
        first_f, first_l = next_first_f, next_first_l
        if stored_index < len(stored_steps) and step == stored_steps[stored_index]:
            ov_flux[stored_index] = f_out
            men_flux[stored_index] = l_out
            cumulative_ov[stored_index] = cumulative_ovulations
            cumulative_men[stored_index] = cumulative_menses
            first_ov[stored_index] = first_f_out
            first_men[stored_index] = first_l_out
            record(stored_index)
            stored_index += 1

    total_mass = f_mass + l_mass
    return CoupledFokkerPlanckResult(
        time=time,
        grid=grid,
        follicular_density=f_density,
        luteal_density=l_density,
        follicular_mass=f_mass,
        luteal_mass=l_mass,
        total_mass=total_mass,
        ovulation_flux=ov_flux,
        menses_flux=men_flux,
        cumulative_ovulations=cumulative_ov,
        cumulative_menses=cumulative_men,
        first_ovulation_density=first_ov,
        first_menses_density=first_men,
        first_cycle_survival=first_survival,
    )
