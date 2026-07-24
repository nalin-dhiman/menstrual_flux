from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FokkerPlanckResult:
    time: np.ndarray
    grid: np.ndarray
    density: np.ndarray
    event_density: np.ndarray
    survival: np.ndarray


def solve_constant_coefficients(
    drift: float = 1 / 14,
    diffusion: float = 0.00015,
    dz: float = 0.01,
    dt: float = 0.02,
    max_time: float = 35.0,
    initial_progress: float = 0.0,
) -> FokkerPlanckResult:
    """Conservative cell-centred finite volume scheme.

    The left face has zero outward flux (reflection) and the right face is
    absorbing. Positive drift is discretized with an upwind flux.
    """
    if drift <= 0 or diffusion < 0 or dz <= 0 or dt <= 0:
        raise ValueError("drift/dz/dt must be positive and diffusion nonnegative")
    if 2 * diffusion * dt / dz**2 + drift * dt / dz > 0.95:
        raise ValueError("unstable grid: reduce dt")
    n_cells = int(round(1 / dz))
    grid = (np.arange(n_cells) + 0.5) * dz
    times = np.arange(0, max_time + dt / 2, dt)
    p = np.zeros(grid.size)
    index = int(np.clip(np.floor(initial_progress / dz), 0, grid.size - 1))
    p[index] = 1 / dz
    densities = np.empty((times.size, grid.size))
    event = np.zeros(times.size)
    survival = np.zeros(times.size)
    densities[0] = p
    survival[0] = np.sum(p) * dz
    c = drift * dt / dz
    r = diffusion * dt / dz**2
    for t in range(1, times.size):
        old_mass = np.sum(p) * dz
        q = p.copy()
        q[1:-1] = p[1:-1] - c * (p[1:-1] - p[:-2]) + r * (p[2:] - 2 * p[1:-1] + p[:-2])
        # No flux through the left face; drift and diffusion can leave cell 0
        # only through its right face.
        q[0] = p[0] - c * p[0] + r * (p[1] - p[0])
        # Zero exterior density at the absorbing right face.
        q[-1] = p[-1] - c * (p[-1] - p[-2]) + r * (p[-2] - 2 * p[-1])
        q = np.maximum(q, 0.0)
        new_mass = np.sum(q) * dz
        event[t] = max((old_mass - new_mass) / dt, 0.0)
        p = q
        densities[t] = p
        survival[t] = new_mass
    return FokkerPlanckResult(times, grid, densities, event, survival)


def monte_carlo_first_passage(
    n: int,
    drift: float = 1 / 14,
    diffusion: float = 0.00015,
    dt: float = 0.02,
    max_time: float = 35.0,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = np.zeros(n)
    times = np.full(n, np.nan)
    crossed = np.zeros(n, dtype=bool)
    for step in range(1, int(max_time / dt) + 1):
        active = ~crossed
        z[active] += drift * dt + np.sqrt(2 * diffusion * dt) * rng.normal(size=active.sum())
        z[active] = np.maximum(z[active], 0)
        new = active & (z >= 1)
        times[new] = step * dt
        crossed |= new
        if crossed.all():
            break
    return times
