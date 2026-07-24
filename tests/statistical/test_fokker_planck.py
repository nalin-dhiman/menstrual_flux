import numpy as np

from digital_twin.fokker_planck import monte_carlo_first_passage, solve_constant_coefficients


def test_fokker_planck_agrees_with_monte_carlo():
    result = solve_constant_coefficients()
    samples = monte_carlo_first_passage(5000, seed=22)
    dt = result.time[1] - result.time[0]
    mass = result.event_density * dt
    pde_mean = np.sum(result.time * mass) / mass.sum()
    assert abs(pde_mean - np.nanmean(samples)) < 0.35
    assert abs(mass.sum() - 1) < 0.01


def test_grid_convergence():
    coarse = solve_constant_coefficients(dz=0.02, dt=0.04)
    fine = solve_constant_coefficients(dz=0.01, dt=0.02)
    def mean(result):
        dt = result.time[1] - result.time[0]
        mass = result.event_density * dt
        return np.sum(result.time * mass) / mass.sum()
    assert abs(mean(coarse) - mean(fine)) < 0.25
