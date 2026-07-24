from __future__ import annotations

import numpy as np
import pandas as pd

from digital_twin.dynamics.coupled_fokker_planck import solve_coupled_cycle
from digital_twin.dynamics.coupled_ou_fokker_planck import (
    solve_coupled_ou_cycle,
)
from digital_twin.dynamics.first_passage import (
    dimensionless_groups,
    first_passage_density,
    first_passage_moments,
    first_passage_survival,
    sample_constant_first_passage,
    simulate_ou_first_passage,
)
from digital_twin.dynamics.identifiability import run_inverse_gaussian_sbc
from digital_twin.dynamics.lifespan import (
    menopause_age_from_reserve,
    ovarian_reserve_log10,
    simulate_reproductive_lifespan,
)
from digital_twin.dynamics.model_comparison import compare_duration_models
from digital_twin.dynamics.regime_map import build_regime_map
from digital_twin.dynamics.signatures import calculate_dynamical_signatures


def test_inverse_gaussian_limit_matches_exact_sampling() -> None:
    drift = 1.0 / 14.0
    diffusion = 0.00015
    theory = first_passage_moments(drift, diffusion)
    samples = sample_constant_first_passage(
        60_000, drift, diffusion, seed=811
    )
    assert abs(samples.mean() - theory.mean) < 0.08
    assert abs(samples.var() - theory.variance) < 0.12
    time = np.linspace(0.01, 50.0, 20_000)
    density = first_passage_density(time, drift, diffusion)
    assert abs(np.trapz(density, time) - 1.0) < 2e-3
    survival = first_passage_survival(time, drift, diffusion)
    assert np.all(np.diff(survival) <= 1e-12)


def test_dimensionless_groups_have_expected_scaling() -> None:
    groups = dimensionless_groups(
        drift=0.1,
        diffusion=0.002,
        kappa=0.4,
        sigma_log_speed=0.1,
    )
    assert groups.peclet == 50.0
    assert groups.relaxation_ratio == 4.0
    assert np.isclose(groups.stationary_log_speed_sd, 0.1 / np.sqrt(0.8))


def test_coupled_solver_conserves_probability_and_transfers_flux() -> None:
    result = solve_coupled_cycle(dt=0.02, dz=0.02, max_time=70.0)
    assert np.max(np.abs(result.total_mass - 1.0)) < 2e-10
    assert result.cumulative_ovulations[-1] > 1.5
    assert result.cumulative_menses[-1] > 1.0
    assert np.trapz(result.first_ovulation_density, result.time) > 0.99
    assert np.trapz(result.first_menses_density, result.time) > 0.98
    assert np.all(np.diff(result.first_cycle_survival) <= 1e-10)


def test_joint_progress_speed_pde_conserves_probability() -> None:
    result = solve_coupled_ou_cycle(
        dz=0.04,
        dell=0.08,
        dt=0.02,
        max_time=35.0,
        store_every=10,
    )
    assert np.max(np.abs(result.total_mass - 1.0)) < 2e-10
    assert np.min(result.final_follicular_density) >= 0
    assert np.min(result.final_luteal_density) >= 0
    assert np.max(result.ovulation_flux) > 0
    assert np.max(result.menses_flux) > 0


def test_ou_limit_reduces_to_nearly_deterministic_passage() -> None:
    samples = simulate_ou_first_passage(
        5_000,
        mean_speed=1.0 / 14.0,
        kappa=0.4,
        sigma_log_speed=0.0,
        diffusion=1e-8,
        dt=0.02,
        seed=92,
    )
    assert np.isfinite(samples).all()
    assert abs(np.mean(samples) - 14.0) < 0.06
    assert np.std(samples) < 0.03


def test_empirical_signatures_and_nested_models_are_finite() -> None:
    rows = []
    for participant in range(8):
        for cycle in range(4):
            rows.append(
                {
                    "participant_id": f"P{participant}",
                    "cycle_id": f"P{participant}-C{cycle}",
                    "cycle_start_day": 1 + 35 * cycle,
                    "cycle_length_days": 27.0 + participant / 2 + cycle % 2,
                    "eligible_for_primary_evaluation": True,
                }
            )
    cycles = pd.DataFrame(rows)
    references = cycles.iloc[:8][["participant_id", "cycle_id"]].copy()
    references["event_type"] = "urinary_lh_positive"
    references["cycle_day"] = 13
    observations = cycles[["participant_id", "cycle_id"]].copy()
    observations["signal_name"] = "signal"
    observations["value"] = 1.0
    result = calculate_dynamical_signatures(
        cycles,
        references,
        observations,
        source="fixture",
        signals=("signal",),
    )
    assert result.summary.iloc[0]["cycles"] == 32
    assert not result.empirical_hazard.empty
    comparison = compare_duration_models(cycles, source="fixture")
    assert len(comparison) == 6
    assert np.isfinite(comparison["aic"]).all()
    assert np.isclose(comparison["akaike_weight"].sum(), 1.0)


def test_regime_map_and_small_sbc_execute() -> None:
    regime = build_regime_map(
        mean_duration=14.0,
        peclet_values=(10.0, 80.0),
        relaxation_values=(0.2, 3.0),
        stationary_log_speed_sd=0.10,
        trajectories=500,
        dt=0.10,
        max_time=50.0,
        seed=5,
    )
    assert len(regime) == 4
    assert regime["regime"].notna().all()
    draws, summary = run_inverse_gaussian_sbc(
        sample_sizes=(10, 30),
        replicates=5,
        drift_range=(1 / 34, 1 / 24),
        diffusion_range=(5e-5, 8e-4),
        grid_size=18,
        seed=7,
    )
    assert len(draws) == 10
    assert len(summary) == 2
    assert summary["drift_contraction"].between(0, 1).all()


def test_lifespan_reserve_and_state_boundaries() -> None:
    ages = np.linspace(0, 60, 500)
    reserve = ovarian_reserve_log10(ages)
    assert np.all(np.diff(reserve) < 0)
    menopause = menopause_age_from_reserve(0.0)
    assert 48.0 < menopause < 51.0
    aggregate = pd.DataFrame(
        {
            "age_midpoint": [18.0, 35.0, 52.0],
            "mean_cycle_days": [29.0, 27.5, 29.0],
            "mean_within_person_sd_days": [4.2, 3.7, 6.0],
        }
    )
    result = simulate_reproductive_lifespan(
        aggregate,
        participants=12,
        seed=14,
    )
    assert not result.cycles.empty
    bounds = result.participants.set_index("participant_id")
    merged = result.cycles.join(
        bounds[["menarche_age_years", "menopause_age_years"]],
        on="participant_id",
    )
    assert (
        merged["start_age_years"] >= merged["menarche_age_years"]
    ).all()
    assert (
        merged["start_age_years"] < merged["menopause_age_years"]
    ).all()
    assert result.participants["uninterrupted_cycle_count"].min() > 300


def test_lifespan_interruption_scenarios_pause_cycles_not_aging() -> None:
    aggregate = pd.DataFrame(
        {
            "age_midpoint": [18.0, 35.0, 52.0],
            "mean_cycle_days": [29.0, 27.5, 29.0],
            "mean_within_person_sd_days": [4.2, 3.7, 6.0],
        }
    )
    baseline = simulate_reproductive_lifespan(
        aggregate, participants=20, seed=19
    )
    interrupted = simulate_reproductive_lifespan(
        aggregate,
        participants=20,
        seed=19,
        interruption_windows=(
            {
                "state": "cycle_suppressing_hormonal_exposure",
                "start_age": 20.0,
                "end_age": 25.0,
            },
        ),
    )
    assert np.allclose(
        baseline.participants["menopause_age_years"],
        interrupted.participants["menopause_age_years"],
    )
    assert (
        interrupted.participants["simulated_cycle_count"].mean()
        < baseline.participants["simulated_cycle_count"].mean() - 50
    )
    assert interrupted.participants["interruption_days"].mean() > 1800
    assert interrupted.participants["uninterrupted_cycle_count"].isna().all()
