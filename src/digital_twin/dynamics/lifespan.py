from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import erf


WALLACE_KELSEY = {
    "amplitude": 5.56,
    "location": 25.6,
    "width": 52.7,
    "growth_scale": 0.074,
    "decline_scale": 24.5,
}


@dataclass
class LifespanSimulation:
    """Cycle-level and participant-level outputs of the lifespan theory."""

    cycles: pd.DataFrame
    participants: pd.DataFrame
    age_summary: pd.DataFrame
    reserve_curve: pd.DataFrame


def ovarian_reserve_log10(
    age_years: np.ndarray | float,
    *,
    log10_offset: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Wallace--Kelsey asymmetric double-Gaussian reserve curve.

    Age is in years from birth; negative values denote gestation. The offset
    represents participant variation in established reserve. This population
    model is not an individual fertility measurement.
    """

    age = np.asarray(age_years, dtype=float)
    p = WALLACE_KELSEY
    first = 1.0 + erf(
        (
            age
            - p["location"]
            + p["width"] / 2.0
        )
        / (p["growth_scale"] * np.sqrt(2.0))
    )
    second = 1.0 - erf(
        (
            age
            - p["location"]
            - p["width"] / 2.0
        )
        / (p["decline_scale"] * np.sqrt(2.0))
    )
    return p["amplitude"] / 4.0 * first * second + np.asarray(
        log10_offset, dtype=float
    )


def menopause_age_from_reserve(
    log10_offset: float,
    *,
    threshold_follicles: float = 1000.0,
) -> float:
    """First post-pubertal age at which the reserve curve crosses a threshold."""

    if threshold_follicles <= 0:
        raise ValueError("threshold_follicles must be positive")
    ages = np.linspace(30.0, 70.0, 8001)
    log_reserve = ovarian_reserve_log10(
        ages, log10_offset=log10_offset
    )
    target = np.log10(threshold_follicles)
    below = np.flatnonzero(log_reserve <= target)
    if not len(below):
        return float(ages[-1])
    index = int(below[0])
    if index == 0:
        return float(ages[0])
    left_age, right_age = ages[index - 1], ages[index]
    left_value, right_value = log_reserve[index - 1], log_reserve[index]
    fraction = (target - left_value) / (right_value - left_value)
    return float(left_age + fraction * (right_age - left_age))


def age_dependent_cycle_moments(
    age_years: np.ndarray | float,
    aggregate: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate published aggregate cycle mean and within-person spread."""

    required = {
        "age_midpoint",
        "mean_cycle_days",
        "mean_within_person_sd_days",
    }
    missing = required - set(aggregate)
    if missing:
        raise ValueError(f"aggregate table missing columns: {sorted(missing)}")
    ordered = aggregate.sort_values("age_midpoint")
    ages = np.asarray(age_years, dtype=float)
    mean = np.interp(
        ages,
        ordered["age_midpoint"],
        ordered["mean_cycle_days"],
        left=float(ordered["mean_cycle_days"].iloc[0]),
        right=float(ordered["mean_cycle_days"].iloc[-1]),
    )
    sd = np.interp(
        ages,
        ordered["age_midpoint"],
        ordered["mean_within_person_sd_days"],
        left=float(ordered["mean_within_person_sd_days"].iloc[0]),
        right=float(ordered["mean_within_person_sd_days"].iloc[-1]),
    )
    return mean, sd


def _inverse_gaussian_draw(
    mean: float,
    variance: float,
    rng: np.random.Generator,
) -> float:
    shape = mean**3 / max(variance, np.finfo(float).eps)
    return float(rng.wald(mean, shape))


def simulate_reproductive_lifespan(
    aggregate: pd.DataFrame,
    *,
    participants: int = 2000,
    seed: int = 20260727,
    menarche_mean_age: float = 12.5,
    menarche_sd_years: float = 1.0,
    reserve_log10_offset_sd: float = 0.30,
    participant_log_duration_sd: float = 0.07,
    menopause_threshold_follicles: float = 1000.0,
    luteal_mean_days: float = 13.5,
    follicular_variance_fraction: float = 0.75,
    maximum_age: float = 65.0,
    interruption_windows: tuple[dict[str, float | str], ...] = (),
) -> LifespanSimulation:
    """Simulate cycling and optional interruptions from menarche to menopause.

    Published age-band moments constrain local passage-time moments.
    ``interruption_windows`` are exogenous no-cycle scenarios with
    ``start_age``, ``end_age``, and ``state`` fields. They do not estimate
    pregnancy, postpartum, or contraceptive biology.
    """

    if participants < 1:
        raise ValueError("participants must be positive")
    if not 0 < follicular_variance_fraction < 1:
        raise ValueError("follicular_variance_fraction must lie in (0, 1)")
    windows = sorted(
        (
            {
                "start_age": float(window["start_age"]),
                "end_age": float(window["end_age"]),
                "state": str(window["state"]),
            }
            for window in interruption_windows
        ),
        key=lambda value: value["start_age"],
    )
    for previous, current in zip(windows, windows[1:]):
        if previous["end_age"] > current["start_age"]:
            raise ValueError("interruption windows must not overlap")
    if any(
        window["end_age"] <= window["start_age"] for window in windows
    ):
        raise ValueError("interruption end_age must exceed start_age")
    participant_seeds = np.random.SeedSequence(seed).spawn(participants)
    rows: list[dict[str, float | int | str]] = []
    people: list[dict[str, float | int]] = []
    ordered_aggregate = aggregate.sort_values("age_midpoint")
    aggregate_age = ordered_aggregate["age_midpoint"].to_numpy(dtype=float)
    aggregate_mean = ordered_aggregate["mean_cycle_days"].to_numpy(
        dtype=float
    )
    aggregate_sd = ordered_aggregate[
        "mean_within_person_sd_days"
    ].to_numpy(dtype=float)

    for participant in range(participants):
        rng = np.random.default_rng(participant_seeds[participant])
        menarche = float(
            np.clip(
                rng.normal(menarche_mean_age, menarche_sd_years),
                8.0,
                18.0,
            )
        )
        reserve_offset = float(
            rng.normal(0.0, reserve_log10_offset_sd)
        )
        duration_scale = float(
            np.exp(
                rng.normal(
                    -0.5 * participant_log_duration_sd**2,
                    participant_log_duration_sd,
                )
            )
        )
        menopause = min(
            menopause_age_from_reserve(
                reserve_offset,
                threshold_follicles=menopause_threshold_follicles,
            ),
            maximum_age,
        )
        age = menarche
        cycle_index = 0
        interruption_days = 0.0
        interruption_episodes = 0
        interrupted_partial_cycles = 0
        while age < menopause:
            active_window = next(
                (
                    window
                    for window in windows
                    if window["start_age"] <= age < window["end_age"]
                ),
                None,
            )
            if active_window is not None:
                interruption_end = min(
                    float(active_window["end_age"]), menopause
                )
                interruption_days += (
                    interruption_end - age
                ) * 365.2425
                interruption_episodes += 1
                age = interruption_end
                continue
            next_interruption = next(
                (
                    float(window["start_age"])
                    for window in windows
                    if float(window["start_age"]) > age
                ),
                np.inf,
            )
            total_mean = float(
                np.interp(
                    age,
                    aggregate_age,
                    aggregate_mean,
                    left=aggregate_mean[0],
                    right=aggregate_mean[-1],
                )
            )
            total_mean *= duration_scale
            total_sd = float(
                np.interp(
                    age,
                    aggregate_age,
                    aggregate_sd,
                    left=aggregate_sd[0],
                    right=aggregate_sd[-1],
                )
            )
            total_sd *= duration_scale
            total_variance = total_sd**2
            luteal_mean = min(luteal_mean_days, total_mean - 5.0)
            follicular_mean = total_mean - luteal_mean
            follicular_days = _inverse_gaussian_draw(
                follicular_mean,
                total_variance * follicular_variance_fraction,
                rng,
            )
            luteal_days = _inverse_gaussian_draw(
                luteal_mean,
                total_variance * (1.0 - follicular_variance_fraction),
                rng,
            )
            duration = follicular_days + luteal_days
            next_age = age + duration / 365.2425
            if next_age > next_interruption:
                interrupted_partial_cycles += 1
                age = next_interruption
                continue
            if next_age > menopause:
                break
            reserve = float(
                10.0
                ** ovarian_reserve_log10(
                    age, log10_offset=reserve_offset
                )
            )
            rows.append(
                {
                    "participant_id": f"L{participant:05d}",
                    "cycle_index": cycle_index,
                    "start_age_years": age,
                    "cycle_length_days": duration,
                    "follicular_days": follicular_days,
                    "luteal_days": luteal_days,
                    "reserve_follicles_population_model": reserve,
                }
            )
            age = next_age
            cycle_index += 1
        people.append(
            {
                "participant_id": f"L{participant:05d}",
                "menarche_age_years": menarche,
                "menopause_age_years": menopause,
                "reproductive_span_years": max(menopause - menarche, 0.0),
                "reserve_log10_offset": reserve_offset,
                "duration_scale": duration_scale,
                "simulated_cycle_count": cycle_index,
                "uninterrupted_cycle_count": (
                    cycle_index if not windows else np.nan
                ),
                "interruption_days": interruption_days,
                "interruption_episodes": interruption_episodes,
                "interrupted_partial_cycles": interrupted_partial_cycles,
            }
        )

    cycles = pd.DataFrame(rows)
    participant_table = pd.DataFrame(people)
    bins = np.arange(8, maximum_age + 1, 1.0)
    cycles["age_year"] = pd.cut(
        cycles["start_age_years"],
        bins=bins,
        right=False,
        labels=bins[:-1].astype(int),
    )
    within_person_age = (
        cycles.groupby(["participant_id", "age_year"], observed=True)
        .agg(
            participant_age_cycles=("cycle_length_days", "size"),
            participant_age_mean=("cycle_length_days", "mean"),
            participant_age_sd=("cycle_length_days", "std"),
        )
        .reset_index()
    )
    within_person_age = within_person_age.loc[
        within_person_age["participant_age_cycles"] >= 3
    ]
    age_summary = (
        cycles.groupby("age_year", observed=True)
        .agg(
            cycles=("cycle_length_days", "size"),
            participants=("participant_id", "nunique"),
            mean_cycle_days=("cycle_length_days", "mean"),
            sd_cycle_days=("cycle_length_days", "std"),
            mean_follicular_days=("follicular_days", "mean"),
            mean_luteal_days=("luteal_days", "mean"),
        )
        .reset_index()
        .rename(columns={"age_year": "age"})
    )
    age_summary["age"] = age_summary["age"].astype(int)
    within_summary = (
        within_person_age.groupby("age_year", observed=True)
        .agg(
            mean_within_person_sd_days=("participant_age_sd", "mean"),
            mean_participant_cycle_days=("participant_age_mean", "mean"),
        )
        .reset_index()
        .rename(columns={"age_year": "age"})
    )
    within_summary["age"] = within_summary["age"].astype(int)
    age_summary = age_summary.merge(within_summary, on="age", how="left")
    ages = np.linspace(-0.75, maximum_age, 1000)
    reserve_curve = pd.DataFrame(
        {
            "age_years": ages,
            "reserve_follicles_population_model": 10.0
            ** ovarian_reserve_log10(ages),
        }
    )
    return LifespanSimulation(
        cycles=cycles,
        participants=participant_table,
        age_summary=age_summary,
        reserve_curve=reserve_curve,
    )
