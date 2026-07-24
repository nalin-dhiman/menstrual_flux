from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .lifespan import (
    age_dependent_cycle_moments,
    ovarian_reserve_log10,
    simulate_reproductive_lifespan,
)

COLORS = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "magenta": "#CC79A7",
    "purple": "#7B3294",
    "red": "#B2182B",
    "neutral": "#4D4D4D",
}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.6,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _style_axis(axis: plt.Axes, *, grid: bool = True) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(direction="out", length=3.5, width=0.8)
    if grid:
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.55)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    required = {
        "experiment",
        "aggregate_age_data",
        "simulation",
        "sensitivity",
        "interruption_scenarios",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"missing lifespan config sections: {sorted(missing)}")
    return config


def _figure(
    result,
    aggregate: pd.DataFrame,
    output: Path,
) -> None:
    figure, axes = plt.subplots(
        3, 2, figsize=(11.0, 12.2), constrained_layout=True
    )
    reserve = result.reserve_curve
    axes[0, 0].semilogy(
        reserve["age_years"],
        reserve["reserve_follicles_population_model"],
        color=COLORS["purple"],
        lw=2.3,
    )
    axes[0, 0].axhline(1000, color="black", ls="--", lw=1)
    axes[0, 0].text(
        64,
        1120,
        "Cessation threshold: 1,000 follicles",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=COLORS["neutral"],
    )
    axes[0, 0].set(
        xlabel="Age (years)",
        ylabel="Population-model reserve",
        title="A  Slow ovarian-reserve coordinate",
        xlim=(-1, 66),
    )

    minimum_participants = max(10, int(0.05 * len(result.participants)))
    age = result.age_summary.loc[
        (result.age_summary["participants"] >= minimum_participants)
        & result.age_summary["age"].between(12, 55)
    ]
    axes[0, 1].plot(
        age["age"],
        age["mean_participant_cycle_days"],
        color=COLORS["blue"],
        lw=2,
    )
    axes[0, 1].scatter(
        aggregate["age_midpoint"],
        aggregate["mean_cycle_days"],
        color=COLORS["red"],
        zorder=3,
    )
    axes[0, 1].set(
        xlabel="Age (years)",
        ylabel="Mean cycle length (days)",
        xlim=(12, 56),
        title="B  Age-dependent cycle duration",
    )

    axes[1, 0].plot(
        age["age"],
        age["mean_within_person_sd_days"],
        color=COLORS["green"],
        lw=2,
    )
    axes[1, 0].scatter(
        aggregate["age_midpoint"],
        aggregate["mean_within_person_sd_days"],
        color=COLORS["red"],
        zorder=3,
    )
    axes[1, 0].set(
        xlabel="Age (years)",
        ylabel="Within-person cycle SD (days)",
        xlim=(12, 56),
        title="C  Age-dependent cycle variability",
    )

    people = result.participants
    grid = np.linspace(8, 65, 400)
    cycling = np.array(
        [
            np.mean(
                (people["menarche_age_years"] <= value)
                & (people["menopause_age_years"] > value)
            )
            for value in grid
        ]
    )
    axes[1, 1].plot(grid, cycling, color=COLORS["orange"], lw=2.3)
    axes[1, 1].set(
        xlabel="Age (years)",
        ylabel="Probability in cycling state",
        ylim=(-0.03, 1.03),
        xlim=(8, 65),
        title="D  Activation and reserve-threshold cessation",
    )

    ordered_people = result.participants.sort_values("duration_scale")
    quantile_specs = [
        (0.10, "10th percentile", COLORS["blue"]),
        (0.50, "50th percentile", COLORS["green"]),
        (0.90, "90th percentile", COLORS["orange"]),
    ]
    trajectory_handles = []
    for quantile, label, color in quantile_specs:
        participant = ordered_people.iloc[
            int(round(quantile * (len(ordered_people) - 1)))
        ]["participant_id"]
        group = result.cycles.loc[
            result.cycles["participant_id"] == participant
        ].sort_values("start_age_years")
        axes[2, 0].plot(
            group["start_age_years"],
            group["cycle_length_days"],
            lw=0.45,
            alpha=0.20,
            color=color,
            rasterized=True,
        )
        rolling = group["cycle_length_days"].rolling(
            12, center=True, min_periods=6
        ).mean()
        axes[2, 0].plot(
            group["start_age_years"],
            rolling,
            lw=1.45,
            color=color,
        )
        trajectory_handles.append(
            Line2D([], [], color=color, label=label)
        )
    axes[2, 0].set(
        xlabel="Age (years)",
        ylabel="Cycle length (days)",
        xlim=(12, 57),
        ylim=(14, 55),
        title="E  Example cycles and 12-cycle moving means",
    )
    axes[2, 0].legend(
        handles=trajectory_handles,
        title="Participant duration-scale quantile",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=3,
        frameon=False,
    )

    counts = people["uninterrupted_cycle_count"].dropna()
    q05, mean_count, q95 = (
        float(counts.quantile(0.05)),
        float(counts.mean()),
        float(counts.quantile(0.95)),
    )
    axes[2, 1].hist(
        counts,
        bins=28,
        color=COLORS["neutral"],
        edgecolor="white",
    )
    axes[2, 1].axvspan(q05, q95, color=COLORS["blue"], alpha=0.12)
    axes[2, 1].axvline(mean_count, color=COLORS["blue"], lw=1.8)
    axes[2, 1].set(
        xlabel="Uninterrupted cycles from menarche to menopause",
        ylabel="Simulated participants",
        title="F  Counterfactual lifetime cycle-count distribution",
    )
    axes[2, 1].legend(
        handles=[
            Line2D(
                [],
                [],
                color=COLORS["blue"],
                lw=1.8,
                label=f"Mean = {mean_count:.0f}",
            ),
            Patch(
                facecolor=COLORS["blue"],
                alpha=0.12,
                label=f"5th–95th = {q05:.0f}–{q95:.0f}",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        frameon=False,
    )
    figure.legend(
        handles=[
            Line2D([], [], color=COLORS["blue"], lw=2, label="Simulation"),
            Line2D(
                [],
                [],
                color=COLORS["red"],
                marker="o",
                linestyle="none",
                label="Published aggregate constraint",
            ),
        ],
        loc="outside upper center",
        ncol=2,
        frameon=False,
    )
    for axis in axes.flat:
        _style_axis(axis)
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _run_sensitivity(
    aggregate: pd.DataFrame,
    simulation: dict[str, Any],
    sensitivity: dict[str, Any],
) -> pd.DataFrame:
    participants = int(sensitivity["participants_per_setting"])
    base = dict(simulation)
    base["participants"] = participants
    base["seed"] = int(simulation["seed"]) + 1000
    settings: list[tuple[str, str, float | None]] = [
        ("baseline", "baseline", None)
    ]
    for parameter, values in sensitivity["parameters"].items():
        for value in values:
            settings.append((f"{parameter}={value}", parameter, float(value)))
    rows = []
    for label, parameter, value in settings:
        arguments = dict(base)
        if value is not None:
            arguments[parameter] = value
        result = simulate_reproductive_lifespan(aggregate, **arguments)
        people = result.participants
        rows.append(
            {
                "setting": label,
                "parameter": parameter,
                "value": value,
                "participants": len(people),
                "mean_menarche_age": people[
                    "menarche_age_years"
                ].mean(),
                "mean_menopause_age": people[
                    "menopause_age_years"
                ].mean(),
                "mean_reproductive_span_years": people[
                    "reproductive_span_years"
                ].mean(),
                "mean_cycle_count": people[
                    "simulated_cycle_count"
                ].mean(),
                "mean_cycle_length_days": result.cycles[
                    "cycle_length_days"
                ].mean(),
            }
        )
    table = pd.DataFrame(rows)
    baseline = table.iloc[0]
    table["delta_mean_cycle_count"] = (
        table["mean_cycle_count"] - baseline["mean_cycle_count"]
    )
    table["delta_mean_menopause_age"] = (
        table["mean_menopause_age"] - baseline["mean_menopause_age"]
    )
    return table


def _run_interruption_scenarios(
    aggregate: pd.DataFrame,
    simulation: dict[str, Any],
    scenario_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    participants = int(scenario_config["participants_per_scenario"])
    base = dict(simulation)
    base["participants"] = participants
    base["seed"] = int(simulation["seed"]) + 2000
    rows = []
    schedules = []
    for scenario, windows in scenario_config["scenarios"].items():
        arguments = dict(base)
        arguments["interruption_windows"] = tuple(windows)
        result = simulate_reproductive_lifespan(aggregate, **arguments)
        people = result.participants
        rows.append(
            {
                "scenario": scenario,
                "participants": len(people),
                "mean_simulated_cycle_count": people[
                    "simulated_cycle_count"
                ].mean(),
                "q05_simulated_cycle_count": people[
                    "simulated_cycle_count"
                ].quantile(0.05),
                "q95_simulated_cycle_count": people[
                    "simulated_cycle_count"
                ].quantile(0.95),
                "mean_interruption_years": people[
                    "interruption_days"
                ].mean()
                / 365.2425,
                "mean_interruption_episodes": people[
                    "interruption_episodes"
                ].mean(),
                "mean_interrupted_partial_cycles": people[
                    "interrupted_partial_cycles"
                ].mean(),
                "mean_menopause_age": people[
                    "menopause_age_years"
                ].mean(),
            }
        )
        for window in windows:
            schedules.append({"scenario": scenario, **window})
    table = pd.DataFrame(rows)
    baseline = float(
        table.loc[
            table["scenario"] == "uninterrupted",
            "mean_simulated_cycle_count",
        ].iloc[0]
    )
    table["mean_cycles_removed_vs_uninterrupted"] = (
        baseline - table["mean_simulated_cycle_count"]
    )
    return table, pd.DataFrame(schedules)


def _extension_figure(
    sensitivity: pd.DataFrame,
    scenarios: pd.DataFrame,
    schedules: pd.DataFrame,
    output: Path,
) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.08, 1.0]},
    )
    nonbaseline = sensitivity.loc[
        sensitivity["setting"] != "baseline"
    ].copy()
    label_formatters = {
        "menarche_mean_age": lambda value: f"Menarche mean: {value:g} y",
        "reserve_log10_offset_sd": lambda value: (
            f"Reserve-offset SD: {value:.2f}"
        ),
        "menopause_threshold_follicles": lambda value: (
            f"Cessation threshold: {value:,.0f}"
        ),
        "participant_log_duration_sd": lambda value: (
            f"Duration heterogeneity SD: {value:.2f}"
        ),
        "luteal_mean_days": lambda value: f"Luteal mean: {value:g} d",
        "follicular_variance_fraction": lambda value: (
            f"Follicular variance: {100 * value:.0f}%"
        ),
    }
    labels = [
        label_formatters[row.parameter](float(row.value))
        for row in nonbaseline.itertuples()
    ]
    y = np.arange(len(nonbaseline))
    cycle_bars = axes[0, 0].barh(
        y,
        nonbaseline["delta_mean_cycle_count"],
        color=np.where(
            nonbaseline["delta_mean_cycle_count"] >= 0,
            COLORS["blue"],
            COLORS["red"],
        ),
        zorder=3,
    )
    axes[0, 0].axvline(0, color="black", lw=0.8)
    axes[0, 0].set_yticks(y, labels, fontsize=8)
    axes[0, 0].set(
        xlabel="Change in lifetime cycle count",
        title="A  One-at-a-time change in cycle count",
    )
    for bar, value in zip(
        cycle_bars, nonbaseline["delta_mean_cycle_count"]
    ):
        value = float(value)
        if abs(value) >= 8:
            axes[0, 0].text(
                value / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{value:+.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                fontweight="bold",
            )
        else:
            axes[0, 0].text(
                value + (1.1 if value >= 0 else -1.1),
                bar.get_y() + bar.get_height() / 2,
                f"{value:+.1f}",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=7,
            )

    cessation_bars = axes[0, 1].barh(
        y,
        nonbaseline["delta_mean_menopause_age"],
        color=COLORS["purple"],
        zorder=3,
    )
    axes[0, 1].axvline(0, color="black", lw=0.8)
    axes[0, 1].set_yticks(y, [])
    axes[0, 1].set(
        xlabel="Change in model menopause age (years)",
        title="B  One-at-a-time change in cessation age",
    )
    for bar, value in zip(
        cessation_bars, nonbaseline["delta_mean_menopause_age"]
    ):
        value = float(value)
        if abs(value) < 0.005:
            continue
        if abs(value) >= 1:
            axes[0, 1].text(
                value / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                fontweight="bold",
            )
        else:
            axes[0, 1].text(
                value + (0.08 if value >= 0 else -0.08),
                bar.get_y() + bar.get_height() / 2,
                f"{value:+.2f}",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=7,
            )

    scenario_display = {
        "uninterrupted": "Uninterrupted",
        "two_births_short_postpartum": "2 births + 42-day postpartum",
        "two_births_six_month_lam": "2 births + 6-month LAM",
        "two_births_twelve_month_postpartum": "2 births + 12-month postpartum",
        "cycle_suppressing_hormonal_exposure": "Suppressing exposure",
        "combined_example": "Combined example",
    }
    scenario_labels = scenarios["scenario"].map(scenario_display)
    horizontal_error = np.vstack(
        [
            scenarios["mean_simulated_cycle_count"]
            - scenarios["q05_simulated_cycle_count"],
            scenarios["q95_simulated_cycle_count"]
            - scenarios["mean_simulated_cycle_count"],
        ]
    )
    scenario_y = np.arange(len(scenarios))[::-1]
    axes[1, 0].errorbar(
        scenarios["mean_simulated_cycle_count"],
        scenario_y,
        xerr=horizontal_error,
        fmt="o",
        markersize=5,
        color=COLORS["green"],
        ecolor=COLORS["neutral"],
        elinewidth=1.2,
        capsize=3,
        zorder=3,
    )
    baseline_count = float(
        scenarios.loc[
            scenarios["scenario"].eq("uninterrupted"),
            "mean_simulated_cycle_count",
        ].iloc[0]
    )
    axes[1, 0].axvline(
        baseline_count,
        color=COLORS["neutral"],
        linestyle="--",
        linewidth=0.9,
    )
    axes[1, 0].set_yticks(scenario_y, scenario_labels, fontsize=8)
    axes[1, 0].set(
        xlabel="Completed cycles from menarche to modeled menopause",
        title="C  Scenario mean and 5th–95th percentiles",
        xlim=(220, 620),
    )
    for x_value, y_value in zip(
        scenarios["mean_simulated_cycle_count"], scenario_y
    ):
        axes[1, 0].text(
            float(x_value) + 7,
            y_value,
            f"{float(x_value):.0f}",
            va="center",
            fontsize=7,
            color=COLORS["neutral"],
        )

    state_colors = {
        "pregnancy": "#E69F00",
        "postpartum_nonlactating_assumption": "#F0E442",
        "postpartum_lactational_amenorrhea": "#0072B2",
        "extended_postpartum_amenorrhea_assumption": "#CC79A7",
        "cycle_suppressing_hormonal_exposure": "#009E73",
    }
    scenario_order = [
        value
        for value in scenarios["scenario"]
        if value != "uninterrupted"
    ]
    schedule_y = np.arange(len(scenario_order))[::-1]
    for row_index, scenario in zip(schedule_y, scenario_order):
        group = schedules.loc[schedules["scenario"] == scenario]
        for _, window in group.iterrows():
            axes[1, 1].broken_barh(
                [
                    (
                        window["start_age"],
                        window["end_age"] - window["start_age"],
                    )
                ],
                (row_index - 0.35, 0.7),
                facecolors=state_colors[window["state"]],
                edgecolors="white",
                linewidth=0.4,
            )
    axes[1, 1].set_yticks(
        schedule_y,
        pd.Series(scenario_order).map(scenario_display),
        fontsize=8,
    )
    axes[1, 1].set(
        xlim=(18, 42),
        ylim=(-0.6, len(scenario_order) - 0.4),
        xlabel="Age (years)",
        title="D  Assumed interruption schedules",
    )
    state_display = {
        "pregnancy": "pregnancy",
        "postpartum_nonlactating_assumption": "short postpartum",
        "postpartum_lactational_amenorrhea": "6-mo LAM",
        "extended_postpartum_amenorrhea_assumption": "12-mo postpartum",
        "cycle_suppressing_hormonal_exposure": "suppressing exposure",
    }
    figure.legend(
        handles=[
            Patch(facecolor=color, label=state_display[state])
            for state, color in state_colors.items()
        ],
        loc="outside upper center",
        ncol=5,
        frameon=False,
    )
    _style_axis(axes[0, 0])
    _style_axis(axes[0, 1])
    _style_axis(axes[1, 0])
    _style_axis(axes[1, 1], grid=False)
    axes[1, 1].grid(axis="x", color="#E6E6E6", linewidth=0.55)
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def run_lifespan_theory(
    config_path: Path,
    project_root: Path,
    *,
    force: bool = False,
) -> Path:
    config_path = Path(config_path)
    project_root = Path(project_root).resolve()
    config = _load(config_path)
    output = project_root / config["experiment"]["output_dir"]
    if output.exists() and any(output.iterdir()) and not force:
        raise FileExistsError(
            f"{output} is not empty; use --force for a theoretical rerun"
        )
    output.mkdir(parents=True, exist_ok=True)
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    aggregate_path = project_root / config["aggregate_age_data"]["path"]
    aggregate = pd.read_csv(aggregate_path)
    result = simulate_reproductive_lifespan(
        aggregate, **config["simulation"]
    )
    result.participants.to_csv(
        output / "lifespan_participants.csv", index=False
    )
    result.age_summary.to_csv(output / "lifespan_age_summary.csv", index=False)
    result.reserve_curve.to_csv(
        output / "ovarian_reserve_population_curve.csv", index=False
    )
    sample_ids = set(
        result.participants["participant_id"].iloc[:25].tolist()
    )
    result.cycles.loc[
        result.cycles["participant_id"].isin(sample_ids)
    ].to_csv(output / "lifespan_cycle_sample.csv", index=False)
    sensitivity = _run_sensitivity(
        aggregate, config["simulation"], config["sensitivity"]
    )
    sensitivity.to_csv(
        output / "lifespan_sensitivity_oat.csv", index=False
    )
    interruption_scenarios, interruption_schedule = (
        _run_interruption_scenarios(
            aggregate,
            config["simulation"],
            config["interruption_scenarios"],
        )
    )
    interruption_scenarios.to_csv(
        output / "lifespan_interruption_scenarios.csv", index=False
    )
    interruption_schedule.to_csv(
        output / "lifespan_interruption_schedule.csv", index=False
    )

    target_mean, target_sd = age_dependent_cycle_moments(
        aggregate["age_midpoint"].to_numpy(), aggregate
    )
    simulated_mean = np.interp(
        aggregate["age_midpoint"],
        result.age_summary["age"],
        result.age_summary["mean_participant_cycle_days"],
    )
    simulated_sd = np.interp(
        aggregate["age_midpoint"],
        result.age_summary["age"],
        result.age_summary["mean_within_person_sd_days"],
    )
    checks = pd.DataFrame(
        [
            {
                "check": "aggregate_mean_cycle_rmse_days",
                "value": float(
                    np.sqrt(np.mean((simulated_mean - target_mean) ** 2))
                ),
                "criterion": "<0.35",
            },
            {
                "check": "aggregate_within_sd_rmse_days",
                "value": float(
                    np.sqrt(np.mean((simulated_sd - target_sd) ** 2))
                ),
                "criterion": "<0.50",
            },
            {
                "check": "reserve_monotone_after_birth",
                "value": float(
                    np.max(
                        np.diff(
                            ovarian_reserve_log10(
                                np.linspace(0, 65, 2000)
                            )
                        )
                    )
                ),
                "criterion": "<=0",
            },
        ]
    )
    checks["passed"] = [
        checks.iloc[0]["value"] < 0.35,
        checks.iloc[1]["value"] < 0.50,
        checks.iloc[2]["value"] <= 1e-12,
    ]
    checks.to_csv(output / "lifespan_theory_checks.csv", index=False)
    _figure(
        result,
        aggregate,
        figure_dir / "Fig6_reproductive_lifespan_theory",
    )
    _extension_figure(
        sensitivity,
        interruption_scenarios,
        interruption_schedule,
        figure_dir / "Fig7_lifespan_sensitivity_and_interruptions",
    )

    people = result.participants
    summary = {
        "simulated_participants": int(len(people)),
        "simulated_cycles": int(len(result.cycles)),
        "mean_menarche_age": float(people["menarche_age_years"].mean()),
        "mean_menopause_age": float(people["menopause_age_years"].mean()),
        "sd_menopause_age": float(people["menopause_age_years"].std()),
        "mean_uninterrupted_cycle_count": float(
            people["uninterrupted_cycle_count"].mean()
        ),
        "q05_uninterrupted_cycle_count": float(
            people["uninterrupted_cycle_count"].quantile(0.05)
        ),
        "q95_uninterrupted_cycle_count": float(
            people["uninterrupted_cycle_count"].quantile(0.95)
        ),
    }
    report = output / "LIFESPAN_THEORY_REPORT.md"
    report.write_text(
        "\n".join(
            [
                "# Theoretical reproductive-lifespan extension",
                "",
                "## Status",
                "",
                "This is a theoretical, aggregate-constrained extension. The local "
                "Salzburg and Soochow/Fudan cohorts contain no individual ages "
                "and do not validate lifespan dynamics.",
                "",
                "The simulator represents premenarche, age-dependent cycling, "
                "and postmenopause. The ovarian-reserve curve controls stochastic "
                "cessation; published age-band cycle moments control local "
                "first-passage durations.",
                "",
                "## Numerical summary",
                "",
                pd.DataFrame([summary]).to_markdown(index=False),
                "",
                "## Checks",
                "",
                checks.to_markdown(index=False),
                "",
                "## One-at-a-time sensitivity",
                "",
                sensitivity.to_markdown(index=False),
                "",
                "These settings vary one assumption at a time. They do not "
                "represent posterior uncertainty or population prevalence.",
                "",
                "## Exogenous interruption scenarios",
                "",
                interruption_scenarios.to_markdown(index=False),
                "",
                "Pregnancy, postpartum amenorrhea, and cycle-suppressing hormonal "
                "exposure are imposed schedules. They are not fitted event rates "
                "and do not represent every contraceptive method or bleeding "
                "pattern.",
                "",
                "## Interpretation boundary",
                "",
                "- The baseline lifetime cycle count assumes uninterrupted "
                "natural cycling.",
                "- Interruption schedules are alternative assumptions, not "
                "participant predictions.",
                "- The hormonal-exposure state represents only a fully "
                "cycle-suppressing scenario; it must not be generalized to all "
                "contraceptive methods or to withdrawal bleeding.",
                "- Illness, surgery, treatment, pregnancy loss, and variable "
                "gestational length remain excluded.",
                "- The reserve curve is a population model, not an individual "
                "fertility measurement.",
                "- Age-band moments constrain the simulator; reproducing them is "
                "not independent validation.",
                "- Ages from menarche through 17 years extrapolate the youngest "
                "available 18--25-year aggregate band.",
                "- Age-summary curves are displayed only where at least 5% of "
                "the simulated cohort remains represented.",
                "- The model does not infer childhood endocrine mechanisms or "
                "claim that ovarian reserve alone causes cycle-length changes.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "lifespan_manifest.json"
    )
    manifest = {
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "claim_scope": "theoretical_aggregate_constrained_lifespan_extension",
        "independent_lifespan_validation": False,
        "config_sha256": _sha256(config_path),
        "aggregate_sha256": _sha256(aggregate_path),
        "protocol_sha256": _sha256(
            project_root / "docs/lifespan_theory_protocol.md"
        ),
        "code_sha256": {
            "src/digital_twin/dynamics/lifespan.py": _sha256(
                project_root / "src/digital_twin/dynamics/lifespan.py"
            ),
            "src/digital_twin/dynamics/lifespan_experiment.py": _sha256(
                project_root
                / "src/digital_twin/dynamics/lifespan_experiment.py"
            ),
        },
        "all_checks_passed": bool(checks["passed"].all()),
        "outputs_sha256": {
            str(path.relative_to(output)): _sha256(path) for path in files
        },
    }
    (output / "lifespan_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report.resolve()
