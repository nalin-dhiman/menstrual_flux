from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"f": "#2A9D8F", "l": "#E76F51", "navy": "#264653", "gold": "#E9C46A", "blue": "#457B9D"}


def _save(fig: plt.Figure, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_milestone_figures(
    truth: pd.DataFrame,
    observations: pd.DataFrame,
    filter_summary: pd.DataFrame,
    forecast_samples: pd.DataFrame,
    calibration: pd.DataFrame,
    metrics: pd.DataFrame,
    parameter_recovery: pd.DataFrame,
    variability: pd.DataFrame,
    robustness: pd.DataFrame,
    fp: pd.DataFrame,
    output_dir: Path,
) -> None:
    participant = truth["participant_id"].iloc[0]
    t = truth[truth["participant_id"] == participant].head(90)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t["day_in_study"], t["progress"], color=COLORS["navy"])
    axes[0].fill_between(t["day_in_study"], 0, 1, where=t["stage"].eq("L"), alpha=0.15, color=COLORS["l"], label="luteal")
    axes[0].set_ylabel("stage progress")
    for signal, ax in zip(("temperature", "lh"), axes[1:]):
        o = observations[(observations["participant_id"] == participant) & observations["signal_name"].eq(signal) & observations["is_observed"]].head(90)
        ax.plot(np.arange(len(o)), o["value"], marker=".", lw=0.8, color=COLORS["blue"])
        ax.set_ylabel(signal)
    axes[-1].set_xlabel("study day")
    fig.suptitle("Synthetic latent trajectory and multimodal observations")
    _save(fig, output_dir, "synthetic_trajectories")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(filter_summary["day"], filter_summary["p_luteal"], color=COLORS["l"], label="P(luteal)")
    ax.fill_between(filter_summary["day"], 0, filter_summary["p_luteal"], color=COLORS["l"], alpha=0.18)
    ax.set(xlabel="day since reported menses", ylabel="posterior probability", ylim=(0, 1), title="Particle-filter stage posterior")
    ax.legend()
    _save(fig, output_dir, "particle_filter_posterior")

    for event, title in (("ovulation", "Ovulation transition posterior"), ("next_menses", "Next-menses forecast")):
        values = forecast_samples.loc[forecast_samples["event_type"].eq(event), "event_day"].dropna()
        fig, ax = plt.subplots(figsize=(8, 4))
        bins = np.arange(np.floor(values.min()) - 0.5, np.ceil(values.max()) + 1.5) if len(values) else np.arange(0, 2)
        ax.hist(values, bins=bins, density=True, color=COLORS["blue"], alpha=0.8)
        ax.set(xlabel="cycle day", ylabel="posterior probability mass", title=title)
        _save(fig, output_dir, f"{event}_posterior")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="0.45", label="ideal")
    ax.plot(calibration["nominal"], calibration["empirical"], "o-", color=COLORS["navy"], label="digital twin")
    ax.set(xlabel="nominal coverage", ylabel="empirical coverage", xlim=(0.4, 1), ylim=(0.4, 1), title="Next-menses interval calibration")
    ax.legend()
    _save(fig, output_dir, "calibration_curve")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    grouped = metrics.groupby("model", as_index=False)["absolute_error"].mean().sort_values("absolute_error")
    ax.bar(grouped["model"], grouped["absolute_error"], color=[COLORS["blue"], COLORS["gold"], COLORS["f"], COLORS["l"]][:len(grouped)])
    ax.set(ylabel="mean absolute error (days)", title="Held-out synthetic next-menses comparison")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, output_dir, "baseline_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(parameter_recovery["true_follicular_log_speed"], parameter_recovery["recovered_follicular_log_speed"], color=COLORS["f"])
    axes[1].scatter(parameter_recovery["true_luteal_log_speed"], parameter_recovery["recovered_luteal_log_speed"], color=COLORS["l"])
    for ax, label in zip(axes, ("follicular", "luteal")):
        low, high = ax.get_xlim()
        ax.plot([low, high], [low, high], "--", color="0.5")
        ax.set(xlabel="true log speed", ylabel="recovered log speed", title=label)
    fig.suptitle("Parameter recovery from simulated event times")
    _save(fig, output_dir, "parameter_recovery")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    components = variability[variability["component"].isin(["follicular_variance", "luteal_variance", "twice_stage_covariance", "between_person", "estimated_drift", "residual_within"])]
    errors = np.vstack([
        components["estimate_days_squared"] - components["lower_95"],
        components["upper_95"] - components["estimate_days_squared"],
    ])
    labels = {
        "follicular_variance": "follicular\nstage",
        "luteal_variance": "luteal\nstage",
        "twice_stage_covariance": "2 × stage\ncovariance",
        "between_person": "between\nperson",
        "estimated_drift": "slow\ndrift",
        "residual_within": "residual\nwithin person",
    }
    ax.bar([labels[x] for x in components["component"]], components["estimate_days_squared"], yerr=errors, color=COLORS["blue"], capsize=3)
    ax.set(ylabel="variance (days²)", title="Synthetic cycle-variability decomposition")
    ax.tick_params(axis="x", rotation=0)
    _save(fig, output_dir, "variability_decomposition")

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(robustness))
    ax.bar(x - 0.18, robustness["complete_width_90"], width=0.36, label="complete", color=COLORS["f"])
    ax.bar(x + 0.18, robustness["missing_width_90"], width=0.36, label="missing", color=COLORS["l"])
    ax.set(xlabel="evaluated cycle", ylabel="90% forecast width (days)", title="Uncertainty under missing observations")
    ax.legend()
    _save(fig, output_dir, "missingness_robustness")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(fp["time"], fp["pde_event_density"], label="Fokker–Planck", color=COLORS["navy"])
    ax.plot(fp["time"], fp["mc_event_density"], label="Monte Carlo", color=COLORS["gold"], alpha=0.9)
    ax.set(xlabel="first-passage time (days)", ylabel="density", title="Simplified first-passage verification")
    ax.legend()
    _save(fig, output_dir, "fokker_planck_vs_monte_carlo")
