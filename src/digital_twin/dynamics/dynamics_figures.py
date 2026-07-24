from __future__ import annotations

from pathlib import Path

from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .coupled_fokker_planck import CoupledFokkerPlanckResult


COLORS = {
    "follicular": "#0072B2",
    "luteal": "#CC79A7",
    "salzburg": "#009E73",
    "soochow": "#D55E00",
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


def _source_color(source: str) -> str:
    return COLORS["salzburg"] if "salzburg" in source else COLORS["soochow"]


def _source_label(source: str) -> str:
    return (
        "Salzburg hormone cohort"
        if "salzburg" in source
        else "Soochow/Fudan wearable cohort"
    )


def _style_axis(axis: plt.Axes, *, grid: bool = True) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(direction="out", length=3.5, width=0.8)
    if grid:
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.55)


def _save(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_coupled_flux(
    result: CoupledFokkerPlanckResult, directory: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.3), constrained_layout=True)
    axes[0, 0].plot(
        result.time,
        result.follicular_mass,
        color=COLORS["follicular"],
    )
    axes[0, 0].plot(
        result.time,
        result.luteal_mass,
        color=COLORS["luteal"],
    )
    axes[0, 0].plot(
        result.time,
        result.total_mass,
        color="black",
        linestyle="--",
        linewidth=1,
    )
    axes[0, 0].set(
        xlabel="Time (days)",
        ylabel="Probability mass",
        title="A  Cyclic occupancy and mass conservation",
        xlim=(0, min(70, result.time[-1])),
        ylim=(-0.03, 1.05),
    )

    axes[0, 1].plot(
        result.time,
        result.first_ovulation_density,
        color=COLORS["follicular"],
        linestyle="--",
    )
    axes[0, 1].plot(
        result.time,
        result.first_menses_density,
        color=COLORS["luteal"],
        linestyle="--",
    )
    axes[0, 1].set(
        xlabel="Time from cycle start (days)",
        ylabel="First-passage density",
        title="B  First transition-time distributions",
        xlim=(0, min(60, result.time[-1])),
    )

    time_mask = result.time <= min(60, result.time[-1])
    extent = [
        result.time[time_mask][0],
        result.time[time_mask][-1],
        result.grid[0],
        result.grid[-1],
    ]
    image_f = axes[1, 0].imshow(
        result.follicular_density[time_mask].T,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="Blues",
        vmin=0,
        vmax=float(
            np.quantile(result.follicular_density[time_mask][1:], 0.995)
        ),
    )
    axes[1, 0].set(
        xlabel="Time (days)",
        ylabel="Progress z",
        title="C  Follicular probability density",
    )
    fig.colorbar(image_f, ax=axes[1, 0], label="Probability density")
    image_l = axes[1, 1].imshow(
        result.luteal_density[time_mask].T,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="RdPu",
        vmin=0,
        vmax=float(
            np.quantile(result.luteal_density[time_mask][1:], 0.995)
        ),
    )
    axes[1, 1].set(
        xlabel="Time (days)",
        ylabel="Progress z",
        title="D  Luteal probability density",
    )
    fig.colorbar(image_l, ax=axes[1, 1], label="Probability density")
    fig.legend(
        handles=[
            Line2D([], [], color=COLORS["follicular"], label="Follicular mass"),
            Line2D([], [], color=COLORS["luteal"], label="Luteal mass"),
            Line2D(
                [],
                [],
                color="black",
                linestyle="--",
                linewidth=1,
                label="Total mass",
            ),
            Line2D(
                [],
                [],
                color=COLORS["follicular"],
                linestyle="--",
                label="First follicular→luteal flux",
            ),
            Line2D(
                [],
                [],
                color=COLORS["luteal"],
                linestyle="--",
                label="First luteal→follicular flux",
            ),
        ],
        loc="outside upper center",
        ncol=5,
        frameon=False,
    )
    _style_axis(axes[0, 0])
    _style_axis(axes[0, 1])
    _style_axis(axes[1, 0], grid=False)
    _style_axis(axes[1, 1], grid=False)
    _save(fig, directory, "Fig1_coupled_probability_flux")


def figure_analytic_limit(
    analytic: pd.DataFrame, samples: np.ndarray, directory: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), constrained_layout=True)
    axes[0].hist(
        samples,
        bins=70,
        density=True,
        color="#B7C9E2",
        edgecolor="none",
    )
    axes[0].plot(
        analytic["time"],
        analytic["analytic_density"],
        color="#173F6F",
        linewidth=2,
    )
    axes[0].plot(
        analytic["time"],
        analytic["finite_volume_first_flux"],
        color="#B04A7A",
        linewidth=1.4,
        linestyle="--",
    )
    density = analytic["analytic_density"].to_numpy()
    support = analytic.loc[density >= density.max() * 0.001, "time"]
    density_xlim = (
        max(0.0, float(support.min()) - 1.0),
        float(support.max()) + 1.0,
    )
    axes[0].set(
        xlabel="Passage time (days)",
        ylabel="Density",
        title="A  Analytic and numerical agreement",
        xlim=density_xlim,
    )
    axes[1].plot(
        analytic["time"],
        analytic["analytic_survival"],
        color="#173F6F",
    )
    axes[1].set(
        xlabel="Time (days)",
        ylabel="P(T > t)",
        title="B  First-passage survival",
        ylim=(-0.02, 1.02),
    )
    axes[2].plot(
        analytic["time"],
        analytic["analytic_hazard"],
        color="#8A4F00",
    )
    axes[2].set(
        xlabel="Time (days)",
        ylabel="Hazard (per day)",
        title="C  First-passage hazard",
    )
    event_xlim = (
        max(0.0, density_xlim[0] - 3.0),
        min(float(analytic["time"].max()), density_xlim[1] + 18.0),
    )
    axes[1].set_xlim(event_xlim)
    axes[2].set_xlim(event_xlim)
    fig.legend(
        handles=[
            Patch(
                facecolor="#B7C9E2",
                edgecolor="none",
                label="Exact inverse-Gaussian samples",
            ),
            Line2D([], [], color="#173F6F", linewidth=2, label="Closed form"),
            Line2D(
                [],
                [],
                color="#B04A7A",
                linewidth=1.4,
                linestyle="--",
                label="Finite-volume flux",
            ),
        ],
        loc="outside upper center",
        ncol=3,
        frameon=False,
    )
    for axis in axes:
        _style_axis(axis)
    _save(fig, directory, "Fig2_analytic_first_passage_limit")


def figure_regime_map(regime: pd.DataFrame, directory: Path) -> None:
    pe = np.sort(regime["peclet"].unique())
    relaxation = np.sort(regime["relaxation_ratio"].unique())
    cv = (
        regime.pivot(
            index="relaxation_ratio",
            columns="peclet",
            values="coefficient_of_variation",
        )
        .reindex(index=relaxation, columns=pe)
        .to_numpy()
    )
    skewness = (
        regime.pivot(
            index="relaxation_ratio", columns="peclet", values="skewness"
        )
        .reindex(index=relaxation, columns=pe)
        .to_numpy()
    )
    regime_order = [
        "broad_first_passage_tail",
        "persistent_speed_heterogeneity",
        "mixed_drift_diffusion",
        "drift_dominated_regular",
    ]
    regime_values = (
        regime.assign(
            regime_code=regime["regime"].map(
                {value: index for index, value in enumerate(regime_order)}
            )
        )
        .pivot(
            index="relaxation_ratio",
            columns="peclet",
            values="regime_code",
        )
        .reindex(index=relaxation, columns=pe)
        .to_numpy()
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0), constrained_layout=True)
    for panel, axis, values, title, label, cmap in (
        (
            "A",
            axes[0],
            cv,
            "Relative passage-time dispersion",
            "Coefficient of variation",
            "viridis",
        ),
        (
            "B",
            axes[1],
            skewness,
            "First-passage tail asymmetry",
            "Skewness",
            "magma",
        ),
    ):
        image = axis.imshow(
            values,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            interpolation="nearest",
        )
        axis.set_xticks(range(len(pe)), [f"{value:g}" for value in pe])
        axis.set_yticks(
            range(len(relaxation)),
            [f"{value:g}" for value in relaxation],
        )
        axis.set(
            xlabel="Péclet number Pe",
            ylabel="Relaxation ratio R",
            title=f"{panel}  {title}",
        )
        fig.colorbar(image, ax=axis, label=label, shrink=0.90)
        _style_axis(axis, grid=False)
    regime_colors = ["#7B3294", "#008837", "#F1A340", "#4D4D4D"]
    axes[2].imshow(
        regime_values,
        aspect="auto",
        origin="lower",
        cmap=ListedColormap(regime_colors),
        vmin=-0.5,
        vmax=len(regime_order) - 0.5,
        interpolation="nearest",
    )
    axes[2].set_xticks(range(len(pe)), [f"{value:g}" for value in pe])
    axes[2].set_yticks(
        range(len(relaxation)),
        [f"{value:g}" for value in relaxation],
    )
    axes[2].set(
        xlabel="Péclet number Pe",
        ylabel="Relaxation ratio R",
        title="C  Prespecified dynamical regime",
    )
    _style_axis(axes[2], grid=False)
    fig.legend(
        handles=[
            Patch(facecolor=regime_colors[0], label="Broad passage-time tail"),
            Patch(facecolor=regime_colors[1], label="Persistent speed"),
            Patch(facecolor=regime_colors[2], label="Mixed drift–diffusion"),
            Patch(facecolor=regime_colors[3], label="Drift-dominated regular"),
        ],
        loc="outside upper center",
        ncol=4,
        frameon=False,
    )
    _save(fig, directory, "Fig3_nondimensional_regime_map")


def figure_empirical_signatures(
    durations: pd.DataFrame,
    hazards: pd.DataFrame,
    participants: pd.DataFrame,
    stages: pd.DataFrame,
    directory: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.2), constrained_layout=True)
    for source, group in durations.groupby("source"):
        color = _source_color(source)
        axes[0, 0].hist(
            group["cycle_length_days"],
            bins=np.arange(14.5, 61.5, 1),
            density=True,
            histtype="step",
            linewidth=1.8,
            color=color,
        )
    axes[0, 0].set(
        xlabel="Cycle duration (days)",
        ylabel="Density",
        title="A  Observed duration distributions",
        xlim=(15, 60),
    )
    for source, group in hazards.groupby("source"):
        color = _source_color(source)
        group = group.loc[group["at_risk"].ge(10)]
        axes[0, 1].plot(
            group["day"],
            group["hazard"],
            marker="o",
            markersize=2,
            linewidth=1,
            color=color,
        )
    axes[0, 1].set(
        xlabel="Cycle day",
        ylabel="Empirical hazard",
        title="B  Menses hazard (≥10 cycles at risk)",
        xlim=(15, 60),
        ylim=(-0.02, 0.64),
    )
    if not stages.empty:
        for source, group in stages.groupby("source"):
            color = _source_color(source)
            axes[1, 0].scatter(
                group["follicular_proxy_days"],
                group["luteal_proxy_days"],
                s=16,
                alpha=0.62,
                color=color,
                edgecolor="white",
                linewidth=0.25,
            )
    axes[1, 0].set(
        xlabel="Menses-to-LH+ duration (days)",
        ylabel="LH+-to-next-menses duration (days)",
        title="C  Urinary-LH stage-duration proxies",
    )
    for source, group in participants.groupby("source"):
        color = _source_color(source)
        valid = group["sd_days"].notna()
        axes[1, 1].scatter(
            group.loc[valid, "mean_days"],
            group.loc[valid, "sd_days"],
            s=18,
            alpha=0.62,
            color=color,
            edgecolor="white",
            linewidth=0.25,
        )
    axes[1, 1].set(
        xlabel="Participant mean duration (days)",
        ylabel="Within-participant SD (days)",
        title="D  Between- and within-person variability",
    )
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color=COLORS["salzburg"],
                marker="o",
                label="Salzburg hormone cohort",
            ),
            Line2D(
                [],
                [],
                color=COLORS["soochow"],
                marker="o",
                label="Soochow/Fudan wearable cohort",
            ),
        ],
        loc="outside upper center",
        ncol=2,
        frameon=False,
    )
    for axis in axes.flat:
        _style_axis(axis)
    _save(fig, directory, "Fig4_empirical_dynamical_signatures")


def figure_identifiability(
    model_comparison: pd.DataFrame,
    profile: pd.DataFrame,
    sbc: pd.DataFrame,
    ou_surface: pd.DataFrame,
    directory: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0), constrained_layout=True)
    models = list(model_comparison["model"].unique())
    sources = list(model_comparison["source"].unique())
    x = np.arange(len(models))
    width = 0.36
    for index, source in enumerate(sources):
        frame = (
            model_comparison.loc[model_comparison["source"].eq(source)]
            .set_index("model")
            .reindex(models)
        )
        values = frame["delta_aic"].to_numpy(dtype=float)
        bars = axes[0, 0].bar(
            x + (index - (len(sources) - 1) / 2) * width,
            values,
            width=width,
            color=_source_color(source),
            alpha=0.85,
            zorder=3,
        )
        axes[0, 0].bar_label(
            bars,
            labels=[f"{value:.1f}" for value in values],
            padding=2,
            fontsize=6.5,
            rotation=90,
        )
    axes[0, 0].set_xticks(
        x,
        [
            {
                "hierarchical_lognormal_random_intercept": "Hierarchical\nlognormal",
                "hierarchical_ar1_cycle_shock": "Hierarchical\nAR(1)",
                "constant_drift_diffusion_first_passage": "Drift–\ndiffusion",
                "lognormal_renewal": "Lognormal\nrenewal",
                "gamma_renewal": "Gamma\nrenewal",
                "gaussian_renewal": "Gaussian\nrenewal",
            }.get(name, name.replace("_", " "))
            for name in models
        ],
        fontsize=7.2,
    )
    axes[0, 0].set(
        ylabel="ΔAIC from cohort-best model",
        title="A  Relative support for duration models",
        ylim=(0, max(130.0, float(model_comparison["delta_aic"].max()) + 8)),
    )

    for (source, parameter), group in profile.groupby(
        ["source", "profile_parameter"]
    ):
        estimate_scale = group.loc[
            group["profile_deviance"].idxmin(), "value"
        ]
        axes[0, 1].plot(
            group["value"] / estimate_scale,
            group["profile_deviance"],
            color=_source_color(source),
            linestyle="-" if parameter == "drift" else "--",
        )
    axes[0, 1].axhline(3.84, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set(
        xlabel="Parameter / profile optimum",
        ylabel="Profile deviance",
        title="B  Constant-limit profile likelihood",
        xlim=(0.5, 1.5),
        ylim=(0, 15),
    )

    axes[1, 0].plot(
        sbc["sample_size"],
        sbc["drift_contraction"],
        marker="o",
        color=COLORS["follicular"],
    )
    axes[1, 0].plot(
        sbc["sample_size"],
        sbc["diffusion_contraction"],
        marker="s",
        linestyle="--",
        color=COLORS["luteal"],
    )
    axes[1, 0].set(
        xlabel="Observed passage times",
        ylabel="Posterior contraction",
        title="C  Analytic-limit identifiability",
        ylim=(0, 1),
        xlim=(5, 105),
    )

    kappa = np.sort(ou_surface["kappa"].unique())
    stationary_sd = np.sort(
        ou_surface["stationary_log_speed_sd"].unique()
    )
    distance = (
        ou_surface.pivot(
            index="stationary_log_speed_sd",
            columns="kappa",
            values="delta_distance",
        )
        .reindex(index=stationary_sd, columns=kappa)
        .to_numpy()
    )
    image = axes[1, 1].imshow(
        np.clip(distance, 0, 10),
        aspect="auto",
        origin="lower",
        cmap="cividis_r",
    )
    axes[1, 1].set_xticks(
        range(len(kappa)), [f"{value:g}" for value in kappa]
    )
    axes[1, 1].set_yticks(
        range(len(stationary_sd)),
        [f"{value:g}" for value in stationary_sd],
    )
    axes[1, 1].set(
        xlabel="OU relaxation κ",
        ylabel="Stationary log-speed SD",
        title="D  OU degeneracy (lower is closer)",
    )
    fig.colorbar(
        image,
        ax=axes[1, 1],
        label="Δ summary distance (clipped at 10)",
        shrink=0.90,
    )
    fig.legend(
        handles=[
            Patch(
                facecolor=COLORS["salzburg"],
                alpha=0.85,
                label="Salzburg cohort",
            ),
            Patch(
                facecolor=COLORS["soochow"],
                alpha=0.85,
                label="Soochow/Fudan cohort",
            ),
            Line2D([], [], color="black", linestyle="-", label="Drift profile"),
            Line2D(
                [], [], color="black", linestyle="--", label="Diffusion profile"
            ),
            Line2D(
                [],
                [],
                color=COLORS["follicular"],
                marker="o",
                label="Drift contraction",
            ),
            Line2D(
                [],
                [],
                color=COLORS["luteal"],
                marker="s",
                linestyle="--",
                label="Diffusion contraction",
            ),
        ],
        loc="outside upper center",
        ncol=3,
        frameon=False,
    )
    for axis in axes.flat:
        _style_axis(axis, grid=axis is not axes[1, 1])
    _save(fig, directory, "Fig5_model_support_and_identifiability")
