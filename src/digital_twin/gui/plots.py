from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .theme import COLORS


PALETTE = [
    COLORS["plum"],
    COLORS["coral"],
    COLORS["teal"],
    COLORS["gold"],
    COLORS["blue"],
    COLORS["lilac"],
]


def _finish(
    figure: go.Figure,
    *,
    height: int = 430,
    legend: bool = True,
) -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=24, r=24, t=78 if legend else 42, b=36),
        font=dict(family="Inter, Aptos, Arial, sans-serif", color=COLORS["ink"]),
        hoverlabel=dict(
            bgcolor="#ffffff",
            font=dict(family="Inter, Aptos, Arial, sans-serif"),
        ),
        legend=(
            dict(
                orientation="h",
                yanchor="bottom",
                y=1.08,
                xanchor="left",
                x=0,
                bgcolor="rgba(0,0,0,0)",
            )
            if legend
            else None
        ),
        hovermode="x unified",
    )
    figure.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="rgba(31,32,51,.16)",
    )
    figure.update_yaxes(
        gridcolor="rgba(31,32,51,.08)",
        zeroline=False,
        linecolor="rgba(31,32,51,.16)",
    )
    return figure


def welcome_cycle_figure() -> go.Figure:
    theta = np.linspace(0, 2 * np.pi, 241)
    radius = 1.0 + 0.09 * np.sin(2 * theta - 0.6)
    color = np.where(theta < np.pi, 0, 1)
    figure = go.Figure()
    for mask, name, shade in (
        (color == 0, "Follicular passage", COLORS["plum"]),
        (color == 1, "Luteal passage", COLORS["coral"]),
    ):
        figure.add_trace(
            go.Scatter(
                x=radius[mask] * np.cos(theta[mask]),
                y=radius[mask] * np.sin(theta[mask]),
                mode="lines",
                name=name,
                line=dict(color=shade, width=14),
                hoverinfo="skip",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=[-1.05, 1.05],
            y=[0.0, 0.0],
            mode="markers+text",
            name="Transitions",
            marker=dict(
                size=[17, 17],
                color=[COLORS["gold"], COLORS["teal"]],
                line=dict(width=3, color="#fff"),
            ),
            text=["reset", "transition"],
            textposition=["bottom center", "top center"],
            hovertemplate="%{text}<extra></extra>",
        )
    )
    figure.update_xaxes(visible=False, range=[-1.35, 1.35])
    figure.update_yaxes(
        visible=False,
        range=[-1.35, 1.35],
        scaleanchor="x",
        scaleratio=1,
    )
    figure.update_layout(
        template="plotly_white",
        height=345,
        margin=dict(l=0, r=0, t=55, b=0),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.02,
            yanchor="bottom",
        ),
        font=dict(family="Inter, Aptos, Arial, sans-serif", color=COLORS["ink"]),
    )
    return figure


def latent_progress_figure(
    states: pd.DataFrame,
    participant_id: str,
) -> go.Figure:
    frame = states.loc[states["participant_id"].eq(participant_id)].copy()
    frame["event_time"] = pd.to_datetime(frame["event_time"])
    figure = go.Figure()
    for stage, label, color in (
        ("F", "Follicular progress", COLORS["plum"]),
        ("L", "Luteal progress", COLORS["coral"]),
    ):
        values = frame["progress"].where(frame["stage"].eq(stage))
        figure.add_trace(
            go.Scatter(
                x=frame["event_time"],
                y=values,
                mode="lines",
                name=label,
                line=dict(color=color, width=2.7),
                connectgaps=False,
                hovertemplate="%{x|%d %b %Y}<br>progress %{y:.2f}<extra></extra>",
            )
        )
    transitions = frame.loc[frame["transition"].ne("none")]
    if not transitions.empty:
        figure.add_trace(
            go.Scatter(
                x=transitions["event_time"],
                y=transitions["progress"],
                mode="markers",
                name="Boundary crossing",
                marker=dict(
                    size=10,
                    color=COLORS["gold"],
                    line=dict(width=2, color="#fff"),
                ),
                customdata=transitions["transition"],
                hovertemplate="%{x|%d %b}<br>%{customdata}<extra></extra>",
            )
        )
    figure.update_yaxes(
        title="Within-stage progress",
        range=[0, 1.04],
        tickformat=".0%",
    )
    figure.update_xaxes(title="Study date")
    return _finish(figure, height=430)


def cycle_distribution_figure(durations: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    participants = list(durations["participant_id"].drop_duplicates())
    if len(participants) <= 12:
        for index, participant in enumerate(participants):
            values = durations.loc[
                durations["participant_id"].eq(participant),
                "cycle_length_days",
            ]
            figure.add_trace(
                go.Box(
                    x=[participant] * len(values),
                    y=values,
                    name=participant,
                    boxpoints="all",
                    jitter=0.25,
                    pointpos=0,
                    marker=dict(size=6, color=PALETTE[index % len(PALETTE)]),
                    line=dict(color=PALETTE[index % len(PALETTE)]),
                    showlegend=False,
                    hovertemplate=(
                        f"{participant}<br>cycle %{{y:.1f}} days<extra></extra>"
                    ),
                )
            )
        figure.update_xaxes(title="Synthetic participant")
    else:
        figure.add_trace(
            go.Histogram(
                x=durations["cycle_length_days"],
                marker=dict(color=COLORS["plum"]),
                name="Cycle intervals",
                hovertemplate="%{x:.1f} days<br>%{y} cycles<extra></extra>",
            )
        )
        figure.update_xaxes(title="Cycle length (days)")
        figure.update_yaxes(title="Intervals")
    if len(participants) <= 12:
        figure.update_yaxes(title="Cycle length (days)")
    return _finish(figure, height=410, legend=False)


def signal_figure(
    observations: pd.DataFrame,
    participant_id: str,
    signals: list[str],
) -> go.Figure:
    signals = list(signals[:4])
    figure = make_subplots(
        rows=len(signals),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[value.replace("_", " ").title() for value in signals],
    )
    frame = observations.loc[
        observations["participant_id"].eq(participant_id)
        & observations["signal_name"].isin(signals)
    ].copy()
    frame["event_time"] = pd.to_datetime(frame["event_time"])
    for index, signal in enumerate(signals, start=1):
        subset = frame.loc[frame["signal_name"].eq(signal)]
        observed = (
            subset["is_observed"].fillna(False)
            if "is_observed" in subset
            else subset["value"].notna()
        )
        figure.add_trace(
            go.Scatter(
                x=subset.loc[observed, "event_time"],
                y=subset.loc[observed, "value"],
                mode="lines+markers",
                name=signal.replace("_", " ").title(),
                line=dict(color=PALETTE[(index - 1) % len(PALETTE)], width=1.8),
                marker=dict(size=5),
                showlegend=False,
                hovertemplate="%{x|%d %b}<br>%{y:.2f}<extra></extra>",
            ),
            row=index,
            col=1,
        )
        missing = subset.loc[~observed]
        if not missing.empty:
            baseline = (
                float(subset.loc[observed, "value"].min())
                if observed.any()
                else 0.0
            )
            figure.add_trace(
                go.Scatter(
                    x=missing["event_time"],
                    y=[baseline] * len(missing),
                    mode="markers",
                    name="Missing",
                    marker=dict(
                        symbol="x",
                        size=6,
                        color="rgba(103,106,125,.55)",
                    ),
                    showlegend=False,
                    customdata=missing["missingness_reason"],
                    hovertemplate="%{x|%d %b}<br>%{customdata}<extra></extra>",
                ),
                row=index,
                col=1,
            )
    figure.update_xaxes(title="Study date", row=len(signals), col=1)
    return _finish(
        figure,
        height=max(430, 185 * len(signals)),
        legend=False,
    )


def missingness_figure(missingness: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    statuses = list(missingness["status"].drop_duplicates())
    for index, status in enumerate(statuses):
        subset = missingness.loc[missingness["status"].eq(status)]
        figure.add_trace(
            go.Bar(
                x=subset["signal_name"],
                y=subset["rows"],
                name=status.replace("_", " ").title(),
                marker=dict(
                    color=(
                        COLORS["teal"]
                        if status == "observed"
                        else PALETTE[(index + 1) % len(PALETTE)]
                    )
                ),
                hovertemplate="%{x}<br>%{y} rows<extra>%{fullData.name}</extra>",
            )
        )
    figure.update_layout(barmode="stack")
    figure.update_xaxes(title="", tickangle=-28)
    figure.update_yaxes(title="Observation rows")
    return _finish(figure, height=440)


def analytic_first_passage_figure(table: pd.DataFrame) -> go.Figure:
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Passage-time density",
            "Probability not yet crossed",
            "Instantaneous crossing rate",
        ),
        horizontal_spacing=0.09,
    )
    for column, label, color, index in (
        ("density", "Density", COLORS["plum"], 1),
        ("survival", "Survival", COLORS["teal"], 2),
        ("hazard", "Hazard", COLORS["coral"], 3),
    ):
        figure.add_trace(
            go.Scatter(
                x=table["time_days"],
                y=table[column],
                mode="lines",
                name=label,
                line=dict(color=color, width=3),
                fill="tozeroy" if column == "density" else None,
                fillcolor=(
                    "rgba(111,59,118,.12)"
                    if column == "density"
                    else None
                ),
                hovertemplate="%{x:.1f} days<br>%{y:.4f}<extra></extra>",
            ),
            row=1,
            col=index,
        )
        figure.update_xaxes(title="Days", row=1, col=index)
    figure.update_yaxes(title="Density", row=1, col=1)
    figure.update_yaxes(title="Probability", row=1, col=2, range=[0, 1.03])
    figure.update_yaxes(title="Rate per day", row=1, col=3)
    return _finish(figure, height=410)


def flux_figure(result) -> go.Figure:
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.14,
        subplot_titles=("Probability mass by stage", "Boundary flux"),
    )
    for values, label, color in (
        (result.follicular_mass, "Follicular mass", COLORS["plum"]),
        (result.luteal_mass, "Luteal mass", COLORS["coral"]),
        (result.total_mass, "Total mass", COLORS["teal"]),
    ):
        figure.add_trace(
            go.Scatter(
                x=result.time,
                y=values,
                mode="lines",
                name=label,
                line=dict(color=color, width=2.4),
            ),
            row=1,
            col=1,
        )
    for values, label, color in (
        (result.ovulation_flux, "Follicular → luteal", COLORS["gold"]),
        (result.menses_flux, "Luteal → reset", COLORS["teal"]),
    ):
        figure.add_trace(
            go.Scatter(
                x=result.time,
                y=values,
                mode="lines",
                name=label,
                line=dict(color=color, width=2.4),
            ),
            row=2,
            col=1,
        )
    figure.update_yaxes(title="Probability", row=1, col=1)
    figure.update_yaxes(title="Flux / day", row=2, col=1)
    figure.update_xaxes(title="Time (days)", row=2, col=1)
    return _finish(figure, height=610)


def density_heatmap_figure(result) -> go.Figure:
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Follicular density", "Luteal density"),
        horizontal_spacing=0.10,
    )
    for density, colorscale, index in (
        (result.follicular_density, "Purples", 1),
        (result.luteal_density, "Oranges", 2),
    ):
        figure.add_trace(
            go.Heatmap(
                x=result.time,
                y=result.grid,
                z=density.T,
                colorscale=colorscale,
                colorbar=(
                    dict(title="density", x=0.44)
                    if index == 1
                    else dict(title="density")
                ),
                hovertemplate=(
                    "time %{x:.1f} days<br>progress %{y:.2f}"
                    "<br>density %{z:.3f}<extra></extra>"
                ),
            ),
            row=1,
            col=index,
        )
        figure.update_xaxes(title="Time (days)", row=1, col=index)
        figure.update_yaxes(title="Stage progress", row=1, col=index)
    return _finish(figure, height=450, legend=False)


def lifespan_cycle_figure(
    age_summary: pd.DataFrame,
    aggregate: pd.DataFrame,
) -> go.Figure:
    figure = go.Figure()
    summary = age_summary.dropna(subset=["mean_cycle_days"]).copy()
    spread = summary["sd_cycle_days"].fillna(0.0)
    figure.add_trace(
        go.Scatter(
            x=pd.concat([summary["age"], summary["age"][::-1]]),
            y=pd.concat(
                [
                    summary["mean_cycle_days"] + spread,
                    (summary["mean_cycle_days"] - spread)[::-1],
                ]
            ),
            fill="toself",
            fillcolor="rgba(111,59,118,.11)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Simulated ±1 SD",
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=summary["age"],
            y=summary["mean_cycle_days"],
            mode="lines",
            name="Simulated mean",
            line=dict(color=COLORS["plum"], width=3),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=aggregate["age_midpoint"],
            y=aggregate["mean_cycle_days"],
            mode="markers",
            name="Aggregate constraint",
            marker=dict(
                color=COLORS["gold"],
                size=11,
                symbol="diamond",
                line=dict(color="#fff", width=2),
            ),
        )
    )
    figure.update_xaxes(title="Age (years)")
    figure.update_yaxes(title="Cycle duration (days)")
    return _finish(figure, height=440)


def reserve_figure(reserve: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=reserve["age_years"],
            y=reserve["reserve_follicles_population_model"],
            mode="lines",
            name="Population reserve curve",
            line=dict(color=COLORS["teal"], width=3),
            fill="tozeroy",
            fillcolor="rgba(22,125,121,.09)",
        )
    )
    figure.add_hline(
        y=1000,
        line=dict(color=COLORS["coral"], dash="dash", width=1.5),
        annotation_text="Scenario threshold",
        annotation_position="top right",
    )
    figure.update_xaxes(title="Age (years)")
    figure.update_yaxes(title="Population-model reserve", type="log")
    return _finish(figure, height=440)


def lifespan_population_figure(participants: pd.DataFrame) -> go.Figure:
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Scenario cessation ages", "Simulated cycle counts"),
        horizontal_spacing=0.12,
    )
    figure.add_trace(
        go.Histogram(
            x=participants["menopause_age_years"],
            marker=dict(color=COLORS["coral"]),
            name="Cessation ages",
            opacity=0.88,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Histogram(
            x=participants["simulated_cycle_count"],
            marker=dict(color=COLORS["teal"]),
            name="Cycle counts",
            opacity=0.88,
        ),
        row=1,
        col=2,
    )
    figure.update_xaxes(title="Age (years)", row=1, col=1)
    figure.update_xaxes(title="Cycles", row=1, col=2)
    figure.update_yaxes(title="Participants", row=1, col=1)
    figure.update_yaxes(title="Participants", row=1, col=2)
    return _finish(figure, height=420)


def upload_completeness_figure(frame: pd.DataFrame) -> go.Figure:
    if {"signal_name", "value"}.issubset(frame):
        completeness = (
            frame.assign(observed=frame["value"].notna())
            .groupby("signal_name")["observed"]
            .mean()
            .sort_values()
        )
        labels = completeness.index.str.replace("_", " ").str.title()
    else:
        candidate_columns = [
            column
            for column in frame.columns
            if column
            not in {
                "participant_id",
                "day_in_study",
                "timestamp_local",
                "missingness_reason",
            }
        ]
        completeness = frame[candidate_columns].notna().mean().sort_values()
        labels = completeness.index.str.replace("_", " ").str.title()
    colors = [
        COLORS["coral"] if value < 0.5 else COLORS["gold"]
        if value < 0.8
        else COLORS["teal"]
        for value in completeness
    ]
    figure = go.Figure(
        go.Bar(
            x=completeness.values,
            y=labels,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{value:.0%}" for value in completeness],
            textposition="outside",
            hovertemplate="%{y}<br>%{x:.1%} complete<extra></extra>",
        )
    )
    figure.update_xaxes(title="Complete fraction", range=[0, 1.08], tickformat=".0%")
    figure.update_yaxes(title="")
    return _finish(
        figure,
        height=max(360, 27 * len(completeness) + 130),
        legend=False,
    )
