from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml

from digital_twin.data.schemas import OBSERVATION_FIELDS
from digital_twin.dynamics.coupled_fokker_planck import solve_coupled_cycle
from digital_twin.dynamics.lifespan import simulate_reproductive_lifespan
from digital_twin.gui.analytics import (
    analytic_stage_table,
    build_research_config,
    cohort_summary,
    cycle_duration_table,
    dataframe_to_csv,
    format_config,
    missingness_table,
    safe_numeric_summary,
    validate_uploaded_frame,
)
from digital_twin.gui.plots import (
    analytic_first_passage_figure,
    cycle_distribution_figure,
    density_heatmap_figure,
    flux_figure,
    latent_progress_figure,
    lifespan_cycle_figure,
    lifespan_population_figure,
    missingness_figure,
    reserve_figure,
    signal_figure,
    upload_completeness_figure,
    welcome_cycle_figure,
)
from digital_twin.gui.theme import (
    COLORS,
    apply_theme,
    callout,
    feature_card,
    metric_strip,
    page_intro,
)
from digital_twin.simulation.cohort import simulate_cohort


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "app" / "assets" / "flux_mark.svg"
AGGREGATE_DATA = ROOT / "data" / "theory" / "menstrual_age_aggregate_2024.csv"


st.set_page_config(
    page_title="Menstrual Flux · Research Explorer",
    page_icon=str(ASSET),
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "About": (
            "Menstrual Flux Research Explorer — a non-clinical interface for "
            "stochastic menstrual-cycle modelling."
        )
    },
)
apply_theme()


def _plot(figure, *, key: str) -> None:
    st.plotly_chart(
        figure,
        width="stretch",
        config={
            "displaylogo": False,
            "responsive": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": f"menstrual_flux_{key}",
                "scale": 2,
            },
        },
        key=key,
    )


def _sidebar() -> str:
    if ASSET.exists():
        st.logo(str(ASSET), size="large")
    st.sidebar.markdown(
        """
        <div class="flux-brand">Menstrual Flux</div>
        <div class="flux-brand-sub">Research Explorer</div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    section = st.sidebar.radio(
        "Explore",
        (
            "✦  Observatory",
            "◌  Synthetic Cycle Lab",
            "↝  Flux & First Passage",
            "⌁  Lifespan Atlas",
            "▦  Data Quality Studio",
            "∑  Methods & Scope",
        ),
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div class="research-stamp">
          <span>◇</span>
          <span><b>Research use only</b><br>
          No diagnosis, fertility guidance, or clinical decisions.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.caption("v0.5 interface · stochastic passage-and-reset model")
    return section


def observatory() -> None:
    st.markdown(
        """
        <section class="flux-hero">
          <div class="flux-kicker" style="color:#ffd28c">A living model of timing and uncertainty</div>
          <h1>See cycles as flows,<br>not calendar dots.</h1>
          <p>
            Explore a two-stage stochastic process in which probability moves,
            crosses boundaries, resets, and accumulates uncertainty. Every view
            is generated from the same tested numerical engine used by the
            command-line workflows.
          </p>
          <span class="flux-pill">◌ synthetic cohorts</span>
          <span class="flux-pill">↝ probability flux</span>
          <span class="flux-pill">⌁ lifespan scenarios</span>
        </section>
        """,
        unsafe_allow_html=True,
    )

    metric_strip(
        (
            ("Model structure", "2 stages", "passage + reset", COLORS["plum"]),
            ("Uncertainty", "Explicit", "density, survival, hazard", COLORS["coral"]),
            ("Observation layer", "14 signals", "wearable · assay · report", COLORS["teal"]),
            ("Safety boundary", "Research", "non-clinical by design", COLORS["gold"]),
        )
    )

    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.subheader("One process, several lenses")
        st.write(
            "The interface keeps mathematical structure, simulated observations, "
            "and empirical data quality separate. That separation makes it easier "
            "to see what the model assumes, what the observations contain, and "
            "where uncertainty enters."
        )
        rows = st.columns(2)
        with rows[0]:
            feature_card(
                "◌",
                "Build a synthetic cohort",
                "Change stage timing, heterogeneity and missingness, then inspect latent and observed trajectories.",
            )
        with rows[1]:
            feature_card(
                "↝",
                "Shape the probability flow",
                "Move drift and diffusion controls and watch density, survival, hazard and cyclic flux respond.",
            )
        st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
        rows = st.columns(2)
        with rows[0]:
            feature_card(
                "⌁",
                "Explore lifespan scenarios",
                "Simulate aggregate-constrained cycling and explicit interruption assumptions across age.",
            )
        with rows[1]:
            feature_card(
                "▦",
                "Audit a data table",
                "Validate schema, timestamp order, completeness and missingness without uploading to a server.",
            )
    with right:
        _plot(welcome_cycle_figure(), key="welcome_cycle")
        callout(
            "Start with the Synthetic Cycle Lab for a guided model run. "
            "Use Flux & First Passage when you want to inspect the equations directly.",
            icon="→",
        )

    st.markdown("### How to read the explorer")
    columns = st.columns(3)
    with columns[0]:
        st.markdown("**1 · Assumptions**")
        st.caption(
            "Controls change a phenomenological process. They are not direct measurements of endocrine mechanisms."
        )
    with columns[1]:
        st.markdown("**2 · Consequences**")
        st.caption(
            "Plots show model-implied timing distributions, latent paths and observation patterns."
        )
    with columns[2]:
        st.markdown("**3 · Limits**")
        st.caption(
            "A visually convincing trajectory is not evidence of biological causality or individual clinical validity."
        )


def synthetic_cycle_lab() -> None:
    page_intro(
        "Interactive generative laboratory",
        "Synthetic Cycle Lab",
        "Create a reproducible cohort, expose its hidden passage dynamics, and then view the noisy measurements that a study would actually observe.",
    )
    callout(
        "All records generated on this page are synthetic. No personal or participant data are required.",
        icon="◇",
    )

    with st.expander("Cohort and process controls", expanded=True):
        one, two, three = st.columns(3)
        with one:
            participants = st.slider("Synthetic participants", 1, 12, 4)
            cycles = st.slider("Cycles per participant", 2, 10, 5)
            scenario = st.selectbox(
                "Dynamical scenario",
                (
                    "mixed",
                    "stable",
                    "short_cycle",
                    "long_cycle",
                    "highly_variable",
                    "follicular_dominant",
                    "luteal_dominant",
                    "slow_drift",
                    "temporary_disruption",
                    "regime_switch",
                    "anovulatory_like",
                    "stalled_transition",
                ),
                format_func=lambda value: value.replace("_", " ").title(),
            )
        with two:
            follicular_days = st.slider(
                "Nominal follicular passage (days)", 8.0, 35.0, 14.0, 0.5
            )
            luteal_days = st.slider(
                "Nominal luteal passage (days)", 8.0, 20.0, 14.0, 0.5
            )
            progress_noise = st.slider(
                "Progress noise", 0.001, 0.025, 0.006, 0.001, format="%.3f"
            )
        with three:
            between_person_sd = st.slider(
                "Between-person variation", 0.00, 0.35, 0.12, 0.01
            )
            missing_mechanism = st.selectbox(
                "Missingness process",
                ("none", "mcar", "mar", "informative", "mixed"),
                index=4,
                format_func=lambda value: {
                    "none": "None",
                    "mcar": "Random technical loss",
                    "mar": "Schedule-related loss",
                    "informative": "Burden-related loss",
                    "mixed": "Mixed realistic loss",
                }[value],
            )
            missing_probability = st.slider(
                "Missingness intensity",
                0.0,
                0.30,
                0.08,
                0.01,
                disabled=missing_mechanism == "none",
            )
        seed = st.number_input(
            "Reproducibility seed",
            min_value=0,
            max_value=2_147_483_647,
            value=20260721,
            step=1,
        )
        run = st.button(
            "Generate cohort",
            type="primary",
            width="stretch",
            key="run_synthetic",
        )

    if run:
        try:
            config = build_research_config(
                participants=participants,
                cycles=cycles,
                seed=int(seed),
                scenario=scenario,
                follicular_days=follicular_days,
                luteal_days=luteal_days,
                progress_noise=progress_noise,
                between_person_sd=between_person_sd,
                missingness_mechanism=missing_mechanism,
                missing_probability=(
                    0.0 if missing_mechanism == "none" else missing_probability
                ),
            )
            with st.spinner("Moving probability through the synthetic cohort…"):
                cohort = simulate_cohort(config)
            st.session_state["flux_cohort"] = cohort
            st.session_state["flux_config"] = config
        except Exception as error:
            st.error(f"Simulation stopped: {error}")

    cohort = st.session_state.get("flux_cohort")
    config = st.session_state.get("flux_config")
    if cohort is None or config is None:
        callout(
            "Choose a scenario and generate a cohort. The resulting state remains available while you explore other pages.",
            tone="warning",
            icon="◌",
        )
        return

    summary = cohort_summary(cohort.observed, cohort.events)
    metric_strip(
        (
            (
                "Participants",
                str(summary["participants"]),
                "synthetic profiles",
                COLORS["plum"],
            ),
            (
                "Complete intervals",
                str(summary["complete_cycles"]),
                "menses-to-menses",
                COLORS["coral"],
            ),
            (
                "Median duration",
                f"{summary['median_cycle_days']:.1f} d",
                f"IQR {summary['cycle_iqr_days']:.1f} d",
                COLORS["gold"],
            ),
            (
                "Measurements retained",
                f"{summary['observed_fraction']:.0%}",
                f"{summary['signals']} signal types",
                COLORS["teal"],
            ),
        )
    )

    participants_available = cohort.participants["participant_id"].tolist()
    selected_participant = st.selectbox(
        "Inspect participant",
        participants_available,
        key="synthetic_participant",
    )
    trajectory_tab, signals_tab, cohort_tab, export_tab = st.tabs(
        (
            "Latent trajectory",
            "Observed signals",
            "Cohort variability",
            "Export & provenance",
        )
    )
    with trajectory_tab:
        callout(
            "This panel can show latent progress because the cohort is simulated. "
            "Latent progress would not be directly observable in a real dataset.",
            icon="✦",
        )
        _plot(
            latent_progress_figure(cohort.truth, selected_participant),
            key=f"latent_{selected_participant}",
        )
        transition_view = cohort.events.loc[
            cohort.events["participant_id"].eq(selected_participant),
            ["cycle_id", "event_type", "event_time", "certainty"],
        ].copy()
        st.dataframe(
            transition_view,
            width="stretch",
            hide_index=True,
        )
    with signals_tab:
        available_signals = sorted(cohort.observed["signal_name"].unique())
        default_signals = [
            value
            for value in (
                "temperature",
                "resting_heart_rate",
                "lh",
                "pdg",
            )
            if value in available_signals
        ]
        selected_signals = st.multiselect(
            "Signals to display (maximum four)",
            available_signals,
            default=default_signals,
            max_selections=4,
        )
        if selected_signals:
            _plot(
                signal_figure(
                    cohort.observed,
                    selected_participant,
                    selected_signals,
                ),
                key=f"signals_{selected_participant}_{len(selected_signals)}",
            )
        missing = missingness_table(
            cohort.observed.loc[
                cohort.observed["participant_id"].eq(selected_participant)
            ]
        )
        _plot(
            missingness_figure(missing),
            key=f"missing_{selected_participant}",
        )
    with cohort_tab:
        durations = cycle_duration_table(cohort.events)
        _plot(
            cycle_distribution_figure(durations),
            key="cycle_distribution",
        )
        st.dataframe(
            durations[
                [
                    "participant_id",
                    "cycle_id",
                    "event_time",
                    "next_event_time",
                    "cycle_length_days",
                ]
            ],
            width="stretch",
            hide_index=True,
        )
    with export_tab:
        st.markdown("#### Reproducibility record")
        st.json(format_config(config), expanded=False)
        first, second, third = st.columns(3)
        with first:
            st.download_button(
                "Download observations",
                dataframe_to_csv(cohort.observed),
                "synthetic_observations.csv",
                "text/csv",
                width="stretch",
            )
        with second:
            st.download_button(
                "Download latent states",
                dataframe_to_csv(cohort.truth),
                "synthetic_latent_states.csv",
                "text/csv",
                width="stretch",
            )
        with third:
            st.download_button(
                "Download configuration",
                yaml.safe_dump(format_config(config), sort_keys=False),
                "synthetic_config.yaml",
                "text/yaml",
                width="stretch",
            )


def _regime_label(metrics: dict[str, float]) -> tuple[str, str]:
    if metrics["cv"] >= 0.35:
        return "Broad passage-time tail", COLORS["coral"]
    if metrics["peclet"] <= 10:
        return "Progress diffusion dominated", COLORS["gold"]
    if metrics["relaxation_ratio"] <= 0.3:
        return "Persistent speed variation", COLORS["blue"]
    if metrics["cv"] <= 0.15 and metrics["peclet"] >= 40:
        return "Drift-dominated regular", COLORS["teal"]
    return "Mixed drift–diffusion", COLORS["plum"]


def flux_studio() -> None:
    page_intro(
        "Equation-level control room",
        "Flux & First Passage",
        "Change the stochastic controls and inspect how a boundary-crossing time distribution and the coupled follicular–luteal probability flow respond.",
    )

    controls, view = st.columns([0.32, 0.68], gap="large")
    with controls:
        st.markdown("### Stage controls")
        follicular_mean = st.slider(
            "Follicular mean (days)", 8.0, 35.0, 14.0, 0.5
        )
        luteal_mean = st.slider(
            "Luteal mean (days)", 8.0, 20.0, 14.0, 0.5
        )
        follicular_diffusion = st.slider(
            "Follicular diffusion",
            0.00002,
            0.00100,
            0.00015,
            0.00001,
            format="%.5f",
        )
        luteal_diffusion = st.slider(
            "Luteal diffusion",
            0.00002,
            0.00100,
            0.00010,
            0.00001,
            format="%.5f",
        )
        st.markdown("### Speed memory")
        kappa = st.slider("OU relaxation κ", 0.01, 1.00, 0.35, 0.01)
        sigma_log_speed = st.slider(
            "Log-speed noise σ", 0.00, 0.12, 0.035, 0.005
        )
        run_flux = st.button(
            "Solve coupled probability flux",
            type="primary",
            width="stretch",
        )

    drift = 1.0 / follicular_mean
    analytic, metrics = analytic_stage_table(
        drift=drift,
        diffusion=follicular_diffusion,
        kappa=kappa,
        sigma_log_speed=sigma_log_speed,
        horizon=max(60.0, 4.0 * follicular_mean),
    )
    label, color = _regime_label(metrics)
    with view:
        metric_strip(
            (
                (
                    "Expected crossing",
                    f"{metrics['mean_days']:.1f} d",
                    "analytic constant-speed limit",
                    COLORS["plum"],
                ),
                (
                    "Timing spread",
                    f"{metrics['sd_days']:.2f} d",
                    f"CV {metrics['cv']:.2f}",
                    COLORS["coral"],
                ),
                (
                    "Péclet number",
                    f"{metrics['peclet']:.1f}",
                    "drift ÷ diffusion",
                    COLORS["gold"],
                ),
                (
                    "Relaxation ratio",
                    f"{metrics['relaxation_ratio']:.2f}",
                    "memory versus passage",
                    COLORS["teal"],
                ),
            )
        )
        callout(
            f"Current qualitative regime: {label}. Regime names describe model behaviour, not biological diagnoses.",
            icon="↝",
        )

    st.markdown("### Analytic first-passage limit")
    _plot(
        analytic_first_passage_figure(analytic),
        key="analytic_first_passage",
    )
    with st.expander("Read the curves"):
        st.markdown(
            """
            - **Density** shows where boundary-crossing times concentrate.
            - **Survival** is the probability that the stage has not crossed by a given day.
            - **Hazard** is the instantaneous crossing rate conditional on not yet crossing.
            - Increasing drift compresses time; increasing diffusion broadens the distribution and its tail.
            """
        )

    if run_flux:
        try:
            with st.spinner("Solving conservative cyclic probability transport…"):
                result = solve_coupled_cycle(
                    follicular_drift=1.0 / follicular_mean,
                    luteal_drift=1.0 / luteal_mean,
                    follicular_diffusion=follicular_diffusion,
                    luteal_diffusion=luteal_diffusion,
                    dz=0.02,
                    dt=0.01,
                    max_time=80.0,
                    store_every=10,
                )
            st.session_state["flux_solution"] = result
        except Exception as error:
            st.error(f"Probability-flux solver stopped: {error}")

    result = st.session_state.get("flux_solution")
    if result is None:
        callout(
            "Run the coupled solver to reveal stage-to-stage transport and density fields.",
            tone="warning",
            icon="↝",
        )
        return

    max_mass_error = float(np.max(np.abs(result.total_mass - 1.0)))
    metric_strip(
        (
            (
                "Mass conservation",
                f"{max_mass_error:.1e}",
                "maximum absolute error",
                COLORS["teal"],
            ),
            (
                "Cumulative transitions",
                f"{result.cumulative_ovulations[-1]:.2f}",
                "follicular exits",
                COLORS["gold"],
            ),
            (
                "Cumulative resets",
                f"{result.cumulative_menses[-1]:.2f}",
                "luteal exits",
                COLORS["coral"],
            ),
        )
    )
    flux_tab, density_tab = st.tabs(("Flux through time", "Density fields"))
    with flux_tab:
        _plot(flux_figure(result), key="coupled_flux")
    with density_tab:
        _plot(density_heatmap_figure(result), key="coupled_density")


def _interruption_scenarios() -> dict[str, tuple[dict[str, float | str], ...]]:
    return {
        "Uninterrupted reference": (),
        "Two pregnancy + six-month postpartum windows": (
            {"state": "pregnancy", "start_age": 27.0, "end_age": 27.767},
            {
                "state": "postpartum_lactational_amenorrhea",
                "start_age": 27.767,
                "end_age": 28.267,
            },
            {"state": "pregnancy", "start_age": 31.0, "end_age": 31.767},
            {
                "state": "postpartum_lactational_amenorrhea",
                "start_age": 31.767,
                "end_age": 32.267,
            },
        ),
        "Two cycle-suppressing exposure windows": (
            {
                "state": "cycle_suppressing_hormonal_exposure",
                "start_age": 20.0,
                "end_age": 27.0,
            },
            {
                "state": "cycle_suppressing_hormonal_exposure",
                "start_age": 34.0,
                "end_age": 39.0,
            },
        ),
    }


def lifespan_atlas() -> None:
    page_intro(
        "Aggregate-constrained scenario engine",
        "Reproductive Lifespan Atlas",
        "Explore how age-dependent timing, participant variation and explicitly imposed no-cycle windows propagate through a stochastic lifespan simulation.",
    )
    callout(
        "This is a population-level scenario model. The reserve curve is not an individual fertility measurement, and interruption windows are imposed assumptions rather than inferred physiology.",
        tone="warning",
        icon="◇",
    )

    with st.expander("Population and scenario controls", expanded=True):
        one, two, three = st.columns(3)
        with one:
            participants = st.slider("Simulated participants", 25, 250, 80, 5)
            seed = st.number_input(
                "Lifespan seed",
                min_value=0,
                max_value=2_147_483_647,
                value=20260727,
            )
        with two:
            menarche_mean = st.slider(
                "Mean menarche age", 10.0, 15.5, 12.5, 0.1
            )
            reserve_sd = st.slider(
                "Reserve log₁₀ variation", 0.05, 0.55, 0.30, 0.01
            )
        with three:
            luteal_mean = st.slider(
                "Mean luteal duration", 10.0, 17.0, 13.5, 0.1
            )
            follicular_variance = st.slider(
                "Follicular variance fraction", 0.50, 0.95, 0.75, 0.01
            )
        scenarios = _interruption_scenarios()
        scenario_name = st.selectbox(
            "Interruption scenario",
            tuple(scenarios),
        )
        run = st.button(
            "Simulate lifespan scenario",
            type="primary",
            width="stretch",
        )

    if run:
        try:
            aggregate = pd.read_csv(AGGREGATE_DATA)
            with st.spinner("Simulating cycle-level passage times across age…"):
                result = simulate_reproductive_lifespan(
                    aggregate,
                    participants=participants,
                    seed=int(seed),
                    menarche_mean_age=menarche_mean,
                    reserve_log10_offset_sd=reserve_sd,
                    luteal_mean_days=luteal_mean,
                    follicular_variance_fraction=follicular_variance,
                    interruption_windows=scenarios[scenario_name],
                )
            st.session_state["lifespan_result"] = result
            st.session_state["lifespan_scenario"] = scenario_name
        except Exception as error:
            st.error(f"Lifespan simulation stopped: {error}")

    result = st.session_state.get("lifespan_result")
    if result is None:
        callout(
            "Run a scenario to generate aggregate age profiles and participant distributions.",
            tone="warning",
            icon="⌁",
        )
        return

    aggregate = pd.read_csv(AGGREGATE_DATA)
    people = result.participants
    scenario_used = st.session_state.get("lifespan_scenario", "Scenario")
    metric_strip(
        (
            (
                "Scenario",
                scenario_used.split()[0],
                scenario_used,
                COLORS["plum"],
            ),
            (
                "Median menarche",
                f"{people['menarche_age_years'].median():.1f} y",
                "simulated distribution",
                COLORS["gold"],
            ),
            (
                "Median cessation",
                f"{people['menopause_age_years'].median():.1f} y",
                "threshold-based scenario",
                COLORS["coral"],
            ),
            (
                "Median cycles",
                f"{people['simulated_cycle_count'].median():.0f}",
                "model-generated count",
                COLORS["teal"],
            ),
        )
    )

    timing_tab, reserve_tab, population_tab, export_tab = st.tabs(
        (
            "Age-dependent timing",
            "Population reserve curve",
            "Participant distributions",
            "Tables & export",
        )
    )
    with timing_tab:
        _plot(
            lifespan_cycle_figure(result.age_summary, aggregate),
            key="lifespan_timing",
        )
    with reserve_tab:
        _plot(reserve_figure(result.reserve_curve), key="reserve_curve")
        st.caption(
            "The plotted curve is a population-model trajectory used to create a cessation threshold. It is not an ovarian-reserve test."
        )
    with population_tab:
        _plot(
            lifespan_population_figure(result.participants),
            key="lifespan_population",
        )
    with export_tab:
        st.dataframe(
            result.age_summary,
            width="stretch",
            hide_index=True,
        )
        first, second = st.columns(2)
        with first:
            st.download_button(
                "Download age summary",
                dataframe_to_csv(result.age_summary),
                "lifespan_age_summary.csv",
                "text/csv",
                width="stretch",
            )
        with second:
            st.download_button(
                "Download participant summary",
                dataframe_to_csv(result.participants),
                "lifespan_participant_summary.csv",
                "text/csv",
                width="stretch",
            )


def data_quality_studio() -> None:
    page_intro(
        "Local schema and completeness audit",
        "Data Quality Studio",
        "Inspect a CSV before modelling. The studio checks the supported schema, timestamp consistency, missing values, basic ranges and table completeness.",
    )
    callout(
        "Uploaded data are processed in the running application session. For sensitive data, run this application locally and do not use an unapproved public deployment.",
        tone="warning",
        icon="▦",
    )

    source = st.radio(
        "Data source",
        ("Included long-form example", "Included wide example", "Upload CSV"),
        horizontal=True,
    )
    frame: pd.DataFrame | None = None
    source_name = ""
    if source == "Included long-form example":
        source_name = "example_common_observations.csv"
        frame = pd.read_csv(ROOT / "data" / source_name)
    elif source == "Included wide example":
        source_name = "example_daily_observations.csv"
        frame = pd.read_csv(ROOT / "data" / source_name)
    else:
        uploaded = st.file_uploader(
            "Choose a CSV file",
            type=("csv",),
            help="The file remains in application memory for this session.",
        )
        if uploaded is not None:
            source_name = uploaded.name
            try:
                frame = pd.read_csv(uploaded)
            except Exception as error:
                st.error(f"Could not read CSV: {error}")

    if frame is None:
        callout(
            "Select an example or upload a CSV to begin the audit.",
            tone="warning",
            icon="▦",
        )
        return

    validation = validate_uploaded_frame(frame)
    date_note = (
        "no valid date range"
        if validation.date_min is None
        else f"{validation.date_min.date()} → {validation.date_max.date()}"
    )
    metric_strip(
        (
            (
                "Validation",
                "Pass" if validation.valid else "Review",
                validation.layout,
                COLORS["teal"] if validation.valid else COLORS["coral"],
            ),
            (
                "Rows",
                f"{validation.rows:,}",
                source_name,
                COLORS["plum"],
            ),
            (
                "Participants",
                f"{validation.participants:,}",
                "pseudonymous IDs",
                COLORS["gold"],
            ),
            (
                "Date coverage",
                (
                    f"{(validation.date_max - validation.date_min).days + 1} d"
                    if validation.date_min is not None
                    else "—"
                ),
                date_note,
                COLORS["blue"],
            ),
        )
    )

    if validation.valid:
        st.success("The table passes the implemented structural checks.")
    else:
        st.error("The table does not yet satisfy the implemented input contract.")
    for message in validation.errors:
        st.error(message)
    for message in validation.warnings:
        st.warning(message)

    completeness_tab, preview_tab, summary_tab, contract_tab = st.tabs(
        ("Completeness", "Preview", "Numeric summary", "Schema contract")
    )
    with completeness_tab:
        _plot(
            upload_completeness_figure(frame),
            key=f"upload_completeness_{validation.layout}",
        )
        if {"signal_name", "missingness_reason"}.issubset(frame):
            missing = missingness_table(frame)
            _plot(missingness_figure(missing), key="upload_missingness")
    with preview_tab:
        st.dataframe(frame.head(500), width="stretch", hide_index=True)
        st.caption(
            f"Showing up to 500 of {len(frame):,} rows. Direct identifiers should never be present."
        )
    with summary_tab:
        st.dataframe(
            safe_numeric_summary(frame),
            width="stretch",
            hide_index=True,
        )
    with contract_tab:
        st.markdown("#### Main long-form observation fields")
        contract = pd.DataFrame(
            {
                "field": OBSERVATION_FIELDS,
                "present": [
                    field in frame.columns for field in OBSERVATION_FIELDS
                ],
            }
        )
        st.dataframe(contract, width="stretch", hide_index=True)
        st.download_button(
            "Download current table",
            dataframe_to_csv(frame),
            source_name,
            "text/csv",
        )


def methods_and_scope() -> None:
    page_intro(
        "Transparent assumptions",
        "Methods & Scope",
        "A compact guide to the process equations, control quantities, observation boundary, reproducibility contract and deliberate non-clinical limits.",
    )
    equation_tab, glossary_tab, workflow_tab, limits_tab = st.tabs(
        ("Equations", "Control glossary", "Reproducibility", "Limits")
    )
    with equation_tab:
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("### Stage passage")
            st.latex(
                r"dZ_t = \exp(\ell_t)\,dt + \sqrt{2D}\,dW_t,"
                r"\qquad Z_t \in [0,1]"
            )
            st.latex(
                r"d\ell_t = \kappa(\log \bar v-\ell_t)\,dt"
                r" + \sigma_\ell\,dB_t"
            )
            st.write(
                "Progress reaches one, probability exits the current stage, "
                "and the next stage begins at progress zero."
            )
        with right:
            st.markdown("### Probability transport")
            st.latex(
                r"\partial_t p = -\partial_z(vp) + D\,\partial_{zz}p"
            )
            st.latex(
                r"J_F(1,t)\rightarrow p_L(0,t),\qquad"
                r"J_L(1,t)\rightarrow p_F(0,t)"
            )
            st.write(
                "The numerical solver moves outgoing boundary flux into the "
                "next stage and checks conservation of total probability."
            )
        callout(
            "These equations are a reduced stochastic representation of timing. Their variables should not be interpreted as measured concentrations or causal endocrine feedback.",
            icon="∑",
        )
    with glossary_tab:
        glossary = pd.DataFrame(
            [
                ("Drift / speed", "Average directional progress toward a transition."),
                ("Diffusion D", "Unstructured short-timescale variability in progress."),
                ("Relaxation κ", "How quickly log speed returns toward its participant mean."),
                ("Speed noise σℓ", "Variability that produces persistent fast or slow passages."),
                ("Péclet number", "Directional progress divided by progress diffusion."),
                ("Relaxation ratio", "Speed-memory timescale relative to passage time."),
                ("First-passage density", "Distribution of the first boundary-crossing time."),
                ("Survival", "Probability that the boundary has not yet been crossed."),
                ("Hazard", "Crossing rate conditional on not yet crossing."),
            ],
            columns=("Term", "Meaning in this model"),
        )
        st.dataframe(glossary, width="stretch", hide_index=True)
    with workflow_tab:
        st.markdown(
            """
            ### Reproducibility contract

            1. Every generated cohort or scenario receives an explicit random seed.
            2. The interface calls functions from `src/digital_twin`; it contains no separate scientific implementation.
            3. Synthetic latent truth remains separate from noisy observations.
            4. Missing measurements remain missing and retain a reason when available.
            5. Exported configurations record the controls needed to recreate a run.
            6. Participant-separated model evaluation remains a command-line workflow.
            """
        )
        architecture = {
            "interface": "app/streamlit_app.py",
            "gui_services": "src/digital_twin/gui/",
            "simulation": "src/digital_twin/simulation/",
            "dynamics": "src/digital_twin/dynamics/",
            "inference": "src/digital_twin/inference/",
            "evaluation": "src/digital_twin/evaluation/",
        }
        st.code(json.dumps(architecture, indent=2), language="json")
    with limits_tab:
        columns = st.columns(2)
        with columns[0]:
            feature_card(
                "✓",
                "Appropriate use",
                "Teaching, exploratory modelling, method development, synthetic experiments and inspection of properly governed research tables.",
            )
        with columns[1]:
            feature_card(
                "×",
                "Unsupported use",
                "Diagnosis, contraception, pregnancy planning, treatment, clinical triage, confirmed ovulation or individual fertility assessment.",
            )
        st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
        callout(
            "A GUI improves accessibility, not evidential strength. Any future clinical claim would require a different validation, governance and regulatory programme.",
            tone="danger",
            icon="!",
        )


section = _sidebar()
if section.startswith("✦"):
    observatory()
elif "Synthetic" in section:
    synthetic_cycle_lab()
elif "First Passage" in section:
    flux_studio()
elif "Lifespan" in section:
    lifespan_atlas()
elif "Data Quality" in section:
    data_quality_studio()
else:
    methods_and_scope()
