from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .coupled_fokker_planck import solve_coupled_cycle
from .coupled_ou_fokker_planck import solve_coupled_ou_cycle
from .first_passage import (
    dimensionless_groups,
    first_passage_density,
    first_passage_hazard,
    first_passage_moments,
    first_passage_survival,
    sample_constant_first_passage,
)
from .identifiability import (
    inverse_gaussian_profile_likelihood,
    ou_identifiability_surface,
    run_inverse_gaussian_sbc,
)
from .model_comparison import compare_duration_models
from .dynamics_figures import (
    figure_analytic_limit,
    figure_coupled_flux,
    figure_empirical_signatures,
    figure_identifiability,
    figure_regime_map,
)
from .regime_map import build_regime_map
from .signatures import calculate_dynamical_signatures


DYNAMICS_CODE = (
    "src/digital_twin/dynamics/coupled_fokker_planck.py",
    "src/digital_twin/dynamics/coupled_ou_fokker_planck.py",
    "src/digital_twin/dynamics/first_passage.py",
    "src/digital_twin/dynamics/signatures.py",
    "src/digital_twin/dynamics/model_comparison.py",
    "src/digital_twin/dynamics/regime_map.py",
    "src/digital_twin/dynamics/identifiability.py",
    "src/digital_twin/dynamics/dynamics_figures.py",
    "src/digital_twin/dynamics/dynamics_workflow.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    required = {
        "experiment",
        "sources",
        "coupled_fokker_planck",
        "coupled_ou_fokker_planck",
        "analytic_limit",
        "regime_map",
        "identifiability",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"missing dynamics config sections: {sorted(missing)}")
    return config


def _read_tables(curated_dir: Path) -> dict[str, pd.DataFrame]:
    required = (
        "cycles",
        "reference_intervals",
        "daily_observations",
    )
    tables = {}
    for name in required:
        path = curated_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        tables[name] = pd.read_csv(path, low_memory=False)
    return tables


def _eligible_durations(cycles: pd.DataFrame, source: str) -> pd.DataFrame:
    eligible = cycles["eligible_for_primary_evaluation"]
    if eligible.dtype != bool:
        eligible = eligible.astype(str).str.lower().eq("true")
    frame = cycles.loc[eligible, ["participant_id", "cycle_id", "cycle_length_days"]].copy()
    frame["cycle_length_days"] = pd.to_numeric(
        frame["cycle_length_days"], errors="coerce"
    )
    frame = frame.loc[frame["cycle_length_days"].gt(0)]
    frame.insert(0, "source", source)
    return frame


def _stage_summary(stages: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, group in stages.groupby("source"):
        follicular = group["follicular_proxy_days"].to_numpy(dtype=float)
        luteal = group["luteal_proxy_days"].to_numpy(dtype=float)
        rows.append(
            {
                "source": source,
                "cycles": int(len(group)),
                "participants": int(group["participant_id"].nunique()),
                "follicular_proxy_mean_days": float(np.mean(follicular)),
                "follicular_proxy_sd_days": float(
                    np.std(follicular, ddof=1)
                ),
                "luteal_proxy_mean_days": float(np.mean(luteal)),
                "luteal_proxy_sd_days": float(np.std(luteal, ddof=1)),
                "stage_duration_correlation": float(
                    np.corrcoef(follicular, luteal)[0, 1]
                )
                if len(group) >= 3
                else np.nan,
                "follicular_variance_fraction_of_stage_sum": float(
                    np.var(follicular, ddof=1)
                    / (
                        np.var(follicular, ddof=1)
                        + np.var(luteal, ddof=1)
                    )
                ),
                "reference_semantics": (
                    "urinary_LH_positive_surrogate_not_confirmed_ovulation"
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_report(
    output_dir: Path,
    config: dict[str, Any],
    numerical_checks: pd.DataFrame,
    groups: pd.DataFrame,
    signatures: pd.DataFrame,
    stages: pd.DataFrame,
    models: pd.DataFrame,
    missingness: pd.DataFrame,
    sbc: pd.DataFrame,
    ou_summary: pd.DataFrame,
    regime: pd.DataFrame,
) -> Path:
    best = (
        models.sort_values(["source", "aic"])
        .groupby("source", as_index=False)
        .first()
    )
    regime_counts = (
        regime.groupby("regime").size().rename("grid_cells").reset_index()
    )
    lines = [
        "# Menstrual cycles as stochastic first-passage dynamics",
        "",
        "## Scope",
        "",
        "This report evaluates a phenomenological stochastic mathematical-biology "
        "model. It does not claim fundamental menstrual physics, causal endocrine "
        "feedback, confirmed ovulation, clinical validity, or a validated personal "
        "digital twin.",
        "",
        "The primary object is a coupled follicular/luteal passage-and-reset process. "
        "Prediction is secondary and the consumed locked prediction benchmarks were "
        "not rerun.",
        "",
        "## Numerical verification",
        "",
        numerical_checks.to_markdown(index=False),
        "",
        "## Nondimensional control groups",
        "",
        groups.to_markdown(index=False),
        "",
        "The Péclet number compares directional stage progression with progress "
        "diffusion. The relaxation ratio compares the OU speed-relaxation timescale "
        "with deterministic passage time.",
        "",
        "## Empirical dynamical signatures",
        "",
        signatures.to_markdown(index=False),
        "",
        "Raw lag correlation mixes persistent participant differences with temporal "
        "dependence. The hierarchical likelihood comparison below is the primary "
        "test of cycle-to-cycle persistence.",
        "",
        "## Urinary-LH-defined stage proxies",
        "",
        stages.to_markdown(index=False)
        if not stages.empty
        else "No usable stage proxies were available.",
        "",
        "These are menses-to-LH+ and LH+-to-next-menses intervals, not confirmed "
        "follicular and luteal biological durations.",
        "",
        "## Nested duration-model comparison",
        "",
        models.to_markdown(index=False),
        "",
        "### Best supported model by source",
        "",
        best[
            ["source", "model", "aic", "delta_aic", "akaike_weight"]
        ].to_markdown(index=False),
        "",
        "A hierarchical random-intercept result supports persistent between-person "
        "heterogeneity. An AR cycle-shock model is interpreted as supported only if "
        "it improves information criteria beyond that simpler hierarchy.",
        "",
        "## Missingness and duration",
        "",
        missingness.to_markdown(index=False)
        if not missingness.empty
        else "No missingness summaries were available.",
        "",
        "These associations are descriptive and cannot determine whether missingness "
        "changes biological dynamics.",
        "",
        "## Identifiability",
        "",
        "### Analytic constant drift-diffusion limit",
        "",
        sbc.to_markdown(index=False),
        "",
        "### OU-speed degeneracy",
        "",
        ou_summary.to_markdown(index=False),
        "",
        "The analytic limiting model and the full OU-speed model have separate "
        "identifiability conclusions. Success in the former never proves the latter.",
        "",
        "## Regime-map inventory",
        "",
        regime_counts.to_markdown(index=False),
        "",
        "Regime labels describe model behavior on the prespecified Pe–R grid. They "
        "are not diagnoses or empirically established menstrual phenotypes.",
        "",
        "## Defensible conclusion",
        "",
        "The coupled solver establishes a conservative probability-flux formulation "
        "of recurrent follicular/luteal passage and reset. Public data support "
        "substantial duration heterogeneity and permit empirical hazard and stage-"
        "proxy characterization. Whether OU speed relaxation and cycle shocks are "
        "uniquely required is decided by the model-comparison and degeneracy tables, "
        "not assumed from the equations.",
        "",
        "The workflow connects hierarchical passage dynamics, cyclic probability "
        "flux, multimodal observation uncertainty, and explicit empirical "
        "identifiability limits.",
        "",
        "## Figures",
        "",
        "- `figures/Fig1_coupled_probability_flux.pdf`",
        "- `figures/Fig2_analytic_first_passage_limit.pdf`",
        "- `figures/Fig3_nondimensional_regime_map.pdf`",
        "- `figures/Fig4_empirical_dynamical_signatures.pdf`",
        "- `figures/Fig5_model_support_and_identifiability.pdf`",
        "",
        "## Frozen configuration",
        "",
        "```yaml",
        yaml.safe_dump(config, sort_keys=False).rstrip(),
        "```",
        "",
    ]
    path = output_dir / "DYNAMICS_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_dynamics_workflow(
    config_path: Path,
    project_root: Path,
    *,
    force: bool = False,
) -> Path:
    config_path = Path(config_path)
    project_root = Path(project_root).resolve()
    config = _load(config_path)
    output_dir = project_root / config["experiment"]["output_dir"]
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(
            f"{output_dir} is not empty; use --force only for a development rerun"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    started = datetime.now(timezone.utc).isoformat()

    fp = config["coupled_fokker_planck"]
    coupled = solve_coupled_cycle(**fp)
    coupled_table = pd.DataFrame(
        {
            "time": coupled.time,
            "follicular_mass": coupled.follicular_mass,
            "luteal_mass": coupled.luteal_mass,
            "total_mass": coupled.total_mass,
            "ovulation_flux": coupled.ovulation_flux,
            "menses_flux": coupled.menses_flux,
            "first_ovulation_density": coupled.first_ovulation_density,
            "first_menses_density": coupled.first_menses_density,
            "first_cycle_survival": coupled.first_cycle_survival,
        }
    )
    coupled_table.to_csv(output_dir / "coupled_probability_flux.csv", index=False)
    np.savez_compressed(
        output_dir / "coupled_density_fields.npz",
        time=coupled.time,
        grid=coupled.grid,
        follicular_density=coupled.follicular_density,
        luteal_density=coupled.luteal_density,
    )

    joint = solve_coupled_ou_cycle(**config["coupled_ou_fokker_planck"])
    pd.DataFrame(
        {
            "time": joint.time,
            "follicular_mass": joint.follicular_mass,
            "luteal_mass": joint.luteal_mass,
            "total_mass": joint.total_mass,
            "ovulation_flux": joint.ovulation_flux,
            "menses_flux": joint.menses_flux,
        }
    ).to_csv(
        output_dir / "coupled_ou_probability_flux.csv", index=False
    )
    np.savez_compressed(
        output_dir / "coupled_ou_marginals.npz",
        time=joint.time,
        progress_grid=joint.progress_grid,
        log_speed_grid=joint.log_speed_grid,
        follicular_progress_marginal=joint.follicular_progress_marginal,
        luteal_progress_marginal=joint.luteal_progress_marginal,
        follicular_speed_marginal=joint.follicular_speed_marginal,
        luteal_speed_marginal=joint.luteal_speed_marginal,
        final_follicular_density=joint.final_follicular_density,
        final_luteal_density=joint.final_luteal_density,
    )

    analytic_cfg = config["analytic_limit"]
    analytic_mask = coupled.time <= 45.0
    analytic_time = coupled.time[analytic_mask]
    analytic = pd.DataFrame(
        {
            "time": analytic_time,
            "analytic_density": first_passage_density(
                analytic_time,
                fp["follicular_drift"],
                fp["follicular_diffusion"],
                analytic_cfg["boundary"],
            ),
            "analytic_survival": first_passage_survival(
                analytic_time,
                fp["follicular_drift"],
                fp["follicular_diffusion"],
                analytic_cfg["boundary"],
            ),
            "analytic_hazard": first_passage_hazard(
                analytic_time,
                fp["follicular_drift"],
                fp["follicular_diffusion"],
                analytic_cfg["boundary"],
            ),
            "finite_volume_first_flux": coupled.first_ovulation_density[
                analytic_mask
            ],
        }
    )
    analytic.to_csv(output_dir / "analytic_limit.csv", index=False)
    samples = sample_constant_first_passage(
        int(analytic_cfg["exact_samples"]),
        fp["follicular_drift"],
        fp["follicular_diffusion"],
        analytic_cfg["boundary"],
        int(config["experiment"]["seed"]),
    )
    moments = first_passage_moments(
        fp["follicular_drift"],
        fp["follicular_diffusion"],
        analytic_cfg["boundary"],
    )
    numerical_checks = pd.DataFrame(
        [
            {
                "check": "cyclic_total_mass_max_abs_error",
                "value": float(np.max(np.abs(coupled.total_mass - 1.0))),
                "criterion": "<1e-8",
                "passed": bool(
                    np.max(np.abs(coupled.total_mass - 1.0)) < 1e-8
                ),
            },
            {
                "check": "first_ovulation_flux_integral",
                "value": float(
                    np.trapz(
                        coupled.first_ovulation_density, coupled.time
                    )
                ),
                "criterion": ">0.99",
                "passed": bool(
                    np.trapz(
                        coupled.first_ovulation_density, coupled.time
                    )
                    > 0.99
                ),
            },
            {
                "check": "first_menses_flux_integral",
                "value": float(
                    np.trapz(coupled.first_menses_density, coupled.time)
                ),
                "criterion": ">0.98",
                "passed": bool(
                    np.trapz(coupled.first_menses_density, coupled.time)
                    > 0.98
                ),
            },
            {
                "check": "exact_sample_mean_relative_error",
                "value": float(abs(np.mean(samples) - moments.mean) / moments.mean),
                "criterion": "<0.01",
                "passed": bool(
                    abs(np.mean(samples) - moments.mean) / moments.mean < 0.01
                ),
            },
            {
                "check": "analytic_vs_finite_volume_density_L1",
                "value": float(
                    np.trapz(
                        np.abs(
                            analytic["analytic_density"]
                            - analytic["finite_volume_first_flux"]
                        ),
                        analytic["time"],
                    )
                ),
                "criterion": "<0.10",
                "passed": bool(
                    np.trapz(
                        np.abs(
                            analytic["analytic_density"]
                            - analytic["finite_volume_first_flux"]
                        ),
                        analytic["time"],
                    )
                    < 0.10
                ),
            },
        ]
    )
    joint_mass_error = float(np.max(np.abs(joint.total_mass - 1.0)))
    joint_dell = float(joint.log_speed_grid[1] - joint.log_speed_grid[0])
    joint_boundary_probability = float(
        np.max(
            (
                joint.follicular_speed_marginal[:, 0]
                + joint.follicular_speed_marginal[:, -1]
                + joint.luteal_speed_marginal[:, 0]
                + joint.luteal_speed_marginal[:, -1]
            )
            * joint_dell
        )
    )
    numerical_checks = pd.concat(
        [
            numerical_checks,
            pd.DataFrame(
                [
                    {
                        "check": "joint_ou_cyclic_total_mass_max_abs_error",
                        "value": joint_mass_error,
                        "criterion": "<1e-8",
                        "passed": bool(joint_mass_error < 1e-8),
                    },
                    {
                        "check": "joint_ou_log_speed_edge_bin_probability",
                        "value": joint_boundary_probability,
                        "criterion": "<1e-6",
                        "passed": bool(joint_boundary_probability < 1e-6),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    numerical_checks.to_csv(
        output_dir / "numerical_verification.csv", index=False
    )

    dimensionless_rows = []
    for stage in ("follicular", "luteal"):
        groups = dimensionless_groups(
            fp[f"{stage}_drift"],
            fp[f"{stage}_diffusion"],
            kappa=0.35 if stage == "follicular" else 0.45,
            sigma_log_speed=0.035 if stage == "follicular" else 0.025,
        )
        dimensionless_rows.append({"stage": stage, **asdict(groups)})
    dimensionless = pd.DataFrame(dimensionless_rows)
    dimensionless.to_csv(
        output_dir / "dimensionless_groups.csv", index=False
    )

    regime_cfg = config["regime_map"]
    regime = build_regime_map(
        mean_duration=float(regime_cfg["mean_duration"]),
        peclet_values=tuple(regime_cfg["peclet_values"]),
        relaxation_values=tuple(regime_cfg["relaxation_values"]),
        stationary_log_speed_sd=float(
            regime_cfg["stationary_log_speed_sd"]
        ),
        trajectories=int(regime_cfg["trajectories"]),
        dt=float(regime_cfg["dt"]),
        max_time=float(regime_cfg["max_time"]),
        seed=int(config["experiment"]["seed"]) + 101,
    )
    regime.to_csv(output_dir / "nondimensional_regime_map.csv", index=False)

    summaries = []
    hazards = []
    participants = []
    stages = []
    missingness = []
    durations = []
    model_comparisons = []
    profiles = []
    source_manifests: dict[str, str] = {}
    for source_name, source_cfg in config["sources"].items():
        curated_dir = project_root / source_cfg["curated_dir"]
        source_manifests[source_name] = _sha256(curated_dir / "manifest.json")
        tables = _read_tables(curated_dir)
        result = calculate_dynamical_signatures(
            tables["cycles"],
            tables["reference_intervals"],
            tables["daily_observations"],
            source=source_name,
            signals=tuple(source_cfg["signals"]),
        )
        summaries.append(result.summary)
        hazards.append(result.empirical_hazard)
        participants.append(result.participant_statistics)
        if not result.stage_durations.empty:
            stages.append(result.stage_durations)
        if not result.missingness_statistics.empty:
            missingness.append(result.missingness_statistics)
        duration_frame = _eligible_durations(tables["cycles"], source_name)
        durations.append(duration_frame)
        model_comparisons.append(
            compare_duration_models(tables["cycles"], source=source_name)
        )
        profile = inverse_gaussian_profile_likelihood(
            duration_frame["cycle_length_days"].to_numpy()
        )
        profile.insert(0, "source", source_name)
        profiles.append(profile)

    signature_summary = pd.concat(summaries, ignore_index=True)
    hazard_table = pd.concat(hazards, ignore_index=True)
    participant_table = pd.concat(participants, ignore_index=True)
    stage_table = (
        pd.concat(stages, ignore_index=True) if stages else pd.DataFrame()
    )
    missingness_table = (
        pd.concat(missingness, ignore_index=True)
        if missingness
        else pd.DataFrame()
    )
    duration_table = pd.concat(durations, ignore_index=True)
    model_table = pd.concat(model_comparisons, ignore_index=True)
    profile_table = pd.concat(profiles, ignore_index=True)
    stage_summary = _stage_summary(stage_table)

    for name, table in (
        ("empirical_signature_summary.csv", signature_summary),
        ("empirical_hazard.csv", hazard_table),
        ("participant_duration_statistics.csv", participant_table),
        ("lh_stage_proxy_durations.csv", stage_table),
        ("lh_stage_proxy_summary.csv", stage_summary),
        ("missingness_duration_statistics.csv", missingness_table),
        ("eligible_cycle_durations.csv", duration_table),
        ("nested_duration_model_comparison.csv", model_table),
        ("inverse_gaussian_profile_likelihood.csv", profile_table),
    ):
        table.to_csv(output_dir / name, index=False)

    ident = config["identifiability"]
    sbc_draws, sbc_summary = run_inverse_gaussian_sbc(
        sample_sizes=tuple(int(value) for value in ident["sample_sizes"]),
        replicates=int(ident["replicates"]),
        drift_range=tuple(float(value) for value in ident["drift_range"]),
        diffusion_range=tuple(
            float(value) for value in ident["diffusion_range"]
        ),
        grid_size=int(ident["posterior_grid_size"]),
        seed=int(config["experiment"]["seed"]) + 202,
    )
    sbc_draws.to_csv(output_dir / "sbc_draws.csv", index=False)
    sbc_summary.to_csv(output_dir / "sbc_summary.csv", index=False)
    ou_surface, ou_summary = ou_identifiability_surface(
        mean_speed=float(ident["target_mean_speed"]),
        target_kappa=float(ident["target_kappa"]),
        target_stationary_log_speed_sd=float(
            ident["target_stationary_log_speed_sd"]
        ),
        diffusion=float(ident["target_diffusion"]),
        kappa_values=tuple(float(value) for value in ident["kappa_values"]),
        stationary_sd_values=tuple(
            float(value) for value in ident["stationary_sd_values"]
        ),
        trajectories=int(ident["ou_trajectories"]),
        dt=float(ident["ou_dt"]),
        max_time=float(ident["ou_max_time"]),
        seed=int(config["experiment"]["seed"]) + 303,
    )
    ou_surface.to_csv(
        output_dir / "ou_identifiability_surface.csv", index=False
    )
    ou_summary.to_csv(
        output_dir / "ou_identifiability_summary.csv", index=False
    )

    figure_coupled_flux(coupled, figure_dir)
    figure_analytic_limit(analytic, samples, figure_dir)
    figure_regime_map(regime, figure_dir)
    figure_empirical_signatures(
        duration_table,
        hazard_table,
        participant_table,
        stage_table,
        figure_dir,
    )
    figure_identifiability(
        model_table,
        profile_table,
        sbc_summary,
        ou_surface,
        figure_dir,
    )
    report = _write_report(
        output_dir,
        config,
        numerical_checks,
        dimensionless,
        signature_summary,
        stage_summary,
        model_table,
        missingness_table,
        sbc_summary,
        ou_summary,
        regime,
    )

    result_files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "experiment_manifest.json"
    )
    manifest = {
        "experiment": config["experiment"]["name"],
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "claim_scope": "phenomenological_stochastic_mathematical_biology",
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "source_manifest_sha256": source_manifests,
        "code_sha256": {
            path: _sha256(project_root / path) for path in DYNAMICS_CODE
        },
        "output_sha256": {
            str(path.relative_to(output_dir)): _sha256(path)
            for path in result_files
        },
        "all_numerical_checks_passed": bool(
            numerical_checks["passed"].all()
        ),
        "ou_speed_identifiable_from_duration_summaries": bool(
            ou_summary.iloc[0][
                "identifiable_from_three_duration_summaries"
            ]
        ),
        "report": str(report.resolve()),
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report.resolve()
