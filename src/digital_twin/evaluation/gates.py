from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GateDecision:
    gate: str
    passed: bool
    value: float
    threshold: str
    interpretation: str


def evaluate_scientific_gates(
    test_summary: pd.DataFrame,
    calibration: pd.DataFrame,
    missingness: pd.DataFrame,
    recovery: pd.DataFrame,
    abstention: pd.DataFrame,
    ablation: pd.DataFrame | None = None,
    scenario_calibration: pd.DataFrame | None = None,
    seed_comparisons: pd.DataFrame | None = None,
    completeness: pd.DataFrame | None = None,
) -> pd.DataFrame:
    twin = test_summary[(test_summary["model"] == "digital_twin") & (test_summary["target"] == "next_menses")].iloc[0]
    distributional_models = {"hierarchical_renewal", "robust_renewal", "lognormal_renewal", "gamma_renewal", "hsmm"}
    baselines = test_summary[test_summary["model"].isin(distributional_models) & (test_summary["target"] == "next_menses")]
    best_baseline_wis = float(baselines["wis"].min())
    rolling_models = {"rolling_mean_k3", "calibrated_rolling_k3"}
    rolling = test_summary[test_summary["model"].isin(rolling_models) & (test_summary["target"] == "next_menses")]
    rolling_wis = float(rolling["wis"].min()) if not rolling.empty else np.inf
    rolling_label = str(rolling.sort_values("wis").iloc[0]["model"]) if not rolling.empty else "rolling_mean_k3"
    row90 = calibration.iloc[(calibration["nominal"] - 0.9).abs().argmin()]
    calibration_error = abs(float(row90["empirical"] - row90["nominal"]))
    width_ratio = float(missingness["high_missing_width_90"].mean() / max(missingness["low_missing_width_90"].mean(), 1e-9))
    low_coverage = float(missingness["low_missing_coverage_90"].mean())
    high_coverage = float(missingness["high_missing_coverage_90"].mean())
    if "participant_duration_correlation" in recovery and np.isfinite(recovery["participant_duration_correlation"].iloc[0]):
        recovery_score = float(recovery["participant_duration_correlation"].iloc[0])
        recovery_threshold = "participant duration-effect correlation >= 0.70"
        recovery_interpretation = "Earlier cycles must recover an observable participant duration effect; stage-specific speeds remain diagnostic only."
    else:
        recovery_score = float(np.nanmin([recovery["follicular_correlation"].iloc[0], recovery["luteal_correlation"].iloc[0]]))
        recovery_threshold = "both speed correlations >= 0.70"
        recovery_interpretation = "Recovery must use model-visible observations, not exact event truth."
    sensitivity = float(abstention.loc[abstention["metric"] == "sensitivity", "value"].iloc[0])
    specificity = float(abstention.loc[abstention["metric"] == "specificity", "value"].iloc[0])
    decisions = [
        GateDecision("locked_test_calibration", calibration_error <= 0.05, calibration_error, "absolute 90% coverage error <= 0.05", "Test intervals must be calibrated without test-set tuning."),
        GateDecision("renewal_hsmm_value", float(twin["wis"]) <= best_baseline_wis, float(twin["wis"] - best_baseline_wis), "digital-twin WIS <= best renewal/HSMM WIS", "A negative value favors the twin; rolling means remain a separately reported challenge."),
        GateDecision("rolling_k3_value", float(twin["wis"]) <= rolling_wis, float(twin["wis"] - rolling_wis), f"digital-twin WIS <= best K3 comparator ({rolling_label})", "The mechanistic forecast must add distributional value beyond a simple history baseline with equal calibration privilege."),
        GateDecision("missingness_uncertainty", width_ratio >= 1.05 and high_coverage >= low_coverage - 0.05, width_ratio, "width ratio >= 1.05 and coverage drop <= 0.05", "Uncertainty should expand without materially losing coverage."),
        GateDecision("participant_variability_recovery", recovery_score >= 0.70, recovery_score, recovery_threshold, recovery_interpretation),
        GateDecision("safe_abstention", sensitivity >= 0.80 and specificity >= 0.80, min(sensitivity, specificity), "sensitivity and specificity >= 0.80", "Ambiguous transitions should not receive forced precise dates."),
    ]
    if ablation is not None and not ablation.empty:
        all_wis = float(ablation.loc[ablation["modality"] == "all_modalities", "wis"].iloc[0])
        best_single = float(ablation.loc[ablation["modality"] != "all_modalities", "wis"].min())
        decisions.append(GateDecision("multimodal_value", all_wis <= 1.05 * best_single, all_wis / best_single, "all-modality WIS <= 1.05 x best subset", "Multimodal fusion must be competitive with the best signal subset."))
    if scenario_calibration is not None and not scenario_calibration.empty:
        worst_error = float(scenario_calibration["absolute_error_90"].max())
        decisions.append(GateDecision("per_scenario_calibration", worst_error <= 0.05 + 1e-12, worst_error, "maximum scenario 90% coverage error <= 0.05", "Aggregate calibration must not hide generator-specific miscalibration."))
    if seed_comparisons is not None and not seed_comparisons.empty:
        renewal = seed_comparisons[seed_comparisons["comparison"] == "best_renewal_hsmm"]
        if not renewal.empty:
            upper = float(renewal["ci_upper_95"].iloc[0])
            decisions.append(GateDecision("seed_cluster_confidence", upper <= 0, upper, "upper 95% seed-bootstrap CI <= 0", "The renewal/HSMM WIS advantage must persist under seed-cluster uncertainty."))
        rolling_confidence = seed_comparisons[seed_comparisons["comparison"] == "rolling_mean_k3"]
        if not rolling_confidence.empty:
            rolling_upper = float(rolling_confidence["ci_upper_95"].iloc[0])
            decisions.append(GateDecision("rolling_seed_confidence", rolling_upper <= 0, rolling_upper, "upper 95% seed-bootstrap CI <= 0", "Value beyond rolling K3 must persist under seed-cluster uncertainty."))
    if completeness is not None and not completeness.empty:
        row = completeness.iloc[0]
        fraction = float(row["eligible_fraction"])
        complete = int(row["administratively_censored"]) == 0 and int(row["eligible_forecasts"]) == int(row["scored_forecasts"]) and fraction >= float(row["minimum_eligible_fraction"])
        decisions.append(GateDecision("forecast_completeness", complete, fraction, "zero administrative censoring; all eligible forecasts scored; eligible fraction >= configured minimum", "Every planned forecast must be accounted for without silent outcome-dependent exclusion."))
    return pd.DataFrame([asdict(decision) for decision in decisions])
