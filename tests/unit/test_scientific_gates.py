import pandas as pd

from digital_twin.evaluation.gates import evaluate_scientific_gates


def test_gate_contract_targets_distributional_baselines_and_safety():
    summary = pd.DataFrame([
        {"model": "digital_twin", "target": "next_menses", "wis": 3.0},
        {"model": "hierarchical_renewal", "target": "next_menses", "wis": 3.4},
        {"model": "hsmm", "target": "next_menses", "wis": 3.6},
        {"model": "rolling_mean_k3", "target": "next_menses", "wis": 3.2},
    ])
    calibration = pd.DataFrame({"nominal": [0.9], "empirical": [0.91]})
    missingness = pd.DataFrame({
        "low_missing_width_90": [6.0, 6.2], "high_missing_width_90": [7.0, 7.2],
        "low_missing_coverage_90": [0.9, 0.9], "high_missing_coverage_90": [0.9, 0.85],
    })
    recovery = pd.DataFrame({"follicular_correlation": [0.8], "luteal_correlation": [0.75]})
    abstention = pd.DataFrame({"metric": ["sensitivity", "specificity"], "value": [0.9, 0.85]})
    ablation = pd.DataFrame({"modality": ["all_modalities", "hormones"], "wis": [3.0, 2.95]})
    gates = evaluate_scientific_gates(summary, calibration, missingness, recovery, abstention, ablation)
    assert gates["passed"].all()
    assert "renewal_hsmm_value" in set(gates["gate"])
    assert "rolling_k3_value" in set(gates["gate"])


def test_extended_gates_reject_hidden_scenario_and_count_failures():
    summary = pd.DataFrame([
        {"model": "digital_twin", "target": "next_menses", "wis": 3.0},
        {"model": "lognormal_renewal", "target": "next_menses", "wis": 3.2},
        {"model": "rolling_mean_k3", "target": "next_menses", "wis": 3.1},
    ])
    calibration = pd.DataFrame({"nominal": [0.9], "empirical": [0.9]})
    missingness = pd.DataFrame({
        "low_missing_width_90": [6.0], "high_missing_width_90": [6.5],
        "low_missing_coverage_90": [0.9], "high_missing_coverage_90": [0.9],
    })
    recovery = pd.DataFrame({"follicular_correlation": [0.8], "luteal_correlation": [0.8]})
    abstention = pd.DataFrame({"metric": ["sensitivity", "specificity"], "value": [0.9, 0.9]})
    scenario = pd.DataFrame({"scenario": ["regular", "heterogeneous"], "absolute_error_90": [0.01, 0.08]})
    comparisons = pd.DataFrame({"comparison": ["best_renewal_hsmm"], "ci_upper_95": [-0.01]})
    completeness = pd.DataFrame({
        "administratively_censored": [1], "eligible_forecasts": [99], "scored_forecasts": [99],
        "eligible_fraction": [0.99], "minimum_eligible_fraction": [0.99],
    })
    gates = evaluate_scientific_gates(
        summary, calibration, missingness, recovery, abstention,
        scenario_calibration=scenario, seed_comparisons=comparisons, completeness=completeness,
    ).set_index("gate")
    assert not bool(gates.loc["per_scenario_calibration", "passed"])
    assert not bool(gates.loc["forecast_completeness", "passed"])
