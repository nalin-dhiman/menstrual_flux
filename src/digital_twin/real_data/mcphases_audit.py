from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


REQUIRED_TABLES = {
    "participants",
    "daily_observations",
    "hormone_measurements",
    "events",
    "reference_intervals",
    "cycles",
    "data_quality",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_verified_tables(curated_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    manifest_path = curated_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("access_classification") != "restricted_health_data":
        raise ValueError("Expected a restricted-health-data curated manifest")
    metadata = manifest.get("tables", {})
    missing = sorted(REQUIRED_TABLES - set(metadata))
    if missing:
        raise ValueError(f"Curated manifest is missing required tables: {missing}")
    tables: dict[str, pd.DataFrame] = {}
    for name in sorted(REQUIRED_TABLES):
        details = metadata[name]
        path = curated_dir / str(details["path"])
        if _sha256(path) != details["sha256"]:
            raise ValueError(f"Curated checksum verification failed: {path.name}")
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
        elif path.suffix == ".csv":
            frame = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported curated table format: {path}")
        if len(frame) != int(details["rows"]):
            raise ValueError(f"Curated row-count verification failed: {path.name}")
        tables[name] = frame
    return tables, manifest


def _forecast_opportunities(cycles: pd.DataFrame, history_cycles: int) -> tuple[int, int]:
    eligible = cycles.loc[cycles["eligible_for_primary_evaluation"]].copy()
    counts = eligible.groupby(["participant_id", "study_interval"]).size()
    opportunities = (counts - history_cycles).clip(lower=0)
    return int(opportunities.sum()), int(opportunities.gt(0).sum())


def build_mcphases_audit(curated_dir: Path) -> dict[str, object]:
    tables, manifest = _load_verified_tables(Path(curated_dir))
    participants = tables["participants"]
    observations = tables["daily_observations"]
    cycles = tables["cycles"]
    events = tables["events"]
    references = tables["reference_intervals"]
    quality = tables["data_quality"]

    coverage = (
        observations.groupby("signal_name")["value"]
        .apply(lambda values: float(values.notna().mean()))
        .sort_index()
        .to_dict()
    )
    interval_coverage = (
        observations.groupby(["study_interval", "signal_name"])["value"]
        .apply(lambda values: float(values.notna().mean()))
        .unstack("signal_name")
    )
    history = {}
    for count in (0, 1, 2, 3):
        forecasts, participant_intervals = _forecast_opportunities(cycles, count)
        history[str(count)] = {
            "forecast_opportunities": forecasts,
            "participant_intervals": participant_intervals,
        }

    complete = cycles.loc[cycles["complete_event_interval"]]
    summary: dict[str, object] = {
        "source": "mcPHASES-1.0.0",
        "access_classification": "restricted_health_data",
        "report_content": "aggregate_only_but_human_disclosure_review_required",
        "curated_manifest_verified": True,
        "participants": int(participants["participant_id"].nunique()),
        "participant_intervals": int(
            quality.groupby(["participant_id", "study_interval"]).ngroups
        ),
        "daily_observation_rows": len(observations),
        "signals": int(observations["signal_name"].nunique()),
        "menstruation_onset_events": len(events),
        "complete_cycle_intervals": len(complete),
        "primary_eligible_cycle_intervals": int(
            complete["eligible_for_primary_evaluation"].sum()
        ),
        "outlier_cycle_intervals_requiring_review": int(
            complete["eligibility_reason"].eq(
                "cycle_interval_outlier_review_required"
            ).sum()
        ),
        "reference_interval_counts": {
            str(key): int(value)
            for key, value in references["event_type"].value_counts().to_dict().items()
        },
        "forecast_opportunities_by_required_history": history,
        "coverage_by_signal": {str(key): float(value) for key, value in coverage.items()},
        "key_interval_coverage": {
            str(int(interval)): {
                signal: float(interval_coverage.loc[interval, signal])
                for signal in ("lh", "e3g", "pdg", "bleeding_reported")
                if signal in interval_coverage
            }
            for interval in interval_coverage.index
        },
        "readiness": {
            "multimodal_device_phase_pilot": "feasible",
            "next_menses_pilot_with_one_history_cycle": "feasible_but_small",
            "strict_rolling_k3_primary_comparison": "underpowered",
            "participant_specific_physiological_learning": "exploratory_only",
            "external_validation": "not_available_in_this_dataset",
            "clinical_ovulation_validation": "not_supported",
        },
        "required_analysis_constraints": [
            "Use participant-separated population training, calibration, and locked testing.",
            "Within each participant interval, use only past cycles and past-day observations.",
            "Treat first and last cycles as censored and retain the two short intervals only in a sensitivity analysis.",
            "Call Mira labels device-defined references, not clinical ovulation truth.",
            "Run interval-specific modality analyses because PdG is absent in 2022 and diaries are mostly absent in 2024.",
            "Report calibrated rolling baselines with the same calibration data as the digital-twin model.",
        ],
        "input_manifest_created_utc": manifest.get("created_utc"),
    }
    return summary


def _markdown(summary: dict[str, object]) -> str:
    history = summary["forecast_opportunities_by_required_history"]
    coverage = summary["coverage_by_signal"]
    reference_counts = summary["reference_interval_counts"]
    readiness = summary["readiness"]
    lines = [
        "# mcPHASES v1.0.0 real-data audit",
        "",
        "> Restricted source data. This report contains aggregates only, but still requires human disclosure review before public release.",
        "",
        "## Verified scope",
        "",
        f"- Participants: {summary['participants']}",
        f"- Participant-intervals: {summary['participant_intervals']}",
        f"- Normalized observation rows: {summary['daily_observation_rows']}",
        f"- Signals: {summary['signals']}",
        f"- Menstruation-onset events: {summary['menstruation_onset_events']}",
        f"- Complete cycle intervals: {summary['complete_cycle_intervals']}",
        f"- Primary-eligible cycle intervals: {summary['primary_eligible_cycle_intervals']}",
        f"- Cycle intervals flagged for sensitivity review: {summary['outlier_cycle_intervals_requiring_review']}",
        "",
        "## Forecast feasibility",
        "",
        "| Required within-interval history | Forecast opportunities | Contributing participant-intervals |",
        "|---:|---:|---:|",
    ]
    for count in ("0", "1", "2", "3"):
        values = history[count]
        lines.append(
            f"| {count} | {values['forecast_opportunities']} | {values['participant_intervals']} |"
        )
    lines.extend(
        [
            "",
            "A strict rolling-K3 comparison is underpowered. The first defensible next-menses pilot should require one prior complete cycle and use an up-to-three-history rolling baseline with a prespecified population fallback.",
            "",
            "## Reference intervals",
            "",
        ]
    )
    for name, count in reference_counts.items():
        lines.append(f"- {name}: {count}")
    lines.extend(
        [
            "",
            "These are proprietary Mira/device-defined references, not ultrasound-confirmed ovulation.",
            "",
            "## Selected signal coverage",
            "",
        ]
    )
    selected = [
        "lh",
        "e3g",
        "pdg",
        "nightly_skin_temperature",
        "resting_heart_rate",
        "sleep_hrv_rmssd",
        "sleep_overall_score",
        "bleeding_reported",
    ]
    for signal in selected:
        if signal in coverage:
            lines.append(f"- {signal}: {coverage[signal]:.1%}")
    lines.extend(["", "## Readiness", ""])
    for claim, status in readiness.items():
        lines.append(f"- {claim}: **{status}**")
    lines.extend(["", "## Required constraints", ""])
    lines.extend(f"- {constraint}" for constraint in summary["required_analysis_constraints"])
    lines.append("")
    return "\n".join(lines)


def run_mcphases_audit(curated_dir: Path, output_dir: Path) -> Path:
    summary = build_mcphases_audit(Path(curated_dir))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "audit_summary.json"
    report_path = output_dir / "REAL_DATA_AUDIT.md"
    existing = [path for path in (summary_path, report_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing audit outputs: "
            + ", ".join(path.name for path in existing)
        )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(_markdown(summary), encoding="utf-8")
    return report_path.resolve()
