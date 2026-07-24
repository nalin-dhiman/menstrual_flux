from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from digital_twin.config import load_experiment_config
from digital_twin.benchmark import run_benchmark
from digital_twin.data_adapters.mcphases import McPhasesAdapter
from digital_twin.data_adapters.salzburg_hormones import SalzburgHormoneAdapter
from digital_twin.data_adapters.soochow_heart_rate import SoochowHeartRateAdapter
from digital_twin.data.schemas import assert_valid, validate_observations
from digital_twin.dynamics.dynamics_workflow import run_dynamics_workflow
from digital_twin.dynamics.lifespan_experiment import run_lifespan_theory
from digital_twin.experiments import run_milestone_1
from digital_twin.inference.forecast import forecast_events
from digital_twin.inference.particle_filter import run_particle_filter
from digital_twin.real_data import (
    freeze_open_protocol,
    freeze_mcphases_protocol,
    run_open_development,
    run_open_locked_test,
    run_mcphases_audit,
    run_mcphases_development,
    run_mcphases_locked_test,
)
from digital_twin.simulation.cohort import simulate_cohort


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="digital-twin", description="Research-only stochastic menstrual-cycle digital twin")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-data", help="validate a common-schema observation CSV")
    validate.add_argument("path")
    inspect_mcphases = sub.add_parser(
        "inspect-mcphases",
        help="verify and summarize a lawfully obtained restricted mcPHASES v1.0.0 release",
    )
    inspect_mcphases.add_argument("--source-dir", required=True)
    convert_mcphases = sub.add_parser(
        "convert-mcphases",
        help="convert a verified restricted mcPHASES v1.0.0 release to curated tables",
    )
    convert_mcphases.add_argument("--source-dir", required=True)
    convert_mcphases.add_argument("--output-dir", required=True)
    convert_mcphases.add_argument("--format", choices=("parquet", "csv"), default="parquet")
    inspect_salzburg = sub.add_parser(
        "inspect-salzburg-hormones",
        help="verify and summarize the public OSF qevzh salivary-hormone release",
    )
    inspect_salzburg.add_argument("--source-dir", required=True)
    convert_salzburg = sub.add_parser(
        "convert-salzburg-hormones",
        help="convert the public OSF qevzh release to curated tables",
    )
    convert_salzburg.add_argument("--source-dir", required=True)
    convert_salzburg.add_argument("--output-dir", required=True)
    convert_salzburg.add_argument("--format", choices=("parquet", "csv"), default="parquet")
    inspect_soochow = sub.add_parser(
        "inspect-soochow-heart-rate",
        help="verify and summarize the public Mendeley v58stpfcnm release",
    )
    inspect_soochow.add_argument("--source-dir", required=True)
    convert_soochow = sub.add_parser(
        "convert-soochow-heart-rate",
        help="convert raw minute-level heart rate and event grids to curated tables",
    )
    convert_soochow.add_argument("--source-dir", required=True)
    convert_soochow.add_argument("--output-dir", required=True)
    convert_soochow.add_argument("--format", choices=("parquet", "csv"), default="parquet")
    audit_mcphases = sub.add_parser(
        "audit-mcphases",
        help="verify curated mcPHASES tables and write an aggregate feasibility audit",
    )
    audit_mcphases.add_argument("--curated-dir", required=True)
    audit_mcphases.add_argument("--output-dir", required=True)
    freeze_mcphases = sub.add_parser(
        "freeze-mcphases-protocol",
        help="freeze outcome-blind participant splits before real-data fitting",
    )
    freeze_mcphases.add_argument("--curated-dir", required=True)
    freeze_mcphases.add_argument("--config", required=True)
    freeze_mcphases.add_argument("--output-dir", required=True)
    develop_mcphases = sub.add_parser(
        "run-mcphases-development",
        help="fit population/calibration artifacts without locked-test outcomes",
    )
    develop_mcphases.add_argument("--curated-dir", required=True)
    develop_mcphases.add_argument("--config", required=True)
    develop_mcphases.add_argument("--protocol-dir", required=True)
    develop_mcphases.add_argument("--output-dir", required=True)
    develop_mcphases.add_argument("--project-root", default=".")
    locked_mcphases = sub.add_parser(
        "run-mcphases-locked",
        help="verify the model freeze and execute the one-shot locked test",
    )
    locked_mcphases.add_argument("--curated-dir", required=True)
    locked_mcphases.add_argument("--config", required=True)
    locked_mcphases.add_argument("--protocol-dir", required=True)
    locked_mcphases.add_argument("--development-dir", required=True)
    locked_mcphases.add_argument("--restricted-output-dir", required=True)
    locked_mcphases.add_argument("--aggregate-output-dir", required=True)
    locked_mcphases.add_argument("--project-root", default=".")
    freeze_open = sub.add_parser(
        "freeze-open-protocol",
        help="freeze participant splits for a curated public cohort",
    )
    freeze_open.add_argument("--curated-dir", required=True)
    freeze_open.add_argument("--config", required=True)
    freeze_open.add_argument("--output-dir", required=True)
    develop_open = sub.add_parser(
        "run-open-development",
        help="fit training and calibration artifacts without locked-test outcomes",
    )
    develop_open.add_argument("--curated-dir", required=True)
    develop_open.add_argument("--config", required=True)
    develop_open.add_argument("--protocol-dir", required=True)
    develop_open.add_argument("--output-dir", required=True)
    develop_open.add_argument("--project-root", default=".")
    locked_open = sub.add_parser(
        "run-open-locked",
        help="verify the model freeze and run the one-shot public-cohort test",
    )
    locked_open.add_argument("--curated-dir", required=True)
    locked_open.add_argument("--config", required=True)
    locked_open.add_argument("--protocol-dir", required=True)
    locked_open.add_argument("--development-dir", required=True)
    locked_open.add_argument("--output-dir", required=True)
    locked_open.add_argument("--project-root", default=".")
    simulate = sub.add_parser("simulate", help="simulate a separated synthetic cohort")
    simulate.add_argument("--config", required=True)
    simulate.add_argument("--output-dir", default="outputs/simulation")
    run = sub.add_parser("run-experiment", help="run the reproducible Milestone 1 experiment")
    run.add_argument("--config", required=True)
    run.add_argument("--project-root", default=".")
    benchmark = sub.add_parser("run-benchmark", help="run or resume the leakage-safe multi-seed synthetic benchmark")
    benchmark.add_argument("--config", required=True)
    benchmark.add_argument("--project-root", default=".")
    benchmark.add_argument("--force", action="store_true", help="rerun completed seed/scenario cells")
    dynamics = sub.add_parser(
        "run-dynamics",
        help="run the reproducible stochastic first-passage dynamics workflow",
    )
    dynamics.add_argument("--config", required=True)
    dynamics.add_argument("--project-root", default=".")
    dynamics.add_argument(
        "--force",
        action="store_true",
        help="overwrite a development output; never use to relabel locked prediction results",
    )
    lifespan = sub.add_parser(
        "run-lifespan-theory",
        help="run the aggregate-constrained reproductive-lifespan theory",
    )
    lifespan.add_argument("--config", required=True)
    lifespan.add_argument("--project-root", default=".")
    lifespan.add_argument("--force", action="store_true")
    status = sub.add_parser("benchmark-status", help="show resumable benchmark progress/status JSON")
    status.add_argument("--experiment-dir", required=True)
    filt = sub.add_parser("filter", help="filter a common-schema observation CSV")
    filt.add_argument("--config", required=True)
    filt.add_argument("--observations", required=True)
    filt.add_argument("--output", required=True)
    forecast = sub.add_parser("forecast", help="filter observations and save event samples")
    forecast.add_argument("--config", required=True)
    forecast.add_argument("--observations", required=True)
    forecast.add_argument("--output", required=True)
    sub.add_parser("fit-baseline", help="baseline fitting is executed and compared by run-experiment")
    sub.add_parser("fit", help="fixed-parameter filtering is implemented; hierarchical joint fitting is future work")
    sub.add_parser("evaluate", help="evaluation is executed by run-experiment using truth-aligned forecasts")
    report = sub.add_parser("build-report", help="show the existing automatic report path")
    report.add_argument("--experiment-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate-data":
        frame = pd.read_csv(args.path)
        assert_valid(validate_observations(frame))
        print(f"valid: {len(frame)} observation rows")
    elif args.command == "inspect-mcphases":
        summary = McPhasesAdapter().inspect(Path(args.source_dir))
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.command == "convert-mcphases":
        adapter = McPhasesAdapter()
        result = adapter.convert(Path(args.source_dir))
        manifest = adapter.write(result, Path(args.output_dir), file_format=args.format)
        summary = {
            "output_dir": str(Path(args.output_dir).resolve()),
            "access_classification": manifest["access_classification"],
            "shareable": manifest["shareable"],
            "table_rows": {
                name: metadata["rows"] for name, metadata in manifest["tables"].items()
            },
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.command == "inspect-salzburg-hormones":
        summary = SalzburgHormoneAdapter().inspect(Path(args.source_dir))
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.command == "convert-salzburg-hormones":
        adapter = SalzburgHormoneAdapter()
        result = adapter.convert(Path(args.source_dir))
        manifest = adapter.write(
            result, Path(args.output_dir), file_format=args.format
        )
        print(
            json.dumps(
                {
                    "output_dir": str(Path(args.output_dir).resolve()),
                    "access_classification": manifest["access_classification"],
                    "shareable": manifest["shareable"],
                    "table_rows": {
                        name: metadata["rows"]
                        for name, metadata in manifest["tables"].items()
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "inspect-soochow-heart-rate":
        summary = SoochowHeartRateAdapter().inspect(Path(args.source_dir))
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.command == "convert-soochow-heart-rate":
        adapter = SoochowHeartRateAdapter()
        result = adapter.convert(Path(args.source_dir))
        manifest = adapter.write(
            result, Path(args.output_dir), file_format=args.format
        )
        print(
            json.dumps(
                {
                    "output_dir": str(Path(args.output_dir).resolve()),
                    "access_classification": manifest["access_classification"],
                    "shareable": manifest["shareable"],
                    "table_rows": {
                        name: metadata["rows"]
                        for name, metadata in manifest["tables"].items()
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "audit-mcphases":
        print(
            run_mcphases_audit(
                Path(args.curated_dir),
                Path(args.output_dir),
            )
        )
    elif args.command == "freeze-mcphases-protocol":
        print(
            freeze_mcphases_protocol(
                Path(args.curated_dir),
                Path(args.config),
                Path(args.output_dir),
            )
        )
    elif args.command == "run-mcphases-development":
        print(
            run_mcphases_development(
                Path(args.project_root).resolve(),
                Path(args.curated_dir),
                Path(args.config),
                Path(args.protocol_dir),
                Path(args.output_dir),
            )
        )
    elif args.command == "run-mcphases-locked":
        print(
            run_mcphases_locked_test(
                Path(args.project_root).resolve(),
                Path(args.curated_dir),
                Path(args.config),
                Path(args.protocol_dir),
                Path(args.development_dir),
                Path(args.restricted_output_dir),
                Path(args.aggregate_output_dir),
            )
        )
    elif args.command == "freeze-open-protocol":
        print(
            freeze_open_protocol(
                Path(args.curated_dir),
                Path(args.config),
                Path(args.output_dir),
            )
        )
    elif args.command == "run-open-development":
        print(
            run_open_development(
                Path(args.project_root).resolve(),
                Path(args.curated_dir),
                Path(args.config),
                Path(args.protocol_dir),
                Path(args.output_dir),
            )
        )
    elif args.command == "run-open-locked":
        print(
            run_open_locked_test(
                Path(args.project_root).resolve(),
                Path(args.curated_dir),
                Path(args.config),
                Path(args.protocol_dir),
                Path(args.development_dir),
                Path(args.output_dir),
            )
        )
    elif args.command == "simulate":
        cfg = load_experiment_config(args.config)
        cohort = simulate_cohort(cfg)
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        cohort.observed.to_csv(output / "observations.csv", index=False)
        cohort.truth.to_csv(output / "latent_truth.csv", index=False)
        cohort.events.to_csv(output / "event_truth.csv", index=False)
        print(output.resolve())
    elif args.command == "run-experiment":
        cfg = load_experiment_config(args.config)
        print(run_milestone_1(cfg, Path(args.project_root).resolve()))
    elif args.command == "run-benchmark":
        print(run_benchmark(args.config, Path(args.project_root).resolve(), force=args.force))
    elif args.command == "run-dynamics":
        print(
            run_dynamics_workflow(
                Path(args.config),
                Path(args.project_root).resolve(),
                force=args.force,
            )
        )
    elif args.command == "run-lifespan-theory":
        print(
            run_lifespan_theory(
                Path(args.config),
                Path(args.project_root).resolve(),
                force=args.force,
            )
        )
    elif args.command == "benchmark-status":
        directory = Path(args.experiment_dir)
        path = directory / "benchmark_status.json" if (directory / "benchmark_status.json").exists() else directory / "benchmark_progress.json"
        if not path.exists():
            raise FileNotFoundError(path)
        print(path.read_text(encoding="utf-8"))
    elif args.command in {"filter", "forecast"}:
        cfg = load_experiment_config(args.config)
        observations = pd.read_csv(args.observations, parse_dates=["event_time", "measurement_time", "report_time", "availability_time"])
        result = run_particle_filter(observations, cfg)
        if args.command == "filter":
            result.summary.to_csv(args.output, index=False)
        else:
            event = forecast_events(result, cfg)
            pd.DataFrame({"ovulation_day": event.ovulation_samples, "next_menses_day": event.next_menses_samples}).to_csv(args.output, index=False)
    elif args.command in {"fit-baseline", "fit", "evaluate"}:
        print("Use `digital-twin run-experiment --config ...` for the reproducible implementation path.")
    elif args.command == "build-report":
        report = Path(args.experiment_dir) / "REPORT.md"
        if not report.exists():
            raise FileNotFoundError(report)
        print(report.resolve())
    return 0
