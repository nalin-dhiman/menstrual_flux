# Menstrual Flux

Menstrual Flux is research software for representing menstrual-cycle timing as
a stochastic passage-and-reset process. The repository contains simulation,
inference, probabilistic forecasting, dynamical analysis, data adapters, and
reproducible evaluation workflows.

The implementation is intended for mathematical and computational research. It
is not a clinical system and must not be used for contraception, pregnancy
planning, diagnosis, treatment, or triage.

## What is included

- A reduced two-stage follicular/luteal first-passage model.
- A coupled probability-flux solver with cyclic transition and reset.
- An Ornstein--Uhlenbeck speed model and analytic drift--diffusion limits.
- Particle filtering, smoothing, parameter learning, and event forecasts.
- Renewal, rolling-history, calendar, and hidden semi-Markov baselines.
- Calibration, scoring, variability decomposition, and scientific checks.
- Synthetic cohort generation with explicit observation noise and missingness.
- Participant-separated workflows for supported public and restricted cohorts.
- Reproductive-lifespan simulations with sensitivity and interruption scenarios.
- Unit, integration, statistical, and regression tests.

The original circular-state prototype is retained in `src/menstrual_twin/`.
The reduced first-passage implementation is in `src/digital_twin/`.

## Installation

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,data,gui]"
```

Conda users can instead create the supplied environment:

```bash
conda env create -f environment.yml
conda activate menstrual-digital-twin
python -m pip install -e .
```

## Quick verification

```bash
make test
make validate-example
```

Run a short end-to-end synthetic example:

```bash
make demo
```

## Research GUI

Launch the local interactive explorer:

```bash
make gui
```

Keep the launch terminal open and visit `http://127.0.0.1:8501`. Pressing
`Ctrl+C` stops the application. Use `make gui-health` from a second terminal to
verify a running server. SSH and alternative-port instructions are in
[docs/gui.md](docs/gui.md).

The interface provides a synthetic cohort laboratory, analytic first-passage
curves, a coupled probability-flux viewer, aggregate lifespan scenarios, data
quality checks, and reproducible downloads. It is designed for research,
teaching and method exploration and does not provide clinical guidance.

See [docs/gui.md](docs/gui.md) for the view-by-view guide and privacy boundary.

Run the reduced first-passage simulator or the resumable smoke benchmark:

```bash
make simulate
make benchmark-smoke
```

Generated files are written beneath `outputs/` and are excluded from version
control.

## Command-line interface

After installation, use either `digital-twin` or
`python -m digital_twin.cli`. List every command with:

```bash
digital-twin --help
```

Core examples:

```bash
digital-twin validate-data data/example_common_observations.csv

digital-twin simulate \
  --config configs/experiments/milestone_1.yaml \
  --output-dir outputs/simulation

digital-twin run-benchmark \
  --config configs/benchmarks/smoke.yaml \
  --project-root .

digital-twin run-dynamics \
  --config configs/dynamics/first_passage.yaml \
  --project-root .

digital-twin run-lifespan-theory \
  --config configs/dynamics/lifespan_v1.yaml \
  --project-root .
```

The full dynamics workflow expects curated source tables described by its
configuration. The analytic solvers and statistical tests do not require those
cohorts.

## Repository layout

```text
src/digital_twin/       Reduced stochastic model and evaluation workflows
src/menstrual_twin/     Original circular-state prototype
app/                    Local Streamlit research explorer
configs/                Versioned model, benchmark, and data configurations
experiments/            Synthetic experiment grids and public-data guidance
data/                   Schemas, templates, and small non-identifying examples
scripts/                Data validation and synthetic demonstration commands
tests/                  Unit, integration, statistical, and regression tests
```

See [docs/architecture.md](docs/architecture.md) for the component boundaries
and [docs/data_interfaces.md](docs/data_interfaces.md) for the data contract.

## Data policy

No raw participant exports, restricted datasets, or derived participant-level
tables are included. Obtain each source under its own access and licence terms,
keep it outside Git, and point the corresponding adapter at the local source
directory. The repository ignores `data/raw/`, `data/external/`,
`data/curated/`, and generated binary tables by default.

The CSV files in `data/` are synthetic examples or templates. They contain no
real participant records. Adapters preserve source provenance and native
missingness and do not invent unavailable modalities.

## Reproducibility

Configurations are resolved before execution. Workflows use explicit random
seeds, participant-separated splits, output manifests, checksums, resumable
benchmark cells, and deterministic regression tests where applicable.

Run the complete suite before changing a model or configuration:

```bash
PYTHONPATH=src python -m pytest -q
```

## Scope and licensing

The models are phenomenological: their latent variables provide a tractable
description of stage progression and uncertainty, not a causal reconstruction
of endocrine physiology. See [NOTICE.md](NOTICE.md) for the research-use,
governance, and current licensing notice.
