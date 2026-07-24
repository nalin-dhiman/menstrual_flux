# Synthetic experiment configurations

The S1--S11 files are runnable configurations for the common deterministic
experiment engine. They vary one primary stressor while retaining the resolved
defaults. For a multi-seed evaluation, expand each configuration into a
predeclared seed and parameter grid.

```bash
PYTHONPATH=src python -m digital_twin.cli run-experiment --config experiments/synthetic/S1_oracle.yaml --project-root .
```

Every run writes a resolved configuration, log, metrics, figures, tables,
automatic report, and experiment card. S8 is a placeholder configuration for
heavy-tailed observations; colored noise, nonlinear dynamics, abrupt bias, and
device-shift generators still require explicit implementations.
