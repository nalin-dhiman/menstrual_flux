# Software architecture

The codebase separates generative assumptions, observation handling, inference,
and evaluation so that each layer can be tested independently.

## Main package

`digital_twin.simulation` generates latent trajectories, observation streams,
events, perturbations, and missingness. Latent truth and observed values are
stored separately.

`digital_twin.inference` implements particle filtering, resampling, smoothing,
parameter learning, personalization, and event-time sampling. Forecasts use only
observations available at the configured issue time.

`digital_twin.baselines` contains renewal and hidden semi-Markov comparators.
Baseline forecasts pass through the same scoring and calibration interfaces as
the stochastic model.

`digital_twin.dynamics` contains analytic first-passage results, coupled
Fokker--Planck solvers, regime maps, dynamical signatures, model comparison,
identifiability checks, and reproductive-lifespan simulations.

`digital_twin.data_adapters` maps supported source layouts to common tables.
`digital_twin.real_data` handles participant splits, development-only fitting,
model freezes, and locked evaluation.

`digital_twin.evaluation` provides probabilistic metrics, calibration,
variability summaries, and pass/fail gates. `digital_twin.visualization`
contains reusable plotting functions.

## Legacy prototype

`menstrual_twin` preserves the earlier circular-state simulator and particle
filter. It remains useful as a compact demonstration and regression target but
is independent of the reduced two-stage implementation.

## Execution flow

```text
configuration
    -> simulation or source adapter
    -> validated common-schema tables
    -> baseline/model inference
    -> event-time samples
    -> calibration and probabilistic metrics
    -> checksummed outputs
```

All generated outputs stay outside the tracked source tree under `outputs/`.
