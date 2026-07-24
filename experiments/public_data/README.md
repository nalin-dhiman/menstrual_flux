# Public-data experiments

This directory contains configuration support for participant-separated
experiments on independently obtained cohorts. Raw or curated participant data
are intentionally not distributed in this repository.

Available adapters:

- `McPhasesAdapter` for a lawfully obtained mcPHASES release;
- `SalzburgHormoneAdapter` for the OSF salivary-hormone release;
- `SoochowHeartRateAdapter` for the Mendeley minute-level heart-rate release.

Each adapter verifies the expected source layout, maps observations to the
common schema, records provenance, and writes a checksum manifest. Missing
signals remain missing; cohorts are never joined to construct artificial
multimodal participants.

The real-data workflows freeze participant splits before fitting, keep
calibration separate from the locked test group, and compare the stochastic
model with simple baselines under the same eligibility rules. Generated
outputs belong under `outputs/`, which is ignored by Git.
