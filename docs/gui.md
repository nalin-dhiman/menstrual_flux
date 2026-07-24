# Menstrual Flux Research Explorer

The Streamlit interface is a local, non-clinical workspace for exploring the
stochastic model. It calls the same functions used by the command-line
workflows and does not contain a separate implementation of the scientific
model.

## Launch

Install the GUI dependencies and start the local server:

```bash
python -m pip install -e ".[gui]"
make gui
```

Streamlit prints a local address, normally `http://localhost:8501`. Data
uploaded to a locally running process remain on that machine for the current
application session.

## Views

- **Observatory:** model orientation and navigation.
- **Synthetic Cycle Lab:** configurable cohorts, latent paths, noisy signals,
  missingness and reproducible exports.
- **Flux & First Passage:** analytic density, survival and hazard curves plus
  the coupled probability-transport solver.
- **Lifespan Atlas:** aggregate-constrained age profiles and explicit
  interruption scenarios.
- **Data Quality Studio:** long- or wide-format CSV validation, completeness
  summaries and schema inspection.
- **Methods & Scope:** equations, parameter glossary, reproducibility contract
  and unsupported uses.

## Privacy and intended use

The application is designed to run locally. Do not deploy it with participant
uploads unless the deployment has appropriate authentication, authorization,
encryption, retention controls, consent and institutional approval.

The interface supports research, teaching, synthetic experiments and
method-development workflows. It does not provide diagnosis, contraception,
pregnancy planning, treatment, confirmed ovulation, fertility assessment or
clinical triage.
