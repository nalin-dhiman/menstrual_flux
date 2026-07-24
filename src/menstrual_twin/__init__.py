"""Research scaffold for a reduced stochastic menstrual-cycle digital twin.

This package is a methodological prototype. It is not a medical device and must not
be used for diagnosis, contraception, or pregnancy planning.
"""

from .config import ModelConfig, load_config
from .simulator import simulate_cohort, simulate_participant
from .particle_filter import ParticleFilterResult, run_particle_filter, forecast_next_menses

__all__ = [
    "ModelConfig",
    "load_config",
    "simulate_cohort",
    "simulate_participant",
    "ParticleFilterResult",
    "run_particle_filter",
    "forecast_next_menses",
]
