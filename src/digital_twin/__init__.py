"""Research implementation of a stochastic menstrual-cycle digital twin.

This package is for simulation and methodological research. It is not a medical
device and must not be used for diagnosis, contraception, or clinical decisions.
"""

from .config import ExperimentConfig, load_experiment_config

__all__ = ["ExperimentConfig", "load_experiment_config"]
__version__ = "0.4.0"
