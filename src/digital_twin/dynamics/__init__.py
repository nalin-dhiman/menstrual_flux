"""Stochastic first-passage and reproductive-lifespan dynamics tools."""

from .coupled_fokker_planck import CoupledFokkerPlanckResult, solve_coupled_cycle
from .coupled_ou_fokker_planck import (
    CoupledOUFokkerPlanckResult,
    solve_coupled_ou_cycle,
)
from .first_passage import (
    DimensionlessGroups,
    FirstPassageMoments,
    dimensionless_groups,
    first_passage_density,
    first_passage_hazard,
    first_passage_moments,
    first_passage_survival,
    simulate_ou_first_passage,
)

__all__ = [
    "CoupledFokkerPlanckResult",
    "CoupledOUFokkerPlanckResult",
    "DimensionlessGroups",
    "FirstPassageMoments",
    "dimensionless_groups",
    "first_passage_density",
    "first_passage_hazard",
    "first_passage_moments",
    "first_passage_survival",
    "simulate_ou_first_passage",
    "solve_coupled_cycle",
    "solve_coupled_ou_cycle",
]
