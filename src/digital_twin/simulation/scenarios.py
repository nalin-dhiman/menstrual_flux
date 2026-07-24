SCENARIOS = (
    "short_cycle",
    "medium_cycle",
    "long_cycle",
    "stable",
    "highly_variable",
    "follicular_dominant",
    "luteal_dominant",
    "slow_drift",
    "temporary_disruption",
    "regime_switch",
    "anovulatory_like",
    "stalled_transition",
)


def scenario_for_index(index: int, requested: str) -> str:
    if requested != "mixed":
        return requested
    return SCENARIOS[index % len(SCENARIOS)]
