from __future__ import annotations

import numpy as np
import pandas as pd


def participant_holdout(participants: pd.Series, test_fraction: float = 0.25, seed: int = 0) -> dict[str, str]:
    unique = np.array(sorted(participants.astype(str).unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_test = max(1, int(round(len(unique) * test_fraction)))
    test = set(unique[:n_test])
    return {p: ("test" if p in test else "train") for p in unique}


def forward_cycle_split(cycles: pd.DataFrame, adaptation_cycles: int = 2) -> pd.Series:
    order = cycles.groupby("participant_id")["cycle_start"].rank(method="dense")
    return pd.Series(np.where(order <= adaptation_cycles, "adaptation", "test"), index=cycles.index)
