from __future__ import annotations

import pandas as pd


def retrospective_smoothed_summary(filtered_summary: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Conservative fixed-window summary smoother for retrospective visualization."""
    out = filtered_summary.copy()
    for col in ("p_follicular", "p_luteal", "mean_progress"):
        if col in out:
            out[f"smoothed_{col}"] = out[col].rolling(window, center=True, min_periods=1).mean()
    return out
