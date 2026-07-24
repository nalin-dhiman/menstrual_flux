from __future__ import annotations

import numpy as np

from .config import ModelConfig
from .math_utils import TWO_PI, circular_difference, sigmoid


def phase_templates(theta: np.ndarray, cfg: ModelConfig) -> dict[str, np.ndarray]:
    """Smooth, deliberately simple templates used by the prototype.

    These are not claims of exact endocrine physiology. In a research model they
    should be estimated hierarchically and validated against reference assays.
    """
    ov = TWO_PI * cfg.process.ovulation_phase_fraction
    # Luteal activation: smooth rise after ovulation and fall approaching menses.
    luteal = sigmoid(circular_difference(theta, ov) / 0.18) * sigmoid((TWO_PI - theta) / 0.16)
    # Localized LH peak at ovulation.
    d_ov = circular_difference(theta, ov)
    lh_peak = np.exp(-0.5 * (d_ov / 0.11) ** 2)
    # Estrogen-metabolite peak before the LH surge.
    d_e3g = circular_difference(theta, ov - 0.30)
    e3g_peak = np.exp(-0.5 * (d_e3g / 0.32) ** 2)
    # Bleeding probability highest just after phase reset.
    d_m = np.minimum(theta, TWO_PI - theta)
    bleeding_peak = np.exp(-0.5 * (d_m / 0.38) ** 2)
    return {
        "luteal": luteal,
        "lh_peak": lh_peak,
        "e3g_peak": e3g_peak,
        "bleeding_peak": bleeding_peak,
    }


def expected_observations(theta: np.ndarray, amplitude: np.ndarray, cfg: ModelConfig) -> dict[str, np.ndarray]:
    t = phase_templates(theta, cfg)
    o = cfg.observation
    temp = o.temperature_baseline_c + o.temperature_luteal_shift_c * t["luteal"] + 0.03 * amplitude
    rhr = o.rhr_baseline_bpm + o.rhr_luteal_shift_bpm * t["luteal"] + 0.4 * amplitude
    lh = o.lh_baseline + o.lh_surge_amplitude * t["lh_peak"]
    e3g = o.e3g_baseline + o.e3g_peak_amplitude * t["e3g_peak"]
    pdg = o.pdg_baseline + o.pdg_luteal_amplitude * t["luteal"]
    logit_bleed = o.bleeding_logit_floor + (o.bleeding_logit_peak - o.bleeding_logit_floor) * t["bleeding_peak"]
    bleed_prob = sigmoid(logit_bleed)
    return {"temperature_c": temp, "rhr_bpm": rhr, "lh": lh, "e3g": e3g, "pdg": pdg, "bleeding_prob": bleed_prob}
