from __future__ import annotations

import numpy as np
import pandas as pd

from digital_twin.config import ExperimentConfig
from digital_twin.simulation.latent_process import ParticipantParameters


SIGNAL_UNITS = {
    "bleeding": "ordinal_0_4",
    "temperature": "degC",
    "resting_heart_rate": "beats/min",
    "hrv": "ms_RMSSD",
    "sleep_duration": "hours",
    "sleep_efficiency": "proportion",
    "lh": "mIU/mL_synthetic",
    "e3g": "ng/mL_synthetic",
    "pdg": "ug/mL_synthetic",
    "estradiol": "pg/mL_synthetic",
    "progesterone": "ng/mL_synthetic",
    "symptom_severity": "ordinal_0_4",
    "wearable_availability": "binary",
    "reporting_availability": "binary",
}


def expected_signal(signal: str, stage: np.ndarray, progress: np.ndarray, baselines: dict[str, float] | None = None) -> np.ndarray:
    """Deliberately parsimonious stage/progress observation templates."""
    baselines = baselines or {}
    stage = np.asarray(stage)
    progress = np.asarray(progress)
    luteal = (stage == 1).astype(float)
    late_luteal = luteal * np.exp(-0.5 * ((progress - 0.92) / 0.16) ** 2)
    menstrual = (stage == 0).astype(float) * np.exp(-0.5 * (progress / 0.13) ** 2)
    ov_peak = np.where(stage == 0, np.exp(-0.5 * ((progress - 0.96) / 0.075) ** 2), 0.0)
    estrogen_peak = np.where(stage == 0, np.exp(-0.5 * ((progress - 0.78) / 0.18) ** 2), 0.18 * np.exp(-0.5 * ((progress - 0.45) / 0.2) ** 2))
    luteal_rise = luteal / (1 + np.exp(-(progress - 0.16) / 0.07))
    if signal == "temperature":
        return baselines.get("temperature", 36.45) + baselines.get("temperature_amplitude", 0.24) * luteal_rise
    if signal == "resting_heart_rate":
        return baselines.get("rhr", 62.0) + 2.2 * luteal_rise
    if signal == "hrv":
        return np.exp(baselines.get("hrv_log", np.log(48.0)) - 0.12 * luteal_rise)
    if signal == "sleep_duration":
        return 7.2 - 0.20 * late_luteal
    if signal == "sleep_efficiency":
        return 0.88 - 0.025 * late_luteal
    if signal == "lh":
        return 3.0 + 27.0 * baselines.get("hormone_amplitude", 1.0) * ov_peak
    if signal == "e3g":
        return 45.0 + 150.0 * baselines.get("hormone_amplitude", 1.0) * estrogen_peak
    if signal == "pdg":
        return 1.4 + 10.0 * baselines.get("hormone_amplitude", 1.0) * luteal_rise
    if signal == "estradiol":
        return 45.0 + 185.0 * baselines.get("hormone_amplitude", 1.0) * estrogen_peak
    if signal == "progesterone":
        return 0.8 + 12.0 * baselines.get("hormone_amplitude", 1.0) * luteal_rise
    if signal == "bleeding":
        return np.clip(0.03 + 3.4 * menstrual + 0.5 * late_luteal, 0, 4)
    if signal == "symptom_severity":
        return np.clip(0.4 + 1.7 * late_luteal, 0, 4)
    if signal in {"wearable_availability", "reporting_availability"}:
        return np.ones_like(progress, dtype=float)
    raise KeyError(f"Unknown signal: {signal}")


def simulate_observations(
    states: pd.DataFrame,
    events: pd.DataFrame,
    parameters: ParticipantParameters,
    cfg: ExperimentConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    o = cfg.observation
    baselines = {
        "temperature": parameters.temperature_baseline,
        "temperature_amplitude": parameters.temperature_amplitude,
        "rhr": parameters.rhr_baseline,
        "hrv_log": parameters.hrv_baseline_log,
        "hormone_amplitude": parameters.hormone_amplitude,
    }
    signals = list(SIGNAL_UNITS)
    rows: list[dict[str, object]] = []
    menses = events[events["event_type"] == "menstruation_onset"]["event_time"].tolist()
    for _, state_row in states.iterrows():
        t = pd.Timestamp(state_row["event_time"])
        stage = np.array([int(state_row["stage_code"])])
        progress = np.array([float(state_row["progress"])])
        days_from_menses = min((abs((t - pd.Timestamp(x)).total_seconds()) / 86400 for x in menses), default=99)
        for signal in signals:
            mean = float(expected_signal(signal, stage, progress, baselines)[0])
            if signal == "bleeding":
                if days_from_menses < 1.0:
                    value = int(rng.choice([2, 3, 4], p=[0.20, 0.55, 0.25]))
                elif days_from_menses < 4.0:
                    value = int(max(0, round(3.2 - 0.75 * days_from_menses + rng.normal(0, 0.55))))
                else:
                    value = int(rng.random() < 0.015)
            elif signal == "temperature":
                value = mean + o.temperature_sd * rng.standard_t(o.heavy_tail_df) / np.sqrt(o.heavy_tail_df / (o.heavy_tail_df - 2))
            elif signal == "resting_heart_rate":
                value = mean + o.rhr_sd * rng.normal()
            elif signal == "hrv":
                value = float(np.exp(rng.normal(np.log(max(mean, 0.1)), o.log_hrv_sd)))
            elif signal == "sleep_duration":
                value = float(np.clip(rng.normal(mean, o.sleep_duration_sd), 2.5, 11.5))
            elif signal == "sleep_efficiency":
                value = float(np.clip(rng.normal(mean, o.sleep_efficiency_sd), 0.45, 1.0))
            elif signal in {"lh", "e3g", "pdg", "estradiol", "progesterone"}:
                hormone_mean = mean
                suppressed_cycle = bool(state_row.get("anovulatory_like_cycle", state_row.get("anovulatory_like", False)))
                stalled_before_transition = bool(state_row.get("stalled_transition_cycle", False)) and int(state_row["stage_code"]) == 0 and float(state_row.get("stage_elapsed_days", 0)) < cfg.process.stalled_follicular_days
                if (suppressed_cycle or stalled_before_transition) and signal in {"lh", "pdg", "progesterone"}:
                    hormone_mean = {"lh": 3.2, "pdg": 1.6, "progesterone": 0.9}[signal]
                value = float(np.exp(rng.normal(np.log(max(hormone_mean, 0.05)), o.log_hormone_sd)))
            elif signal == "symptom_severity":
                value = int(np.clip(round(mean + rng.normal(0, 0.7)), 0, 4))
            else:
                value = 1
            rows.append({
                "participant_id": parameters.participant_id,
                "source_dataset": "synthetic_milestone_1",
                "source_record_id": f"{parameters.participant_id}-{int(state_row['day_in_study']):04d}-{signal}",
                "cycle_id": f"{parameters.participant_id}-C{int(state_row['cycle_id']):03d}",
                "signal_name": signal,
                "value": value,
                "unit": SIGNAL_UNITS[signal],
                "event_time": t,
                "measurement_time": t + pd.Timedelta(hours=7 if signal != "bleeding" else 21),
                "report_time": t + pd.Timedelta(hours=8 if signal != "bleeding" else 22),
                "availability_time": t + pd.Timedelta(hours=8 if signal != "bleeding" else 22),
                "device": "synthetic_wearable_v1" if signal in {"temperature", "resting_heart_rate", "hrv", "sleep_duration", "sleep_efficiency"} else "none",
                "assay": "synthetic_assay_v1" if signal in {"lh", "e3g", "pdg", "estradiol", "progesterone"} else "none",
                "quality_flag": "pass",
                "missingness_reason": "",
                "raw_column": signal,
                "transformation_version": "synthetic-v1",
                "is_observed": True,
            })
    return pd.DataFrame(rows)
