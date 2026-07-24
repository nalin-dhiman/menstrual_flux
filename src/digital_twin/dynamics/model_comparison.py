from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import gamma, lognorm, norm

from .first_passage import fit_constant_first_passage


def _information_row(
    source: str,
    model: str,
    log_likelihood: float,
    parameters: int,
    observations: int,
    details: dict[str, float],
) -> dict[str, float | str | int]:
    return {
        "source": source,
        "model": model,
        "log_likelihood": float(log_likelihood),
        "parameters": int(parameters),
        "observations": int(observations),
        "aic": float(2 * parameters - 2 * log_likelihood),
        "bic": float(
            np.log(max(observations, 1)) * parameters - 2 * log_likelihood
        ),
        **details,
    }


def _grouped_loglikelihood(
    theta: np.ndarray,
    groups: list[np.ndarray],
    *,
    ar1: bool,
) -> float:
    mu = theta[0]
    tau = np.exp(theta[1])
    sigma = np.exp(theta[2])
    rho = np.tanh(theta[3]) if ar1 else 0.0
    total = 0.0
    for y in groups:
        n = len(y)
        distance = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
        residual_covariance = sigma**2 * (rho**distance if ar1 else np.eye(n))
        covariance = tau**2 * np.ones((n, n)) + residual_covariance
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            return np.inf
        delta = y - mu
        try:
            quadratic = float(delta @ np.linalg.solve(covariance, delta))
        except np.linalg.LinAlgError:
            return np.inf
        # The Gaussian model is on log duration. Subtract log(x)=y for the
        # Jacobian so information criteria are comparable on the day scale.
        total += (
            -0.5 * (n * np.log(2 * np.pi) + logdet + quadratic)
            - float(np.sum(y))
        )
    return -total


def compare_duration_models(
    cycles: pd.DataFrame, *, source: str
) -> pd.DataFrame:
    """Compare renewal, drift-diffusion, and persistent-duration hypotheses.

    The comparison concerns the distribution and dependence of observed cycle
    durations. It does not establish that the latent coordinate is biological.
    """

    frame = cycles.copy()
    eligible = frame["eligible_for_primary_evaluation"]
    if eligible.dtype != bool:
        eligible = eligible.astype(str).str.lower().eq("true")
    frame = frame.loc[eligible].copy()
    frame["cycle_length_days"] = pd.to_numeric(
        frame["cycle_length_days"], errors="coerce"
    )
    frame = frame.loc[frame["cycle_length_days"].gt(0)].sort_values(
        ["participant_id", "cycle_start_day"]
    )
    x = frame["cycle_length_days"].to_numpy(dtype=float)
    n = len(x)
    rows: list[dict[str, float | str | int]] = []

    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=0))
    rows.append(
        _information_row(
            source,
            "gaussian_renewal",
            float(np.sum(norm.logpdf(x, loc=mean, scale=max(sd, 1e-8)))),
            2,
            n,
            {"location": mean, "scale": sd},
        )
    )
    log_shape, _, log_scale = lognorm.fit(x, floc=0)
    rows.append(
        _information_row(
            source,
            "lognormal_renewal",
            float(
                np.sum(
                    lognorm.logpdf(
                        x, s=log_shape, loc=0, scale=log_scale
                    )
                )
            ),
            2,
            n,
            {"location": float(log_scale), "scale": float(log_shape)},
        )
    )
    gamma_shape, _, gamma_scale = gamma.fit(x, floc=0)
    rows.append(
        _information_row(
            source,
            "gamma_renewal",
            float(
                np.sum(
                    gamma.logpdf(
                        x, a=gamma_shape, loc=0, scale=gamma_scale
                    )
                )
            ),
            2,
            n,
            {"location": float(gamma_scale), "scale": float(gamma_shape)},
        )
    )
    passage = fit_constant_first_passage(x)
    rows.append(
        _information_row(
            source,
            "constant_drift_diffusion_first_passage",
            passage["log_likelihood"],
            2,
            n,
            {
                "location": passage["drift"],
                "scale": passage["diffusion"],
            },
        )
    )

    groups = [
        np.log(group["cycle_length_days"].to_numpy(dtype=float))
        for _, group in frame.groupby("participant_id")
    ]
    initial = np.array(
        [
            np.log(mean),
            np.log(max(np.std([g.mean() for g in groups]), 0.05)),
            np.log(max(np.std(np.concatenate(groups)), 0.05)),
        ]
    )
    random_intercept = minimize(
        lambda value: _grouped_loglikelihood(value, groups, ar1=False),
        initial,
        method="L-BFGS-B",
        bounds=(
            (np.log(5.0), np.log(100.0)),
            (np.log(1e-4), np.log(1.0)),
            (np.log(1e-4), np.log(1.0)),
        ),
    )
    rows.append(
        _information_row(
            source,
            "hierarchical_lognormal_random_intercept",
            -float(random_intercept.fun),
            3,
            n,
            {
                "location": float(np.exp(random_intercept.x[0])),
                "scale": float(np.exp(random_intercept.x[2])),
                "between_person_sd": float(np.exp(random_intercept.x[1])),
            },
        )
    )
    ar1_initial = np.r_[random_intercept.x, 0.0]
    hierarchical_ar1 = minimize(
        lambda value: _grouped_loglikelihood(value, groups, ar1=True),
        ar1_initial,
        method="L-BFGS-B",
        bounds=(
            (np.log(5.0), np.log(100.0)),
            (np.log(1e-4), np.log(1.0)),
            (np.log(1e-4), np.log(1.0)),
            (-2.0, 2.0),
        ),
    )
    rows.append(
        _information_row(
            source,
            "hierarchical_ar1_cycle_shock",
            -float(hierarchical_ar1.fun),
            4,
            n,
            {
                "location": float(np.exp(hierarchical_ar1.x[0])),
                "scale": float(np.exp(hierarchical_ar1.x[2])),
                "between_person_sd": float(np.exp(hierarchical_ar1.x[1])),
                "ar1": float(np.tanh(hierarchical_ar1.x[3])),
            },
        )
    )
    result = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)
    result["delta_aic"] = result["aic"] - result["aic"].min()
    result["akaike_weight"] = np.exp(-0.5 * result["delta_aic"])
    result["akaike_weight"] /= result["akaike_weight"].sum()
    return result
