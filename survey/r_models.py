"""Thin Python wrappers around ``survey/fit_models.R`` (lme4).

All numerical fitting happens in R. This module only reshapes the analysis
frame, calls ``Rscript`` and returns the same dictionaries the notebook used
before, so the table code does not need to know about lme4.

Requires R with ``lme4`` and ``jsonlite`` installed.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R_SCRIPT = HERE / "fit_models.R"
CACHE_DIR = HERE / "output" / "model-cache"


def _cache_key(frame: pd.DataFrame, models: list[dict], family: str) -> str:
    """Hash of the data, the model formulas and the R script.

    Any change to the analysis frame, a formula or fit_models.R produces a new
    key, so a stale fit is never reused. The cache stores the exact JSON the R
    script wrote; delete ``survey/output/model-cache/`` to force a refit.
    """
    payload = (
        frame.to_csv(index=False)
        + json.dumps({"family": family, "models": models}, sort_keys=True)
        + R_SCRIPT.read_text()
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _run_r(
    frame: pd.DataFrame,
    models: list[dict],
    family: str,
    use_cache: bool = True,
) -> list[dict]:
    """Write a spec, run lme4 in R (or load a cached fit), return the models."""
    cache_file = CACHE_DIR / f"{_cache_key(frame, models, family)}.json"
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text())

    with tempfile.TemporaryDirectory() as tmp:
        data_path = Path(tmp) / "data.csv"
        spec_path = Path(tmp) / "spec.json"
        out_path = Path(tmp) / "out.json"
        frame.to_csv(data_path, index=False)
        spec_path.write_text(
            json.dumps({"data": str(data_path), "family": family, "models": models})
        )
        result = subprocess.run(
            ["Rscript", str(R_SCRIPT), str(spec_path), str(out_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"lme4 fit failed:\n{result.stdout}\n{result.stderr}")
        parsed = json.loads(out_path.read_text())

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        temporary = cache_file.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(parsed))
        temporary.replace(cache_file)
    return parsed


def _frame_with_groups(data, groups):
    """Safe column names for the grouping factors (R dislikes ``_recordId``)."""
    frame = pd.DataFrame(index=data.index)
    group_map, terms = {}, []
    for i, column in enumerate(groups):
        safe = f"g{i}"
        frame[safe] = data[column].astype(str).to_numpy()
        group_map[safe] = column
        terms.append(f"(1|{safe})")
    return frame, group_map, terms


def fit_lmm(
    data,
    covariates=(),
    groups=("_recordId", "dilemma"),
    y_col="ft_score",
    include_order=True,
):
    """REML LMM (or OLS) fitted by ``lme4::lmer`` / ``lm``.

    Returns the same shape as the old hand-rolled fit: fixed-effect names,
    ``beta``, ``se`` and ``cov`` arrays, the intercept estimate/CI/p, the
    residual SD and the random-effect SDs keyed by the original column names.
    ``include_order=False`` drops the order covariate.
    """
    frame, group_map, terms = _frame_with_groups(data, groups)
    frame.insert(0, "y", data[y_col].to_numpy(float))
    names = ["Intercept"]
    fixed_terms = []
    if include_order:
        frame["order_effect"] = data["order_effect"].to_numpy(float)
        fixed_terms.append("order_effect")
        names.append("order_effect")
    for i, (name, values) in enumerate(covariates):
        safe = f"x{i}"
        frame[safe] = np.asarray(values, float)
        fixed_terms.append(safe)
        names.append(name)

    formula = "y ~ " + (" + ".join(fixed_terms) if fixed_terms else "1")
    formula += "".join(f" + {term}" for term in terms)
    model = _run_r(frame, [{"name": "model", "formula": formula}], "gaussian")[0]

    beta = np.atleast_1d(np.asarray(model["coef_estimate"], float))
    se = np.atleast_1d(np.asarray(model["coef_se"], float))
    cov = np.atleast_2d(np.asarray(model["cov"], float))
    z = NormalDist().inv_cdf(0.975)
    estimate, std_error = float(beta[0]), float(se[0])
    sigma_raw = model.get("sigma") or {}
    sigma_items = sigma_raw.items() if isinstance(sigma_raw, dict) else []
    return {
        "names": names,
        "beta": beta,
        "se": se,
        "cov": cov,
        "estimate": estimate,
        "ci": (estimate - z * std_error, estimate + z * std_error),
        "p": 2 * NormalDist().cdf(-abs(estimate / std_error)),
        "n": int(model["n"]),
        "respondents": int(data["_recordId"].nunique()),
        "sigma": {group_map[key]: float(value) for key, value in sigma_items},
        "sigma_resid": (
            None if model["sigma_resid"] is None else float(model["sigma_resid"])
        ),
        "loglik": float(model["loglik"]),
    }


def fit_bt_glmm(data, groups, include_order=True):
    """Bradley--Terry binomial GLMM fitted by ``lme4::glmer``.

    ``data`` must contain decided comparisons only (``ft_score != 0``).
    Returns the win probability ``expit(beta[0])``, its interval, the Wald
    p-value against no preference, and the random-effect SDs on the log-odds
    scale, keyed by the original column names. ``include_order=False`` drops
    the order covariate.
    """
    frame, group_map, terms = _frame_with_groups(data, groups)
    frame.insert(0, "y", data["win"].to_numpy(float))
    fixed_terms = []
    if include_order:
        frame["order_effect"] = data["order_effect"].to_numpy(float)
        fixed_terms.append("order_effect")
    formula = "y ~ " + (" + ".join(fixed_terms) if fixed_terms else "1")
    formula += "".join(f" + {term}" for term in terms)
    model = _run_r(frame, [{"name": "model", "formula": formula}], "binomial")[0]

    beta = np.atleast_1d(np.asarray(model["coef_estimate"], float))
    se = np.atleast_1d(np.asarray(model["coef_se"], float))
    estimate = float(beta[0])
    std_error = float(se[0])
    sigma_raw = model.get("sigma") or {}
    sigma_items = sigma_raw.items() if isinstance(sigma_raw, dict) else []
    return {
        "estimate": float(1.0 / (1.0 + np.exp(-estimate))),
        "ci": (
            float(1.0 / (1.0 + np.exp(-(estimate - 1.96 * std_error)))),
            float(1.0 / (1.0 + np.exp(-(estimate + 1.96 * std_error)))),
        ),
        "p": 2 * NormalDist().cdf(-abs(estimate / std_error)),
        "sigma": {group_map[key]: float(value) for key, value in sigma_items},
        "beta": beta,
        "se": se,
        "n": int(model["n"]),
        "loglik": float(model["loglik"]),
    }
