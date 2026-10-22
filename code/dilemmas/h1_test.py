"""H1: do the fine-tuned models answer the dilemmas differently from the base?

The analysis, as specified in the analysis plan:

* Each of the four fine-tuned models is compared separately with the base
  model it was fine-tuned from.
* Per dilemma, the judge-classified action counts are compared with a Monte
  Carlo permutation test: 10,000 replicates with a fixed seed, permuting the
  model labels without replacement within the dilemma.
* The effect size is the total variation distance (TVD) between the two
  models' distributions over the dilemma's answer options.
* A model's overall effect size is the mean TVD across all dilemmas, compared
  with the mean TVD of the permutation null; the four overall p-values are
  Benjamini-Hochberg corrected.
* H1 is supported for a model when its BH-adjusted overall p is below .05
  (and overall when at least one of the four models passes).
* The rate of "Other or unclear" judge classifications is reported
  separately, and the whole test is repeated with those runs excluded as a
  sensitivity check.

The module holds the statistics only: it takes the exported runs (one row per
run, holding the judge-classified answer option) and returns pandas frames.
``analyze.py`` supplies the data and writes the tables and the figures.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

SEED = 42  # the fixed seed of the permutation test
REPLICATES = 10_000  # the Monte Carlo replicates of the permutation test
ALPHA = 0.05  # the BH-adjusted p-value H1 is decided on
OTHER_PREFIX = "other"  # the catch-all option starts with this


@dataclass(frozen=True)
class H1Results:
    """Everything one run of the test produces.

    - summary: one row per fine-tuned model (the overall effect, the
      permutation null it is compared with, the p-values and the verdict).
    - per_dilemma: one row per (model, dilemma): the TVD and its p-value.
    - other_rates: the "Other or unclear" rate of every model, from the runs
      as they were handed in (so the sensitivity run still reports the rates
      of the full run set).
    - null_overall: the permutation null distribution of the overall mean TVD,
      one array per fine-tuned model (for the figure).
    """

    summary: pd.DataFrame
    per_dilemma: pd.DataFrame
    other_rates: pd.DataFrame
    null_overall: dict[str, np.ndarray]


def other_option_index(options: list[str]) -> int | None:
    """The index of the "Other or unclear" catch-all option, if there is one."""
    for index, option in enumerate(options):
        if option.strip().lower().startswith(OTHER_PREFIX):
            return index
    return None


def total_variation_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """The TVD between two sets of distributions (one row per set).

    The TVD is half the sum of the absolute differences: 0 when the
    distributions are the same, 1 when they share no mass.
    """
    return 0.5 * np.abs(left - right).sum(axis=-1)


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """The Benjamini-Hochberg adjusted p-values (step-up, capped at 1)."""
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    order = np.argsort(p)
    adjusted = p[order] * n / np.arange(1, n + 1)
    # The step-up step: the adjusted values are monotone in the ranked p.
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def _model_seed(seed: int, model: str) -> list[int]:
    """A reproducible seed per model.

    The model name is hashed, so a model's permutation does not depend on
    the order the models are tested in.
    """
    digest = hashlib.sha256(model.encode("utf-8")).digest()
    return [seed, int.from_bytes(digest[:4], "big")]


def choice_counts(
    runs: pd.DataFrame, options: dict[str, list[str]]
) -> dict[tuple[str, str], np.ndarray]:
    """The judge-classified action counts, keyed by (model, dilemma).

    ``runs`` holds one row per run with a ``Dilemma`` (the situation id), a
    ``Model`` (the label) and an ``Index`` (the 1-based position of the
    judged answer in the dilemma's answer options). Each count is over the
    dilemma's own options, so the counts of two models can be compared
    directly.
    """
    counts: dict[tuple[str, str], np.ndarray] = {}
    for (model, dilemma), group in runs.groupby(["Model", "Dilemma"]):
        option_list = options[dilemma]
        values = np.zeros(len(option_list), dtype=int)
        for index, count in group["Index"].value_counts().items():
            position = int(index) - 1
            if 0 <= position < len(option_list):
                values[position] = int(count)
        counts[(model, dilemma)] = values
    return counts


def other_rates(runs: pd.DataFrame, options: dict[str, list[str]]) -> pd.DataFrame:
    """The share of runs the judge classified as "Other or unclear".

    One row per model: how many of its runs the judge could not place on one
    of the dilemma's substantive options. Reported separately from the test,
    because a catch-all classification carries no decision.
    """
    rows = []
    for model, group in runs.groupby("Model"):
        unclear = 0
        for dilemma, dilemma_runs in group.groupby("Dilemma"):
            index = other_option_index(options[dilemma])
            if index is not None:
                unclear += int((dilemma_runs["Index"] == index + 1).sum())
        total = len(group)
        rows.append(
            {
                "Model": model,
                "Runs": total,
                "Other or unclear": unclear,
                "Rate": unclear / total if total else np.nan,
            }
        )
    return pd.DataFrame(rows)


def drop_unclear(runs: pd.DataFrame, options: dict[str, list[str]]) -> pd.DataFrame:
    """The runs with the "Other or unclear" classifications removed.

    The sensitivity check of the test: the same permutation test on the runs
    the judge could place on a substantive option only.
    """
    kept = []
    for dilemma, group in runs.groupby("Dilemma"):
        index = other_option_index(options[dilemma])
        kept.append(group if index is None else group[group["Index"] != index + 1])
    return pd.concat(kept) if kept else runs.iloc[0:0]


def _observed_tvd(left: np.ndarray, right: np.ndarray) -> float:
    """The TVD between the two models' observed distributions."""
    n_left, n_right = left.sum(), right.sum()
    if not n_left or not n_right:
        return np.nan
    return float(total_variation_distance(left / n_left, right / n_right))


def _permutation_null(
    rng: np.random.Generator, left: np.ndarray, right: np.ndarray, replicates: int
) -> np.ndarray:
    """The null TVD distribution when the model labels are permuted.

    The labels are permuted within the dilemma and without replacement, so
    each group keeps its number of runs: the null is the distribution of the
    TVD under the two models' runs being exchangeable.
    """
    n_options = left.size
    totals = left + right
    n_total = int(totals.sum())
    n_left = int(left.sum())
    n_right = n_total - n_left
    if not n_left or not n_right:
        # Nothing to permute: every labelling gives the same (empty) contrast.
        return np.zeros(replicates)
    # The dilemma's runs as option indices, so the labels can be shuffled.
    labels = np.repeat(np.arange(n_options), totals)
    # A random subset of the runs per replicate: the labels are permuted
    # without replacement and the two group sizes are fixed.
    order = np.argsort(rng.random((replicates, n_total)), axis=1)
    chosen = order[:, :n_left]
    shuffled_left = np.stack(
        [(labels[chosen] == option).sum(axis=1) for option in range(n_options)],
        axis=1,
    )
    shuffled_right = totals - shuffled_left
    return total_variation_distance(shuffled_left / n_left, shuffled_right / n_right)


def run_h1(
    runs: pd.DataFrame,
    options: dict[str, list[str]],
    base_label: str,
    model_order: list[str] | None = None,
    replicates: int = REPLICATES,
    seed: int = SEED,
    alpha: float = ALPHA,
) -> H1Results:
    """The H1 permutation test of every fine-tuned model against the base.

    ``runs`` and ``options`` are the exported runs and the dilemmas' answer
    options (see ``choice_counts``). ``base_label`` is the label of the base
    model in ``runs``; every other model in ``runs`` is a fine-tuned model
    and is tested against it separately.

    Returns the summary (one row per fine-tuned model), the per-dilemma
    results, the "Other or unclear" rates and the permutation null
    distributions of the overall effect.
    """
    if runs.empty:
        # Nothing to test yet: the notebook still runs on a fresh checkout.
        return H1Results(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {})
    labels = list(runs["Model"].unique())
    if base_label not in labels:
        raise ValueError(f"the base model {base_label!r} is not in the runs")
    fine_tuned = [label for label in labels if label != base_label]
    if model_order:
        fine_tuned = [label for label in model_order if label in fine_tuned]
    counts = choice_counts(runs, options)
    dilemmas = sorted({dilemma for _, dilemma in counts})

    summary_rows: list[dict] = []
    per_dilemma_rows: list[dict] = []
    null_overall: dict[str, np.ndarray] = {}
    for model in fine_tuned:
        rng = np.random.default_rng(_model_seed(seed, model))
        # The dilemmas both models answered: a dilemma one of them missed
        # cannot be compared.
        shared = [
            dilemma
            for dilemma in dilemmas
            if (model, dilemma) in counts and (base_label, dilemma) in counts
        ]
        observed = np.zeros(len(shared))
        null = np.zeros((replicates, len(shared)))
        for position, dilemma in enumerate(shared):
            left = counts[(model, dilemma)]
            right = counts[(base_label, dilemma)]
            observed[position] = _observed_tvd(left, right)
            null[:, position] = _permutation_null(rng, left, right, replicates)
        # The overall effect: the mean TVD across the dilemmas, against the
        # permutation null of that same mean.
        null_mean = null.mean(axis=1) if len(shared) else np.full(replicates, np.nan)
        observed_mean = float(observed.mean()) if len(shared) else np.nan
        p_overall = (
            (1 + int((null_mean >= observed_mean).sum())) / (replicates + 1)
            if len(shared)
            else np.nan
        )
        p_dilemma = (1 + (null >= observed).sum(axis=0)) / (replicates + 1)
        null_overall[model] = null_mean
        for position, dilemma in enumerate(shared):
            per_dilemma_rows.append(
                {
                    "Model": model,
                    "Dilemma": dilemma,
                    "Runs": int(
                        counts[(model, dilemma)].sum()
                        + counts[(base_label, dilemma)].sum()
                    ),
                    "TVD": float(observed[position]),
                    "p": float(p_dilemma[position]),
                }
            )
        summary_rows.append(
            {
                "Model": model,
                "Dilemmas": len(shared),
                "Runs": int(
                    sum(
                        counts[(model, dilemma)].sum()
                        + counts[(base_label, dilemma)].sum()
                        for dilemma in shared
                    )
                ),
                "Mean TVD": observed_mean,
                "Null mean TVD": float(null_mean.mean()),
                "p (overall)": p_overall,
            }
        )

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["p (BH)"] = benjamini_hochberg(
            summary["p (overall)"].to_numpy(dtype=float)
        )
        summary["Supported"] = summary["p (BH)"] < alpha
    return H1Results(
        summary=summary,
        per_dilemma=pd.DataFrame(per_dilemma_rows),
        other_rates=other_rates(runs, options),
        null_overall=null_overall,
    )


def null_calibration(trials: int = 100, replicates: int = 1000, seed: int = 0) -> float:
    """The rejection rate of the test when the null holds.

    Two models with the same answer distribution are simulated on the same
    dilemmas, so the test should reject at about the alpha rate. This is the
    check that the permutation test is calibrated.
    """
    rng = np.random.default_rng(seed)
    options = {f"d{dilemma}": ["a", "b", "c"] for dilemma in range(30)}
    rejections = 0
    for trial in range(trials):
        rows = []
        for dilemma in range(30):
            for model in ("base", "fine-tuned"):
                for index in rng.choice(3, size=5, p=[0.4, 0.4, 0.2]) + 1:
                    rows.append(
                        {
                            "Dilemma": f"d{dilemma}",
                            "Model": model,
                            "Index": int(index),
                        }
                    )
        results = run_h1(
            pd.DataFrame(rows),
            options,
            base_label="base",
            replicates=replicates,
            seed=trial,
        )
        rejections += int(bool(results.summary["p (BH)"].iloc[0] < ALPHA))
    return rejections / trials


if __name__ == "__main__":
    rate = null_calibration()
    print(
        f"null calibration: {rate:.1%} of the null trials were rejected (alpha {ALPHA})"
    )
