"""Power analysis for the pairwise preference survey (report: survey.tex).

Design
------
The survey draws one pair per sample index, keeps the 18 dilemmas whose
response distributions differ (H1, p < .05), and takes the most divergent
pairs from each. Each respondent sees 5 comparisons - the outer
``BlockRandomizer`` draws 5 of the 8 profiles and each profile's inner
randomiser draws 1 pair - so 500 respondents give 2,500 ratings. See
``survey_definition.py`` and the "Survey design" section of
``code/dilemmas/analyze.py``; the pair counts are read from the survey subset
so the tables follow the subset that is actually built.

Model
-----
A rating is the preference for the fine-tune over the base model on one pair.
Write the win probability of pair ``i`` in dilemma ``d`` rated by respondent
``j`` as

    y_ijd = 1 with probability p_id + a_j,    p_id = mu + u_d + v_i

with ``u_d ~ (0, tau_d^2)`` the between-dilemma spread, ``v_i ~ (0,
tau_p^2)`` the between-pair spread within a dilemma, ``a_j ~ (0, omega^2)``
the respondent's leniency/position bias, and a Bernoulli residual. Every
latent probability is near 0.5, so a single binary rating has variance
pinned at 1/4 whatever generates it; the layers only split that budget:

    Var(mean) = f_d/4 / D + f_p/4 / K + f_w/4 / J + (1 - f_d - f_p - f_w)/4 / N

for ``D`` dilemmas, ``K`` pairs, ``J`` respondents and ``N = J*s = K*m``
ratings. The 5-point Likert outcome uses the same form with total variance
1.25^2. The split (``f_d``, ``f_p``, ``f_w``) is the only assumption; the
dilemma share is varied in the table because it turns out to dominate.

Three estimands, one per sampling layer - the chapter has to say which one
its claim is about:

* ``pairs``      these 18 situations and these pairs (only respondents and
                 ratings are sampled);
* ``situations`` these 18 situations, but the pairs are a sample of the
                 response pairs those situations could produce;
* ``general``    new situations, i.e. the claim extends beyond the dilemmas
                 that were written.

Coverage vs replication
-----------------------
At a fixed budget ``N``, ``K`` and ``m = N/K`` trade off. Only the pair term
depends on the split; the rating term is invariant in ``N``. More pairs
therefore never hurt the aggregate estimate, while ratings per pair only buy
per-pair precision (``SE ~ 0.5 / sqrt(m)``). The script prints the frontier
and writes the report tables and figure:

    uv run survey/power_analysis.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from latex_tables import FIGURES_DIR, write_tex  # noqa: E402

# --- Design constants -------------------------------------------------------
# N_PROFILES_SHOWN comes from survey/survey_definition.py; the pair counts
# come from the survey subset written by code/dilemmas/analyze.py.
SUBSET = (
    ROOT
    / "code/dilemmas/output/comparisons/2026-10-23_dc41eb9e/survey"
    / "modal_response-overall/comparisons.json"
)
N_RESPONDENTS = 500
RATINGS_PER_RESPONDENT = 5
N_PROFILES_SHOWN = 5
N_RATINGS = N_RESPONDENTS * RATINGS_PER_RESPONDENT
ALPHA = 0.05
POWER = 0.80
Z_ALPHA = NormalDist().inv_cdf(1 - ALPHA / 2)
Z_POWER = NormalDist().inv_cdf(POWER)

# Total variance of one rating, and the central split of it across layers.
# A binary rating is pinned at 1/4; the 5-point scale is the pilot SD 1.25.
#   situations: the public backs a different model in different dilemmas
#   pairs:      responses within a dilemma differ in how well they land
#   respondent: some respondents lean Agent 1 or prefer longer answers
#   rating:     binomial/Likert noise plus the respondent x pair interaction
BINARY_VARIANCE = 0.25
LIKERT_VARIANCE = 1.25**2
SPLIT = {
    "binary": {"situations": 0.08, "pairs": 0.12, "respondent": 0.04},
    "likert": {"situations": 0.16, "pairs": 0.24, "respondent": 0.05},
}
# The between-dilemma spread is the assumption that matters, so the table
# runs it over three levels. The share is the same in both outcomes, which
# gives tau_d = 0.10/0.14/0.20 win-share and 0.25/0.35/0.50 Likert points.
SITUATION_SHARES = (0.04, 0.08, 0.16)
TARGETS = {"binary": 0.05, "likert": 0.25}


def phi(x: float) -> float:
    return NormalDist().cdf(x)


def variance_by_layer(
    outcome, layer, dilemmas, pairs, respondents, ratings, situation_share=None
):
    """Variance of the mean preference with the sampling layers above
    ``layer`` dropped (``pairs`` < ``situations`` < ``general``)."""
    split = dict(SPLIT[outcome])
    if situation_share is not None:
        split["situations"] = situation_share
    total = BINARY_VARIANCE if outcome == "binary" else LIKERT_VARIANCE
    terms = {
        "respondent": split["respondent"] / respondents,
        "rating": (1 - sum(split.values())) / ratings,
    }
    if layer in ("situations", "general"):
        terms["pairs"] = split["pairs"] / pairs
    if layer == "general":
        terms["situations"] = split["situations"] / dilemmas
    return total * sum(terms.values())


def mde(se, power=POWER):
    """Minimum detectable effect at ``power`` (two-sided alpha)."""
    return (Z_ALPHA + NormalDist().inv_cdf(power)) * se


def power_for(se, effect):
    """Power to detect ``effect`` given the SE of the estimate."""
    return phi(effect / se - Z_ALPHA)


def load_design(subset=SUBSET):
    """Pairs per profile and dilemma in the survey subset."""
    manifest = json.loads(Path(subset).read_text())
    counts: dict[str, dict[str, int]] = {}
    for record in manifest["comparisons"]:
        counts.setdefault(record["profile_id"], {})
        counts[record["profile_id"]][record["dilemma_id"]] = (
            counts[record["profile_id"]].get(record["dilemma_id"], 0) + 1
        )
    return counts


def design_sizes(counts):
    """Dilemmas and pairs in the subset."""
    return sum(len(v) for v in counts.values()), sum(
        sum(v.values()) for v in counts.values()
    )


def ratings_per_pair(counts):
    """Expected ratings per pair under the two-stage randomiser.

    Each respondent draws 5 of the 8 profiles, then 1 pair from the chosen
    profile, so a pair in a profile is rated ``500 * (5/8) / n_pairs`` times.
    """
    share = N_PROFILES_SHOWN / len(counts)
    rows = []
    for profile, dilemmas in sorted(counts.items()):
        n_pairs = sum(dilemmas.values())
        rows.append(
            {
                "profile": profile,
                "dilemmas": len(dilemmas),
                "pairs": n_pairs,
                "ratings_per_pair": N_RESPONDENTS * share / n_pairs,
            }
        )
    return rows


def estimand_table():
    """MDE and power for the three estimands and three dilemma spreads."""
    counts = load_design()
    dilemmas, pairs = design_sizes(counts)
    rows = []
    for layer, label, tex in (
        ("pairs", "these pairs and situations", "these pairs and situations"),
        ("situations", "these situations, new pairs", "these situations, new pairs"),
    ):
        row = {"label": label, "tex": tex}
        for outcome in ("binary", "likert"):
            se = math.sqrt(
                variance_by_layer(
                    outcome, layer, dilemmas, pairs, N_RESPONDENTS, N_RATINGS
                )
            )
            row[outcome] = (se, mde(se), power_for(se, TARGETS[outcome]))
        rows.append(row)
    for share in SITUATION_SHARES:
        tau = math.sqrt(share * BINARY_VARIANCE)
        row = {
            "label": f"new situations, spread {share:.2f}",
            "tex": f"new situations, $\\tau_d={tau:.2f}$",
        }
        for outcome in ("binary", "likert"):
            se = math.sqrt(
                variance_by_layer(
                    outcome, "general", dilemmas, pairs, N_RESPONDENTS, N_RATINGS, share
                )
            )
            row[outcome] = (se, mde(se), power_for(se, TARGETS[outcome]))
        rows.append(row)
    return rows


def coverage_frontier():
    """Fixed 2,500 ratings spread over more or fewer pairs.

    The pool holds 50 index-matched pairs per dilemma, so 50 per dilemma
    (900 pairs) is full coverage of the pool the selection draws from.
    """
    dilemmas, pairs = design_sizes(load_design())
    planned = round(pairs / dilemmas)
    rows = []
    for per_dilemma in (planned, 20, 35, 50):
        pairs = dilemmas * per_dilemma
        m = N_RATINGS / pairs
        se_scoped = math.sqrt(
            variance_by_layer(
                "binary", "situations", dilemmas, pairs, N_RESPONDENTS, N_RATINGS
            )
        )
        se_general = math.sqrt(
            variance_by_layer(
                "binary", "general", dilemmas, pairs, N_RESPONDENTS, N_RATINGS
            )
        )
        rows.append(
            {
                "per_dilemma": per_dilemma,
                "pairs": pairs,
                "ratings_per_pair": m,
                "mde_scoped": mde(se_scoped),
                "mde_general": mde(se_general),
                "pair_ci": 1.96 * 0.5 / math.sqrt(m),
            }
        )
    return rows


def required_respondents(target=0.05, layer="situations"):
    """Respondents needed for 80% power at ``target`` under ``layer``.

    Five ratings per respondent makes the respondent and rating terms
    ``(f_w + f_e/5) / J``; the pairs (and dilemmas) are fixed by the subset.
    """
    counts = load_design()
    dilemmas, pairs = design_sizes(counts)
    target_var = (target / (Z_ALPHA + Z_POWER)) ** 2
    fixed = 0.25 * (SPLIT["binary"]["pairs"] / pairs)
    if layer == "general":
        fixed += 0.25 * SPLIT["binary"]["situations"] / dilemmas
    per_respondent = 0.25 * (
        SPLIT["binary"]["respondent"]
        + (1 - sum(SPLIT["binary"].values())) / RATINGS_PER_RESPONDENT
    )
    remaining = target_var - fixed
    return None if remaining <= 0 else math.ceil(per_respondent / remaining)


def required_dilemmas(target=0.05, pairs_per_dilemma=10):
    """Dilemmas needed to generalise at 80% power, at a fixed rating budget.

    Pairs scale with dilemmas, so the pair term shrinks as dilemmas are added.
    """
    target_var = (target / (Z_ALPHA + Z_POWER)) ** 2
    fixed = 0.25 * (
        SPLIT["binary"]["respondent"] / N_RESPONDENTS
        + (1 - sum(SPLIT["binary"].values())) / N_RATINGS
    )
    per_dilemma = 0.25 * (
        SPLIT["binary"]["situations"] + SPLIT["binary"]["pairs"] / pairs_per_dilemma
    )
    return math.ceil(per_dilemma / (target_var - fixed))


def simulate(layer="situations", effect=0.05, n_sims=2000, seed=0):
    """Monte Carlo check of the two-stage randomiser.

    Redraws the population (dilemma and pair effects) for every survey, which
    is the superpopulation view, and drops the layers ``layer`` fixes. Returns
    the empirical SD of the mean and the rejection rate at ``effect``.
    """
    counts = load_design()
    profiles = list(counts)
    dilemma_ids = sorted({d for v in counts.values() for d in v})
    dilemma_of = {d: i for i, d in enumerate(dilemma_ids)}
    pair_dilemma = np.repeat(
        [dilemma_of[dilemma] for profile in profiles for dilemma in counts[profile]],
        [counts[p][d] for p in profiles for d in counts[p]],
    )
    n_pairs = {p: sum(counts[p].values()) for p in profiles}
    starts, offset = {}, 0
    for profile in profiles:
        starts[profile], offset = offset, offset + n_pairs[profile]

    rng = np.random.default_rng(seed)
    variance = BINARY_VARIANCE
    sd_situation = (
        math.sqrt(variance * SPLIT["binary"]["situations"])
        if layer == "general"
        else 0.0
    )
    sd_pair = (
        math.sqrt(variance * SPLIT["binary"]["pairs"]) if layer != "pairs" else 0.0
    )
    sd_respondent = math.sqrt(variance * SPLIT["binary"]["respondent"])
    sd_rating = math.sqrt(variance * (1 - sum(SPLIT["binary"].values())))

    estimates = np.empty(n_sims)
    for sim in range(n_sims):
        # A uniform 5-of-8 draw per respondent: rank the profiles by a random
        # key and take the first five.
        chosen = np.argsort(rng.random((N_RESPONDENTS, len(profiles))), axis=1)[
            :, :N_PROFILES_SHOWN
        ]
        slot_pair = np.empty((N_RESPONDENTS, N_PROFILES_SHOWN), dtype=int)
        for index, profile in enumerate(profiles):
            mask = chosen == index
            slot_pair[mask] = starts[profile] + rng.integers(
                0, n_pairs[profile], mask.sum()
            )
        situation_effect = rng.normal(0, sd_situation, size=len(dilemma_ids))
        pair_effect = situation_effect[pair_dilemma] + rng.normal(
            0, sd_pair, len(pair_dilemma)
        )
        bias = rng.normal(0, sd_respondent, size=N_RESPONDENTS)
        y = (
            effect
            + pair_effect[slot_pair]
            + bias[:, None]
            + rng.normal(0, sd_rating, (N_RESPONDENTS, N_PROFILES_SHOWN))
        )
        estimates[sim] = y.mean()
    sd = estimates.std(ddof=1)
    return sd, float(np.mean(np.abs(estimates) > Z_ALPHA * sd))


# --- Report outputs ---------------------------------------------------------
def write_estimand_table(rows, name="survey-power-estimands.tex"):
    """Table 1: the three estimands and the dilemma-spread sensitivity."""
    body = "\n".join(
        f"{r['tex']} & {100 * r['binary'][1]:.1f} & "
        f"{100 * r['binary'][2]:.0f} & {r['likert'][1]:.2f} & "
        f"{100 * r['likert'][2]:.0f} \\\\"
        for r in rows
    )
    tex = (
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{lrrrr}\n"
        "\\toprule\n"
        " & \\multicolumn{2}{c}{Binary preference} & \\multicolumn{2}{c}{Likert (points)} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
        "Sampling layers considered & MDE (pp) & Power at 5pp & "
        "MDE (pts) & Power at 0.25 \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "}%"
    )
    return write_tex(tex, name)


def write_coverage_table(rows, name="survey-power-coverage.tex"):
    """Table 2: coverage vs replication at a fixed 2,500 ratings."""
    body = "\n".join(
        f"{r['per_dilemma']} & {r['pairs']} & {r['ratings_per_pair']:.1f} & "
        f"{100 * r['mde_scoped']:.1f} & {100 * r['mde_general']:.1f} & "
        f"$\\pm${100 * r['pair_ci']:.0f} \\\\"
        for r in rows
    )
    tex = (
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{rrrrrr}\n"
        "\\toprule\n"
        "Pairs per dilemma & Pairs & Ratings per pair & "
        "MDE, situations fixed (pp) & MDE, new situations (pp) & "
        "One pair's 95\\% CI (pp) \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "}%"
    )
    return write_tex(tex, name)


def plot_frontier(rows, path=None):
    """MDE and pair-level precision against the number of pairs."""
    path = path or FIGURES_DIR / "survey-power-coverage.png"
    grid = np.linspace(rows[0]["pairs"], rows[-1]["pairs"], 200)

    def curve(layer):
        return [
            100
            * mde(
                math.sqrt(
                    variance_by_layer(
                        "binary", layer, 18, pairs, N_RESPONDENTS, N_RATINGS
                    )
                )
            )
            for pairs in grid
        ]

    plt.rcParams.update({"font.size": 9})
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].plot(
        grid, curve("situations"), color="tab:blue", label="Dilemmas fixed (new pairs)"
    )
    axes[0].plot(
        grid, curve("general"), color="tab:orange", ls="--", label="New dilemmas"
    )
    axes[0].axhline(5, color="grey", lw=1, ls=":")
    axes[0].plot(
        [r["pairs"] for r in rows],
        [100 * r["mde_scoped"] for r in rows],
        "o",
        color="tab:blue",
        ms=4,
    )
    axes[0].plot(
        [r["pairs"] for r in rows],
        [100 * r["mde_general"] for r in rows],
        "o",
        color="tab:orange",
        ms=4,
    )
    axes[0].set_xlabel("Pairs in the survey")
    axes[0].set_ylabel("MDE (percentage points)")
    axes[0].set_ylim(0, 12)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Aggregate test (80% power)")

    axes[1].plot(grid, 100 * 1.96 * 0.5 / np.sqrt(N_RATINGS / grid), color="tab:red")
    axes[1].plot(
        [r["pairs"] for r in rows],
        [100 * r["pair_ci"] for r in rows],
        "o",
        color="tab:red",
        ms=4,
    )
    axes[1].set_xlabel("Pairs in the survey")
    axes[1].set_ylabel("95% CI half-width (pp)")
    axes[1].set_title("One pair's preference")

    for axis in axes:
        axis.grid(alpha=0.3, lw=0.5)
        axis.set_xlim(grid[0] - 15, grid[-1] + 30)
        top = axis.secondary_xaxis(
            "top",
            functions=(
                lambda k: N_RATINGS / np.maximum(k, 1),
                lambda r: N_RATINGS / np.maximum(r, 1),
            ),
        )
        top.set_xticks([10, 8, 6, 4])
        top.set_xlabel("Ratings per pair", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    return path


def respondent_curve():
    """MDE against the number of respondents (pairs and dilemmas fixed)."""
    counts = load_design()
    dilemmas, pairs = design_sizes(counts)
    rows = []
    for respondents in (250, 500, 750, 1000, 2000):
        ratings = respondents * RATINGS_PER_RESPONDENT
        row = {"respondents": respondents, "ratings": ratings}
        for layer in ("situations", "general"):
            se = math.sqrt(
                variance_by_layer(
                    "binary", layer, dilemmas, pairs, respondents, ratings
                )
            )
            row[layer] = mde(se)
        rows.append(row)
    return rows


def main():
    counts = load_design()
    dilemmas, pairs = design_sizes(counts)
    print("Planned design")
    print(
        f"  dilemmas={dilemmas}  pairs={pairs}  respondents={N_RESPONDENTS}  "
        f"ratings={N_RATINGS}  ratings per pair={N_RATINGS / pairs:.1f}"
    )
    print(
        "  per profile: "
        + ", ".join(
            f"{r['profile']}={r['ratings_per_pair']:.1f}"
            for r in ratings_per_pair(counts)
        )
    )
    print()

    print("MDE by estimand and between-dilemma spread")
    for row in estimand_table():
        print(
            f"  {row['label']:<34} binary MDE={100 * row['binary'][1]:5.1f}pp "
            f"(power {100 * row['binary'][2]:3.0f}%)   "
            f"likert MDE={row['likert'][1]:.2f}pts "
            f"(power {100 * row['likert'][2]:3.0f}%)"
        )
    print()

    print("Coverage vs replication (fixed 2,500 ratings, 18 dilemmas)")
    for row in coverage_frontier():
        print(
            f"  {row['per_dilemma']:>3}/dilemma  pairs={row['pairs']:>4}  "
            f"m={row['ratings_per_pair']:>5.1f}  "
            f"MDE scoped={100 * row['mde_scoped']:.1f}pp  "
            f"general={100 * row['mde_general']:.1f}pp  "
            f"pair CI=+/-{100 * row['pair_ci']:.0f}pp"
        )
    print()

    print("What it takes for 80% power at 5pp")
    print(f"  respondents, situations fixed : {required_respondents()}")
    print(f"  dilemmas, generalising        : {required_dilemmas()}")
    print()

    print("MDE against the number of respondents")
    for row in respondent_curve():
        print(
            f"  {row['respondents']:>5} respondents ({row['ratings']:>6} ratings)  "
            f"scoped={100 * row['situations']:.1f}pp  "
            f"general={100 * row['general']:.1f}pp"
        )
    print()

    print("Monte Carlo check of the two-stage randomiser (2,000 surveys)")
    for layer in ("pairs", "situations", "general"):
        sd, rejection = simulate(layer)
        analytic = math.sqrt(
            variance_by_layer(
                "binary", layer, dilemmas, pairs, N_RESPONDENTS, N_RATINGS
            )
        )
        print(
            f"  {layer:<10} analytic SD={analytic:.4f}  simulated SD={sd:.4f}  "
            f"rejection rate={rejection:.1%}"
        )
    print()

    print(f"wrote {write_estimand_table(estimand_table())}")
    print(f"wrote {write_coverage_table(coverage_frontier())}")
    print(f"wrote {plot_frontier(coverage_frontier())}")


if __name__ == "__main__":
    main()
