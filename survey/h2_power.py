"""Minimal power analysis for H2 (report: survey.tex).

H2 has two outcomes:

* LMM on the -2..+2 preference rating, with a fixed intercept and random
  intercepts for respondent and dilemma. H2 is supported if the mean
  preference is positive and its 95% interval excludes zero.
* Bradley-Terry win score (win = 1, tie = 0.5, loss = 0), with a clustered
  interval. H2 is supported if the interval excludes 0.50.

This script answers one question: with the current design (500 respondents x
5 comparisons = 2,500 ratings, 18 dilemmas, 306 pairs), what average
preference can H2 detect at 80% power? It reuses the variance model in
``power_analysis.py`` and keeps only the four rows that matter:

* the scoped claim - these 18 dilemmas, new pairs;
* the general claim - new dilemmas, at low / medium / high between-dilemma
  spread.

The between-dilemma spread is the assumption that drives everything. It is
shown both in Likert points and in win-rate percentage points, because the
two H2 outcomes use different scales.

Run with::

    uv run survey/h2_power.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ``power_analysis`` is the single source of the variance model. It lives in
# this directory, so put this directory on the path before importing it.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import power_analysis as pa

# The rows of the simple table. ``share`` is the between-dilemma share of the
# total variance in ``power_analysis.SITUATION_SHARES``; ``None`` means the
# 18 dilemmas are fixed (the scoped claim).
ROWS = (
    ("These 18 dilemmas, new pairs", "situations", None),
    ("New dilemmas, low spread", "general", 0.04),
    ("New dilemmas, medium spread", "general", 0.08),
    ("New dilemmas, high spread", "general", 0.16),
)
COLOURS = ("0.35", "tab:blue", "tab:orange", "tab:red")


def h2_rows():
    """MDE and power for both H2 outcomes in each scope."""
    dilemmas, pairs = pa.design_sizes(pa.load_design())
    rows = []
    for label, layer, share in ROWS:
        row = {"label": label, "layer": layer, "share": share}
        for outcome in ("binary", "likert"):
            se = math.sqrt(
                pa.variance_by_layer(
                    outcome,
                    layer,
                    dilemmas,
                    pairs,
                    pa.N_RESPONDENTS,
                    pa.N_RATINGS,
                    share,
                )
            )
            row[outcome] = {
                "se": se,
                "mde": pa.mde(se),
                "power": pa.power_for(se, pa.TARGETS[outcome]),
            }
        rows.append(row)
    return rows


def spread_label(share):
    """Between-dilemma spread in both H2 units."""
    if share is None:
        return "--"
    points = math.sqrt(share * pa.LIKERT_VARIANCE)
    win = 100 * math.sqrt(share * pa.BINARY_VARIANCE)
    return f"{points:.2f} pts / {win:.0f} pp"


def dilemmas_needed(outcome, effect, share, max_dilemmas=500):
    """Dilemmas needed for 80% power at ``effect``, holding the current
    number of pairs per dilemma."""
    planned_dilemmas, planned_pairs = pa.design_sizes(pa.load_design())
    pairs_per_dilemma = planned_pairs / planned_dilemmas
    for dilemmas in range(1, max_dilemmas + 1):
        pairs = round(pairs_per_dilemma * dilemmas)
        se = math.sqrt(
            pa.variance_by_layer(
                outcome,
                "general",
                dilemmas,
                pairs,
                pa.N_RESPONDENTS,
                pa.N_RATINGS,
                share,
            )
        )
        if pa.mde(se) <= effect:
            return dilemmas
    return None


def print_table(rows):
    print("H2 power, current survey plan")
    print(
        f"  {pa.N_RESPONDENTS} respondents x {pa.RATINGS_PER_RESPONDENT} "
        f"comparisons = {pa.N_RATINGS:,} ratings; 18 dilemmas; 306 pairs"
    )
    print("  80% power, two-sided 5%\n")
    header = (
        f"{'Scope':<30} {'Spread':<17} {'LMM MDE':>9} {'@0.25':>7} "
        f"{'BT MDE':>9} {'@5pp':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        likert, binary = row["likert"], row["binary"]
        print(
            f"{row['label']:<30} {spread_label(row['share']):<17} "
            f"{likert['mde']:>8.2f}p {100 * likert['power']:>6.0f}% "
            f"{100 * binary['mde']:>7.1f}pp {100 * binary['power']:>6.0f}%"
        )
    print("\nDilemmas needed for 80% power (current pairs per dilemma):")
    for row in rows[1:]:
        likert = dilemmas_needed("likert", pa.TARGETS["likert"], row["share"])
        binary = dilemmas_needed("binary", pa.TARGETS["binary"], row["share"])
        print(
            f"  {row['label']:<30} LMM at 0.25 pts: {likert:>3}   "
            f"BT at 5 pp: {binary:>3}"
        )


def write_table(rows, name="survey-h2-power.tex"):
    """Write the minimal H2 table for the report."""
    body = "\n".join(
        f"{row['label']} & {spread_label(row['share'])} & "
        f"{row['likert']['mde']:.2f} & {100 * row['likert']['power']:.0f} & "
        f"{100 * row['binary']['mde']:.1f} & {100 * row['binary']['power']:.0f} \\\\"
        for row in rows
    )
    tex = (
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{lrrrrr}\n"
        "\\toprule\n"
        " & & \\multicolumn{2}{c}{LMM (points)} & "
        "\\multicolumn{2}{c}{Bradley-Terry (pp)} \\\\\n"
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
        "Scope & Dilemma spread & MDE at 80\\% & Power at 0.25 & "
        "MDE at 80\\% & Power at 5 pp \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "}%"
    )
    return pa.write_tex(tex, name)


def plot_power(rows, path=None):
    """Power curves for the LMM and Bradley-Terry outcomes."""
    path = path or pa.FIGURES_DIR / "survey-h2-power.png"
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))

    points = np.linspace(0, 0.6, 300)
    for row, colour in zip(rows, COLOURS):
        axes[0].plot(
            points,
            [pa.power_for(row["likert"]["se"], value) for value in points],
            color=colour,
            label=row["label"],
        )

    win_shift = np.linspace(0, 0.20, 300)
    for row, colour in zip(rows, COLOURS):
        axes[1].plot(
            100 * win_shift,
            [pa.power_for(row["binary"]["se"], value) for value in win_shift],
            color=colour,
        )

    for axis in axes:
        axis.axhline(0.8, color="grey", ls=":", lw=1)
        axis.set_ylim(0, 1.02)
        axis.set_ylabel("Power")
        axis.grid(alpha=0.3, lw=0.5)

    axes[0].axvline(0.25, color="black", ls="--", lw=1)
    axes[0].annotate("target 0.25", (0.255, 0.05), fontsize=7)
    axes[0].set_xlabel("Average preference (points, -2 to +2)")
    axes[0].set_title("LMM (primary)")

    axes[1].axvline(5, color="black", ls="--", lw=1)
    axes[1].annotate("target 5 pp", (5.2, 0.05), fontsize=7)
    axes[1].set_xlabel("Average win-rate shift (percentage points)")
    axes[1].set_title("Bradley-Terry")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=7,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(path, dpi=200, bbox_inches="tight")
    return path


def main():
    rows = h2_rows()
    print_table(rows)
    print(f"\nwrote {write_table(rows)}")
    print(f"wrote {plot_power(rows)}")


if __name__ == "__main__":
    main()
