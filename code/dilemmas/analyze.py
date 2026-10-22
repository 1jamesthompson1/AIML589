import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Dilemmas analysis

    What the models answered, how far apart the two models' answers are, and
    how much they noticed they were being tested.

    Open it with `uv run marimo edit code/dilemmas/analyze.py`, or run it
    headless with `uv run marimo export html code/dilemmas/analyze.py -o
    analyze.html`.
    """)
    return


@app.cell
def _():
    import json
    import random
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from inspect_scout import scan_results_df

    # Shared LaTeX table helpers live in code/latex_tables.py.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from latex_tables import escape_header, write_latex_table

    HERE = Path(__file__).resolve().parent
    return (
        HERE,
        Path,
        escape_header,
        json,
        mo,
        np,
        pd,
        plt,
        random,
        scan_results_df,
        sys,
        write_latex_table,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Shared helpers

    The two analyses at the end of this notebook both look for the newest data
    directory under the artifacts tree, read a JSON file, pivot a model x
    category table and write a figure to `code/figures/`, so those four steps
    live here.
    """)
    return


@app.cell
def _(json, plt):
    def newest_matching(parent, pattern, needs=()):
        """The newest directory under parent matching pattern.

        A directory only counts when every path in needs (relative to it)
        exists, so a run that was never finished is skipped. None when
        nothing matches.
        """
        if not parent.is_dir():
            return None
        candidates = [
            directory
            for directory in parent.glob(pattern)
            if directory.is_dir() and all((directory / need).exists() for need in needs)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda directory: directory.stat().st_mtime)

    def read_json(path, default=None):
        """The JSON at path, or default when the file is not there."""
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def pivot_means_counts(df, column, value="value"):
        """(means, counts) of a score by model x column.

        The unscored rows are dropped first, so the counts are the scored
        responses only.
        """
        valid = df.dropna(subset=[value])
        means = valid.pivot_table(
            index="model", columns=column, values=value, aggfunc="mean"
        )
        counts = valid.pivot_table(
            index="model", columns=column, values=value, aggfunc="size"
        )
        return means, counts

    def save_figure(figure, path, dpi=200):
        """Write the figure to path and close it."""
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        return path

    return newest_matching, pivot_means_counts, read_json, save_figure


@app.cell
def _(mo):
    mo.md(r"""
    ## Per-dilemma answer statistics

    The per-dilemma answer statistics from the exported runs, ranked by how far
    the models diverge on the dilemma.

    **Mean answer** is the mean chosen option index (1 = the first option, up
    to and including "Other or unclear"), which locates where the answers sit.
    **Stability** is `1 - H / ln(k)`, where `H` is the Shannon entropy of the
    chosen indices over the repetitions and `k` is the number of options the
    dilemma offered: 1.00 means every run chose the same option, 0.00 means
    the answers were spread evenly across the options. Entropy reads the whole
    distribution, so fading in and out of one option is scored differently from
    a clean split, which a modal share cannot show. With three repetitions the
    metric is coarse.

    Each model's column gives the mean answer with its stability in
    parentheses, e.g. `1.80 (0.54)`; the highest stability on a dilemma is
    bolded in the LaTeX table.

    **Divergence** is the mean pairwise total variation distance between the
    models' distributions of chosen option indices: 0.00 = identical
    distributions, 1.00 = no shared option. Rows are sorted by it, so the most
    model-divergent scenarios sit at the top.

    Edited dilemmas keep their superseded runs in the manifest; each (profile,
    dilemma, model) is scored on its most recent run set only, so the
    statistics use the option list the index currently publishes.

    The tables are written to `code/figures/dilemmas-answer-stats.tex` and
    `code/figures/dilemmas-answer-stats-frontier.tex` for the report to
    import with `\ctable{...}`.
    """)
    return


@app.cell
def _(HERE, Path, json, np, pd, write_latex_table):
    def load_run_choices(model_labels: dict[str, str]) -> tuple[pd.DataFrame, dict]:
        """The exported runs of the requested models, one row per run.

        The bundle is picked the way the answer statistics pick it: the
        export holding the most of the requested models, and inside it the
        most recent run set per (profile, dilemma, model), so superseded
        runs are not counted twice. Both the answer statistics and the
        aggregate below read their data from here, so the two always
        describe the same runs.

        Returns (runs, options):
        - runs: Profile, Dilemma, Model (the label), Created and Index,
          where Index is the 1-based position of the judged answer in the
          dilemma's answer options. Runs without a judged answer are
          dropped.
        - options: dilemma id -> the dilemma's answer options.
        """
        # The model set is passed in: see the two sets defined below the
        # function (the fine-tuned adapters and the frontier comparison
        # models). The two sets live in different bundles, so the bundle
        # holding the set is looked up rather than the active export alone
        # (which holds the active batch only, so a set the active batch does
        # not carry would find nothing).
        runs_dir = HERE / "output" / "runs"

        def candidates() -> list[tuple[Path, Path]]:
            """(manifest, base dir) pairs, active export first."""
            found = [(runs_dir / "index.json", runs_dir)]
            bundles = sorted(
                (p for p in (runs_dir / "full").glob("*/index.json")),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            found += [(p, p.parent) for p in bundles]
            return found

        def load(manifest_path: Path, base: Path):
            if not manifest_path.exists():
                return None
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            present = sum(1 for run in manifest["runs"] if run["model"] in model_labels)
            return (present, manifest, base)

        chosen = None
        for manifest_path, base in candidates():
            loaded = load(manifest_path, base)
            if loaded is None:
                continue
            if loaded[0] and (chosen is None or loaded[0] > chosen[0]):
                chosen = loaded
        if chosen is None:
            print(
                f"No exported runs yet: {runs_dir / 'index.json'} is missing or "
                f"holds none of the models. Run `uv run python "
                f"code/dilemmas/export_results.py` first."
            )
            return pd.DataFrame(), {}
        _, manifest, base = chosen
        options = {d["situation_id"]: d["answer_options"] for d in manifest["dilemmas"]}
        rows = []
        for run in manifest["runs"]:
            if run["model"] not in model_labels:
                continue
            judge = json.loads(
                (base / run["path"] / "judge.json").read_text(encoding="utf-8")
            )
            choice = (judge.get("metadata") or {}).get("choice") or judge.get("answer")
            answer_options = options[run["situation_id"]]
            rows.append(
                {
                    "Profile": run["profile_id"],
                    "Dilemma": run["situation_id"],
                    "Model": model_labels[run["model"]],
                    "Created": run["created"],
                    "Index": float(answer_options.index(choice) + 1)
                    if choice in answer_options
                    else np.nan,
                }
            )
        if not rows:
            return pd.DataFrame(), options
        runs = pd.DataFrame(rows).dropna(subset=["Index"])
        keys = ["Profile", "Dilemma", "Model"]
        # Superseded runs stay in the manifest for history; score each
        # (profile, dilemma, model) on its most recent run set only.
        latest = runs.groupby(keys)["Created"].transform("max")
        runs = runs[runs["Created"] == latest]
        return runs, options

    def answer_stats(
        model_labels: dict[str, str], model_order: list[str]
    ) -> pd.DataFrame:
        runs, options = load_run_choices(model_labels)
        if runs.empty:
            return pd.DataFrame()
        keys = ["Profile", "Dilemma", "Model"]
        mean_answer = runs.groupby(keys)["Index"].mean().rename("Mean answer")
        counts = runs.groupby(keys + ["Index"]).size()
        probabilities = counts / counts.groupby(level=[0, 1, 2]).sum()
        entropy = (
            (-probabilities * np.log(probabilities)).groupby(level=[0, 1, 2]).sum()
        )
        entropy = entropy.reindex(mean_answer.index)
        option_count = np.array(
            [len(options[d]) for d in mean_answer.index.get_level_values("Dilemma")],
            dtype=float,
        )
        stability = pd.Series(
            1 - entropy.to_numpy() / np.log(option_count),
            index=mean_answer.index,
            name="Stability",
        )
        # Mean pairwise total variation distance between the models'
        # distributions of chosen option indices for the dilemma.
        divergence = {}
        for (profile, dilemma), group in probabilities.groupby(level=[0, 1]):
            per_model = group.droplevel([0, 1]).unstack("Model").fillna(0.0)
            pair_tvds = [
                0.5 * (per_model[left] - per_model[right]).abs().sum()
                for position, left in enumerate(per_model.columns)
                for right in per_model.columns[position + 1 :]
            ]
            divergence[(profile, dilemma)] = np.mean(pair_tvds) if pair_tvds else np.nan
        divergence = pd.Series(divergence, name="Divergence")
        stats = pd.concat([mean_answer, stability], axis=1).unstack("Model")
        stats.insert(0, ("Divergence", "TVD"), divergence.reindex(stats.index))
        # Keep the requested model order (pandas would otherwise sort the
        # columns alphabetically by label).
        columns = [("Divergence", "TVD")]
        columns += [
            (stat, model)
            for stat in ("Mean answer", "Stability")
            for model in model_order
        ]
        stats = stats.reindex(columns=columns)
        stats = stats.sort_values(
            ("Divergence", "TVD"), ascending=False, na_position="last", kind="stable"
        )
        return stats.round(2)

    def merge_answer_stats(
        stats: pd.DataFrame, model_order: list[str], *, latex: bool = False
    ) -> pd.DataFrame:
        """One column per model: mean answer with its stability in parentheses.

        The notebook shows plain text ("1.80 (0.54)"). The LaTeX table also
        bolds the stability where it is the highest on the dilemma (the
        caption's highlight), leaving the mean answer plain.
        """
        merged = {("Divergence", "TVD"): stats[("Divergence", "TVD")]}
        stability = stats["Stability"][model_order]
        best = stability.max(axis=1)
        for model in model_order:
            means = stats[("Mean answer", model)]
            cells = []
            for mean, value, top in zip(means, stability[model], best):
                if pd.isna(mean):
                    cells.append("--")
                elif pd.isna(value):
                    cells.append(f"{mean:.2f} (--)")
                else:
                    bold = "\\bfseries " if latex and np.isclose(value, top) else ""
                    cells.append(f"{mean:.2f} ({bold}{value:.2f})")
            merged[("Mean answer (stability)", model)] = cells
        table = pd.DataFrame(merged, index=stats.index)
        table.index = table.index.set_names([None, None])
        table.columns = pd.MultiIndex.from_tuples(table.columns, names=[None, None])
        return table

    MODEL_SETS = {
        # The study set: all four fine-tuned Qwen3.8-27B adapter variants
        # ("overall" adapters served from the local vLLM server) plus their
        # base model.
        "dilemmas-answer-stats": {
            "description": "fine-tuned Qwen3.8-27B adapters and base",
            # The base model the adapters were fine-tuned from. Its pairs
            # come first in the aggregate table.
            "base": "Qwen3.8-27B base",
            "labels": {
                "openai/Qwen3.8-27B-nz-wvs-full_string_distribution-overall": (
                    "full-string"
                ),
                "openai/Qwen3.8-27B-nz-wvs-first_token_distribution-overall": (
                    "first-token"
                ),
                "openai/Qwen3.8-27B-nz-wvs-modal_response-overall": "modal",
                "openai/Qwen3.8-27B-nz-wvs-sampled_response-overall": "sampled",
                "openai/Qwen/Qwen3.8-27B": "Qwen3.8-27B base",
            },
            "order": [
                "full-string",
                "sampled",
                "modal",
                "first-token",
                "Qwen3.8-27B base",
            ],
        },
        # For curiosity: the frontier comparison models the same dilemmas
        # were also run on.
        "dilemmas-answer-stats-frontier": {
            "description": "frontier comparison models",
            "labels": {
                "openrouter/qwen/qwen3.8-27b": "Qwen3.8-27b",
                "openrouter/deepseek/deepseek-v4.1-flash": "deepseek-v4.1-flash",
                "openrouter/anthropic/claude-haiku-5.5": "claude-haiku",
                "openrouter/openai/gpt-6-luna": "gpt-6-luna",
                "openrouter/z-ai/glm-5.3-flash": "glm-5.3-flash",
            },
            "order": [
                "Qwen3.8-27b",
                "claude-haiku",
                "deepseek-v4.1-flash",
                "gpt-6-luna",
                "glm-5.3-flash",
            ],
        },
    }

    def write_stats_table(
        stats: pd.DataFrame, model_order: list[str], filename: str
    ) -> None:
        # Mean answer and stability share one column per model, with the
        # highest stability on each dilemma bolded in the parentheses. The
        # stub columns are unnamed so the index names do not add a header
        # row of their own.
        stats_tex_df = merge_answer_stats(stats, model_order, latex=True)
        write_latex_table(
            stats_tex_df,
            {("Divergence", "TVD"): "{:.2f}"},
            stats_tex_df.shape[1],
            filename,
            # No automatic highlights: the stability highlight is already in
            # the merged cells, and the numeric divergence is left plain.
            senses={},
            column_format="ll" + "r" * stats_tex_df.shape[1],
            stub_label="",
        )

    result: dict[str, pd.DataFrame] = {}
    for filename, spec in MODEL_SETS.items():
        df = answer_stats(spec["labels"], spec["order"])
        if df.empty:
            result[spec["description"]] = df
            continue
        write_stats_table(df, spec["order"], filename + ".tex")
        result[spec["description"]] = merge_answer_stats(df, spec["order"])
    answer_stats_df = result["fine-tuned Qwen3.8-27B adapters and base"]

    answer_stats_df
    return MODEL_SETS, load_run_choices


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Answer stats aggregate

    Which model answers most like the other, and which models are most
    different from each other. Every model answers every dilemma several
    times (the run set), so a pair of models is compared two ways:

    **Same / Different (modal)** — each model's modal answer on a dilemma
    (the option the most runs chose) either matches the other model's or
    it does not, so every dilemma is one "same" or one "different" for the
    pair. This is the headline count: the model pair that agrees on most
    dilemmas is the most alike, and the pair that agrees on fewest is the
    most different.

    **Same (run pairs)** — the share of run pairs (every run of one model
    against every run of the other on the same dilemma) that chose the same
    option. It sees the run-to-run wobble the modal count hides: two models
    that both lean the same way but wobble differently score lower here.

    **Mean TVD** — the mean total variation distance between the two
    models' answer distributions on a dilemma: 0.00 means the same answers,
    1.00 means no shared answer. It is the graded version of "different"
    and the same measure as the divergence column of the table above.

    The table covers the study set (the four fine-tuned adapters and the
    base model). The pairs against the base model come first — each adapter
    against the model it was fine-tuned from — and the fine-tuned models
    against each other come below them, the most alike pair first in each
    group. The lowest agreement (the most different pair) is highlighted in
    the LaTeX table. The frontier set is the other key of `MODEL_SETS`, so
    the same table can be rebuilt for those models by swapping the set.
    """)
    return


@app.cell
def _(MODEL_SETS, load_run_choices, mo, np, pd, write_latex_table):
    def answer_agreement(
        model_labels: dict[str, str],
        model_order: list[str],
        base: str | None = None,
    ) -> pd.DataFrame:
        """The same/different counts for every pair of models.

        One row per model pair: the number of dilemmas compared, how many
        the two models' modal answers matched ("Same") and how many did not
        ("Different"), the same counts as a percentage, the run-pair
        agreement percentage and the mean TVD.

        The pairs against the base model come first (each adapter against
        the model it was fine-tuned from), then the fine-tuned models
        against each other; within each group the most alike pair is
        first.
        """
        runs, _ = load_run_choices(model_labels)
        if runs.empty:
            return pd.DataFrame()
        keys = ["Profile", "Dilemma", "Model"]
        # How many runs each model chose each answer option.
        counts = runs.groupby(keys + ["Index"]).size().rename("Runs").reset_index()
        # The modal answer per model per dilemma: the option with the most
        # runs. Ties go to the lowest option index, so the pick is
        # deterministic.
        ordered = counts.sort_values(keys + ["Index"])
        modal = (
            ordered.loc[ordered.groupby(keys)["Runs"].idxmax()]
            .pivot(index=["Profile", "Dilemma"], columns="Model", values="Index")
            .reindex(columns=model_order)
        )
        # The same counts as a wide table: one column per (model, option).
        wide = counts.pivot_table(
            index=["Profile", "Dilemma"],
            columns=["Model", "Index"],
            values="Runs",
            aggfunc="sum",
            fill_value=0,
        )
        options = sorted({option for _, option in wide.columns})
        rows = []
        for position, left in enumerate(model_order):
            for right in model_order[position + 1 :]:
                # The modal answers, on the dilemmas both models answered.
                both = modal[[left, right]].dropna()
                same = int((both[left] == both[right]).sum())
                compared = len(both)
                # The run-pair agreement and the TVD, per dilemma.
                run_same = 0.0
                run_total = 0.0
                tvds = []
                for _, dilemma_row in wide.iterrows():
                    left_counts = dilemma_row[left].reindex(options).fillna(0.0)
                    right_counts = dilemma_row[right].reindex(options).fillna(0.0)
                    n_left, n_right = left_counts.sum(), right_counts.sum()
                    if not n_left or not n_right:
                        continue
                    run_same += float((left_counts * right_counts).sum())
                    run_total += float(n_left * n_right)
                    tvds.append(
                        0.5
                        * (left_counts / n_left - right_counts / n_right).abs().sum()
                    )
                rows.append(
                    {
                        "Left": left,
                        "Right": right,
                        "Dilemmas": compared,
                        "Same": same,
                        "Different": compared - same,
                        "Same %": 100 * same / compared if compared else np.nan,
                        "Same % (run pairs)": (
                            100 * run_same / run_total if run_total else np.nan
                        ),
                        "Mean TVD": float(np.mean(tvds)) if tvds else np.nan,
                    }
                )
        pairs = pd.DataFrame(rows)
        pairs["Pair"] = pairs["Left"] + " vs " + pairs["Right"]
        # The pairs against the base model come first, the fine-tuned models
        # against each other below them; the most alike pair first in each
        # group.
        pairs["Group"] = [
            0 if base and base in (left, right) else 1
            for left, right in zip(pairs["Left"], pairs["Right"])
        ]
        pairs = pairs.sort_values(
            ["Group", "Same %", "Mean TVD"],
            ascending=[True, False, True],
            kind="stable",
        )
        return pairs.set_index("Pair")[
            [
                "Dilemmas",
                "Same",
                "Different",
                "Same %",
                "Same % (run pairs)",
                "Mean TVD",
            ]
        ].round(2)

    def write_agreement_table(pairs: pd.DataFrame) -> None:
        """The LaTeX table the report can import with \\ctable{...}."""
        # The pair table keeps a two-level header (the top row groups the
        # measures, the second row holds the measure) and the stub column
        # is unnamed so the index names do not add a header row of their own.
        # The columns are re-ordered to match the header labels, which are
        # assigned positionally.
        pairs_tex_df = pairs[
            [
                "Dilemmas",
                "Same",
                "Same %",
                "Same % (run pairs)",
                "Different",
                "Mean TVD",
            ]
        ].copy()
        pairs_tex_df.columns = pd.MultiIndex.from_tuples(
            [
                ("Dilemmas", "compared"),
                ("Same", "modal"),
                ("Same", "modal %"),
                ("Same", "run pairs %"),
                ("Different", "modal"),
                ("Distance", "mean TVD"),
            ]
        )
        pairs_tex_df.columns = pairs_tex_df.columns.set_names([None, None])
        pairs_tex_df.index = pairs_tex_df.index.set_names([None])
        # The format keys are the escaped labels the Styler sees: the "%"
        # signs are escaped by write_latex_table before it styles the table.
        # The senses highlight the lowest agreement: the fewest "same"
        # dilemmas, the most "different" ones and the largest distance.
        write_latex_table(
            pairs_tex_df,
            {
                ("Dilemmas", "compared"): "{:.0f}",
                ("Same", "modal"): "{:.0f}",
                ("Same", "modal \\%"): "{:.1f}",
                ("Same", "run pairs \\%"): "{:.1f}",
                ("Different", "modal"): "{:.0f}",
                ("Distance", "mean TVD"): "{:.2f}",
            },
            pairs_tex_df.shape[1],
            "dilemmas-answer-agreement.tex",
            senses={"Same": "min", "Different": "max", "Distance": "max"},
            column_format="l" + "r" * pairs_tex_df.shape[1],
            stub_label="",
        )

    study_set = MODEL_SETS["dilemmas-answer-stats"]
    agreement_pairs_df = answer_agreement(
        study_set["labels"], study_set["order"], study_set.get("base")
    )
    if not agreement_pairs_df.empty:
        write_agreement_table(agreement_pairs_df)
        # The least alike pair: the lowest agreement, and the largest
        # distance when the agreement is tied.
        least = agreement_pairs_df.sort_values(
            ["Same %", "Mean TVD"], ascending=[True, False]
        ).iloc[0]
        least_note = mo.md(
            f"The least alike pair is **{least.name}**: the two models' modal "
            f"answers matched on {int(least['Same'])} of "
            f"{int(least['Dilemmas'])} dilemmas ({least['Same %']:.1f}%), "
            f"mean TVD {least['Mean TVD']:.2f}."
        )
    else:
        least_note = mo.md("No exported runs to compare yet.")
    return least_note


@app.cell
def _(mo):
    mo.md(r"""
    ## H1: do the fine-tuned models answer differently from the base model?

    The pre-specified test of H1. Each of the four fine-tuned models is
    compared separately with the base model it was fine-tuned from.

    **Per dilemma** the judge-classified action counts of the two models are
    compared with a Monte Carlo permutation test: 10,000 replicates with a
    fixed seed, permuting the model labels without replacement within the
    dilemma (each model keeps its number of runs), so the null is the
    distribution of the statistic when the two models' runs are
    exchangeable. The effect size is the **total variation distance** (TVD)
    between the two models' distributions over the dilemma's answer options:
    0.00 when the distributions are the same, 1.00 when they share no mass.

    **Overall** a model's effect size is the mean TVD across all the
    dilemmas, and it is compared with the mean TVD of the permutation null
    (the same permutations, averaged across the dilemmas). The four overall
    p-values are Benjamini-Hochberg corrected, and H1 is supported for a
    model when its BH-adjusted p is below .05 - overall when at least one of
    the four models passes.

    The p-values are Monte Carlo p-values, `(1 + #{null >= observed}) /
    (replicates + 1)`, so the smallest possible p is 1/10,001. The
    per-dilemma p-values are uncorrected: the pre-specified correction is
    the Benjamini-Hochberg step across the four overall effects.

    The rate of **"Other or unclear"** judge classifications is reported
    separately (a catch-all classification carries no decision), and the
    whole test is repeated with those runs excluded as a sensitivity check.
    """)
    return


@app.cell
def _(
    HERE,
    MODEL_SETS,
    json,
    load_run_choices,
    mo,
    pd,
    sys,
    write_latex_table,
):
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import h1_test

    h1_set = MODEL_SETS["dilemmas-answer-stats"]
    h1_runs, h1_options = load_run_choices(h1_set["labels"])
    # The fine-tuned models, in the study set's order.
    h1_order = [label for label in h1_set["order"] if label != h1_set["base"]]
    # The test, and the sensitivity check: the same test with the "Other or
    # unclear" runs dropped, so only the runs the judge could place on one
    # of the dilemma's substantive options count.
    h1_results = h1_test.run_h1(
        h1_runs,
        h1_options,
        base_label=h1_set["base"],
        model_order=h1_order,
        replicates=h1_test.REPLICATES,
        seed=h1_test.SEED,
    )
    h1_sensitivity = h1_test.run_h1(
        h1_test.drop_unclear(h1_runs, h1_options),
        h1_options,
        base_label=h1_set["base"],
        model_order=h1_order,
        replicates=h1_test.REPLICATES,
        seed=h1_test.SEED,
    )

    def h1_summary_frame(results):
        """The summary of the test: one row per fine-tuned model."""
        if results.summary.empty:
            return pd.DataFrame()
        frame = results.summary.set_index("Model")[
            [
                "Mean TVD",
                "Null mean TVD",
                "p (overall)",
                "p (BH)",
            ]
        ].copy()
        frame.index = frame.index.set_names(["Model"])
        return frame

    def h1_write_summary(frame, filename):
        """The summary table the report can import with \\ctable{...}."""
        write_latex_table(
            frame,
            {
                "Mean TVD": "{:.3f}",
                "Null mean TVD": "{:.3f}",
                "p (overall)": "{:.4f}",
                "p (BH)": "{:.4f}",
            },
            frame.shape[1],
            filename,
            senses={"p (BH)": "min"},
            column_format="l" + "r" * frame.shape[1],
            stub_label="Model",
        )

    def h1_dilemma_frame(results):
        """The per-dilemma results: the TVD and its p-value per model."""
        if results.per_dilemma.empty:
            return pd.DataFrame()
        frame = pd.DataFrame(index=sorted(results.per_dilemma["Dilemma"].unique()))
        for model in h1_order:
            subset = results.per_dilemma[
                results.per_dilemma["Model"] == model
            ].set_index("Dilemma")
            frame[f"{model} TVD"] = subset["TVD"]
            frame[f"{model} p"] = subset["p"]
        frame.index = frame.index.set_names(["Dilemma"])
        return frame

    h1_summary_df = h1_summary_frame(h1_results)
    h1_sensitivity_df = h1_summary_frame(h1_sensitivity)
    h1_dilemma_df = h1_dilemma_frame(h1_results)
    # The base model first, then the fine-tuned models in the study set's
    # order, so the table reads like the rest of the notebook.
    h1_other_df = (
        h1_results.other_rates.set_index("Model").reindex([h1_set["base"]] + h1_order)
        if not h1_results.other_rates.empty
        else pd.DataFrame()
    )
    if not h1_summary_df.empty:
        h1_write_summary(h1_summary_df, "dilemmas-h1-summary.tex")
        h1_write_summary(h1_sensitivity_df, "dilemmas-h1-sensitivity.tex")
        write_latex_table(
            h1_other_df[["Runs", "Other or unclear", "Rate"]],
            {"Runs": "{:.0f}", "Other or unclear": "{:.0f}", "Rate": "{:.1%}"},
            3,
            "dilemmas-h1-other-rates.tex",
            senses={"Rate": "max"},
            column_format="lrr",
            stub_label="Model",
        )
        # The per-dilemma table: the largest TVD and the smallest p per
        # column are highlighted.
        h1_senses = {
            column: ("max" if column.endswith("TVD") else "min")
            for column in h1_dilemma_df.columns
        }
        write_latex_table(
            h1_dilemma_df,
            {
                column: ("{:.2f}" if column.endswith("TVD") else "{:.3f}")
                for column in h1_dilemma_df.columns
            },
            h1_dilemma_df.shape[1],
            "dilemmas-h1-per-dilemma.tex",
            senses=h1_senses,
            column_format="l" + "r" * h1_dilemma_df.shape[1],
            stub_label="Dilemma",
        )

    # The results are written out as JSON as well, so the numbers can be
    # checked without rerunning the notebook.
    h1_payload = {
        "settings": {
            "replicates": h1_test.REPLICATES,
            "seed": h1_test.SEED,
            "alpha": h1_test.ALPHA,
            "base_model": h1_set["base"],
            "models": h1_order,
        },
        "summary": json.loads(h1_results.summary.to_json(orient="records")),
        "per_dilemma": json.loads(h1_results.per_dilemma.to_json(orient="records")),
        "other_rates": json.loads(h1_results.other_rates.to_json(orient="records")),
        "sensitivity": {
            "summary": json.loads(h1_sensitivity.summary.to_json(orient="records")),
            "per_dilemma": json.loads(
                h1_sensitivity.per_dilemma.to_json(orient="records")
            ),
        },
    }
    h1_output = HERE / "output" / "h1-results.json"
    h1_output.parent.mkdir(parents=True, exist_ok=True)
    h1_output.write_text(json.dumps(h1_payload, indent=2) + "\n", encoding="utf-8")

    h1_supported = (
        h1_results.summary.loc[h1_results.summary["Supported"], "Model"].tolist()
        if not h1_results.summary.empty
        else []
    )
    if h1_summary_df.empty:
        h1_note = mo.md("No exported runs to test yet.")
    else:
        h1_note = mo.md(
            f"H1 is supported for **{len(h1_supported)} of "
            f"{len(h1_summary_df)}** fine-tuned models: "
            + (
                ", ".join(f"**{model}**" for model in h1_supported)
                if h1_supported
                else "none"
            )
            + ". The overall test is the mean TVD across the "
            f"{int(h1_results.summary['Dilemmas'].iloc[0])} dilemmas against the "
            f"permutation null (10,000 replicates, seed {h1_test.SEED}), "
            f"Benjamini-Hochberg corrected across the four models, so H1 is "
            f"supported overall (at least one model passes). The sensitivity "
            f"check (excluding the 'Other or unclear' runs) supports "
            f"{int(h1_sensitivity.summary['Supported'].sum())} of "
            f"{len(h1_sensitivity.summary)} models."
        )
    h1_note
    return h1_order, h1_results


@app.cell
def _(HERE, h1_order, h1_results, plt, save_figure):
    h1_fig_dir = HERE.parent / "figures"

    if h1_results.per_dilemma.empty:
        print("no exported runs to plot yet")
    else:
        # The permutation null of each model's overall effect: the observed
        # mean TVD against the 10,000 permuted ones.
        figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        summary = h1_results.summary.set_index("Model")
        for axis, model in zip(axes.ravel(), h1_order):
            null = h1_results.null_overall[model]
            observed = float(summary.loc[model, "Mean TVD"])
            axis.hist(null, bins=40, color="#c9c9c9", edgecolor="white")
            axis.axvline(
                null.mean(),
                color="#555555",
                linestyle="--",
                linewidth=1.2,
                label="null mean",
            )
            axis.axvline(observed, color="#d1642f", linewidth=2, label="observed")
            axis.set_title(
                f"{model}: observed {observed:.3f}, "
                f"BH p = {summary.loc[model, 'p (BH)']:.4f}",
                fontsize=9,
            )
            axis.set_xlabel("mean TVD across the dilemmas")
            axis.legend(fontsize=8)
        figure.suptitle(
            "H1: the observed mean TVD against the permutation null", fontsize=11
        )
        figure.tight_layout()
        h1_null_figure = save_figure(figure, h1_fig_dir / "dilemmas-h1-permutation.png")

        # The per-dilemma TVD of each model: where the differences sit.
        tvds = h1_results.per_dilemma.pivot(
            index="Dilemma", columns="Model", values="TVD"
        )[h1_order]
        ps = h1_results.per_dilemma.pivot(index="Dilemma", columns="Model", values="p")[
            h1_order
        ]
        order = tvds.mean(axis=1).sort_values(ascending=False).index
        tvds, ps = tvds.loc[order], ps.loc[order]
        grid = tvds.to_numpy(dtype=float)
        figure, axis = plt.subplots(figsize=(7, 10))
        image = axis.imshow(grid, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        figure.colorbar(image, ax=axis, label="TVD between the two models")
        axis.set_xticks(range(len(h1_order)))
        axis.set_xticklabels(h1_order, fontsize=9)
        axis.set_yticks(range(len(tvds)))
        axis.set_yticklabels(tvds.index, fontsize=6)
        for row in range(len(tvds)):
            for column in range(len(h1_order)):
                significant = ps.iloc[row, column] < 0.05
                axis.text(
                    column,
                    row,
                    f"{grid[row, column]:.2f}" + ("*" if significant else ""),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black" if grid[row, column] < 0.6 else "white",
                    fontweight="bold" if significant else "normal",
                )
        axis.set_title(
            "H1: the per-dilemma TVD of each fine-tuned model against the base\n"
            "(* marks a dilemma whose permutation p is below .05)",
            fontsize=10,
        )
        figure.tight_layout()
        h1_dilemma_figure = save_figure(
            figure, h1_fig_dir / "dilemmas-h1-per-dilemma.png"
        )
        print(f"Figures written to {h1_null_figure} and {h1_dilemma_figure}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Answer distribution

    The overall answer distribution per dilemma, pooled across the models'
    latest run sets: the debugging view for how balanced a dilemma is. A
    one-sided dilemma (balance 0.00 - every run picked the same option) says
    little about a model; a spread means the models are actually splitting.
    Run sets whose stored text/options differ from the current profile file
    are skipped as stale (they measured an earlier version of the scenario).

    Columns 1-4 are option indices in the dilemma's `answer_options` ("--" when
    it has fewer options). **Balance** is `H / ln(k)`: 1.00 = even spread, 0.00
    = one option. **Balance (not other)** recomputes that `H / ln(k)` with the
    "Other or unclear" catch-all dropped and the remaining shares renormalised,
    so a dilemma looks as balanced as its real decisions warrant; it is NaN
    when there is no catch-all option, fewer than two other options, or every
    run was "Other or unclear".
    """)
    return


@app.cell
def _(HERE, json, np, pd):
    def answer_distribution() -> pd.DataFrame:
        runs_dir = HERE / "output" / "runs"
        manifest = json.loads((runs_dir / "index.json").read_text(encoding="utf-8"))
        current = {}
        for profile_path in sorted((HERE / "profiles").glob("*.json")):
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            for dilemma in profile["dilemmas"]:
                current[dilemma["id"]] = {
                    "profile_id": profile["profile"],
                    "situation": dilemma["situation"],
                    "question": dilemma["question"],
                    "answer_options": dilemma["answer_options"],
                }
        groups: dict[tuple[str, str], list[dict]] = {}
        for run in manifest["runs"]:
            groups.setdefault((run["model"], run["situation_id"]), []).append(run)

        picked: dict[str, dict[int, int]] = {situation: {} for situation in current}
        for (model, situation_id), runs in groups.items():
            entry = current.get(situation_id)
            if entry is None or not runs:
                continue
            newest = max(run["created"] for run in runs)
            for run in (r for r in runs if r["created"] == newest):
                run_dir = runs_dir / run["path"]
                try:
                    config = json.loads(
                        (run_dir / "config.json").read_text(encoding="utf-8")
                    )
                    judge = json.loads(
                        (run_dir / "judge.json").read_text(encoding="utf-8")
                    )
                except (FileNotFoundError, json.JSONDecodeError):
                    continue  # exported run dir missing (interrupted spot test)
                stored = config.get("dilemma", {})
                if (
                    stored.get("situation"),
                    stored.get("question"),
                    stored.get("answer_options"),
                ) != (
                    entry["situation"],
                    entry["question"],
                    entry["answer_options"],
                ):
                    continue  # stale run set: earlier text/options
                judge = json.loads((run_dir / "judge.json").read_text(encoding="utf-8"))
                choice = (judge.get("metadata") or {}).get("choice") or judge.get(
                    "answer"
                )
                if choice not in entry["answer_options"]:
                    continue
                index = entry["answer_options"].index(choice) + 1
                picked[situation_id][index] = picked[situation_id].get(index, 0) + 1

        rows = []
        for situation_id, entry in current.items():
            counts = picked[situation_id]
            total = sum(counts.values())
            option_count = len(entry["answer_options"])
            shares = (
                [counts.get(i, 0) / total for i in range(1, option_count + 1)]
                if total
                else []
            )
            entropy = -sum(p * np.log(p) for p in shares if p > 0)
            balance = (
                float(entropy / np.log(option_count))
                if total and option_count > 1
                else 0.0
            )
            if not balance:  # normalise a rounded -0.0
                balance = 0.0
            # Balance with the "Other or unclear" catch-all dropped.
            other_index = next(
                (
                    i
                    for i, option in enumerate(entry["answer_options"])
                    if option.strip().lower().startswith("other")
                ),
                None,
            )
            decided = [
                counts.get(i, 0)
                for i in range(1, option_count + 1)
                if (i - 1) != other_index
            ]
            decided_total = sum(decided)
            if other_index is not None and len(decided) > 1 and decided_total:
                decided_shares = [n / decided_total for n in decided]
                decided_entropy = -sum(p * np.log(p) for p in decided_shares if p > 0)
                balance_not_other = float(decided_entropy / np.log(len(decided)))
                if not balance_not_other:  # normalise a rounded -0.0
                    balance_not_other = 0.0
            else:
                balance_not_other = None
            row = {
                "Profile": entry["profile_id"],
                "Dilemma": situation_id,
                "Runs": total,
                "Balance": round(max(balance, 0.0), 2),
                "Balance (not other)": (
                    None if balance_not_other is None else round(balance_not_other, 2)
                ),
            }
            for index in range(1, 5):
                row[str(index)] = (
                    f"{counts.get(index, 0) / total:.0%}"
                    if total and index <= option_count
                    else ("--" if index > option_count else "-")
                )
            rows.append(row)
        table = pd.DataFrame(rows)
        return table.sort_values(["Balance", "Profile", "Dilemma"], kind="stable")

    answer_distribution_df = answer_distribution()
    answer_distribution_df
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Comparison distance

    How far apart the two models' answers are, per dilemma.

    The judge gives one answer option per response, which is coarse: two
    answers can pick the same option and still differ in what they say. The
    responses were embedded with a local embedding model (see
    `embed_responses.py`) and this cell measures the cosine distance between
    the two agents' answers for every comparison: `1 - cos(a, b)`, where 0
    means the two answers say the same thing and 1 means they are unrelated.

    The data is the newest comparison run's per-model pools: one pool per
    fine-tune against the base model, each holding every pair of the two
    models' runs on each dilemma. Each pool gets its own figure, named after
    the fine-tune, so the four models can be compared side by side.
    """)
    return


@app.cell
def _(HERE, np, pd, read_json):
    def pair_rows(manifest, vectors, run_ids):
        """One row per comparison: the distance and the judge's verdict."""
        index = {run_id: position for position, run_id in enumerate(run_ids)}
        rows = []
        for record in manifest.get("comparisons", []):
            agent1 = record["agents"]["Agent 1"]
            agent2 = record["agents"]["Agent 2"]
            if agent1["run_id"] not in index or agent2["run_id"] not in index:
                continue
            similarity = float(
                vectors[index[agent1["run_id"]]] @ vectors[index[agent2["run_id"]]]
            )
            choice1, choice2 = agent1.get("choice", ""), agent2.get("choice", "")
            if not (choice1 and choice2):
                agreement = "unjudged"
            else:
                agreement = "same" if choice1 == choice2 else "different"
            rows.append(
                {
                    "comparison_id": record["comparison_id"],
                    "dilemma_id": record["dilemma_id"],
                    "profile_id": record["profile_id"],
                    "situation_id": record["situation_id"],
                    "distance": 1.0 - similarity,
                    "agreement": agreement,
                }
            )
        return pd.DataFrame(rows)

    def model_label(manifest):
        """The fine-tune's short name, e.g. 'modal_response-overall'."""
        model = (manifest.get("models") or {}).get("model_b") or ""
        name = model.rsplit("/", 1)[-1]
        marker = "-nz-wvs-"
        return name.split(marker, 1)[1] if marker in name else (name or "complete")

    def pool_paths(directory):
        """(name, manifest path, embeddings dir) for one run's pools.

        Per-model subdirectories win over a leftover top-level pool, so an
        old single-pool run is not analysed next to its replacements.
        """
        complete = directory / "complete"
        pools = []
        if complete.is_dir():
            pools.extend(
                (
                    child.name,
                    child / "comparisons.json",
                    directory / "embeddings" / "complete" / child.name,
                )
                for child in sorted(complete.iterdir())
                if child.is_dir() and (child / "comparisons.json").exists()
            )
        if not pools and (complete / "comparisons.json").exists():
            pools.append(
                (
                    "complete",
                    complete / "comparisons.json",
                    directory / "embeddings" / "complete",
                )
            )
        return pools

    def load_pools(comparisons_dir):
        """Every embedded comparison pool of the newest per-model run.

        Each entry is the pool manifest, its embeddings path, the pair
        distances and the model label. The figure, the agreement correlation
        and the survey design all walk this one list, so they always describe
        the same data. A run with per-model pools wins over a newer legacy
        single-pool run, so a scratch single-pool test does not hide the main
        comparison run; the legacy pool is still the fallback when no
        per-model run has embeddings.
        """
        run_dirs = (
            sorted(
                (path for path in comparisons_dir.iterdir() if path.is_dir()),
                key=lambda path: path.name,
                reverse=True,
            )
            if comparisons_dir.is_dir()
            else []
        )
        legacy_fallback = None
        for directory in run_dirs:
            pools = []
            for name, manifest_path, embeddings_dir in pool_paths(directory):
                vectors_path = embeddings_dir / "vectors.npz"
                if not vectors_path.exists():
                    continue
                manifest = read_json(manifest_path)
                cache = np.load(vectors_path)
                rows = pair_rows(
                    manifest,
                    cache["vectors"],
                    [str(value) for value in cache["run_ids"]],
                )
                pools.append(
                    {
                        "name": name,
                        "label": model_label(manifest),
                        "directory": directory,
                        "manifest": manifest,
                        "vectors_path": vectors_path,
                        "rows": rows,
                    }
                )
            if not pools:
                continue
            if any(pool["name"] != "complete" for pool in pools):
                return pools
            if legacy_fallback is None:
                legacy_fallback = pools
        if legacy_fallback is not None:
            return legacy_fallback
        print(
            f"no embedded comparison pools under {comparisons_dir} - run "
            "generate_comparisons.py and embed_responses.py first"
        )
        return []

    pools = load_pools(HERE / "output" / "comparisons")

    pd.DataFrame(
        [
            {
                "pool": pool["name"],
                "model": pool["label"],
                "comparisons": len(pool["rows"]),
            }
            for pool in pools
        ]
    )
    return (pools,)


@app.cell
def _(HERE, np, pd, plt, pools, save_figure):
    COLOUR_AGREE = "#3b7dd8"
    COLOUR_DISAGREE = "#d1642f"

    def dilemma_stats(rows):
        """The per-dilemma statistics the figure is drawn from."""
        return (
            rows.groupby("dilemma_id")
            .agg(
                mean=("distance", "mean"),
                sd=("distance", "std"),
                n=("distance", "size"),
                agree=("agreement", lambda values: (values == "same").mean()),
            )
            .sort_values("mean")
        )

    def dilemma_figure(stats, rows, out_path, title, subtitle):
        """The ranked per-dilemma figure: the mean distance on the left, the
        judge's agreement rate on the right, both on the dilemma order."""
        labels = [dilemma.rsplit("-", 1)[-1] for dilemma in stats.index]
        positions = np.arange(len(stats))

        figure, axes = plt.subplots(
            1, 2, figsize=(13, 7), sharey=True, gridspec_kw={"width_ratios": [2, 1]}
        )

        axis = axes[0]
        axis.barh(
            positions,
            stats["mean"],
            xerr=stats["sd"],
            height=0.65,
            color=COLOUR_DISAGREE,
            error_kw={"ecolor": "#555", "elinewidth": 0.8, "capsize": 2},
        )
        axis.set_xlabel("mean cosine distance between the two answers (sd bars)")
        axis.set_title("How far apart the two models' answers are", fontsize=10)
        axis.axvline(rows["distance"].mean(), color="black", linestyle=":", linewidth=1)
        axis.text(
            rows["distance"].mean(),
            len(stats) * 0.99,
            f" overall mean {rows['distance'].mean():.3f}",
            va="top",
            fontsize=8,
        )

        axis = axes[1]
        axis.barh(positions, stats["agree"], height=0.65, color=COLOUR_AGREE)
        axis.set_xlabel("share of pairs the judge gave the same answer")
        axis.set_title("Did the two models pick the same answer?", fontsize=10)
        axis.set_xlim(0, 1)
        axis.axvline(
            (rows["agreement"] == "same").mean(),
            color="black",
            linestyle=":",
            linewidth=1,
        )
        axis.text(
            (rows["agreement"] == "same").mean(),
            len(stats) * 0.99,
            f" overall {(rows['agreement'] == 'same').mean():.0%}",
            va="top",
            fontsize=8,
        )

        axes[0].set_yticks(positions)
        axes[0].set_yticklabels(labels, fontsize=8)

        figure.suptitle(title, fontsize=11)
        figure.tight_layout(rect=(0, 0.04, 1, 1))
        figure.text(
            0.5,
            0.01,
            subtitle,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#555",
        )
        save_figure(figure, out_path)

    def comparison_distance_figure(pool, fig_dir):
        """The figure and the per-dilemma stats it is drawn from.

        One figure per model-vs-base pool, named after the pool, so the four
        fine-tunes can be compared side by side. Writes nothing when the
        comparisons have not been generated or embedded yet, so the notebook
        still runs on a fresh checkout.
        """
        rows = pool["rows"]
        if rows.empty:
            print(
                f"no comparisons could be matched to the embeddings for {pool['name']}"
            )
            return pd.DataFrame(), None
        stats = dilemma_stats(rows)
        title = "Embedding distance between the two models' answers"
        subtitle = f"Fine-tuned model: {pool['label']}"
        out_path = fig_dir / f"dilemmas-comparison-distance-{pool['name']}.png"
        dilemma_figure(stats, rows, out_path, title, subtitle)
        same = int((rows["agreement"] == "same").sum())
        print(f"Figure written to {out_path} (comparison run {pool['directory'].name})")
        print(
            f"  distance: mean {rows['distance'].mean():.3f}, "
            f"same answer {same} of {len(rows)} pairs"
        )
        return stats, out_path

    def write_comparison_figures(pools, fig_dir):
        """One distance figure per model-vs-base pool."""
        frames = []
        for pool in pools:
            stats, _ = comparison_distance_figure(pool, fig_dir)
            if stats is None:
                continue
            frames.append(stats.reset_index().assign(model=pool["label"]))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # Histogram of the comparison distances, one colour per fine-tune.
    def write_comparison_histogram(pools, fig_dir):
        """The distance distribution of every fine-tune on one coloured plot."""
        usable = [pool for pool in pools if not pool["rows"].empty]
        if not usable:
            print("no comparisons could be matched to the embeddings")
            return None
        figure, axis = plt.subplots(figsize=(8, 5))
        colours = plt.get_cmap("tab10").colors
        for position, pool in enumerate(usable):
            values = pool["rows"]["distance"]
            colour = colours[position % len(colours)]
            label = (
                pool["label"]
                .replace("_distribution-overall", "")
                .replace("_response-overall", "")
                .replace("_", " ")
            )
            axis.hist(
                values,
                bins=40,
                range=(0.0, 1.0),
                density=True,
                histtype="step",
                linewidth=1.8,
                color=colour,
                label=f"{label} (mean {values.mean():.3f})",
            )
            axis.axvline(values.mean(), color=colour, linestyle=":", linewidth=1.2)
        axis.set_xlabel("cosine distance between the two answers")
        axis.set_ylabel("share of pairs")
        axis.set_title("Comparison distance by fine-tuned model", fontsize=10)
        axis.legend(fontsize=8)
        figure.tight_layout()
        out_path = fig_dir / "dilemmas-comparison-distance-histogram.png"
        save_figure(figure, out_path)
        print(f"Figure written to {out_path}")
        return out_path

    comparison_stats_df = write_comparison_figures(pools, HERE.parent / "figures")

    write_comparison_histogram(pools, HERE.parent / "figures")

    comparison_stats_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Embeddings vs judge agreement
    How correlated are the embeddings distances with the judge's answer option agreement?
    """)
    return


@app.cell
def _(mo, pd, pools):
    def correlation_metrics(rows):
        """Correlation between embedding distance and judge option agreement."""
        scored = rows[rows["agreement"].isin(["same", "different"])].copy()
        scored["same_option"] = (scored["agreement"] == "same").astype(float)

        def metrics(group):
            distance = group["distance"]
            same = group["same_option"]
            return pd.Series(
                {
                    "n": len(group),
                    "same_option_rate": same.mean(),
                    "mean_distance": distance.mean(),
                    "mean_distance_same": distance[same == 1].mean(),
                    "mean_distance_different": distance[same == 0].mean(),
                    # Negative values mean that same-option answers tend to be
                    # closer in embedding space.
                    "pearson_r": distance.corr(same, method="pearson"),
                    "spearman_r": distance.corr(same, method="spearman"),
                }
            )

        per_dilemma = scored.groupby("dilemma_id", sort=True).apply(
            metrics, include_groups=False
        )
        overall = metrics(scored).to_frame().T
        overall.index = ["overall"]
        return per_dilemma, overall

    correlation_by_dilemma_frames = []
    correlation_overall_frames = []
    for pool in pools:
        if pool["rows"].empty:
            continue
        per_dilemma, overall = correlation_metrics(pool["rows"])
        correlation_by_dilemma_frames.append(
            per_dilemma.reset_index().assign(model=pool["label"])
        )
        correlation_overall_frames.append(
            overall.reset_index(drop=True).assign(model=pool["label"])
        )
    if correlation_overall_frames:
        correlation_by_dilemma_df = pd.concat(
            correlation_by_dilemma_frames, ignore_index=True
        ).round(3)
        correlation_overall_df = pd.concat(
            correlation_overall_frames, ignore_index=True
        ).round(3)
    else:
        print("No comparison data with embeddings is available.")
        correlation_by_dilemma_df = pd.DataFrame()
        correlation_overall_df = pd.DataFrame()

    mo.vstack([correlation_by_dilemma_df, correlation_overall_df])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Survey design

    The survey's questions are drawn from the comparison pool like this:

    1. Each model's runs on a dilemma are paired one-to-one by sample index
       (the run's `version`), so a 50-run set gives 50 pairwise comparisons
       per dilemma rather than the full cross product. This is the comparison
       a respondent sees: the same dilemma answered at the same time by two
       different agents.
    2. A dilemma where every index pair was judged the same answer option is
       removed: every comparison is an agreement, so there is no value choice
       to put to the respondent.
    3. Each pair's cosine distance is measured from the two answers'
       embeddings. A pair is dropped when its distance is below 0.20 **and**
       the judge gave both answers the same option; a close pair that the
       judge also scored the same would not show a real difference.
    4. Per dilemma, a seeded random draw takes up to 10 of the remaining
       pairs as the survey's questions.

    The tables are grouped by profile and ranked by mean cosine distance
    within each profile. **Agreement %** is the share of pairs the judge gave
    the same answer option; disagreement is the remainder. The complete table
    still lists the full-agreement dilemmas (at 100%); the selected table
    only lists dilemmas that contribute questions.

    The cells below write two tables per model - `survey-dilemma-stats-<model>.tex`
    (every index pair) and `survey-dilemma-selected-stats-<model>.tex` (the
    drawn questions) - and write each model's selected pairs as its own
    `survey/<model>/` subset. The live Qualtrics survey reads the subset
    chosen with `--model` (`survey/survey_definition.py`).
    """)
    return


@app.cell
def _(HERE, pools, random, sys):
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from generate_comparisons import write_subset

    # --- The survey design -----------------------------------------------------
    # Each pool holds every cross pair of the base model's runs and one
    # fine-tune's runs on a dilemma. The survey pairs each run of one model
    # with the run of the other at the same sample index (the run's
    # `version`), so the comparisons are one-to-one rather than the full cross
    # product. The selection then:
    #   1. drops a dilemma where every index pair was judged the same option
    #      (every pair is an agreement, so there is no value choice to ask);
    #   2. drops a pair when the cosine distance between the two answers is
    #      below the threshold and the judge gave both answers the same
    #      option;
    #   3. per dilemma, draws up to the per-dilemma cap with a seeded RNG (one
    #      RNG per dilemma, so the draw does not depend on the order the
    #      dilemmas are walked in).
    DISTANCE_THRESHOLD = 0.20
    PER_DILEMMA = 10
    SEED = 0

    def index_pairs(records):
        """One pair per sample index: the runs with the same ``version``.

        The complete pool holds every cross pair, so the same-version pairs
        are the one-to-one pairing the survey uses.
        """
        return [
            record
            for record in records
            if record["agents"]["Agent 1"].get("version")
            == record["agents"]["Agent 2"].get("version")
        ]

    def full_agreement_dilemmas(records):
        """Dilemma ids where every index pair was judged the same option."""
        by_dilemma = {}
        for record in records:
            by_dilemma.setdefault(record["dilemma_id"], []).append(record)
        return {
            dilemma_id
            for dilemma_id, group in by_dilemma.items()
            if all(record.get("same_choice") for record in group)
        }

    def select_comparisons(records, distances, threshold, per_dilemma, seed, pair):
        """The survey's questions: drop close same-judged pairs, then draw."""
        by_dilemma = {}
        for record in records:
            distance = distances.get(record["comparison_id"])
            if distance is None:
                continue
            if distance < threshold and record.get("same_choice"):
                continue
            by_dilemma.setdefault(record["dilemma_id"], []).append(record)
        chosen = set()
        for dilemma_id in sorted(by_dilemma):
            candidates = by_dilemma[dilemma_id]
            rng = random.Random(f"{seed}:{pair}:{dilemma_id}")
            picked = rng.sample(candidates, min(per_dilemma, len(candidates)))
            chosen.update(record["comparison_id"] for record in picked)
        return [record for record in records if record["comparison_id"] in chosen]

    def build_survey_designs(pools):
        """The survey selection for every model-vs-base pool."""
        designs = []
        for pool in pools:
            manifest = pool["manifest"]
            distances = dict(
                zip(
                    pool["rows"]["comparison_id"],
                    pool["rows"]["distance"],
                    strict=False,
                )
            )
            # The complete dataset: one pair per sample index.
            pool_records = index_pairs(manifest.get("comparisons", []))
            # A full-agreement dilemma is left out of the draw entirely.
            full_agreement = full_agreement_dilemmas(pool_records)
            eligible = [
                record
                for record in pool_records
                if record["dilemma_id"] not in full_agreement
            ]
            models = manifest.get("models") or {}
            pair = f"{models.get('model_a')}|{models.get('model_b')}"
            # The draw: the questions the survey actually asks.
            records = select_comparisons(
                eligible, distances, DISTANCE_THRESHOLD, PER_DILEMMA, SEED, pair
            )
            designs.append(
                {
                    "pool": pool,
                    "records": records,
                    "pool_records": pool_records,
                    "distances": distances,
                    "full_agreement": full_agreement,
                }
            )
            print(
                f"{pool['label']}: excluded {len(full_agreement)} "
                f"full-agreement dilemmas, selected {len(records)} of "
                f"{len(pool_records)} index pairs"
            )
        return designs

    survey_designs = build_survey_designs(pools)

    len(survey_designs)
    return DISTANCE_THRESHOLD, PER_DILEMMA, SEED, survey_designs, write_subset


@app.cell
def _(
    DISTANCE_THRESHOLD,
    PER_DILEMMA,
    SEED,
    escape_header,
    mo,
    np,
    pd,
    survey_designs,
    write_latex_table,
):
    def format_columns(frame):
        """Distances to two decimals, percentages to one, counts plain."""
        fmt = {}
        for column in frame.columns:
            label = escape_header(str(column))
            if column == "Mean cosine distance":
                fmt[label] = "{:.2f}"
            elif column == "Agreement %":
                fmt[label] = "{:.1f}"
            elif frame[column].dtype.kind in "iu":
                fmt[label] = "{:d}"
            else:
                fmt[label] = "{}"
        return fmt

    def survey_dilemma_stats(records, distances, with_below):
        """One row per dilemma: mean distance and the judge's split.

        ``with_below`` adds the number of pairs below the distance threshold,
        which is only meaningful for the complete dataset (the selected
        subset has already dropped some of those pairs).
        """
        rows = []
        for dilemma_id in sorted({record["dilemma_id"] for record in records}):
            group = [record for record in records if record["dilemma_id"] == dilemma_id]
            values = np.array([distances[record["comparison_id"]] for record in group])
            agreement = sum(1 for record in group if record.get("same_choice"))
            row = {
                "Profile": group[0]["profile_id"],
                "Dilemma": group[0]["situation_id"],
                "Mean cosine distance": float(values.mean()),
                "Agreement %": 100.0 * agreement / len(group) if group else np.nan,
            }
            if with_below:
                row[f"Below {DISTANCE_THRESHOLD:.2f}"] = int(
                    (values < DISTANCE_THRESHOLD).sum()
                )
            rows.append(row)
        columns = ["Profile", "Dilemma", "Mean cosine distance"]
        if with_below:
            columns.append(f"Below {DISTANCE_THRESHOLD:.2f}")
        columns.append("Agreement %")
        return pd.DataFrame(rows, columns=columns).round({"Mean cosine distance": 2})

    def ranked_table(records, distances, with_below):
        """Group by profile, then rank the dilemmas within a profile by mean
        distance. The profile/dilemma pair is the table's two-level stub."""
        return (
            survey_dilemma_stats(records, distances, with_below)
            .sort_values(
                ["Profile", "Mean cosine distance"],
                ascending=[True, False],
                kind="stable",
            )
            .set_index(["Profile", "Dilemma"])
        )

    # --- The tables ------------------------------------------------------------
    def write_survey_tables(survey_designs):
        """The complete/selected tables and the rendered notebook view."""

        def file_label(label):
            """The short table suffix: modal_response-overall -> modal."""
            return (
                label.replace("_response-overall", "")
                .replace("_distribution-overall", "")
                .replace("_", "-")
            )

        rendered_tables = []
        for design in survey_designs:
            pool = design["pool"]
            complete_table = ranked_table(
                design["pool_records"], design["distances"], with_below=True
            )
            # Only dilemmas that actually contribute questions appear in the
            # selected table (full-agreement dilemmas are excluded before the
            # draw).
            selected_table = ranked_table(
                design["records"], design["distances"], with_below=False
            )
            for frame, stem in (
                (complete_table, "survey-dilemma-stats"),
                (selected_table, "survey-dilemma-selected-stats"),
            ):
                table = frame.copy()
                table.index = table.index.set_names([None, None])
                filename = f"{stem}-{file_label(pool['label'])}.tex"
                write_latex_table(
                    table,
                    format_columns(table),
                    table.shape[1],
                    filename,
                    senses={},
                    column_format="ll" + "r" * table.shape[1],
                    stub_label="",
                )
                print(f"wrote code/figures/{filename}")
            rendered_tables.extend(
                [
                    mo.md(f"**{pool['label']}**"),
                    mo.md("Complete dataset (one pair per sample index)"),
                    complete_table,
                    mo.md("Selected subset"),
                    selected_table,
                ]
            )
        return rendered_tables

    print(
        f"design: one pair per sample index; drop distance < "
        f"{DISTANCE_THRESHOLD:.2f} judged the same; up to {PER_DILEMMA} per "
        f"dilemma (seed {SEED})"
    )
    mo.vstack(write_survey_tables(survey_designs))
    return


@app.cell
def _(DISTANCE_THRESHOLD, PER_DILEMMA, SEED, survey_designs, write_subset):

    # --- The survey's questions ------------------------------------------------
    # Each model's selected comparisons become its own `survey/<model>/`
    # subset; the live Qualtrics survey reads the one chosen with
    # `--model` (survey/survey_definition.py).
    def write_survey_subsets(survey_designs):
        """Write one selected-question subset per model."""
        for design in survey_designs:
            pool = design["pool"]
            manifest = pool["manifest"]
            models = manifest.get("models") or {}
            write_subset(
                pool["directory"] / "survey" / pool["name"],
                pool["name"],
                design["records"],
                manifest.get("dilemmas", []),
                {
                    "run_id": manifest.get("run_id", pool["directory"].name),
                    "label": pool["label"],
                    "models": {
                        "model_a": models.get("model_a"),
                        "model_b": models.get("model_b"),
                    },
                    "selection": {
                        "rule": (
                            "one pair per sample index; remove full-agreement "
                            "dilemmas; drop pairs below the distance threshold "
                            "judged the same; then a seeded random draw per "
                            "dilemma"
                        ),
                        "distance_threshold": DISTANCE_THRESHOLD,
                        "per_dilemma": PER_DILEMMA,
                        "seed": SEED,
                        "pool": "complete (one pair per sample index)",
                        "pool_size": len(design["pool_records"]),
                        "selected": len(design["records"]),
                    },
                },
            )
            print(
                f"wrote the {pool['label']} survey subset to "
                f"{pool['directory'] / 'survey' / pool['name'] / 'comparisons.json'} "
                f"({len(design['records'])} questions)"
            )

    write_survey_subsets(survey_designs)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Situational awareness

    How much the models notice that they are being tested rather than acting as
    deployed agents.

    The scan (see `sa_scanner.py` and the README) scores every scanned response
    0-4. This cell pools those scores by model x work profile and writes the
    heatmap to `code/figures/awareness-profile-heatmap.png`. The score says how
    often a model broke frame; it is not a measure of how good its answers
    were.
    """)
    return


@app.cell
def _(
    HERE,
    json,
    newest_matching,
    np,
    pd,
    pivot_means_counts,
    plt,
    read_json,
    save_figure,
    scan_results_df,
):
    SCANNER_KEY = "situational_awareness"  # scanner name from sa_scanner.py

    def as_mapping(value):
        """Scout result metadata columns arrive as dict or JSON string."""
        if isinstance(value, str):
            return json.loads(value)
        return value if isinstance(value, dict) else {}

    def scan_frame(scan_dir):
        """The scanned rows joined with the eval sample metadata, and the
        number of scans that failed."""
        results = scan_results_df(str(scan_dir))
        df = results.scanners[SCANNER_KEY].copy()
        failed = int(df["scan_error"].notna().sum())
        df = df[df["scan_error"].isna()].copy()
        sample_metadata = [
            as_mapping(meta).get("sample_metadata") or {}
            for meta in df["transcript_metadata"]
        ]
        df["profile_id"] = [meta.get("profile_id", "") for meta in sample_metadata]
        df["dilemma_id"] = [meta.get("dilemma_id", "") for meta in sample_metadata]
        df["model"] = [model.split("/")[-1] for model in df["transcript_model"]]
        return df, failed

    def profile_order(profiles_dir):
        """The live profile ids in profiles/ file order. The file stem and
        the profile id inside the JSON differ for some profiles
        (clergy_advisor.json holds profile 'parish_advisor'), and the scans
        are keyed by the latter."""
        return [
            read_json(path)["profile"] for path in sorted(profiles_dir.glob("*.json"))
        ]

    def plot_heatmap(means, counts, out_path, title, xlabel, dpi=150):
        """The heatmap: a row per model, a column per category, the cell
        colour and the number both the mean score. Grey cells have no data."""
        grid = means.to_numpy(dtype=float)
        blocked = np.ma.masked_invalid(grid)
        figure, axis = plt.subplots(
            figsize=(
                max(4.0, 0.3 * len(means.columns) + 2.8),
                max(2.2, 0.45 * len(means.index) + 1.4),
            )
        )
        cmap = plt.colormaps["YlOrRd"].copy()
        cmap.set_bad("#e8e8e8")
        image = axis.imshow(blocked, cmap=cmap, vmin=0, vmax=4, aspect="auto")
        figure.colorbar(image, ax=axis, label="mean awareness score (0-4)")
        axis.set_xticks(range(len(means.columns)))
        axis.set_xticklabels(means.columns, rotation=90, fontsize=6.5)
        axis.set_yticks(range(len(means.index)))
        axis.set_yticklabels(means.index, fontsize=8)
        n_table = counts.notna().to_numpy()
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                if n_table[row, col]:
                    axis.text(
                        col,
                        row,
                        f"{grid[row, col]:.1f}",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                    )
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("model")
        figure.tight_layout()
        save_figure(figure, out_path, dpi=dpi)
        return out_path

    def awareness_profile_heatmap(scans_dir, profiles_dir, fig_dir):
        """The model x profile mean awareness heatmap.

        Returns an empty frame and writes nothing when the logs have not
        been scanned yet, so the notebook still runs on a fresh checkout.
        """
        scan_dir = newest_matching(scans_dir, "scan_id=*")
        if scan_dir is None:
            print(f"no scan under {scans_dir} - run scout scan first")
            return pd.DataFrame(), None
        df, failed = scan_frame(scan_dir)
        if df.empty:
            print(f"no scored transcripts in {scan_dir}")
            return pd.DataFrame(), None
        print(f"scan {scan_dir.name}: {len(df)} transcripts scored, {failed} failed")
        # Rows are ordered by each model's mean score across the dilemmas;
        # the columns are the live profiles in profiles/ file order, then
        # any retired ones.
        dilemma_means, _ = pivot_means_counts(df, "dilemma_id")
        row_order = dilemma_means.mean(axis=1).sort_values(ascending=False).index
        means, counts = pivot_means_counts(df, "profile_id")
        live = [
            profile
            for profile in profile_order(profiles_dir)
            if profile in means.columns
        ]
        retired = sorted(profile for profile in means.columns if profile not in live)
        means = means.reindex(index=row_order, columns=live + retired)
        counts = counts.reindex(index=means.index, columns=means.columns)
        out_path = fig_dir / "awareness-profile-heatmap.png"
        plot_heatmap(
            means,
            counts,
            out_path,
            "Situational-awareness scan: model x profile (pooled means)",
            "profile",
        )
        print(f"Figure written to {out_path}")
        return means, out_path

    awareness_profile_df, awareness_figure_path = awareness_profile_heatmap(
        HERE / "output" / "scans", HERE / "profiles", HERE.parent / "figures"
    )

    awareness_profile_df
    return


if __name__ == "__main__":
    app.run()
