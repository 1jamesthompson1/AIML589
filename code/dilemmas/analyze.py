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

    ## The runs the analysis reads

    The run ids are set in the cell below, and every other cell reads its
    data through them, so pointing the analysis at another run is a one-line
    change:

    - **MAIN_RUN_ID** - the headline run. The modal adapter and the base
      model both have 50 runs per dilemma, so the main H1 test, the
      per-dilemma results and the survey design come from here.
    - **PREVIOUS_RUN_ID** - the run before it, which holds all four
      fine-tuned adapters at 10 runs per dilemma, so the four-model H1 test
      and the answer-statistics and agreement tables come from here.
    - **FRONTIER_RUN_ID** - the only run holding the frontier comparison
      models.
    """)
    return


@app.cell
def _():
    # The exported runs the analysis reads. Every cell takes its data from
    # these ids, so changing the analysis to another run is a one-line edit.
    MAIN_RUN_ID = (
        "2026-10-23_dc41eb9e"  # headline run (base + modal at 50 runs per dilemma)
    )
    PREVIOUS_RUN_ID = (
        "2026-10-22_0a63e425"  # the run holding all four adapters (10 runs per dilemma)
    )
    FRONTIER_RUN_ID = "2026-10-16_initial-full-runs"  # the frontier comparison models
    return FRONTIER_RUN_ID, MAIN_RUN_ID, PREVIOUS_RUN_ID


@app.cell
def _():
    import json
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
        escape_header,
        json,
        mo,
        np,
        pd,
        plt,
        scan_results_df,
        sys,
        write_latex_table,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Shared helpers

    The two analyses at the end of this notebook both read one exported run,
    read a JSON file, pivot a model x category table and write a figure to
    `code/figures/`, so those four steps live here. The run each analysis
    reads is named at the top of the notebook.
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
    the models diverge on the dilemma. The runs come from the run named for
    the model set at the top of the notebook (the study set uses the run
    holding all five models, the frontier set the run holding the frontier
    models).

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
def _(FRONTIER_RUN_ID, HERE, PREVIOUS_RUN_ID, json, np, pd, write_latex_table):
    def load_manifest(run_id: str):
        """The manifest of one exported run, and the directory its files sit in.

        A run's manifest lives under `output/runs/full/<run-id>/index.json`,
        with the run paths relative to that directory. The active export
        (`output/runs/index.json`) holds the same manifest for the active
        run, with the paths relative to `output/runs/`, and is used when it
        is the run asked for and the bundle's own manifest is missing.

        Returns (manifest, base, path); manifest is None when the run has
        not been exported.
        """
        runs_dir = HERE / "output" / "runs"
        bundle = runs_dir / "full" / run_id / "index.json"
        if bundle.exists():
            return json.loads(bundle.read_text(encoding="utf-8")), bundle.parent, bundle
        active = runs_dir / "index.json"
        if active.exists():
            manifest = json.loads(active.read_text(encoding="utf-8"))
            if manifest.get("active_run_id") == run_id:
                return manifest, runs_dir, active
        return None, None, None

    def load_run_choices(
        model_labels: dict[str, str], run_id: str
    ) -> tuple[pd.DataFrame, dict]:
        """The exported runs of one run, one row per run.

        The run is the one asked for (the ids at the top of the notebook),
        and inside it the most recent run set per (profile, dilemma, model),
        so superseded runs are not counted twice. Both the answer statistics
        and the aggregate below read their data from here, so the two always
        describe the same runs.

        Returns (runs, options):
        - runs: Profile, Dilemma, Model (the label), Created and Index,
          where Index is the 1-based position of the judged answer in the
          dilemma's answer options. Runs without a judged answer are
          dropped.
        - options: dilemma id -> the dilemma's answer options.
        """
        manifest, base, manifest_path = load_manifest(run_id)
        if manifest is None:
            print(
                f"No exported runs for run {run_id!r} under "
                f"{HERE / 'output' / 'runs'}. Run `uv run python "
                f"code/dilemmas/export_results.py` first."
            )
            return pd.DataFrame(), {}
        # The models the run does not hold: the tables are built from the
        # ones it has, and the missing ones are named so the gap is visible.
        missing = [
            f"{model_labels[model]} ({model})"
            for model in model_labels
            if model not in {run["model"] for run in manifest["runs"]}
        ]
        if missing:
            print(
                f"note: run {run_id} does not hold {len(missing)} of the "
                f"{len(model_labels)} requested models, so they are left out of "
                f"the tables: {', '.join(missing)}"
            )
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
        model_labels: dict[str, str], model_order: list[str], run_id: str
    ) -> pd.DataFrame:
        runs, options = load_run_choices(model_labels, run_id)
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
        # base model. The run is the one that holds all five models.
        "dilemmas-answer-stats": {
            "description": "fine-tuned Qwen3.8-27B adapters and base",
            "run": PREVIOUS_RUN_ID,
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
        # were also run on. Only this run holds them.
        "dilemmas-answer-stats-frontier": {
            "description": "frontier comparison models",
            "run": FRONTIER_RUN_ID,
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
        df = answer_stats(spec["labels"], spec["order"], spec["run"])
        if df.empty:
            result[spec["description"]] = df
            continue
        write_stats_table(df, spec["order"], filename + ".tex")
        result[spec["description"]] = merge_answer_stats(df, spec["order"])
    answer_stats_df = result["fine-tuned Qwen3.8-27B adapters and base"]

    answer_stats_df
    return MODEL_SETS, load_manifest, load_run_choices


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
def _(mo):
    mo.md(r"""
    def answer_agreement(
        model_labels: dict[str, str],
        model_order: list[str],
        base: str | None = None,
        run_id: str = "\",
    ) -> pd.DataFrame:
        "\"\"The same/different counts for every pair of models.

        One row per model pair: the number of dilemmas compared, how many
        the two models' modal answers matched ("Same") and how many did not
        ("Different"), the same counts as a percentage, the run-pair
        agreement percentage and the mean TVD.

        The pairs against the base model come first (each adapter against
        the model it was fine-tuned from), then the fine-tuned models
        against each other; within each group the most alike pair is
        first.
        "\"\"
        runs, _ = load_run_choices(model_labels, run_id)
        if runs.empty:
            return pd.DataFrame()
        keys = ["Profile", "Dilemma", "Model"]
        # The models the export holds: a model that is missing is left out
        # of the aggregate (and named), so a partial export still produces a
        # table of the pairs it can compare.
        available = set(runs["Model"].unique())
        missing = [model for model in model_order if model not in available]
        if missing:
            print(
                f"note: {len(missing)} of the {len(model_order)} models are not "
                f"in the export, so their pairs are left out of the agreement "
                f"table: {', '.join(missing)}"
            )
        model_order = [model for model in model_order if model in available]
        if len(model_order) < 2:
            return pd.DataFrame()
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
        # The answer options the runs chose between.
        options = sorted(counts["Index"].unique())
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
                for _, dilemma_row in counts.groupby(["Profile", "Dilemma"]):
                    left_counts = (
                        dilemma_row[dilemma_row["Model"] == left]
                        .set_index("Index")["Runs"]
                        .reindex(options)
                        .fillna(0.0)
                    )
                    right_counts = (
                        dilemma_row[dilemma_row["Model"] == right]
                        .set_index("Index")["Runs"]
                        .reindex(options)
                        .fillna(0.0)
                    )
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
        "\"\"The LaTeX table the report can import with \\ctable{...}."\"\"
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
            stub_label="\",
        )

    study_set = MODEL_SETS["dilemmas-answer-stats"]
    agreement_pairs_df = answer_agreement(
        study_set["labels"],
        study_set["order"],
        study_set.get("base"),
        study_set["run"],
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
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## H1: do the fine-tuned models answer differently from the base model?

    The pre-specified test of H1. Each of the four fine-tuned models is
    compared separately with the base model it was fine-tuned from. The
    section runs the test twice, on the two runs named at the top of the
    notebook:

    - the **four-model test** on the run that holds all four fine-tuned
      adapters (10 runs per dilemma): the overall effect and the p-value of
      each adapter against the base model;
    - the **main test** on the main run, where the modal adapter was re-run
      at 50 runs per dilemma: the headline result for the modal adapter, and
      the per-dilemma results (the per-dilemma table and figure are written
      from this run only).

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
    MAIN_RUN_ID,
    MODEL_SETS,
    PREVIOUS_RUN_ID,
    json,
    load_run_choices,
    pd,
    sys,
    write_latex_table,
):
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import h1_test

    h1_set = MODEL_SETS["dilemmas-answer-stats"]

    def h1_run_test(runs, options, model_order):
        """The test and its sensitivity check on one run's runs."""
        if runs.empty or h1_set["base"] not in set(runs["Model"].unique()):
            print(
                f"note: the base model {h1_set['base']!r} is not in the run, "
                f"so the H1 test is skipped."
            )
            empty = h1_test.H1Results(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
            return empty, empty
        results = h1_test.run_h1(
            runs,
            options,
            base_label=h1_set["base"],
            model_order=model_order,
            replicates=h1_test.REPLICATES,
            seed=h1_test.SEED,
        )
        # The sensitivity check: the same test with the "Other or unclear"
        # runs dropped, so only the runs the judge could place on one of the
        # dilemma's substantive options count.
        sensitivity = h1_test.run_h1(
            h1_test.drop_unclear(runs, options),
            options,
            base_label=h1_set["base"],
            model_order=model_order,
            replicates=h1_test.REPLICATES,
            seed=h1_test.SEED,
        )
        return results, sensitivity

    # The four-model test: every fine-tuned adapter against the base model,
    # from the run that holds all five models.
    previous_runs, previous_options = load_run_choices(
        h1_set["labels"], PREVIOUS_RUN_ID
    )
    h1_order = [label for label in h1_set["order"] if label != h1_set["base"]]
    h1_results, h1_sensitivity = h1_run_test(previous_runs, previous_options, h1_order)

    # The main test: the modal adapter against the base model from the main
    # run (50 runs per dilemma). Only the models that run holds are tested.
    main_runs, main_options = load_run_choices(h1_set["labels"], MAIN_RUN_ID)
    main_available = set(main_runs["Model"].unique()) if not main_runs.empty else set()
    main_order = [
        label
        for label in h1_set["order"]
        if label != h1_set["base"] and label in main_available
    ]
    h1_main_results, h1_main_sensitivity = h1_run_test(
        main_runs, main_options, main_order
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

    def h1_dilemma_frame(results, model_order):
        """The per-dilemma results: the TVD and its p-value per model."""
        if results.per_dilemma.empty:
            return pd.DataFrame()
        frame = pd.DataFrame(index=sorted(results.per_dilemma["Dilemma"].unique()))
        for model in model_order:
            subset = results.per_dilemma[
                results.per_dilemma["Model"] == model
            ].set_index("Dilemma")
            frame[f"{model} TVD"] = subset["TVD"]
            frame[f"{model} p"] = subset["p"]
        frame.index = frame.index.set_names(["Dilemma"])
        return frame

    h1_summary_df = h1_summary_frame(h1_results)
    h1_sensitivity_df = h1_summary_frame(h1_sensitivity)
    # The per-dilemma results come from the main run only.
    h1_dilemma_df = h1_dilemma_frame(h1_main_results, main_order)
    h1_main_summary_df = h1_summary_frame(h1_main_results)
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
    if not h1_main_summary_df.empty:
        # The main test: the modal adapter on the main run.
        h1_write_summary(h1_main_summary_df, "dilemmas-h1-summary-main.tex")
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
            "four_model_run": PREVIOUS_RUN_ID,
            "main_run": MAIN_RUN_ID,
            "main_models": main_order,
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
        "main": {
            "summary": json.loads(h1_main_results.summary.to_json(orient="records")),
            "per_dilemma": json.loads(
                h1_main_results.per_dilemma.to_json(orient="records")
            ),
            "other_rates": json.loads(
                h1_main_results.other_rates.to_json(orient="records")
            ),
            "sensitivity": {
                "summary": json.loads(
                    h1_main_sensitivity.summary.to_json(orient="records")
                ),
                "per_dilemma": json.loads(
                    h1_main_sensitivity.per_dilemma.to_json(orient="records")
                ),
            },
        },
    }
    h1_output = HERE / "output" / "h1-results.json"
    h1_output.parent.mkdir(parents=True, exist_ok=True)
    h1_output.write_text(json.dumps(h1_payload, indent=2) + "\n", encoding="utf-8")
    return h1_main_results, main_order


@app.cell
def _(HERE, h1_main_results, main_order, plt, save_figure):
    h1_fig_dir = HERE.parent / "figures"

    if h1_main_results.per_dilemma.empty:
        print("no exported runs to plot yet")
    else:
        # The per-dilemma TVD of each model: where the differences sit. The
        # per-dilemma figure is written from the main run only.
        models = main_order or sorted(h1_main_results.per_dilemma["Model"].unique())
        tvds = h1_main_results.per_dilemma.pivot(
            index="Dilemma", columns="Model", values="TVD"
        )[models]
        ps = h1_main_results.per_dilemma.pivot(
            index="Dilemma", columns="Model", values="p"
        )[models]
        order = tvds.mean(axis=1).sort_values(ascending=False).index
        tvds, ps = tvds.loc[order], ps.loc[order]
        grid = tvds.to_numpy(dtype=float)
        figure, axis = plt.subplots(figsize=(7, 10))
        image = axis.imshow(grid, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        figure.colorbar(image, ax=axis, label="TVD between the two models")
        axis.set_xticks(range(len(models)))
        axis.set_xticklabels(models, fontsize=9)
        axis.set_yticks(range(len(tvds)))
        axis.set_yticklabels(tvds.index, fontsize=6)
        for row in range(len(tvds)):
            for column in range(len(models)):
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
        print(f"Figures written to {h1_dilemma_figure}")
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
def _(HERE, PREVIOUS_RUN_ID, json, load_manifest, np, pd):
    def answer_distribution(run_id: str) -> pd.DataFrame:
        manifest, base, _ = load_manifest(run_id)
        if manifest is None:
            print(f"no exported runs for run {run_id!r}")
            return pd.DataFrame()
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
                run_dir = base / run["path"]
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

    answer_distribution_df = answer_distribution(PREVIOUS_RUN_ID)
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

    The data is the main run's per-model pools (the run id at the top of the
    notebook): one pool per fine-tune against the base model, each holding
    every pair of the two models' runs on each dilemma. Each pool gets its
    own figure, named after the fine-tune, so the four models can be
    compared side by side.
    """)
    return


@app.cell
def _(HERE, MAIN_RUN_ID, json, np, pd, read_json):
    def pair_rows(manifest, vectors, run_ids):
        """One row per comparison: the distance and the judge's verdict."""
        index = {run_id: position for position, run_id in enumerate(run_ids)}
        rows = []
        for record in manifest.get("comparisons", []):
            base = record["agents"]["base"]
            ft = record["agents"]["ft"]
            if base["run_id"] not in index or ft["run_id"] not in index:
                continue
            similarity = float(
                vectors[index[base["run_id"]]] @ vectors[index[ft["run_id"]]]
            )
            choice1, choice2 = base.get("choice", ""), ft.get("choice", "")
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

    def load_quality(directory, name):
        """The judge-quality scores for one pool's index pairs, if present.

        ``judge_quality.py`` writes the single-judge ``scores.json`` when it
        finishes and the multi-judge ``ensemble/scores.json`` beside it. Prefer
        the ensemble when it exists, and use the ensemble's mean metric as the
        ranking/display value (the median and per-judge scores stay in the
        frame for audit). Otherwise use the single-judge compiled file, then
        the JSONL cache with the latest prompt version per comparison. An empty
        frame means the pool has no judge scores yet.
        """
        quality_dir = directory / "quality" / "index" / name
        ensemble = quality_dir / "ensemble" / "scores.json"
        if ensemble.exists():
            frame = pd.DataFrame(read_json(ensemble).get("scores", []))
            for metric in ("value_divergence", "survey_quality", "interest"):
                mean_column = f"{metric}_mean"
                if mean_column not in frame.columns:
                    raise ValueError(
                        f"{ensemble} has no {mean_column!r}; the survey ranks "
                        "by the ensemble mean, so this file is not usable"
                    )
                frame[f"{metric}_median"] = frame[metric]
                frame[metric] = frame[mean_column]
            return frame
        compiled = quality_dir / "scores.json"
        if compiled.exists():
            return pd.DataFrame(read_json(compiled).get("scores", []))
        cache = quality_dir / "scores.jsonl"
        if not cache.exists():
            return pd.DataFrame()
        rows = [
            json.loads(line)
            for line in cache.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows)
        if "prompt_version" in frame.columns and len(frame):
            frame = frame[frame["prompt_version"] == frame["prompt_version"].iloc[-1]]
        frame = frame.drop_duplicates("comparison_id", keep="last")
        return frame

    def load_pools(comparisons_dir, run_id):
        """Every embedded comparison pool of the run named at the top.

        Each entry is the pool manifest, its embeddings path, the pair
        distances and the model label. The figure, the agreement correlation
        and the survey design all walk this one list, so they always describe
        the same data.

        Only the pinned run is read. A missing run, a missing pool, or a pool
        without ``vectors.npz`` is a hard error: silently reading another
        run's pools can make the figures and the live survey describe a
        different run from the one named in the notebook.
        """
        directory = comparisons_dir / run_id
        if not directory.is_dir():
            raise RuntimeError(
                f"no comparison run {run_id!r} under {comparisons_dir}; "
                "set MAIN_RUN_ID or run generate_comparisons.py first"
            )
        pool_specs = pool_paths(directory)
        if not pool_specs:
            raise RuntimeError(
                f"run {run_id!r} has no comparison pools under {directory}; "
                "run generate_comparisons.py first"
            )
        missing = [
            name
            for name, _manifest_path, embeddings_dir in pool_specs
            if not (embeddings_dir / "vectors.npz").exists()
        ]
        if missing:
            raise RuntimeError(
                f"run {run_id!r} is missing embeddings for pool(s) "
                f"{missing}; run embed_responses.py for this run, or fix "
                "MAIN_RUN_ID. Refusing to use another run's pools."
            )
        pools = []
        for name, manifest_path, embeddings_dir in pool_specs:
            vectors_path = embeddings_dir / "vectors.npz"
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
                    "quality": load_quality(directory, name),
                }
            )
        return pools

    pools = load_pools(HERE / "output" / "comparisons", MAIN_RUN_ID)

    pd.DataFrame(
        [
            {
                "pool": pool["name"],
                "model": pool["label"],
                "comparisons": len(pool["rows"]),
                "quality scores": len(pool["quality"]),
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Judge quality

    Using a LLM-as-Judge to evaluate the quality of the comparisons
    """)
    return


@app.cell
def _(h1_main_results):
    main_per_dilemma = h1_main_results.per_dilemma
    model_rows = main_per_dilemma[
        main_per_dilemma["Model"].str.lower().str.contains("modal")
    ]
    significance = dict(zip(model_rows["Dilemma"], model_rows["p"] < 0.05))
    significance
    return


@app.cell
def _(h1_main_results, pd, pools, write_latex_table):
    # Table of each dilemmas survey quality, interest and difference scores, one row per dilemma with mean and variance of each value (variance in braces). Just for the modal model in mainline run. A column has a true and false for if the model is statistically significant in divergence from the base model.

    def quality_dilemma_table(pools, model_marker="modal"):
        """Mean (variance) judge scores plus H1 significance per dilemma."""
        modal_pools = [pool for pool in pools if model_marker in pool["label"].lower()]
        if not modal_pools:
            print("No modal-model comparison pool found.")
            return pd.DataFrame()

        quality = modal_pools[0]["quality"].copy()
        if quality.empty:
            print("No judge-quality scores found for the modal model.")
            return pd.DataFrame()

        significance = {}
        per_dilemma = h1_main_results.per_dilemma
        if not per_dilemma.empty:
            model_rows = per_dilemma[
                per_dilemma["Model"].str.lower().str.contains(model_marker)
            ]
            significance = dict(zip(model_rows["Dilemma"], model_rows["p"] < 0.05))

        score_columns = {
            "Survey quality": "survey_quality",
            "Interest": "interest",
            "Difference": "value_divergence",
        }
        rows = []
        for dilemma, group in quality.groupby("dilemma_id", sort=True):
            row = {"Dilemma": dilemma}
            for label, column in score_columns.items():
                values = pd.to_numeric(group[column], errors="coerce").dropna()
                row[label] = (
                    f"{values.mean():.2f} ({values.var(ddof=1):.2f})"
                    if len(values) > 1
                    else "--"
                )
            row["H1 significant"] = significance.get(dilemma.split("-")[1], "error")
            rows.append(row)

        return pd.DataFrame(rows).set_index("Dilemma")

    modal_quality_table = quality_dilemma_table(pools)

    write_latex_table(
        modal_quality_table,
        {col: "{}" for col in modal_quality_table.columns},
        len(modal_quality_table.columns),
        "dilemmas-modal-quality.tex",
        senses={col: "max" for col in modal_quality_table.columns},
        column_format="l" + "r" * len(modal_quality_table.columns),
        stub_label="Dilemma",
    )

    significance_effect = modal_quality_table.groupby("H1 significant").agg(
        count=("H1 significant", "count"),
        mean_survey_quality=(
            "Survey quality",
            lambda x: x.str.extract(r"([\d.]+)").astype(float).mean(),
        ),
        mean_interest=(
            "Interest",
            lambda x: x.str.extract(r"([\d.]+)").astype(float).mean(),
        ),
        mean_difference=(
            "Difference",
            lambda x: x.str.extract(r"([\d.]+)").astype(float).mean(),
        ),
    )
    write_latex_table(
        significance_effect,
        {col: "{}" for col in significance_effect.columns},
        len(significance_effect.columns),
        "dilemmas-modal-quality-significance.tex",
        senses={col: "max" for col in significance_effect.columns},
        column_format="l" + "r" * len(significance_effect.columns),
        stub_label="H1 significant",
    )

    modal_quality_table
    return


@app.cell
def _(HERE, MAIN_RUN_ID, np, pd, plt, save_figure, survey_designs):
    # One figure per dilemma: the judge's value-divergence and survey-quality
    # scores for the complete index pool and for the pairs selected into the
    # survey. The survey pairs are a subset of the complete pool, so the two
    # histograms share the same 1-5 bins. Written under
    # code/dilemmas/output/quality/<run-id>/<pool>/<dilemma>.png.
    def quality_histogram_rows(records, quality):
        rows = []
        for record in records:
            score = quality.get(record["comparison_id"])
            if not score:
                continue
            divergence = pd.to_numeric(score.get("value_divergence"), errors="coerce")
            survey_quality = pd.to_numeric(score.get("survey_quality"), errors="coerce")
            if pd.isna(divergence) or pd.isna(survey_quality):
                continue
            rows.append(
                {
                    "dilemma_id": record["situation_id"],
                    "value_divergence": float(divergence),
                    "survey_quality": float(survey_quality),
                }
            )
        return pd.DataFrame(
            rows, columns=["dilemma_id", "value_divergence", "survey_quality"]
        )

    def write_quality_histograms(survey_designs, fig_dir):
        """Write one complete-vs-survey quality figure per dilemma."""
        bins = np.arange(0.5, 6.5, 1.0)
        paths = []
        for design in survey_designs:
            quality = design["quality"]
            complete = quality_histogram_rows(design["pool_records"], quality)
            survey = quality_histogram_rows(design["records"], quality)
            if complete.empty and survey.empty:
                print(
                    f"{design['pool']['label']}: no scored pairs, so no "
                    "quality histograms were written"
                )
                continue
            model_dir = fig_dir / design["pool"]["name"]
            for dilemma_id in sorted(
                set(complete["dilemma_id"]) | set(survey["dilemma_id"])
            ):
                complete_rows = complete[complete["dilemma_id"] == dilemma_id]
                survey_rows = survey[survey["dilemma_id"] == dilemma_id]
                figure, axes = plt.subplots(1, 2, figsize=(10, 4))
                for axis, column, title in (
                    (axes[0], "value_divergence", "Value divergence"),
                    (axes[1], "survey_quality", "Survey quality"),
                ):
                    complete_values = complete_rows[column].to_numpy()
                    survey_values = survey_rows[column].to_numpy()
                    if len(complete_values):
                        axis.hist(
                            complete_values,
                            bins=bins - 0.2,
                            width=0.38,
                            color="#3b7dd8",
                            label=(
                                f"complete index pool (n={len(complete_values)}, "
                                f"mean {complete_values.mean():.2f})"
                            ),
                        )
                    if len(survey_values):
                        axis.hist(
                            survey_values,
                            bins=bins + 0.2,
                            width=0.38,
                            color="#d1642f",
                            label=(
                                f"survey selection (n={len(survey_values)}, "
                                f"mean {survey_values.mean():.2f})"
                            ),
                        )
                    else:
                        axis.text(
                            0.5,
                            0.5,
                            "no scored survey pairs",
                            transform=axis.transAxes,
                            ha="center",
                            va="center",
                            color="grey",
                            fontsize=9,
                        )
                    axis.set_xlim(0.4, 5.6)
                    axis.set_xticks([1, 2, 3, 4, 5])
                    axis.set_xlabel("judge score (1-5)")
                    axis.set_ylabel("pairs")
                    axis.set_title(title, fontsize=10)
                    axis.grid(axis="y", alpha=0.2)
                    if len(complete_values) or len(survey_values):
                        axis.legend(fontsize=7, loc="upper right")
                figure.suptitle(
                    f"{dilemma_id} · {design['pool']['label']}", fontsize=11
                )
                figure.tight_layout()
                out_path = model_dir / f"{dilemma_id}.png"
                save_figure(figure, out_path)
                paths.append(out_path)
        return paths

    quality_histogram_dir = HERE / "output" / "quality" / MAIN_RUN_ID
    quality_histogram_paths = write_quality_histograms(
        survey_designs, quality_histogram_dir
    )
    print(
        f"wrote {len(quality_histogram_paths)} quality histograms to "
        f"{quality_histogram_dir}"
    )
    pd.DataFrame(
        {"figure": [str(path.relative_to(HERE)) for path in quality_histogram_paths]}
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Survey design

    The survey's questions are drawn from the comparison pool like this:

        \item Pair each run of one model with the run of the other model at the same sample index. This gives one pairwise comparison per sample index, so a 50-run set gives 50 pairwise comparisons for each dilemma (1,500 across the 30 dilemmas when both models have 50 runs).
        \item Remove any dilemma where the two models' response distributions do not differ significantly. Significance is the H1 test's per-dilemma permutation p-value from the main run (p < .05).
        \item For each remaining dilemma, keep the 17 pairwise comparisons the judge rated most divergent (highest `value_divergence`). When the high-reasoning ensemble is available, its mean `value_divergence` is the ranking score; otherwise the single-judge score is used.

    The selection deliberately favours the pairs whose value judgements differ
    most, so it is not a random sample of the dilemma's pairs. Ties are broken
    by the judge's survey quality, then interest, then the comparison id. A
    dilemma with fewer than 17 scored pairs aborts the build instead of being
    filled with an unranked pair, so every live question is genuinely in the
    top 17 by the judge/ensemble score. The judge's scores are also reported: **Mean
    divergence** and **Mean survey quality** are the judge/ensemble means, and
    **Strong %** is the share meeting divergence >= 4, survey quality >= 4 and
    both competent. **Agreement %** is the share of pairs the judge gave the
    same answer option; the complete table still lists non-significant
    dilemmas so they can be checked even though they are not drawn.

    The cells below write five tables per model: the two survey-dilemma
    tables, the filtering-stage summary and the per-dilemma and per-profile
    survivor matrices. They also write each model's selected pairs as its own
    `survey/<model>/` subset. The live Qualtrics survey reads the subset
    chosen with `--model` (`survey/survey_definition.py`).
    """)
    return


@app.cell
def _(HERE, h1_main_results, pd, pools, sys):
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from generate_comparisons import validate_comparison_roles, write_subset

    # --- The survey design -----------------------------------------------------
    # Each pool holds every cross pair of the base model's runs and one
    # fine-tune's runs on a dilemma. The survey pairs each run of one model
    # with the run of the other at the same sample index (the run's inspect
    # `epoch`, falling back to `version` for older manifests), so the
    # comparisons are one-to-one rather than the full cross product. The
    # selection then:
    #   1. keeps only the dilemmas where the fine-tune's response distribution
    #      differs significantly from the base model's (the H1 per-dilemma
    #      permutation test, p < P_THRESHOLD);
    #   2. keeps the PER_DILEMMA pairs per dilemma with the highest LLM judge
    #      value divergence, so the survey asks about the sharpest value
    #      disagreements.
    PER_DILEMMA = 17
    P_THRESHOLD = 0.05
    # The judge's value_divergence ranks the selection; "strong" is the
    # judge's own summary of a pair (divergence >= 4 and survey quality >= 4).
    STRONG_QUALITY = 4

    def run_index_key(agent):
        """The run's pairing key: inspect ``epoch``, else export ``version``.

        ``epoch`` is the original sample index. Older exports only carry
        ``version`` (the chronological export number), so keep that as a
        compatibility fallback; mixing the two would pair different runs.
        """
        epoch = agent.get("epoch")
        if epoch is not None:
            return ("epoch", epoch)
        version = agent.get("version")
        return ("version", version) if version is not None else (None, None)

    def index_pairs(records):
        """One pair per sample index: runs with the same ``epoch`` (or version).

        The complete pool holds every cross pair, so the same-index pairs are
        the one-to-one pairing the survey uses. A record whose two sides use
        different key spaces (one epoch, one version) is not a pair.
        """
        pairs = []
        for record in records:
            base_key = run_index_key(record["agents"]["base"])
            ft_key = run_index_key(record["agents"]["ft"])
            if base_key[0] is not None and base_key == ft_key:
                pairs.append(record)
        return pairs

    def h1_model_label(pool_label):
        """The H1 model label for a pool label.

        The pools are named after the adapter
        (``modal_response-overall``) and the H1 test labels the models by
        their short name (``modal``), so the pool label is reduced to that
        name.
        """
        short = (
            pool_label.replace("_response", "")
            .replace("_distribution", "")
            .replace("_", "-")
        )
        return short.removesuffix("-overall")

    def significant_dilemmas(h1_results, pool_label):
        """The situation ids where the pool's fine-tune differs from the base.

        The test is the H1 per-dilemma permutation test from the main run:
        a dilemma is kept when its p-value is below ``P_THRESHOLD``. None is
        returned when the run has no H1 result for the model, so the caller
        can say so instead of silently keeping every dilemma.
        """
        if h1_results is None or h1_results.per_dilemma.empty:
            return None
        model = h1_model_label(pool_label)
        rows = h1_results.per_dilemma[
            h1_results.per_dilemma["Model"].str.lower() == model.lower()
        ]
        if rows.empty:
            return None
        return set(rows.loc[rows["p"] < P_THRESHOLD, "Dilemma"])

    def modal_answers(manifest):
        """The modal judged answer per model per dilemma.

        Each run of a model is counted once (the pool holds every cross pair,
        so a run shows up in many records), and the option with the most
        runs wins. A tie goes to the option that comes first in the
        dilemma's answer options - the same rule the answer-agreement
        table uses - so the pick is deterministic.
        """
        options = {
            f"{d['profile_id']}-{d['situation_id']}": d.get("answer_options", [])
            for d in manifest.get("dilemmas", [])
        }
        # (dilemma, model, run) -> the judged option.
        choices = {}
        for record in manifest.get("comparisons", []):
            for agent in record.get("agents", {}).values():
                choice = agent.get("choice")
                if not choice:
                    continue
                key = (record["dilemma_id"], agent.get("model", ""), agent["run_id"])
                choices[key] = choice
        counts = {}
        for (dilemma_id, model, _), choice in choices.items():
            counts.setdefault(dilemma_id, {}).setdefault(model, {})
            counts[dilemma_id][model][choice] = (
                counts[dilemma_id][model].get(choice, 0) + 1
            )
        modal = {}
        for dilemma_id, by_model in counts.items():
            order = {
                option: position
                for position, option in enumerate(options.get(dilemma_id, []))
            }
            modal[dilemma_id] = {}
            for model, counter in by_model.items():
                modal[dilemma_id][model] = max(
                    counter.items(),
                    key=lambda item: (
                        item[1],
                        -order.get(item[0], len(order)),
                    ),
                )[0]
        return modal

    def modal_match(modal, dilemma_id):
        """Whether the pool's models share a modal answer on the dilemma.

        True when every model's modal answer is the same, False when they
        differ, and None when the dilemma has fewer than two models (so the
        table shows ``--`` rather than a meaningless match).
        """
        answers = list(modal.get(dilemma_id, {}).values())
        if len(answers) < 2:
            return None
        return len(set(answers)) == 1

    def quality_map(frame):
        """comparison_id -> the judge's quality row as a plain dict."""
        if frame is None or frame.empty:
            return {}
        return frame.set_index("comparison_id").to_dict("index")

    def divergence_rank(record, quality):
        """The within-dilemma order: highest LLM value divergence first.

        Ties are broken by the judge's survey quality, then its interest
        score, then the comparison id, so the pick is deterministic.
        """
        row = quality.get(record["comparison_id"]) or {}
        return (
            float(row.get("value_divergence") or 0),
            float(row.get("survey_quality") or 0),
            float(row.get("interest") or 0),
            record["comparison_id"],
        )

    def top_divergence_records(records, quality, per_dilemma, pair):
        """Keep the ``per_dilemma`` highest-divergence questions per dilemma.

        The rank is the LLM judge's/ensemble's ``value_divergence`` score.
        Every selected slot must be a real score: if a dilemma has fewer than
        ``per_dilemma`` scored pairs, the build fails loudly instead of
        filling slots with an unranked random pair (which would silently stop
        the live survey being "top N by divergence").
        """
        by_dilemma = {}
        for record in records:
            by_dilemma.setdefault(record["dilemma_id"], []).append(record)
        chosen = set()
        for dilemma_id in sorted(by_dilemma):
            candidates = by_dilemma[dilemma_id]

            def has_divergence(record):
                value = (quality.get(record["comparison_id"]) or {}).get(
                    "value_divergence"
                )
                return value is not None and not pd.isna(value)

            scored = [record for record in candidates if has_divergence(record)]
            if len(scored) < per_dilemma:
                raise ValueError(
                    f"{pair}: only {len(scored)} scored comparison(s) for "
                    f"{dilemma_id}; {per_dilemma} are required for the survey. "
                    "Run judge_quality.py for this pool (and make sure every "
                    "judge succeeded) before rebuilding the survey subset."
                )
            picked = sorted(
                scored,
                key=lambda record: divergence_rank(record, quality),
                reverse=True,
            )[:per_dilemma]
            chosen.update(record["comparison_id"] for record in picked)
        return [record for record in records if record["comparison_id"] in chosen]

    def stage_counts(records, key):
        counts = {}
        for record in records:
            value = record[key]
            counts[value] = counts.get(value, 0) + 1
        return counts

    def filter_stages(pool_records, significant, quality, pair):
        """Counts and per-dilemma/profile survivors at every filtering stage.

        The stage order matches ``top_divergence_records``: the significance
        filter, then the top-divergence selection.
        """
        stages = []
        current = list(pool_records)

        def add_stage(name, before, after):
            before_dilemmas = stage_counts(before, "dilemma_id")
            after_dilemmas = stage_counts(after, "dilemma_id")
            before_profiles = stage_counts(before, "profile_id")
            after_profiles = stage_counts(after, "profile_id")
            dropped_dilemmas = [
                (dilemma_id, count)
                for dilemma_id, count in before_dilemmas.items()
                if dilemma_id not in after_dilemmas
            ]
            dropped_profiles = [
                (profile, count)
                for profile, count in before_profiles.items()
                if profile not in after_profiles
            ]
            affected_dilemmas = [
                dilemma_id
                for dilemma_id, count in before_dilemmas.items()
                if after_dilemmas.get(dilemma_id, 0) < count
            ]
            affected_profiles = [
                profile
                for profile, count in before_profiles.items()
                if after_profiles.get(profile, 0) < count
            ]
            stages.append(
                {
                    "Stage": name,
                    "Start": len(before),
                    "Removed": len(before) - len(after),
                    "Remaining": len(after),
                    "Dilemmas": len(after_dilemmas),
                    "Profiles": len(after_profiles),
                    "Dilemmas affected": len(affected_dilemmas),
                    "Profiles affected": len(affected_profiles),
                    "Dropped dilemmas": ", ".join(
                        f"{dilemma_id} ({count})"
                        for dilemma_id, count in sorted(dropped_dilemmas)
                    )
                    or "--",
                    "Dropped profiles": ", ".join(
                        f"{profile} ({count})"
                        for profile, count in sorted(dropped_profiles)
                    )
                    or "--",
                    "dilemma_counts": after_dilemmas,
                    "profile_counts": after_profiles,
                }
            )

        add_stage("Index pairs", current, current)

        before = current
        current = [
            record for record in current if record["situation_id"] in significant
        ]
        add_stage(f"Not significant (p >= {P_THRESHOLD})", before, current)

        before = current
        current = top_divergence_records(current, quality, PER_DILEMMA, pair)
        add_stage(f"Top {PER_DILEMMA} by divergence", before, current)
        return stages

    def stage_matrices(stages, pool_records):
        """Survivors per dilemma/profile at each stage, with zeros visible."""
        stage_names = [stage["Stage"] for stage in stages]
        dilemma_index = {}
        for record in pool_records:
            dilemma_index[record["dilemma_id"]] = (
                record["profile_id"],
                record["situation_id"],
            )
        dilemma_rows = [
            [stage["dilemma_counts"].get(dilemma_id, 0) for stage in stages]
            for dilemma_id in sorted(dilemma_index)
        ]
        dilemma_matrix = pd.DataFrame(
            dilemma_rows,
            index=pd.MultiIndex.from_tuples(
                [dilemma_index[dilemma_id] for dilemma_id in sorted(dilemma_index)],
                names=["Profile", "Dilemma"],
            ),
            columns=stage_names,
        )
        profiles = sorted({record["profile_id"] for record in pool_records})
        profile_matrix = pd.DataFrame(
            [
                [stage["profile_counts"].get(profile, 0) for stage in stages]
                for profile in profiles
            ],
            index=pd.Index(profiles, name="Profile"),
            columns=stage_names,
        )
        return dilemma_matrix, profile_matrix

    def build_survey_designs(pools, h1_results):
        """The survey selection for every model-vs-base pool."""
        designs = []
        for pool in pools:
            manifest = pool["manifest"]
            rows = pool["rows"]
            distances = (
                dict(
                    zip(
                        rows["comparison_id"],
                        rows["distance"],
                        strict=False,
                    )
                )
                if {"comparison_id", "distance"} <= set(rows.columns)
                else {}
            )
            quality = quality_map(pool.get("quality"))
            # The complete dataset: one pair per sample index.
            pool_records = index_pairs(manifest.get("comparisons", []))
            # The dilemmas where the fine-tune differs from the base. A
            # missing H1 result selects nothing, so the gap is visible in the
            # stage table instead of being filled with every dilemma.
            significant = significant_dilemmas(h1_results, pool["label"])
            if significant is None:
                print(
                    f"{pool['label']}: no H1 per-dilemma result for this model "
                    "in the main run, so no questions are selected."
                )
                significant = set()
            models = manifest.get("models") or {}
            pair = f"{models.get('model_a')}|{models.get('model_b')}"
            records = top_divergence_records(
                [
                    record
                    for record in pool_records
                    if record["situation_id"] in significant
                ],
                quality,
                PER_DILEMMA,
                pair,
            )
            stages = filter_stages(pool_records, significant, quality, pair)
            dilemma_matrix, profile_matrix = stage_matrices(stages, pool_records)
            selection_mode = (
                "top-divergence-ensemble"
                if pool.get("quality") is not None
                and not pool["quality"].empty
                and "value_divergence_mean" in pool["quality"].columns
                else "top-divergence"
            )
            designs.append(
                {
                    "pool": pool,
                    "records": records,
                    "pool_records": pool_records,
                    "distances": distances,
                    "quality": quality,
                    "selection_mode": selection_mode,
                    "significant": significant,
                    "stages": stages,
                    "dilemma_matrix": dilemma_matrix,
                    "profile_matrix": profile_matrix,
                    "modal": modal_answers(manifest),
                }
            )
            dilemmas = {record["situation_id"] for record in pool_records}
            print(
                f"{pool['label']}: {selection_mode} selection, kept "
                f"{len(significant)} of {len(dilemmas)} significant dilemmas, "
                f"selected {len(records)} of {len(pool_records)} index pairs"
            )
        return designs

    survey_designs = build_survey_designs(pools, h1_main_results)

    len(survey_designs)
    return (
        PER_DILEMMA,
        P_THRESHOLD,
        STRONG_QUALITY,
        modal_match,
        survey_designs,
        validate_comparison_roles,
        write_subset,
    )


@app.cell
def _(
    PER_DILEMMA,
    P_THRESHOLD,
    STRONG_QUALITY,
    escape_header,
    mo,
    modal_match,
    np,
    pd,
    survey_designs,
    write_latex_table,
):
    def format_columns(frame):
        """Distances/divergence to two decimals, percentages to one, counts plain."""
        fmt = {}
        for column in frame.columns:
            label = escape_header(str(column))
            if column in (
                "Mean cosine distance",
                "Mean divergence",
                "Mean survey quality",
            ):
                fmt[label] = "{:.2f}"
            elif column in ("Agreement %", "Strong %"):
                fmt[label] = "{:.1f}"
            elif frame[column].dtype.kind in "iu":
                fmt[label] = "{:d}"
            else:
                fmt[label] = "{}"
        return fmt

    def survey_dilemma_stats(records, distances, quality, modal):
        """One row per dilemma: distance, judge divergence and agreement.

        ``quality`` is the comparison_id -> judge-score mapping. The means
        are blank when the pool has no judge scores. ``Strong %`` is the
        share of pairs with divergence >= 4, survey quality >= 4 and both
        answers competent.
        """
        rows = []
        for dilemma_id in sorted({record["dilemma_id"] for record in records}):
            group = [record for record in records if record["dilemma_id"] == dilemma_id]
            values = np.array(
                [distances.get(record["comparison_id"], np.nan) for record in group],
                dtype=float,
            )
            agreement = sum(1 for record in group if record.get("same_choice"))
            scored = [
                quality.get(record["comparison_id"])
                for record in group
                if quality.get(record["comparison_id"])
            ]
            divergences = [
                float(row["value_divergence"])
                for row in scored
                if pd.notna(row.get("value_divergence"))
            ]
            survey_values = [
                float(row["survey_quality"])
                for row in scored
                if pd.notna(row.get("survey_quality"))
            ]
            strong = sum(
                1
                for row in scored
                if row.get("both_competent")
                and float(row.get("value_divergence") or 0) >= 4
                and float(row.get("survey_quality") or 0) >= STRONG_QUALITY
            )
            row = {
                "Profile": group[0]["profile_id"],
                "Dilemma": group[0]["situation_id"],
                "Mean cosine distance": (
                    float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
                ),
                "Mean divergence": float(np.mean(divergences))
                if divergences
                else np.nan,
                "Mean survey quality": float(np.mean(survey_values))
                if survey_values
                else np.nan,
                "Strong %": 100.0 * strong / len(scored) if scored else np.nan,
                "Agreement %": 100.0 * agreement / len(group) if group else np.nan,
            }
            # Whether the models' modal answers on this dilemma match.
            match = modal_match(modal, dilemma_id)
            row["Modal match"] = pd.NA if match is None else match
            rows.append(row)
        columns = [
            "Profile",
            "Dilemma",
            "Mean cosine distance",
            "Mean divergence",
            "Mean survey quality",
            "Strong %",
            "Agreement %",
            "Modal match",
        ]
        frame = pd.DataFrame(rows, columns=columns)
        # The match column holds True/False with a missing value, so it
        # stays an object column (the table shows `--` for the missing one).
        frame["Modal match"] = frame["Modal match"].astype(object)
        return frame.round(
            {
                "Mean cosine distance": 2,
                "Mean divergence": 2,
                "Mean survey quality": 2,
            }
        )

    def ranked_table(records, distances, quality, modal):
        """Group by profile, then rank the dilemmas within a profile by mean
        distance. The profile/dilemma pair is the table's two-level stub."""
        return (
            survey_dilemma_stats(records, distances, quality, modal)
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
                design["pool_records"],
                design["distances"],
                design["quality"],
                design["modal"],
            )
            # Only dilemmas that actually contribute questions appear in the
            # selected table (non-significant dilemmas are excluded before the
            # draw).
            selected_table = ranked_table(
                design["records"],
                design["distances"],
                design["quality"],
                design["modal"],
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

            # The stage summary: how many comparisons survive each filter.
            stage_table = pd.DataFrame(
                [
                    {
                        key: value
                        for key, value in stage.items()
                        if key not in ("dilemma_counts", "profile_counts")
                    }
                    for stage in design["stages"]
                ]
            ).set_index("Stage")
            stage_table.index.name = None
            stage_tex = stage_table.copy()
            for column in ("Dropped dilemmas", "Dropped profiles"):
                stage_tex[column] = stage_tex[column].map(escape_header)
            stage_name = f"survey-filter-stages-{file_label(pool['label'])}.tex"
            write_latex_table(
                stage_tex,
                format_columns(stage_tex),
                stage_tex.shape[1],
                stage_name,
                senses={},
                column_format="l" + "r" * 7 + "p{4.2cm}p{4.2cm}",
                stub_label="",
            )
            print(f"wrote code/figures/{stage_name}")

            # The same numbers per dilemma and per profile, with zeros for
            # the stage that drops them.
            dilemma_matrix = design["dilemma_matrix"].copy()
            dilemma_matrix.index = dilemma_matrix.index.set_names([None, None])
            dilemma_name = f"survey-filter-dilemmas-{file_label(pool['label'])}.tex"
            write_latex_table(
                dilemma_matrix,
                {escape_header(str(c)): "{:d}" for c in dilemma_matrix.columns},
                dilemma_matrix.shape[1],
                dilemma_name,
                senses={},
                column_format="ll" + "r" * dilemma_matrix.shape[1],
                stub_label="",
            )
            print(f"wrote code/figures/{dilemma_name}")

            profile_matrix = design["profile_matrix"].copy()
            profile_matrix.index.name = None
            profile_name = f"survey-filter-profiles-{file_label(pool['label'])}.tex"
            write_latex_table(
                profile_matrix,
                {escape_header(str(c)): "{:d}" for c in profile_matrix.columns},
                profile_matrix.shape[1],
                profile_name,
                senses={},
                column_format="l" + "r" * profile_matrix.shape[1],
                stub_label="",
            )
            print(f"wrote code/figures/{profile_name}")

            rendered_tables.extend(
                [
                    mo.md(f"**{pool['label']}**"),
                    mo.md(
                        f"Selection mode: **{design['selection_mode']}** "
                        f"(judge quality coverage: "
                        f"{len(design['quality'])}/{len(design['pool_records'])} index pairs)"
                    ),
                    mo.md("Complete dataset (one pair per sample index)"),
                    complete_table,
                    mo.md("Selected subset"),
                    selected_table,
                    mo.md("Filtering stages"),
                    stage_table,
                ]
            )
        return rendered_tables

    print(
        "design: one pair per sample index; keep only dilemmas whose H1 "
        f"per-dilemma permutation p is below {P_THRESHOLD}, then keep the top "
        f"{PER_DILEMMA} pairs per dilemma by the judge/ensemble mean "
        "value_divergence"
    )
    mo.vstack(write_survey_tables(survey_designs))
    return


@app.cell
def _():
    # Give examples of the different levels of divergence.
    return


@app.cell
def _(
    PER_DILEMMA,
    P_THRESHOLD,
    survey_designs,
    validate_comparison_roles,
    write_subset,
):

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
            for record in design["records"]:
                validate_comparison_roles(record)
            write_subset(
                pool["directory"] / "survey" / pool["name"],
                pool["name"],
                design["records"],
                manifest.get("dilemmas", []),
                {
                    "run_id": manifest.get("run_id", pool["directory"].name),
                    "label": pool["label"],
                    "models": {
                        # Role names the survey reads; model_a/model_b keep
                        # the raw pool provenance.
                        "base": models.get("base") or models.get("model_a"),
                        "ft": models.get("ft") or models.get("model_b"),
                        "model_a": models.get("model_a"),
                        "model_b": models.get("model_b"),
                    },
                    "selection": {
                        "rule": (
                            "one pair per sample index; keep only dilemmas "
                            "where the H1 per-dilemma permutation p is below "
                            f"{P_THRESHOLD}; then keep the top {PER_DILEMMA} "
                            "pairs per dilemma by the "
                            + (
                                "3-judge high-reasoning ensemble mean value_divergence"
                                if design["selection_mode"].endswith("ensemble")
                                else "LLM judge's value_divergence"
                            )
                        ),
                        "selection_mode": design["selection_mode"],
                        "p_threshold": P_THRESHOLD,
                        "divergence_field": (
                            "value_divergence_mean"
                            if design["selection_mode"].endswith("ensemble")
                            else "value_divergence"
                        ),
                        "significant_dilemmas": len(design["significant"]),
                        "judge_quality_scores": len(design["quality"]),
                        "per_dilemma": PER_DILEMMA,
                        "unscored_fallback": "none: a dilemma with fewer than "
                        "per_dilemma scored pairs aborts the survey build",
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
