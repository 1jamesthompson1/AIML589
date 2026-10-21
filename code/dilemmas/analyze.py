import marimo

__generated_with = "0.23.14"
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
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from inspect_scout import scan_results_df

    # Shared LaTeX table helpers live in code/latex_tables.py.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from latex_tables import write_latex_table

    HERE = Path(__file__).resolve().parent
    return (
        HERE,
        Path,
        json,
        mo,
        np,
        pd,
        plt,
        scan_results_df,
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
    def answer_stats(
        model_labels: dict[str, str], model_order: list[str]
    ) -> pd.DataFrame:
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
            return pd.DataFrame()
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
            return pd.DataFrame()
        runs = pd.DataFrame(rows).dropna(subset=["Index"])
        keys = ["Profile", "Dilemma", "Model"]
        # Superseded runs stay in the manifest for history; score each
        # (profile, dilemma, model) on its most recent run set only.
        latest = runs.groupby(keys)["Created"].transform("max")
        runs = runs[runs["Created"] == latest]
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

    MODEL_SETS = {
        # The study set: all four fine-tuned Qwen3.8-27B adapter variants
        # ("overall" adapters served from the local vLLM server) plus their
        # base model.
        "dilemmas-answer-stats": {
            "description": "fine-tuned Qwen3.8-27B adapters and base",
            "labels": {
                "openai/Qwen3.8-27B-nz-wvs-full_string_distribution-overall": (
                    "FT full-string"
                ),
                "openai/Qwen3.8-27B-nz-wvs-first_token_distribution-overall": (
                    "FT first-token"
                ),
                "openai/Qwen3.8-27B-nz-wvs-modal_response-overall": "FT modal",
                "openai/Qwen3.8-27B-nz-wvs-sampled_response-overall": "FT sampled",
                "openai/Qwen/Qwen3.8-27B": "Qwen3.8-27B base",
            },
            "order": [
                "FT full-string",
                "FT sampled",
                "FT modal",
                "FT first-token",
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

    def write_stats_table(stats: pd.DataFrame, filename: str) -> None:
        # The LaTeX figure keeps multi-level columns: the top header row
        # groups the statistics and the second row holds the models, like
        # the fine-tuning tables. The stub columns are unnamed so the index
        # names do not add a header row of their own.
        stats_tex_df = stats.copy()
        stats_tex_df.index = stats_tex_df.index.set_names([None, None])
        stats_tex_df.columns = stats_tex_df.columns.set_names([None, None])
        write_latex_table(
            stats_tex_df,
            "{:.2f}",
            stats_tex_df.shape[1],
            filename,
            senses={"Stability": "max"},
            column_format="ll" + "r" * stats_tex_df.shape[1],
            stub_label="",
        )

    result: dict[str, pd.DataFrame] = {}
    for filename, spec in MODEL_SETS.items():
        df = answer_stats(spec["labels"], spec["order"])
        if not df.empty:
            write_stats_table(df, filename + ".tex")
        result[spec["description"]] = df
    answer_stats_df = result["fine-tuned Qwen3.8-27B adapters and base"]

    answer_stats_df
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

    The figure (`code/figures/dilemmas-comparison-distance-dilemmas.png`) ranks
    the dilemmas by their mean distance and puts the judge's agreement rate
    beside it. The agreement rate is the check on the measure: a dilemma where
    the two models pick different answers should sit further apart than one
    where they agree. The distance says how different the answers are, not
    which answer is better - the survey is what asks that.

    The data is the newest comparison run's "complete" subset: every pair of
    the two models' runs on each dilemma.
    """)
    return


@app.cell
def _(HERE, newest_matching, np, pd, plt, read_json, save_figure):
    COLOUR_AGREE = "#3b7dd8"
    COLOUR_DISAGREE = "#d1642f"

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
                    "dilemma_id": record["dilemma_id"],
                    "distance": 1.0 - similarity,
                    "agreement": agreement,
                }
            )
        return pd.DataFrame(rows)

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

    def dilemma_figure(stats, rows, out_path, title):
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
        figure.tight_layout()
        save_figure(figure, out_path)

    def comparison_distance_figure(comparisons_dir, fig_dir):
        """The figure and the per-dilemma stats it is drawn from.

        Returns an empty frame and writes nothing when the comparisons have
        not been generated or embedded yet, so the notebook still runs on a
        fresh checkout.
        """
        directory = newest_matching(
            comparisons_dir,
            "*",
            needs=("complete/comparisons.json", "embeddings/complete/vectors.npz"),
        )
        if directory is None:
            print(
                f"no complete comparisons with embeddings under {comparisons_dir} - "
                "run generate_comparisons.py and embed_responses.py first"
            )
            return pd.DataFrame(), None
        manifest_path = directory / "complete" / "comparisons.json"
        embeddings_path = directory / "embeddings" / "complete" / "vectors.npz"
        manifest = read_json(manifest_path)
        cache = np.load(embeddings_path)
        rows = pair_rows(manifest, cache["vectors"], [str(v) for v in cache["run_ids"]])
        if rows.empty:
            print(f"no comparisons could be matched to {embeddings_path}")
            return pd.DataFrame(), None
        stats = dilemma_stats(rows)
        meta = read_json(embeddings_path.with_name("meta.json"), {})
        model = meta.get("model", "the embedding model")
        revision = meta.get("revision", "")
        title = (
            f"Embedding distance between the two models' answers "
            f"({model}{' @ ' + revision[:8] if revision else ''}, "
            f"complete set, {len(rows)} comparisons)"
        )
        out_path = fig_dir / "dilemmas-comparison-distance-dilemmas.png"
        dilemma_figure(stats, rows, out_path, title)
        same = int((rows["agreement"] == "same").sum())
        print(f"Figure written to {out_path} (comparison run {directory.name})")
        print(
            f"  distance: mean {rows['distance'].mean():.3f}, "
            f"same answer {same} of {len(rows)} pairs"
        )
        return stats, out_path

    comparison_stats_df, comparison_figure_path = comparison_distance_figure(
        HERE / "output" / "comparisons", HERE.parent / "figures"
    )

    comparison_stats_df
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
