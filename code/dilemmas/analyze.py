import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd

    # Shared LaTeX table helpers live in code/latex_tables.py.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from latex_tables import write_latex_table, write_text_table

    HERE = Path(__file__).resolve().parent
    return HERE, json, np, pd, write_latex_table, write_text_table


@app.cell
def _(HERE, json, pd, write_text_table):
    # Reference table for the rest of the analysis: the eleven value tensions
    # every handwritten dilemma is written around. Dilemmas, judge results and
    # comparisons all point at a tension by id (the `tension` field), so this
    # table is the shared key for the sections below. The third value option
    # in the JSON is always "Other or unclear", so it is left out here. The
    # named form ("Security–personal: ...") stays in tensions.json for the
    # judge and comparisons; the report cells carry only the definition, so
    # the columns stay narrow.
    # The table is also written to code/figures/dilemmas-tensions.tex for the
    # report to import with \ctable{dilemmas-tensions}{...}.

    def value_definition(value: str) -> str:
        """Drop the leading value name so the cell is just the definition."""
        return value.split(": ", 1)[1] if ": " in value else value

    def tensions_table() -> pd.DataFrame:
        """One row per tension: the two sides it sets against each other and
        the guidance for reading a decision as that tension."""
        tensions = json.loads((HERE / "tensions.json").read_text(encoding="utf-8"))
        rows = []
        for tension in tensions:
            first, second = tension["value_options"][:2]
            rows.append(
                {
                    "#": tension["id"],
                    "Tension": tension["name"],
                    "First value": value_definition(first),
                    "Second value": value_definition(second),
                    "Guidance": tension["guidance"],
                }
            )
        return pd.DataFrame(rows)

    tensions_df = tensions_table()
    write_text_table(
        tensions_df,
        "dilemmas-tensions.tex",
        column_format="@{}rp{3.0cm}p{3.3cm}p{3.3cm}p{5.6cm}@{}",
        ragged_columns=("Tension", "First value", "Second value", "Guidance"),
    )

    tensions_df
    return (tensions_df,)


@app.cell
def _(HERE, json, np, pd, write_latex_table):
    # Per-dilemma answer statistics from the exported runs.
    #
    # "Mean answer" is the mean chosen option index (1 = the first option, up
    # to and including "Other or unclear"), which locates where the answers
    # sit. "Stability" is 1 - H / ln(k), where H is the Shannon entropy of the
    # chosen indices over the repetitions and k is the number of options the
    # dilemma offered: 1.00 means every run chose the same option, 0.00 means
    # the answers were spread evenly across the options. Entropy reads the
    # whole distribution, so fading in and out of one option is scored
    # differently from a clean split, which a modal share cannot show. With
    # three repetitions the metric is coarse.
    #
    # The table is also written to code/figures/dilemmas-answer-stats.tex for
    # the report to import with \ctable{dilemmas-answer-stats}{...}.
    def answer_stats() -> pd.DataFrame:
        runs_dir = HERE / "output" / "runs"
        manifest_path = runs_dir / "index.json"
        if not manifest_path.exists():
            print(
                f"No exported runs yet: {manifest_path} is missing. "
                "Run `uv run python code/dilemmas/export_results.py` first."
            )
            return pd.DataFrame()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        options = {d["situation_id"]: d["answer_options"] for d in manifest["dilemmas"]}
        rows = []
        for run in manifest["runs"]:
            judge = json.loads(
                (runs_dir / run["path"] / "judge.json").read_text(encoding="utf-8")
            )
            choice = (judge.get("metadata") or {}).get("choice") or judge.get("answer")
            answer_options = options[run["situation_id"]]
            rows.append(
                {
                    "Profile": run["profile_id"],
                    "Dilemma": run["situation_id"],
                    "Model": run["model"].split("/")[-1],
                    "Index": float(answer_options.index(choice) + 1)
                    if choice in answer_options
                    else np.nan,
                }
            )
        runs = pd.DataFrame(rows).dropna(subset=["Index"])
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
        stats = pd.concat([mean_answer, stability], axis=1).unstack("Model")
        return stats.round(2)

    answer_stats_df = answer_stats()

    if not answer_stats_df.empty:
        # The LaTeX figure keeps multi-level columns: the top header row
        # groups the two statistics and the second row holds the models, like
        # the fine-tuning tables. The stub columns are unnamed so the index
        # names do not add a header row of their own.
        answer_stats_tex_df = answer_stats_df.copy()
        answer_stats_tex_df.index = answer_stats_tex_df.index.set_names([None, None])
        answer_stats_tex_df.columns = answer_stats_tex_df.columns.set_names(
            [None, None]
        )
        write_latex_table(
            answer_stats_tex_df,
            "{:.2f}",
            answer_stats_tex_df.shape[1],
            "dilemmas-answer-stats.tex",
            senses={"Stability": "max"},
            column_format="ll" + "r" * answer_stats_tex_df.shape[1],
            stub_label="",
        )

    answer_stats_df
    return (answer_stats_df,)


if __name__ == "__main__":
    app.run()
