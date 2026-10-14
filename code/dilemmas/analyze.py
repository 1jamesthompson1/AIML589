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
    # every handwritten dilemma is written around. Dilemmas and judge results
    # point at a tension by id (the `tension` field), so this table is the
    # shared key for the sections below. The third value option
    # in the JSON is always "Other or unclear", so it is left out here. The
    # named form ("Security–personal: ...") stays in tensions.json for the
    # judge; the report cells carry only the definition, so
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
    # Per-dilemma answer statistics from the exported runs, ranked by how far
    # the two models diverge on the dilemma.
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
    # "Divergence" is the mean pairwise total variation distance between the
    # models' distributions of chosen option indices: 0.00 = identical
    # distributions, 1.00 = no shared option. Rows are sorted by it, so the
    # most model-divergent scenarios sit at the top.
    #
    # Edited dilemmas keep their superseded runs in the manifest; each
    # (profile, dilemma, model) is scored on its most recent run set only, so
    # the statistics use the option list the index currently publishes.
    #
    # The table is also written to code/figures/dilemmas-answer-stats.tex for
    # the report to import with \ctable{dilemmas-answer-stats}{...}.
    def answer_stats() -> pd.DataFrame:
        # Only the study models appear in the table: the fine-tuned
        # Qwen3.8-27B, its base (local vLLM or the OpenRouter service), and
        # the two frontier comparison models. Spot-test extras
        # (claude-haiku-5.5, gpt-5.6-luna) stay out; add an entry here when
        # the study models change (the label is the column name, so keep the
        # Qwen entries sorting before the frontier ones).
        model_labels = {
            "openai/Qwen3.8-27B-nz-wvs-full_string_distribution-overall": (
                "Qwen3.8-27B FT"
            ),
            "openai/Qwen/Qwen3.8-27B": "Qwen3.8-27B base",
            "openrouter/qwen/qwen3.8-27b": "Qwen3.8-27b",
            "openrouter/deepseek/deepseek-v4.1-flash": "deepseek-v4.1-flash",
            "openrouter/z-ai/glm-5.3-flash": "glm-5.3-flash",
        }
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
            if run["model"] not in model_labels:
                continue
            judge = json.loads(
                (runs_dir / run["path"] / "judge.json").read_text(encoding="utf-8")
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
        stats = stats.sort_values(
            ("Divergence", "TVD"), ascending=False, na_position="last", kind="stable"
        )
        return stats.round(2)

    answer_stats_df = answer_stats()

    if not answer_stats_df.empty:
        # The LaTeX figure keeps multi-level columns: the top header row
        # groups the statistics and the second row holds the models, like
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


@app.cell
def _(HERE, json, np, pd):
    # Overall answer distribution per dilemma, pooled across the models' latest
    # run sets: the debugging view for how balanced a dilemma is. A one-sided
    # dilemma (balance 0.00 - every run picked the same option) says little
    # about a model; a spread means the models are actually splitting. Run
    # sets whose stored text/options differ from the current profile file are
    # skipped as stale (they measured an earlier version of the scenario).
    # Columns 1-4 are option indices in the dilemma's answer_options ("--"
    # when it has fewer options); Balance is H / ln(k): 1.00 = even spread,
    # 0.00 = one option.
    def answer_distribution() -> pd.DataFrame:
        runs_dir = HERE / "output" / "runs"
        manifest = json.loads((runs_dir / "index.json").read_text(encoding="utf-8"))
        current = {}
        for profile_path in sorted((HERE / "profiles").glob("*.json")):
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            for dilemma in profile["dilemmas"]:
                current[dilemma["id"]] = {
                    "profile_id": profile["profile"],
                    "tension": dilemma["tension"],
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
            row = {
                "Profile": entry["profile_id"],
                "Dilemma": situation_id,
                "T": entry["tension"],
                "Runs": total,
                "Balance": round(max(balance, 0.0), 2),
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
    return (answer_distribution_df,)


if __name__ == "__main__":
    app.run()
