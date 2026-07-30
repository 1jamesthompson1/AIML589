import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import json
    from pathlib import Path

    return Path, json, mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Analyze

    The purpose of this notebook is to interrogate the evaluation output found in `output/evals` and generate figures and statistics for reports.
    """)
    return


@app.cell
def _(Path, json):
    # Reading in the evaluation results
    EVALS_DIR = Path("output/evals")

    def load_evals():
        evals = []
        if not EVALS_DIR.exists():
            return evals
        for model_dir in sorted(EVALS_DIR.iterdir()):
            if not model_dir.is_dir():
                continue
            for run_dir in sorted(model_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                config_path = run_dir / "config.json"
                if not config_path.exists():
                    continue
                config = json.loads(config_path.read_text())
                evals.append(
                    {
                        "model": model_dir.name,
                        "model_full": config.get("target", ""),
                        "dataset": config.get("dataset", ""),
                        "population": config.get("population", ""),
                        "run_name": run_dir.name,
                        "timestamp": config.get("timestamp", ""),
                        "reasoning": config.get("reasoning", None),
                        "num_test_examples": config.get("num_test_examples", None),
                        "config_path": config_path,
                        "results_path": run_dir / "per_question_results.csv",
                        "figures_path": run_dir / "figures",
                    }
                )
        return evals

    evals = load_evals()
    len(evals)
    return (evals,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Is fine-tuning robust to reasoning vs non-reasoning prompts?

    The finetuning is done with reasoning turned off. Does this finetuning have an effect even when reasoning is turned on?

    To answer this question, we can compare the evaluation results of a model and see if the effect of finetuning is consistent across reasoning and non-reasoning runs.
    """)
    return


@app.cell
def _(evals, mo, pd):
    def _make_pairs(evals):
        result = []
        for e in evals:
            if e["reasoning"]:
                continue
            key = (e["model"], e["dataset"], e["population"])
            for other in evals:
                if (
                    other["reasoning"]
                    and (other["model"], other["dataset"], other["population"]) == key
                ):
                    result.append({"no_reasoning": e, "with_reasoning": other})
        return result

    _pairs = _make_pairs(evals)
    rows = []
    for pair in _pairs:
        nr = pair["no_reasoning"]
        wr = pair["with_reasoning"]
        try:
            nr_df = pd.read_csv(nr["results_path"])
            wr_df = pd.read_csv(wr["results_path"])
        except FileNotFoundError:
            continue

        def _summarize(df):
            ce = (
                df["cross_entropy"].dropna().mean()
                if "cross_entropy" in df.columns
                else None
            )
            kl = (
                df["kl_divergence"].dropna().mean()
                if "kl_divergence" in df.columns
                else None
            )
            if "expected_text" in df.columns and "model_answer" in df.columns:
                acc = (
                    df["expected_text"].str.strip().str.lower()
                    == df["model_answer"].str.strip().str.lower()
                ).mean() * 100
            else:
                acc = None
            return ce, kl, acc

        nr_ce, nr_kl, nr_acc = _summarize(nr_df)
        wr_ce, wr_kl, wr_acc = _summarize(wr_df)

        def pick(v_nr, v_wr, lower_better=True):
            if v_nr is None or v_wr is None:
                return ""
            if lower_better:
                return (
                    "reasoning"
                    if v_wr < v_nr
                    else "no_reasoning"
                    if v_wr > v_nr
                    else "tie"
                )
            return (
                "reasoning" if v_wr > v_nr else "no_reasoning" if v_wr < v_nr else "tie"
            )

        rows.append(
            {
                "model": nr["model"],
                "dataset": nr["dataset"],
                "population": nr["population"],
                "no_reasoning_run": nr["run_name"],
                "with_reasoning_run": wr["run_name"],
                "nr_cross_entropy": round(nr_ce, 4) if nr_ce is not None else "",
                "wr_cross_entropy": round(wr_ce, 4) if wr_ce is not None else "",
                "ce_better": pick(nr_ce, wr_ce),
                "nr_kl": round(nr_kl, 4) if nr_kl is not None else "",
                "wr_kl": round(wr_kl, 4) if wr_kl is not None else "",
                "kl_better": pick(nr_kl, wr_kl),
                "nr_acc": f"{nr_acc:.1f}%" if nr_acc is not None else "",
                "wr_acc": f"{wr_acc:.1f}%" if wr_acc is not None else "",
                "acc_better": pick(nr_acc, wr_acc, lower_better=False),
            }
        )

    performance_comparison = pd.DataFrame(rows)
    mo.ui.table(performance_comparison)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Efficacy of fine-tuning

    There are three sorts of fine-tuning that has been explored. SFT using modal response expected text, SFT using sampled response text and distributional fine-tuning using the ground truth distribution of the population response.

    To understand efficacy we can look at three metrics:

    - Accuracy: How often does the model get the correct answer?
    - KL Divergence: How close is the model's predicted distribution to the ground truth distribution?
    - Cross-Entropy: How well does the model's predicted distribution match the ground truth distribution?

    The first metric is uniquely relevant to the first training method as the modal is only ever seen the modal response so its distribution is unlikely to match the ground truth distribution, yet it should be better at the responding with the right answer.

    The second and third metrics are uniquely relevant the other training methods. These method are trying ot not just get the model to have the right answer but to also have the same level of uncertainty as the population. Therefore its accuracy might be lower but its KL divergence and cross-entropy should be better.

    All comparisons will be made against the base model. Furthermore there are some evaluation time parameters that are varied:
    - Reasoning vs non-reasoning
    - Dataset split
    """)
    return


if __name__ == "__main__":
    app.run()
