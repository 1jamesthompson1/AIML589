import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import marimo as mo
    import numpy as np
    import pandas as pd
    from pathlib import Path

    EVALS_ROOT = Path(__file__).resolve().parent / "output" / "evals"
    FIGS_DIR = Path(__file__).resolve().parent.parent / "figures"
    return EVALS_ROOT, FIGS_DIR, json, mo, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Find tune results table

    For each base model: one row per fine-tuned version (base model at the
    top), columns = accuracy / cross-entropy / KL divergence on the train,
    validation and overall splits. All metrics come from the modal eval config
    (``modal_response`` so accuracy
    always means exact match against the modal response and CE/KL are against
    the empirical response distribution. Every fine-tuned version is matched
    to the subpopulation it was trained on (cluster\_0 / cluster\_1 /
    overall); the base model is scored on the overall population. The LaTeX
    tables (pandas ``to_latex``, booktabs) are written to
    ``code/figures/`` as ``ft-results-<model>.tex`` and included in the report
    with ``\ctable{...}``.
    """)
    return


@app.cell
def _(EVALS_ROOT, json, np, pd):
    # Building the main table data.

    MAIN_MODAL_CONFIGS = ("modal_response",)

    METHOD_PRETTY = {
        "modal_response": "Modal response",
        "sampled_response": "Sampled response",
        "first_token_distribution": "First token",
        "full_string_distribution": "Full string",
    }
    POP_PRETTY = {
        "cluster_0": "cluster 0",
        "cluster_1": "cluster 1",
        "overall": "overall",
    }
    METHOD_ORDER = [
        "modal_response",
        "sampled_response",
        "first_token_distribution",
        "full_string_distribution",
    ]
    POP_ORDER = {"cluster_0": 0, "cluster_1": 1, "overall": 2}
    SPLITS = ["train", "validation", "overall"]
    METRICS = ["accuracy", "cross_entropy", "kl_divergence"]

    def base_and_method(model_name):
        # "{base}-nz-wvs-{method}" -> (base, method); no suffix -> (name, "base")
        if "nz-wvs-" in model_name:
            base, method = model_name.split("nz-wvs-", 1)
            return base.rstrip("-"), method
        return model_name, "base"

    def collect_modal_metrics():
        """Per (model, subpopulation, split) mean metrics, averaged over the
        system prompts, from the newest non-reasoning modal-config eval run.

        Older eval runs did not tag every row with a split/subpopulation:
        they are treated as validation-only over the whole population, so
        their train columns come out as NaN (rendered as "--" in the table).
        """
        per_prompt = []
        for model_dir in sorted(EVALS_ROOT.iterdir()):
            if not model_dir.is_dir():
                continue
            runs = []
            for run_dir in model_dir.iterdir():
                cfg_path = run_dir / "config.json"
                if not cfg_path.exists():
                    continue
                cfg = json.loads(cfg_path.read_text())
                if cfg.get("dataset") not in MAIN_MODAL_CONFIGS or cfg.get("reasoning"):
                    continue
                runs.append((cfg.get("timestamp", ""), run_dir))
            if not runs:
                continue
            _, run_dir = sorted(runs)[-1]
            df = pd.read_csv(run_dir / "per_question_results.csv")
            if df.empty or not {"expected_text", "model_answer"}.issubset(df.columns):
                continue
            df = df.assign(
                subpopulation=df["subpopulation"]
                if "subpopulation" in df
                else "overall",
                split=df["split"] if "split" in df else "validation",
            )
            base, method = base_and_method(model_dir.name)
            for (pop, split, sp_id), h in df.groupby(
                ["subpopulation", "split", "system_prompt_id"], dropna=False
            ):
                acc = (
                    h["expected_text"].str.strip().str.lower()
                    == h["model_answer"].str.strip().str.lower()
                ).mean() * 100
                per_prompt.append(
                    {
                        "model": model_dir.name,
                        "base_model": base,
                        "method": method,
                        "subpopulation": pop,
                        "split": split,
                        "system_prompt_id": sp_id,
                        "n": len(h),
                        "accuracy": acc,
                        "cross_entropy": h["cross_entropy"].mean(),
                        "kl_divergence": h["kl_divergence"].mean(),
                    }
                )
        df = pd.DataFrame(per_prompt)
        if df.empty:
            return df

        def mean_over_prompts(sub):
            return (
                sub.groupby(
                    ["model", "base_model", "method", "subpopulation", "split"],
                    dropna=False,
                )[METRICS]
                .mean()
                .reset_index()
            )

        by_split = mean_over_prompts(df)
        # Overall = train + validation pooled per prompt, weighted by question
        # count so each question contributes equally.
        combined = []
        for (model, base, method, pop, sp_id), h in df.groupby(
            ["model", "base_model", "method", "subpopulation", "system_prompt_id"],
            dropna=False,
        ):
            combined.append(
                {
                    "model": model,
                    "base_model": base,
                    "method": method,
                    "subpopulation": pop,
                    "system_prompt_id": sp_id,
                    "accuracy": np.average(h["accuracy"], weights=h["n"]),
                    "cross_entropy": np.average(h["cross_entropy"], weights=h["n"]),
                    "kl_divergence": np.average(h["kl_divergence"], weights=h["n"]),
                }
            )
        combined = pd.DataFrame(combined)
        combined["split"] = "overall"
        return pd.concat([by_split, mean_over_prompts(combined)], ignore_index=True)

    def build_main_tables():
        """dict base_model -> DataFrame, one row per fine-tuned version (base
        first), 9 columns = 3 metrics x 3 splits. Each adapter is matched to
        its own training subpopulation."""
        metrics = collect_modal_metrics()
        tables = {}
        for base_model in sorted(metrics["base_model"].unique()):
            sub = metrics[metrics["base_model"] == base_model]
            rows = []
            base_row = sub[
                (sub["method"] == "base") & (sub["subpopulation"] == "overall")
            ]
            if not base_row.empty:
                rows.append((0, 0, "Base", base_row))
            adapters = []
            for method, g in sub[sub["method"] != "base"].groupby(
                "method", dropna=False
            ):
                target_pop = method.rsplit("-", 1)[-1]
                matched = g[g["subpopulation"] == target_pop]
                if matched.empty:
                    continue
                stem = method.rsplit("-", 1)[0]
                adapters.append(
                    (
                        METHOD_ORDER.index(stem)
                        if stem in METHOD_ORDER
                        else len(METHOD_ORDER),
                        POP_ORDER.get(target_pop, len(POP_ORDER)),
                        f"{METHOD_PRETTY.get(stem, stem)} ({POP_PRETTY.get(target_pop, target_pop)})",
                        matched,
                    )
                )
            rows += sorted(adapters, key=lambda r: (r[0], r[1]))

            table_rows = []
            for _, _, label, g in rows:
                piv = g.set_index("split")
                row = {"label": label}
                for split in SPLITS:
                    for metric in METRICS:
                        row[f"{metric}_{split}"] = (
                            float("nan")
                            if split not in piv.index
                            else piv.loc[split, metric]
                        )
                table_rows.append(row)
            tables[base_model] = pd.DataFrame(table_rows)
        return tables

    def show_main_tables():
        for base_model, table in build_main_tables().items():
            print(
                f"\n=== {base_model} — matched to training subpopulation, modal eval config ==="
            )
            print(table.round(3).to_string(index=False))

    show_main_tables()
    tables = build_main_tables()
    return (tables,)


@app.cell
def _(FIGS_DIR, pd, tables):
    # Rendering the tables to LaTeX (pandas to_latex, booktabs).
    # Each file holds only the tabular; the report wraps it via
    # \ctable{fine-tuning/main-table-<model>.tex}{<caption>}.

    def render_tables_to_tex():
        METRIC_LABELS = ["Accuracy", "Cross-entropy", "KL divergence"]
        METRIC_COLS = ["accuracy", "cross_entropy", "kl_divergence"]
        SPLIT_LABELS = ["train", "validation", "overall"]
        # accuracy higher-better, CE and KL lower-better
        SENSES = ["max", "min", "min"]
        FORMATS = ["{:.1f}", "{:.2f}", "{:.3f}"]
        written = []
        for base_model, table in tables.items():
            columns = pd.MultiIndex.from_product([METRIC_LABELS, SPLIT_LABELS])
            out = pd.DataFrame(index=table["label"], columns=columns)
            out.index.name = "Model"
            # Position of the best value per column (first on ties); the base
            # row competes like any other. NaN cells never win.
            best_pos = {}
            for mcol, sense in zip(METRIC_COLS, SENSES):
                for slabel in SPLIT_LABELS:
                    col = table[f"{mcol}_{slabel}"].dropna()
                    best_pos[(mcol, slabel)] = (
                        None
                        if col.empty
                        else (col.idxmax() if sense == "max" else col.idxmin())
                    )
            for ri, row_label in enumerate(table["label"]):
                for mi, (mcol, fmt) in enumerate(zip(METRIC_COLS, FORMATS)):
                    for si, slabel in enumerate(SPLIT_LABELS):
                        val = table[f"{mcol}_{slabel}"].iloc[ri]
                        if pd.isna(val):
                            text = "--"
                        else:
                            text = fmt.format(val)
                            if (
                                best_pos[(mcol, slabel)] is not None
                                and ri == best_pos[(mcol, slabel)]
                            ):
                                text = f"\\textbf{{{text}}}"
                        out.loc[row_label, (METRIC_LABELS[mi], slabel)] = text
            tex = out.to_latex(
                column_format="l" + "r" * 9,
                escape=False,
                index_names=False,
                multicolumn_format="l",
            )
            # Put the "Model" header in the top-left corner of the first header row.
            tex = tex.replace(
                " & \\multicolumn{3}{l}{Accuracy}",
                "Model & \\multicolumn{3}{l}{Accuracy}",
                1,
            )
            out_path = FIGS_DIR / f"ft-results-{base_model}.tex"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(tex)
            written.append(out_path)
        return written

    table_files = render_tables_to_tex()
    table_files
    return


if __name__ == "__main__":
    app.run()
