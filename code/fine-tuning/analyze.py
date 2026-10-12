import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import ast
    import csv
    import json
    import math
    import textwrap
    from datetime import datetime, timezone

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from pathlib import Path

    EVALS_ROOT = Path(__file__).resolve().parent / "output" / "evals"
    FIGS_DIR = Path(__file__).resolve().parent.parent / "figures"
    return (
        EVALS_ROOT,
        FIGS_DIR,
        Path,
        ast,
        csv,
        datetime,
        json,
        math,
        mo,
        np,
        pd,
        plt,
        textwrap,
        timezone,
    )


@app.cell
def _(EVALS_ROOT, ast, json, pd):
    # Shared data-loading layer. Every analysis section below reads the same
    # thing: the newest non-reasoning modal_response eval run per model
    # directory, overall population rows only. All models carry their
    # (base_model, method) classification; adapter version ordering /
    # de-duplication is shared via adapter_rows + collapse_versions.
    MAIN_MODAL_CONFIGS = ("modal_response",)

    METHOD_PRETTY = {
        "modal_response": "Modal response",
        "sampled_response": "Sampled response",
        "first_token_distribution": "First token",
        "full_string_distribution": "Full string",
    }
    METHOD_ORDER = [
        "modal_response",
        "sampled_response",
        "first_token_distribution",
        "full_string_distribution",
    ]
    SPLITS = ["train", "validation", "overall"]
    METRICS = ["accuracy", "cross_entropy", "kl_divergence"]
    CAPABILITY_ROOT = EVALS_ROOT.parent / "capability"
    TASK_LABELS = {
        "mmlu_pro": "MMLU-Pro",
        "gpqa_diamond": "GPQA Diamond",
        "swe_bench": "SWE-bench",
    }
    TASK_ORDER = ["mmlu_pro", "gpqa_diamond", "swe_bench"]

    def base_and_method(model_name):
        # "{base}-nz-wvs-{method}" -> (base, method); no suffix -> (name, "base")
        if "nz-wvs-" in model_name:
            base, method = model_name.split("nz-wvs-", 1)
            return base.rstrip("-"), method
        return model_name, "base"

    def newest_run(model_dir, keep=None):
        """Newest run dir by config timestamp; keep(cfg) optionally filters."""
        runs = []
        for run_dir in model_dir.iterdir():
            cfg_path = run_dir / "config.json"
            if not cfg_path.exists():
                continue
            cfg = json.loads(cfg_path.read_text())
            if keep and not keep(cfg):
                continue
            runs.append((cfg.get("timestamp", ""), run_dir))
        return sorted(runs)[-1][1] if runs else None

    def newest_eval_run(model_dir):
        """Newest non-reasoning modal_response run dir (by config timestamp)."""
        return newest_run(
            model_dir,
            keep=lambda cfg: (
                cfg.get("dataset") in MAIN_MODAL_CONFIGS and not cfg.get("reasoning")
            ),
        )

    def load_overall_per_question(model_dir):
        """Per-question rows (overall population only) of the newest
        modal_response eval run of one model dir; None when there is none."""
        run_dir = newest_eval_run(model_dir)
        if run_dir is None:
            return None
        csv_path = run_dir / "per_question_results.csv"
        if not csv_path.exists():
            return None
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
        df = df.assign(
            subpopulation=df["subpopulation"] if "subpopulation" in df else "overall",
            split=df["split"] if "split" in df else "validation",
        )
        return df[df["subpopulation"] == "overall"]

    def parse_list(value):
        """CSV cells hold list-like values as strings; parse them into real
        lists regardless of quoting style (repr or JSON)."""
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except Exception:
                return None
        return list(value) if isinstance(value, (list, tuple)) else None

    def per_question_frames():
        """dict model dir name -> (base_model, method, per-question overall df)
        for every eval dir with a usable newest modal_response run."""
        frames = {}
        for model_dir in sorted(EVALS_ROOT.iterdir()):
            if not model_dir.is_dir():
                continue
            df = load_overall_per_question(model_dir)
            if df is None:
                continue
            base, method = base_and_method(model_dir.name)
            frames[model_dir.name] = (base, method, df)
        return frames

    def adapter_rows(sub):
        """Ordered [(key, label, group)] of one base model's versions: base
        first, then adapters in METHOD_ORDER; duplicates later collapsed by
        collapse_versions."""
        rows = []
        base_g = sub[sub["method"] == "base"]
        if not base_g.empty:
            rows.append((0, "Base", base_g))
        adapters = []
        for method, g in sub[sub["method"] != "base"].groupby("method", dropna=False):
            stem = method.rsplit("-", 1)[0]
            key = (
                METHOD_ORDER.index(stem) if stem in METHOD_ORDER else len(METHOD_ORDER)
            )
            adapters.append((key, METHOD_PRETTY.get(stem, stem), g))
        rows += sorted(adapters, key=lambda r: (r[0], r[1]))
        return rows

    def collapse_versions(rows):
        """Average duplicated (key, label) entries (several eval dirs mapping
        to the same method, e.g. reruns) into one concat group."""
        by_label = {}
        for key, label, g in rows:
            by_label.setdefault((key, label), []).append(g)
        return [
            (key, label, pd.concat(gs)) for (key, label), gs in sorted(by_label.items())
        ]

    model_frames = per_question_frames()
    return (
        CAPABILITY_ROOT,
        METRICS,
        SPLITS,
        TASK_LABELS,
        TASK_ORDER,
        adapter_rows,
        base_and_method,
        collapse_versions,
        load_overall_per_question,
        model_frames,
        newest_run,
        parse_list,
    )


@app.cell
def _(Path):
    # Shared LaTeX table rendering (code/latex_tables.py), used by all
    # sections below. Styler .to_latex (convert_css) turns CSS font-weight
    # into \bfseries; pandas 3 does not escape its output, so header labels
    # and text cells are escaped explicitly with escape_header/escape_latex.
    # Tables are wrapped in \resizebox{\textwidth}{!}{...} so they never
    # overflow the page; the report pulls them in with \ctable{...}{...}.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from latex_tables import (
        escape_header,
        escape_latex,
        resizebox,
        write_latex_table,
        write_tex,
    )

    return escape_header, escape_latex, resizebox, write_latex_table, write_tex


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Distance to the NZ population (TVD)

    For every model (base and fine-tuned): per question, the model's
    response distribution is averaged across system prompts, then compared
    with the NZ population distribution via TVD (%) = 0.5 * sum |model -
    true| over the response categories (0 = answers exactly like NZ).
    The table below shows the mean question-level TVD per split plus a
    "Δ vs Base" column (adapter minus its base on the held-out validation
    split) so fine tuning can be judged by whether it lowered the TVD.
    The LaTeX table is written to ``code/figures/ft-tvd-<model>.tex``.
    """)
    return


@app.cell
def _(adapter_rows, collapse_versions, model_frames, parse_list, pd):
    # Building the TVD table data.

    def collect_question_tvd(frames):
        """Per (model, split, question) TVD (%) between the model's response
        distribution averaged across system prompts and the NZ population."""
        rows = []
        for model, (base, method, df) in frames.items():
            if "question_id" not in df.columns or not {
                "model_distribution",
                "true_distribution",
            }.issubset(df.columns):
                continue
            for (split, qid), g in df.groupby(["split", "question_id"], dropna=False):
                dists = [d for d in g["model_distribution"].map(parse_list) if d]
                true_dist = next(
                    (d for d in g["true_distribution"].map(parse_list) if d), None
                )
                if not dists or true_dist is None:
                    continue
                avg_dist = [sum(vals) / len(vals) for vals in zip(*dists)]
                tvd = 0.5 * sum(abs(a - b) for a, b in zip(avg_dist, true_dist)) * 100
                rows.append(
                    {
                        "model": model,
                        "base_model": base,
                        "method": method,
                        "split": split,
                        "question_id": qid,
                        "n_prompts": len(dists),
                        "tvd": tvd,
                    }
                )
        return pd.DataFrame(rows)

    def build_tvd_tables(qtvd):
        """dict base_model -> DataFrame with one row per model version (base
        first), columns = mean question-level TVD per split."""
        tables = {}
        if qtvd.empty:
            return tables, {}
        for base_model in sorted(qtvd["base_model"].unique()):
            rows = []
            for _, label, g in collapse_versions(
                adapter_rows(qtvd[qtvd["base_model"] == base_model])
            ):
                rows.append(
                    {
                        "label": label,
                        "tvd_validation": g.loc[
                            g["split"] == "validation", "tvd"
                        ].mean(),
                        "tvd_train": g.loc[g["split"] == "train", "tvd"].mean(),
                        "tvd_all": g["tvd"].mean(),
                    }
                )
            table = pd.DataFrame(rows)
            tables[base_model] = table
        return tables

    def show_tvd_tables(tables):
        """Console view: mean question-level TVD per split + Δ vs Base."""
        for base_model, table in tables.items():
            base_row = table[table["label"] == "Base"]
            base_val = float(base_row["tvd_validation"].iloc[0])
            print(f"\n=== {base_model} — mean TVD (%) vs NZ population ===")
            out = pd.DataFrame(index=table["label"])
            out["Validation"] = table["tvd_validation"].round(1).values
            out["Train"] = table["tvd_train"].round(1).values
            out["All"] = table["tvd_all"].round(1).values
            delta = (table["tvd_validation"] - base_val).where(table["label"] != "Base")
            out["Δ vs Base"] = ["--" if pd.isna(v) else f"{v:+.1f}" for v in delta]
            print(out.to_string())
        print(
            "\nTVD (%) averaged over questions, model distribution averaged "
            "across system prompts. Δ vs Base = adapter minus base on "
            "validation (negative means fine tuning moved the model closer "
            "to the NZ population)."
        )

    tvd_tables = build_tvd_tables(collect_question_tvd(model_frames))
    show_tvd_tables(tvd_tables)
    return (tvd_tables,)


@app.cell
def _(pd, tvd_tables, write_latex_table):
    # Rendering the TVD tables to LaTeX (booktabs, lowest TVD per column
    # bolded; deltas formatted with an explicit sign).

    def render_tvd_tables_to_tex(tables):
        # 4 columns only: TVD per split, plus a single delta column.
        columns = pd.MultiIndex.from_tuples(
            [
                ("TVD (%)", "Validation"),
                ("TVD (%)", "Train"),
                ("TVD (%)", "All"),
                ("Δ vs Base", "vs Base (val)"),
            ]
        )
        senses = {"TVD (%)": "min", "Δ vs Base": "min"}
        written = []
        for base_model, table in tables.items():
            base_row = table[table["label"] == "Base"]
            base_val = float(base_row["tvd_validation"].iloc[0])
            out = pd.DataFrame(index=table["label"], columns=columns)
            out.index.name = None
            fmt = {}
            for col, vals in [
                ("Validation", table["tvd_validation"]),
                ("Train", table["tvd_train"]),
                ("All", table["tvd_all"]),
            ]:
                out[("TVD (%)", col)] = vals.values
                fmt[("TVD (%)", col)] = "{:.1f}"
            delta = (table["tvd_validation"] - base_val).where(table["label"] != "Base")
            out[("Δ vs Base", "vs Base (val)")] = delta.values
            fmt[("Δ vs Base", "vs Base (val)")] = "{:+.1f}"
            written.append(
                write_latex_table(
                    out, fmt, out.shape[1], f"ft-tvd-{base_model}.tex", senses=senses
                )
            )
        return written

    tvd_tex_files = render_tvd_tables_to_tex(tvd_tables)
    tvd_tex_files
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Most divergent questions (base model vs NZ population)

    For each base model evaluated: ranks the 25 survey sub-items where the
    base model (prompt-averaged response distribution, overall population,
    train and validation pooled) differs most from the NZ population
    distribution, measured by TVD (%). For each question the modal (most
    likely) model answer and the modal NZ answer are shown, so the
    direction of the disagreement is visible. The LaTeX (booktabs) tables
    are written to ``code/figures/ft-divergent-questions-<model>.tex``.
    """)
    return


@app.cell
def _(escape_latex, model_frames, np, parse_list, pd, write_tex):
    # Top-N most divergent questions per base model: per (question_id,
    # column_name) sub-item, TVD (%) between the model's response distribution
    # averaged across system prompts (all splits pooled) and the NZ population,
    # plus the modal answers of each side.
    TOP_N = 25

    def collect_question_divergence(df):
        """DataFrame with one row per survey sub-item: label, TVD (%) and the
        modal model / NZ answer categories."""
        rows = []
        for (_qid, col), g in df.groupby(["question_id", "column_name"], dropna=False):
            dists = [d for d in g["model_distribution"].map(parse_list) if d]
            true_dist = next(
                (d for d in g["true_distribution"].map(parse_list) if d), None
            )
            if not dists or true_dist is None:
                continue
            avg_dist = [sum(vals) / len(vals) for vals in zip(*dists)]
            cats = parse_list(g.iloc[0].get("categories"))
            if not cats or len(cats) != len(true_dist) or len(cats) != len(avg_dist):
                continue
            sub_q = g.iloc[0].get("sub_question")
            question_text = (
                sub_q
                if isinstance(sub_q, str) and sub_q.strip()
                else str(g.iloc[0].get("question", ""))
            )
            rows.append(
                {
                    "label": f"{col}: {question_text}",
                    "tvd": 0.5
                    * sum(abs(a - b) for a, b in zip(avg_dist, true_dist))
                    * 100,
                    "modal_model": cats[int(np.argmax(avg_dist))],
                    "modal_nz": cats[int(np.argmax(true_dist))],
                }
            )
        return pd.DataFrame(rows).sort_values("tvd", ascending=False)

    def render_divergent_tex(base_model, top, kind="divergent"):
        """Booktabs table: question, TVD (%) and modal answers of both sides.
        Question text is the FULL stem, word-wrapped inside LaTeX paragraph
        columns. Cells are pre-escaped (escape=False in to_latex)."""
        out = pd.DataFrame(
            {
                "Question": [escape_latex(t) for t in top["label"]],
                "TVD (\\%)": [f"{v:.1f}" for v in top["tvd"]],
                "Model answer": [escape_latex(t) for t in top["modal_model"]],
                "NZ answer": [escape_latex(t) for t in top["modal_nz"]],
            }
        )
        tex = out.style.hide(axis="index").to_latex(
            column_format="p{3.1in}lp{1.05in}p{1.05in}",
            hrules=True,
            convert_css=True,
        )
        return write_tex(tex, f"ft-{kind}-questions-{base_model}.tex")

    divergent_tex_paths = []
    similar_tex_paths = []
    for divergent_base, divergent_method, divergent_df in model_frames.values():
        if divergent_method != "base":
            continue
        divergence = collect_question_divergence(divergent_df)
        divergent_top = divergence.head(TOP_N)
        similar_top = divergence.sort_values("tvd", ascending=True).head(TOP_N)
        if divergent_top.empty:
            continue
        divergent_tex_paths.append(
            render_divergent_tex(divergent_base, divergent_top, kind="divergent")
        )
        similar_tex_paths.append(
            render_divergent_tex(divergent_base, similar_top, kind="similar")
        )
        print(
            f"\n=== {divergent_base} — top {TOP_N} most divergent questions vs NZ ==="
        )
        print(
            divergent_top.assign(
                label=lambda d: d["label"].str.slice(0, 70),
                tvd=lambda d: d["tvd"].round(1),
            ).to_string(index=False)
        )
    divergent_tex_paths
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Find tune results table

    For each base model: one row per fine-tuned version (base model at the
    top), columns = accuracy / cross-entropy / KL divergence on the train,
    validation and overall splits, plus a "vs Base" column block with the
    adapter-minus-base deltas on the held-out validation split. All metrics
    come from the modal eval config
    (``modal_response`` so accuracy
    always means exact match against the modal response and CE/KL are
    against the empirical response distribution. Every model (base
    and fine-tuned) is
    scored on the overall population. The LaTeX
    tables (pandas ``to_latex``, booktabs) are written to
    ``code/figures/`` as ``ft-results-<model>.tex`` and included in the report
    with ``\ctable{...}``.
    """)
    return


@app.cell
def _(METRICS, SPLITS, adapter_rows, collapse_versions, model_frames, np, pd):
    # Building the main table data.

    def collect_modal_metrics():
        """Per (model, split) mean metrics, averaged over the system prompts,
        from the newest non-reasoning modal-config eval run. Only the
        overall population is kept; rows tagged with a cluster
        subpopulation are dropped.

        Older eval runs did not tag every row with a split/subpopulation:
        they are treated as validation-only over the whole population, so
        their train columns come out as NaN (rendered as "--" in the table).
        """
        per_prompt = []
        for model, (base, method, df) in model_frames.items():
            if not {"expected_text", "model_answer"}.issubset(df.columns):
                continue
            for (split, sp_id), h in df.groupby(
                ["split", "system_prompt_id"], dropna=False
            ):
                acc = (
                    h["expected_text"].str.strip().str.lower()
                    == h["model_answer"].str.strip().str.lower()
                ).mean() * 100
                per_prompt.append(
                    {
                        "model": model,
                        "base_model": base,
                        "method": method,
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
                    ["model", "base_model", "method", "split"],
                    dropna=False,
                )[METRICS]
                .mean()
                .reset_index()
            )

        by_split = mean_over_prompts(df)
        # Overall = train + validation pooled per prompt, weighted by question
        # count so each question contributes equally.
        combined = []
        for (model, base, method, sp_id), h in df.groupby(
            ["model", "base_model", "method", "system_prompt_id"],
            dropna=False,
        ):
            combined.append(
                {
                    "model": model,
                    "base_model": base,
                    "method": method,
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
        first), 9 columns = 3 metrics x 3 splits. All models are matched to
        the overall population. Duplicate entries (several eval dirs mapping
        to the same method) are averaged per split."""
        metrics = collect_modal_metrics()
        tables = {}
        base_refs = {}
        for base_model in sorted(metrics["base_model"].unique()):
            collapsed = collapse_versions(
                adapter_rows(metrics[metrics["base_model"] == base_model])
            )
            table_rows = []
            for _, label, g in collapsed:
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
            table = pd.DataFrame(table_rows)
            tables[base_model] = table
            # Reference values of the base model, for the "vs base" deltas.
            base_vals = table[table["label"] == "Base"]
            if not base_vals.empty:
                base_refs[base_model] = {
                    f"{metric}_{split}": base_vals[f"{metric}_{split}"].iloc[0]
                    for split in SPLITS
                    for metric in METRICS
                }
        return tables, base_refs

    def show_main_tables():
        """Console view: metrics x {Validation, Train, All, Δ vs Base}.
        Δ vs Base = adapter minus base on the held-out validation split
        (positive Δ accuracy = better; negative Δ CE/KL = better). The
        base row shows "--" for the deltas."""
        metric_labels = {
            "accuracy": "Acc (%)",
            "cross_entropy": "CE",
            "kl_divergence": "KL",
        }
        split_cols = ["Validation", "Train", "All", "Δ vs Base"]
        for base_model, table in tables.items():
            base = base_refs[base_model]
            print(
                f"\n=== {base_model} — overall population "
                "(validation = held-out questions/prompts) ==="
            )
            cols = pd.MultiIndex.from_product(
                [[metric_labels[m] for m in METRICS], split_cols]
            )
            out = pd.DataFrame(index=table["label"], columns=cols)
            for mi, m in enumerate(METRICS):
                # .values: assign positionally, otherwise pandas aligns on
                # the mismatched row labels and silently produces NaN.
                out[(metric_labels[m], "Validation")] = (
                    table[f"{m}_validation"].round(3).values
                )
                out[(metric_labels[m], "Train")] = table[f"{m}_train"].round(3).values
                out[(metric_labels[m], "All")] = table[f"{m}_overall"].round(3).values
                delta = table[f"{m}_validation"] - base[f"{m}_validation"]
                delta = delta.where(table["label"] != "Base").round(3)
                out[(metric_labels[m], "Δ vs Base")] = delta.values
            print(out.to_string())
            print(
                "Δ vs Base = adapter minus base on validation "
                "(+Acc / −CE / −KL means improvement)"
            )

    tables, base_refs = build_main_tables()
    show_main_tables()
    return base_refs, tables


@app.cell
def _(base_refs, pd, tables, write_latex_table):
    # Columns: per metric {Validation, Train, All, vs Base} — the validation
    # split (held-out questions/prompts) comes first as the honest measure of
    # generalisation, and "vs Base" = adapter minus base on validation
    # (+Acc / -CE / -KL means the fine-tuned model moved towards the NZ
    # response distribution).

    def render_tables_to_tex():
        METRIC_LABELS = ["Accuracy", "CE", "KL"]
        METRIC_COLS = ["accuracy", "cross_entropy", "kl_divergence"]
        SPLIT_LABELS = ["Validation", "Train", "All", "vs Base"]
        # accuracy higher-better, CE and KL lower-better (also for the deltas)
        SENSES = {"Accuracy": "max", "CE": "min", "KL": "min"}
        FORMATS = {"Accuracy": "{:.1f}", "CE": "{:.2f}", "KL": "{:.3f}"}
        DELTA_FORMATS = {k: v.replace("{:", "{:+") for k, v in FORMATS.items()}

        written = []
        for base_model, table in tables.items():
            base = base_refs[base_model]
            columns = pd.MultiIndex.from_product([METRIC_LABELS, SPLIT_LABELS])
            out = pd.DataFrame(index=table["label"], columns=columns)
            # Index name suppressed (Styler would emit an extra header row);
            # the "Model" header is re-added by write_latex_table.
            out.index.name = None
            fmt = {}
            for mi, mcol in enumerate(METRIC_COLS):
                label = METRIC_LABELS[mi]
                out[(label, "Validation")] = table[f"{mcol}_validation"].values
                out[(label, "Train")] = table[f"{mcol}_train"].values
                out[(label, "All")] = table[f"{mcol}_overall"].values
                delta = table[f"{mcol}_validation"] - base[f"{mcol}_validation"]
                # No delta for the base row itself (rendered as "--").
                delta = delta.where(table["label"] != "Base")
                out[(label, "vs Base")] = delta.values
                for slabel in SPLIT_LABELS[:-1]:
                    fmt[(label, slabel)] = FORMATS[label]
                fmt[(label, "vs Base")] = DELTA_FORMATS[label]
            written.append(
                write_latex_table(
                    out,
                    fmt,
                    out.shape[1],
                    f"ft-results-{base_model}.tex",
                    senses=SENSES,
                )
            )
        return written

    table_files = render_tables_to_tex()
    table_files
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Capability results table

    Same layout as the main table, but for the capability evals: one row per
    fine-tuned version (base model at the top), columns = accuracy (with
    standard error) on MMLU-Pro, GPQA Diamond and SWE-bench. Scores are read
    from the newest completed run per model in
    ``output/capability/<model>/<run>/summary.json`` (written by
    ``capability_eval.py``). The LaTeX (booktabs) tables are written to
    ``code/figures/ft-capability-<model>.tex`` and included in the report
    with ``\ctable{ft-capability-<model>.tex}{<caption>}``.
    """)
    return


@app.cell
def _(
    CAPABILITY_ROOT,
    TASK_LABELS,
    TASK_ORDER,
    adapter_rows,
    base_and_method,
    json,
    newest_run,
    pd,
):
    # Building the capability table data from summary.json of the newest run
    # per model.

    def collect_capability_scores():
        """Per (model, task) accuracy/stderr from the newest run of each model."""
        rows = []
        if not CAPABILITY_ROOT.exists():
            return pd.DataFrame(rows)
        for model_dir in sorted(CAPABILITY_ROOT.iterdir()):
            if not model_dir.is_dir():
                continue
            run_dir = newest_run(model_dir)
            if run_dir is None:
                continue
            summary_path = run_dir / "summary.json"
            if not summary_path.exists():
                continue
            base, method = base_and_method(model_dir.name)
            summary = json.loads(summary_path.read_text())
            # Only the overall population adapters (plus the base model):
            # cluster-trained adapters are out of scope.
            if method != "base" and not method.endswith("-overall"):
                continue
            for task in TASK_ORDER:
                entry = summary.get(task) or {}
                if "accuracy" not in entry:
                    continue
                rows.append(
                    {
                        "model": model_dir.name,
                        "base_model": base,
                        "method": method,
                        "task": task,
                        "n": entry.get("n"),
                        "accuracy": entry["accuracy"],
                        "stderr": entry.get("stderr"),
                    }
                )
        return pd.DataFrame(rows)

    def build_capability_tables(scores):
        """dict base_model -> DataFrame, one row per fine-tuned version (base
        first), columns = accuracy/stderr per task. Adapters are labelled and
        ordered exactly like in the main table."""
        tables = {}
        if scores.empty:
            return tables
        for base_model in sorted(scores["base_model"].unique()):
            table_rows = []
            for _, label, g in sorted(
                adapter_rows(scores[scores["base_model"] == base_model]),
                key=lambda r: (r[0], r[1]),
            ):
                row = {"label": label}
                for task in TASK_ORDER:
                    h = g[g["task"] == task]
                    row[f"{task}_acc"] = (
                        h["accuracy"].iloc[0] if not h.empty else float("nan")
                    )
                    row[f"{task}_se"] = (
                        h["stderr"].iloc[0]
                        if not h.empty and not pd.isna(h["stderr"].iloc[0])
                        else float("nan")
                    )
                table_rows.append(row)
            tables[base_model] = pd.DataFrame(table_rows)
        return tables

    capability_scores = collect_capability_scores()
    capability_tables = build_capability_tables(capability_scores)
    for base_model, table in capability_tables.items():
        print(f"\n=== {base_model} — capability evals (accuracy in %, with 95% CI) ===")
        disp = pd.DataFrame(index=table["label"])
        for task in TASK_ORDER:
            cells = []
            for a, s in zip(table[f"{task}_acc"], table[f"{task}_se"]):
                if pd.isna(a) or pd.isna(s):
                    cells.append("--")
                else:
                    lo = (a - 1.96 * s) * 100
                    hi = (a + 1.96 * s) * 100
                    cells.append(f"{a * 100:.1f} [{lo:.1f}, {hi:.1f}]")
            disp[TASK_LABELS[task]] = cells
        print(disp.to_string())
    return (capability_tables,)


@app.cell
def _(CAPABILITY_ROOT, json, pd):
    def collect_capability_run_overview():
        """Rows for all inspect logs under output/capability/*/*/inspect-logs."""
        rows = []
        if not CAPABILITY_ROOT.exists():
            return pd.DataFrame(rows)
        for model_dir in sorted(CAPABILITY_ROOT.iterdir()):
            if not model_dir.is_dir():
                continue
            for run_dir in sorted(model_dir.iterdir()):
                log_dir = run_dir / "inspect-logs"
                if not log_dir.is_dir():
                    continue
                cfg = {}
                cfg_path = run_dir / "config.json"
                if cfg_path.exists():
                    cfg = json.loads(cfg_path.read_text())
                for log_path in sorted(log_dir.glob("*.json")):
                    log = json.loads(log_path.read_text())
                    status = log.get("status") or "unknown"
                    results = log.get("results") or {}
                    error = log.get("error") or {}
                    samples = len(log.get("samples") or [])
                    completed = results.get("completed_samples")
                    total = results.get("total_samples")
                    task = (log.get("eval") or {}).get("task") or log_path.stem
                    task = task.replace("inspect_evals/", "").replace(
                        "inspect_harbor/", ""
                    )
                    if status == "success":
                        progress = (
                            f"{completed}/{total}"
                            if completed is not None
                            else str(samples)
                        )
                        reason = "Completed"
                    elif status in ("cancelled", "error", "started"):
                        progress = (
                            str(samples) if not results else f"{samples} (unscored)"
                        )
                        reason = {
                            "cancelled": "Cancelled",
                            "started": "Not finalised",
                            "error": "Error",
                        }.get(status, status)
                    else:
                        progress = str(samples)
                        reason = status
                    if error.get("message"):
                        reason = f"{reason}: {error['message']}"
                    rows.append(
                        {
                            "model": model_dir.name,
                            "run": (cfg.get("timestamp") or run_dir.name)[:19].replace(
                                "T", " "
                            ),
                            "task": task,
                            "status": status,
                            "samples": progress,
                            "reason": reason,
                        }
                    )
        return pd.DataFrame(rows)

    cap_run_overview = collect_capability_run_overview()
    print(
        cap_run_overview.to_string(index=False)
        if not cap_run_overview.empty
        else "no capability eval logs found"
    )
    return


@app.cell
def _(
    TASK_LABELS,
    TASK_ORDER,
    capability_tables,
    escape_header,
    pd,
    write_latex_table,
):
    # Rendering the capability tables to LaTeX (booktabs, best accuracy per
    # column bolded). Each benchmark gets two sub-columns: accuracy (%) and its
    # 95% confidence interval (acc +- 1.96 x s.e. from the inspect metrics).

    def render_capability_tables_to_tex():
        headers = [TASK_LABELS[t] for t in TASK_ORDER]

        subs = ["Acc (%)", "95% CI"]
        escaped = {s: escape_header(s) for s in subs}
        written = []
        for base_model, table in capability_tables.items():
            columns = pd.MultiIndex.from_product([headers, [escaped[s] for s in subs]])
            out = pd.DataFrame(index=table["label"].values, columns=columns)
            out.index.name = None
            fmt = {}
            for task, header in zip(TASK_ORDER, headers):
                acc = table[f"{task}_acc"]
                se = table[f"{task}_se"]
                lo = (acc - 1.96 * se) * 100
                hi = (acc + 1.96 * se) * 100
                out[(header, escaped["Acc (%)"])] = (acc * 100).round(1).values
                out[(header, escaped["95% CI"])] = [
                    f"{lo_v:.1f}-{hi_v:.1f}"
                    if pd.notna(lo_v) and pd.notna(hi_v)
                    else "--"
                    for lo_v, hi_v in zip(lo, hi)
                ]
                fmt[(header, escaped["Acc (%)"])] = "{:.1f}"
            written.append(
                write_latex_table(
                    out, fmt, out.shape[1], f"ft-capability-{base_model}.tex"
                )
            )
        return written

    capability_table_files = render_capability_tables_to_tex()
    capability_table_files
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Distribution comparison figures

    For each model's newest ``modal_response`` run (overall population
    only) we pick the 3 questions whose TVD of the prompt-averaged model
    distribution vs the NZ population sits at the 3 percentiles across questions. Each figure has two bars per answer
    category: the NZ baseline and the average across system prompts,
    annotated with the TVD to the baseline. Figures are written to
    ``code/fine-tuning/output/figures/distributions/<model>/<column>_<p10|p50|p90>.png``.
    """)
    return


@app.cell
def _(
    EVALS_ROOT,
    load_overall_per_question,
    math,
    np,
    parse_list,
    plt,
    textwrap,
):
    # Distribution summary figures — read the newest run's saved
    # per_question_results.csv per model (nothing recomputed during eval).
    DIST_FIGS_ROOT = EVALS_ROOT.parent / "figures" / "distributions"
    PERCENTILE_PICKS = [("p5", 5), ("p50", 50), ("p95", 95)]
    BASELINE_COLOR = "#DD8452"
    AVG_COLOR = "#55A868"

    def _question_rows(df):
        """All overall-population rows. The split column is assigned per
        question-prompt pair, so filtering by it would drop prompts; the
        figures are descriptive, so train+validation rows are combined."""
        return df

    def _question_stats(group):
        """(title_row, cats, true_dist, {prompt_id: dist}, avg_dist, tvd_pct)
        for one question column; None when the stored data is unusable."""
        row = group.iloc[0]
        cats = parse_list(row.get("categories"))
        true_dist = parse_list(row.get("true_distribution"))
        if not cats or not true_dist or len(cats) != len(true_dist):
            return None
        per_prompt = {}
        for _, r in group.iterrows():
            d = parse_list(r.get("model_distribution"))
            if d and len(d) == len(cats):
                per_prompt[r.get("system_prompt_id", "unknown")] = d
        if not per_prompt:
            return None
        avg_dist = [sum(vals) / len(per_prompt) for vals in zip(*per_prompt.values())]
        tvd = 0.5 * sum(abs(a - b) for a, b in zip(avg_dist, true_dist)) * 100
        return row, cats, true_dist, per_prompt, avg_dist, tvd

    def _plot_question(model_name, col, stats, pct_label, figs_dir):
        """Two bars per category: the NZ baseline and the plain average
        across system prompts. Returns the figure path."""
        row, cats, true_dist, per_prompt, avg_dist, tvd = stats

        title_parts = []
        qtext = row.get("question", "")
        if qtext:
            title_parts.append(textwrap.fill(str(qtext), width=70))
        sq = row.get("sub_question", "")
        if sq and not (isinstance(sq, float) and math.isnan(sq)):
            title_parts.append(str(sq))

        x = np.arange(len(cats))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(9, len(cats) * 0.9), 6))

        bars = ax.bar(
            x - width / 2,
            true_dist,
            width * 0.92,
            color=BASELINE_COLOR,
            alpha=0.85,
        )
        for b in bars:
            h = b.get_height()
            if h > 0.01:
                ax.annotate(
                    f"{h:.0%}",
                    xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

        bars = ax.bar(
            x + width / 2,
            avg_dist,
            width * 0.92,
            color=AVG_COLOR,
            alpha=0.85,
        )
        for b in bars:
            h = b.get_height()
            if h > 0.01:
                ax.annotate(
                    f"{h:.0%}",
                    xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(cats, fontsize=8, rotation=20, ha="right")
        ax.set_ylabel("Probability")
        if title_parts:
            ax.set_title("\n".join(title_parts), fontsize=10, linespacing=1.3, pad=12)
        ax.set_ylim(0, max(max(true_dist), max(avg_dist)) * 1.25)
        ax.legend(
            ["Baseline\n(NZ pop.)", "Avg across\nprompts"],
            fontsize=7,
            loc="upper right",
        )
        ax.text(
            0.02,
            0.98,
            f"TVD={tvd:.1f}%",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7),
        )

        footer = f"{model_name} | {col} | {pct_label}"
        fig.text(
            0.5,
            0.01,
            footer,
            ha="center",
            va="bottom",
            fontsize=7,
            color="gray",
            style="italic",
        )
        plt.tight_layout(rect=[0, 0.04, 1, 1])

        safe_name = f"{col}_{pct_label}".replace(" ", "_").replace("/", "_")[:80]
        figs_dir.mkdir(parents=True, exist_ok=True)
        figpath = figs_dir / f"{safe_name}.png"
        fig.savefig(figpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plots] wrote {figpath.name}")
        return figpath

    def generate_distribution_figures():
        """For every model: 3 figures at the p10/p50/p90 percentiles of the
        question-level average-TVD distribution."""
        out_files = []
        for model_dir in sorted(EVALS_ROOT.iterdir()):
            if not model_dir.is_dir():
                continue
            df = load_overall_per_question(model_dir)
            if df is None or "categories" not in df.columns:
                continue
            entries = []
            for col, g in _question_rows(df).groupby("column_name"):
                stats = _question_stats(g)
                if stats:
                    entries.append((col, stats))
            if not entries:
                continue
            tvds = np.array([stats[5] for _, stats in entries])
            figs_dir = DIST_FIGS_ROOT / model_dir.name
            used = set()
            for pct_label, pct in PERCENTILE_PICKS:
                order = np.argsort(np.abs(tvds - np.percentile(tvds, pct)))
                idx = next(int(i) for i in order if int(i) not in used)
                used.add(idx)
                col, stats = entries[idx]
                out_files.append(
                    _plot_question(model_dir.name, col, stats, pct_label, figs_dir)
                )
        return out_files

    distribution_fig_files = generate_distribution_figures()
    distribution_fig_files
    return


@app.cell
def _(EVALS_ROOT, Path, csv, datetime, json, pd, timezone):
    """Write the webapp manifest (output/evals/index.json).

    The website fetches this at runtime so nothing is bundled at build time.
    It lists every model/run (with config + bucket-relative path) plus a
    question inventory keyed by (question_id, column_name) — the 251 distinct
    survey sub-items, each with in_training / in_eval flags.
    """
    TRAIN_PARQUET = (
        Path(__file__).resolve().parents[1]
        / "training-dataset"
        / "output"
        / "dataset"
        / "sampled_response"
        / "train"
        / "overall.parquet"
    )
    VAL_PARQUET = (
        Path(__file__).resolve().parents[1]
        / "training-dataset"
        / "output"
        / "dataset"
        / "sampled_response"
        / "validation"
        / "overall.parquet"
    )
    QUESTION_MAPPING = (
        Path(__file__).resolve().parents[1]
        / "training-dataset"
        / "output"
        / "question_mapping.json"
    )
    BUCKET_ID = "1jamesthompson1/wvs-nz-value-alignment-evals"

    def _read_parquet_items(parquet_path: Path) -> set[tuple[str, str]]:
        """Return the set of (question_id, column_name) pairs in a parquet."""
        df = pd.read_parquet(parquet_path)
        return {(str(r["question_id"]), r["column_name"]) for _, r in df.iterrows()}

    def _load_question_mapping(path: Path) -> dict[str, str]:
        """question_id -> question text from the canonical mapping."""
        mapping = json.loads(path.read_text())
        out: dict[str, str] = {}
        for entry in mapping:
            if entry.get("question_type") != "value survey":
                continue
            out[str(entry["id"])] = entry.get("question", "")
        return out

    def build_webapp_manifest() -> dict:
        models: dict[str, list[dict]] = {}
        eval_qcols: set[tuple[str, str]] = set()

        if EVALS_ROOT.is_dir():
            for model_dir in sorted(p for p in EVALS_ROOT.iterdir() if p.is_dir()):
                runs = []
                for run_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                    cfg_path = run_dir / "config.json"
                    csv_path = run_dir / "per_question_results.csv"
                    if not cfg_path.exists() or not csv_path.exists():
                        continue
                    config = json.loads(cfg_path.read_text())
                    runs.append(
                        {
                            "run_name": run_dir.name,
                            "config": {
                                "target": config.get("target"),
                                "dataset": config.get("dataset"),
                                "run_name": config.get("run_name"),
                                "timestamp": config.get("timestamp"),
                                "model_sha": config.get("model_sha"),
                                "elapsed_seconds": config.get("elapsed_seconds"),
                                "aborted": config.get("aborted"),
                                "reasoning": config.get("reasoning") is True,
                            },
                            "path": f"{model_dir.name}/{run_dir.name}",
                        }
                    )
                    with open(csv_path, newline="") as fh:
                        for row in csv.DictReader(fh):
                            qid = str(row.get("question_id", "")).strip()
                            col = row.get("column_name", "").strip()
                            if qid and col:
                                eval_qcols.add((qid, col))
                if runs:
                    models[model_dir.name] = runs

        # Question inventory: every (question_id, column_name) pair from the
        # validation split, with in_training / in_eval flags.
        train_qcols: set[tuple[str, str]] = set()
        if TRAIN_PARQUET.exists():
            train_qcols = _read_parquet_items(TRAIN_PARQUET)

        # Validation split is the canonical full list of sub-items.
        val_qcols: set[tuple[str, str]] = set()
        if VAL_PARQUET.exists():
            val_qcols = _read_parquet_items(VAL_PARQUET)

        q_text = (
            _load_question_mapping(QUESTION_MAPPING)
            if QUESTION_MAPPING.exists()
            else {}
        )

        # Build a lookup for sub_question text from the validation parquet.
        val_df = (
            pd.read_parquet(VAL_PARQUET) if VAL_PARQUET.exists() else pd.DataFrame()
        )
        sub_q_map: dict[tuple[str, str], str] = {}
        if not val_df.empty:
            for _, r in val_df.iterrows():
                key = (str(r["question_id"]), r["column_name"])
                if key not in sub_q_map:
                    sub_q_map[key] = r.get("sub_question", "")
                    if (
                        not isinstance(sub_q_map[key], str)
                        or sub_q_map[key] != sub_q_map[key]
                    ):
                        sub_q_map[key] = ""

        questions: list[dict] = []
        all_qcols = sorted(
            val_qcols | train_qcols | eval_qcols,
            key=lambda qc: (int(qc[0]) if qc[0].isdigit() else 10**9, qc[0], qc[1]),
        )
        for qid, col_name in all_qcols:
            questions.append(
                {
                    "question_id": qid,
                    "column_name": col_name,
                    "question": q_text.get(qid, ""),
                    "sub_question": sub_q_map.get((qid, col_name), ""),
                    "in_training": (qid, col_name) in train_qcols,
                    "in_eval": (qid, col_name) in eval_qcols,
                }
            )

        return {
            "schema": "wvs-ft-evals-index/v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "bucket": BUCKET_ID,
            "base_url": f"https://huggingface.co/buckets/{BUCKET_ID}/resolve/ft/evals/",
            "models": models,
            "questions": questions,
        }

    manifest = build_webapp_manifest()
    manifest_out = EVALS_ROOT / "index.json"
    manifest_out.write_text(json.dumps(manifest, indent=2) + "\n")
    n_runs = sum(len(r) for r in manifest["models"].values())
    print(f"manifest: {manifest_out}")
    print(
        f"  models: {len(manifest['models'])}  runs: {n_runs}  questions: {len(manifest['questions'])}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Schwartz value map and value tensions

    The WVS items are read through Schwartz's (2012) refined value circle.
    `item_map.json`, next to this notebook, assigns each value-survey column
    to the value it measures: DIRECT means the item asks about the value's
    own motivational content, PROXY means it is adjacent to it.

    The circular map shows how far the base models sit from the NZ
    population on each value. The angle is the value's place on the circle,
    the bar is the mean TVD across base models over the value's DIRECT
    items, and each dot is one base model on one value (colour = model; see
    the legend). Values in grey have no DIRECT items. The tension table
    ranks opposed pairs by Schwartz opposition x base-model divergence from
    NZ and is written to ``code/figures/value-tensions-<model>.tex`` for the
    report - it is the table used to justify which tensions the behavioural
    simulations test.
    """)
    return


@app.cell
def _(Path, json):
    # The Schwartz (2012) refined value circle: circular order, third-order
    # dimension, and which values an agent can act on. Circular order is the
    # motivational order - index 0 and the last index are neighbours.
    CIRCLE = (
        ("SDT", "Self-Direction (Thought)", "Openness"),
        ("SDA", "Self-Direction (Action)", "Openness"),
        ("ST", "Stimulation", "Openness"),
        ("HE", "Hedonism", "Openness"),
        ("AC", "Achievement", "Self-Enhancement"),
        ("POD", "Power (Dominance)", "Self-Enhancement"),
        ("POR", "Power (Resources)", "Self-Enhancement"),
        ("FA", "Face", "Conservation"),
        ("SEP", "Security (Personal)", "Conservation"),
        ("SES", "Security (Societal)", "Conservation"),
        ("COI", "Conformity (Interpersonal)", "Conservation"),
        ("COR", "Conformity (Rules)", "Conservation"),
        ("TR", "Tradition", "Conservation"),
        ("HU", "Humility", "Self-Transcendence"),
        ("BED", "Benevolence (Dependability)", "Self-Transcendence"),
        ("BEC", "Benevolence (Caring)", "Self-Transcendence"),
        ("UNC", "Universalism (Concern)", "Self-Transcendence"),
        ("UNN", "Universalism (Nature)", "Self-Transcendence"),
        ("UNT", "Universalism (Tolerance)", "Self-Transcendence"),
    )
    # Hedonism, Face and Universalism (Nature) are in the circle but not
    # usable tension poles for a public-service agent (they stay on the map).
    NON_ACTIONABLE = {"HE", "FA", "UNN"}
    AGENT_ACTIONABLE = {code for code, _, _ in CIRCLE} - NON_ACTIONABLE
    HIGHER_ORDER_CONFLICTS = (
        ("Openness", "Conservation"),
        ("Self-Transcendence", "Self-Enhancement"),
    )

    def opposition(code_a, code_b):
        """Circular distance, 1.0 = diametrically opposite."""
        codes = [code for code, _, _ in CIRCLE]
        n = len(codes)
        distance = abs(codes.index(code_a) - codes.index(code_b)) % n
        return min(distance, n - distance) / (n / 2)

    def dimension_conflict(dim_a, dim_b):
        """One of Schwartz's two stated higher-order conflicts?"""
        return (dim_a, dim_b) in HIGHER_ORDER_CONFLICTS or (
            dim_b,
            dim_a,
        ) in HIGHER_ORDER_CONFLICTS

    # item map: value-survey column -> [(value, strength, note), ...]
    raw = json.loads((Path(__file__).resolve().parent / "item_map.json").read_text())
    VALUE_ITEMS = {
        col: [(entry["value"], entry["strength"], entry["note"]) for entry in assigns]
        for col, assigns in raw["items"].items()
    }
    unknown_codes = sorted(
        {code for assigns in VALUE_ITEMS.values() for code, _, _ in assigns}
        - {code for code, _, _ in CIRCLE}
    )
    if unknown_codes:
        raise ValueError(f"item_map.json uses unknown Schwartz codes: {unknown_codes}")

    def value_items(code, available=None, proxy_weight=0.0):
        """Columns measuring `code`: DIRECT always, PROXY only if weighted."""
        columns = []
        for col, assigns in VALUE_ITEMS.items():
            if available is not None and col not in available:
                continue
            if any(
                value == code and (strength == "DIRECT" or proxy_weight > 0)
                for value, strength, _note in assigns
            ):
                columns.append(col)
        return sorted(columns)

    return (
        AGENT_ACTIONABLE,
        CIRCLE,
        VALUE_ITEMS,
        dimension_conflict,
        opposition,
        value_items,
    )


@app.cell
def _(CIRCLE, Path, VALUE_ITEMS, json, pd):
    # Coverage check: every mapped key must be a value-survey column, and the
    # per-value counts show which poles are measured at all.
    question_mapping_path = (
        Path(__file__).resolve().parents[1]
        / "training-dataset"
        / "output"
        / "question_mapping.json"
    )
    question_mapping = json.loads(question_mapping_path.read_text())
    value_columns = {
        col
        for entry in question_mapping
        if entry["question_type"] == "value survey"
        for col in entry["column_names"]
    }
    unknown_columns = sorted(set(VALUE_ITEMS) - value_columns)
    if unknown_columns:
        raise ValueError(f"item_map.json: not value-survey columns: {unknown_columns}")

    coverage_rows = []
    for code, name, _dim in CIRCLE:
        strengths = [
            strength
            for assigns in VALUE_ITEMS.values()
            for value, strength, _note in assigns
            if value == code
        ]
        coverage_rows.append(
            {
                "Code": code,
                "Value": name,
                "DIRECT": strengths.count("DIRECT"),
                "PROXY": strengths.count("PROXY"),
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    measured = coverage[coverage[["DIRECT", "PROXY"]].sum(axis=1) > 0]
    print(
        f"item map: {len(VALUE_ITEMS)} of {len(value_columns)} value-survey "
        "columns mapped"
    )
    print(measured.to_string(index=False))
    print(
        "unmeasured values: "
        + ", ".join(coverage.loc[~coverage.index.isin(measured.index), "Value"])
    )
    return


@app.cell
def _(EVALS_ROOT, json, parse_list, pd):
    # Base models: the newest completed full_string_distribution run per
    # model directory (OpenRouter, reasoning off, overall population). vLLM
    # runs in the same tree are fine-tuned adapters, not base models.
    def base_model_runs():
        runs = {}
        for model_dir in sorted(p for p in EVALS_ROOT.iterdir() if p.is_dir()):
            newest = None
            for run_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                cfg_path = run_dir / "config.json"
                csv_path = run_dir / "per_question_results.csv"
                if not (cfg_path.exists() and csv_path.exists()):
                    continue
                cfg = json.loads(cfg_path.read_text())
                if (
                    cfg.get("provider") != "openrouter"
                    or cfg.get("dataset") != "full_string_distribution"
                    or cfg.get("subpopulation") != "overall"
                    or cfg.get("use_logprobs") is not True
                    or cfg.get("reasoning")
                    or cfg.get("reasoning_mandatory")
                ):
                    continue
                stamp = cfg.get("timestamp", "")
                if newest is None or stamp > newest[0]:
                    newest = (stamp, run_dir, cfg)
            if newest is not None:
                runs[model_dir.name] = {
                    "run_dir": newest[1],
                    "config": newest[2],
                    "rows": pd.read_csv(newest[1] / "per_question_results.csv"),
                }
        return runs

    def column_tvd(rows):
        """column_name -> TVD (%) vs NZ, averaged over system prompts first."""
        out = {}
        for col, group in rows.groupby("column_name", dropna=False):
            dists = [d for d in group["model_distribution"].map(parse_list) if d]
            true = next(
                (d for d in group["true_distribution"].map(parse_list) if d), None
            )
            if not dists or true is None:
                continue
            average = [sum(vals) / len(vals) for vals in zip(*dists)]
            if len(average) != len(true):
                continue
            out[col] = 0.5 * sum(abs(a - b) for a, b in zip(average, true)) * 100
        return out

    BASE_RUNS = base_model_runs()
    BASE_TVD = {label: column_tvd(run["rows"]) for label, run in BASE_RUNS.items()}
    print(
        f"base models: {len(BASE_RUNS)} full_string_distribution runs, "
        f"{len(next(iter(BASE_TVD.values())))} questions each"
    )
    return (BASE_TVD,)


@app.cell
def _(BASE_TVD, CIRCLE, FIGS_DIR, np, plt, value_items):
    # The circular value map: angle = value position on the Schwartz circle,
    # radius = TVD from the NZ population over the value's items. The bar is
    # the mean across base models, the dots the individual models.
    DIMENSION_COLORS = {
        "Openness": "#1f77b4",
        "Self-Enhancement": "#d62728",
        "Conservation": "#2ca02c",
        "Self-Transcendence": "#9467bd",
    }

    def value_divergence():
        """Per value: {model: mean TVD} over the value's DIRECT items."""
        rows = []
        for code, name, dimension in CIRCLE:
            per_model = {}
            for model, tvd in BASE_TVD.items():
                columns = value_items(code, available=tvd)
                if columns:
                    per_model[model] = float(np.mean([tvd[col] for col in columns]))
            rows.append(
                {
                    "code": code,
                    "name": name,
                    "dimension": dimension,
                    "models": per_model,
                }
            )
        return rows

    def circular_value_map():
        rows = value_divergence()
        model_colors = {
            model: color
            for model, color in zip(
                sorted(BASE_TVD),
                plt.get_cmap("tab20")(np.linspace(0, 1, len(BASE_TVD))),
            )
        }
        angles = np.linspace(0, 2 * np.pi, len(rows), endpoint=False)
        means = np.array(
            [
                np.mean(list(row["models"].values())) if row["models"] else 0.0
                for row in rows
            ]
        )
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(17, 10))
        ax.bar(
            angles,
            means,
            width=2 * np.pi / len(rows) * 0.9,
            color=[DIMENSION_COLORS[row["dimension"]] for row in rows],
            alpha=0.22,
            zorder=1,
        )
        for angle, row in zip(angles, rows):
            for index, model in enumerate(sorted(BASE_TVD)):
                if model not in row["models"]:
                    continue
                jitter = (index - (len(BASE_TVD) - 1) / 2) * 0.016
                ax.scatter(
                    angle + jitter,
                    row["models"][model],
                    s=16,
                    color=model_colors[model],
                    alpha=0.9,
                    zorder=3,
                    linewidths=0,
                )
        ax.set_xticks(angles)
        ax.set_xticklabels(
            [row["name"].replace(" (", "\n(") for row in rows], fontsize=8
        )
        for label, angle, row in zip(ax.get_xticklabels(), angles, rows):
            label.set_color("#444444" if row["models"] else "#aaaaaa")
            screen = (90 - np.degrees(angle)) % 360
            label.set_ha("left" if screen < 180 else "right")
            rotation = screen if screen <= 90 or screen >= 270 else screen - 180
            label.set_rotation(rotation)
            label.set_va("center")
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(["25", "50", "75", "100"], fontsize=7, color="#777777")
        ax.set_rlabel_position(135)
        ax.set_title(
            "Base-model divergence around the Schwartz value circle\n"
            f"mean TVD (%) from the NZ population; each dot is one base model "
            f"on one value ({len(BASE_TVD)} models), grey values have no "
            "DIRECT items",
            fontsize=13,
        )
        dimension_legend = ax.legend(
            handles=[
                plt.Line2D([0], [0], color=color, lw=7, alpha=0.4, label=dimension)
                for dimension, color in DIMENSION_COLORS.items()
            ],
            title="Third-order dimension",
            loc="upper left",
            bbox_to_anchor=(1.20, 1.0),
            fontsize=8.5,
            frameon=False,
        )
        ax.add_artist(dimension_legend)
        ax.legend(
            handles=[
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    color=model_colors[model],
                    label=model,
                    markersize=7,
                )
                for model in sorted(BASE_TVD)
            ],
            title="Base models",
            loc="upper left",
            bbox_to_anchor=(1.20, 0.84),
            fontsize=8,
            frameon=False,
        )
        fig.subplots_adjust(right=0.56)
        FIGS_DIR.mkdir(parents=True, exist_ok=True)
        figure_path = FIGS_DIR / "value-circle.png"
        fig.savefig(figure_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return figure_path, rows

    value_circle_figure, VALUE_DIVERGENCE = circular_value_map()
    print(f"value map: wrote {value_circle_figure}")
    unmeasured = [row["code"] for row in VALUE_DIVERGENCE if not row["models"]]
    if unmeasured:
        print(f"value map: no DIRECT items for {', '.join(unmeasured)}")
    return


@app.cell
def _(
    AGENT_ACTIONABLE,
    BASE_TVD,
    CIRCLE,
    dimension_conflict,
    model_frames,
    np,
    opposition,
    parse_list,
    pd,
    resizebox,
    value_items,
    write_tex,
):
    # Tension table for one base model (every model is available in
    # BASE_TVD; the report's table uses qwen3.8-27b). Only agent-actionable
    # pairs above the opposition threshold with at least two DIRECT items on
    # each pole are ranked. The fine-tuned column is the same pair TVD for
    # the adapter trained on the full-string distribution, from its
    # modal_response eval (validation rows only); the Base TVD column is the
    # OpenRouter full-string ranking, so compare the delta, not the level.
    TENSION_MODEL = "qwen3.8-27b"
    MIN_OPPOSITION = 0.6
    MIN_ITEMS_PER_POLE = 2
    FT_MODEL = "Qwen3.8-27B-nz-wvs-full_string_distribution-overall"
    FT_COLUMN = "FT full string (val)"

    def per_column_tvd(model, split="validation"):
        """column_name -> TVD (%) for one model dir on one split, the model
        distribution averaged across system prompts."""
        _, _, df = model_frames[model]
        rows = df[df["split"] == split]
        out = {}
        for col, group in rows.groupby("column_name", dropna=False):
            dists = [d for d in group["model_distribution"].map(parse_list) if d]
            true = next(
                (d for d in group["true_distribution"].map(parse_list) if d), None
            )
            if not dists or true is None:
                continue
            avg = [sum(vals) / len(dists) for vals in zip(*dists)]
            if len(avg) != len(true):
                continue
            out[col] = 0.5 * sum(abs(a - b) for a, b in zip(avg, true)) * 100
        return out

    def pair_tvd(code_a, code_b, tvd):
        """Mean item TVD over the union of the pair's DIRECT items."""
        used = sorted(
            set(value_items(code_a, available=tvd))
            | set(value_items(code_b, available=tvd))
        )
        return round(sum(tvd[col] for col in used) / len(used), 1) if used else None

    def build_tension_table(model):
        tvd = BASE_TVD[model]
        ft_tvd = per_column_tvd(FT_MODEL)
        rows = []
        for index, (code_a, name_a, dim_a) in enumerate(CIRCLE):
            for code_b, name_b, dim_b in CIRCLE[index + 1 :]:
                if code_a not in AGENT_ACTIONABLE or code_b not in AGENT_ACTIONABLE:
                    continue
                score = opposition(code_a, code_b)
                if score < MIN_OPPOSITION:
                    continue
                items_a = value_items(code_a, available=tvd)
                items_b = value_items(code_b, available=tvd)
                if min(len(items_a), len(items_b)) < MIN_ITEMS_PER_POLE:
                    continue
                used = sorted(set(items_a) | set(items_b))
                values = [tvd[col] for col in used]
                base_tvd = float(np.mean(values))
                rows.append(
                    {
                        "tension": f"{name_a} / {name_b}",
                        "a": code_a,
                        "b": code_b,
                        "Opp.": round(score, 2),
                        "Base TVD": round(base_tvd, 1),
                        FT_COLUMN: pair_tvd(code_a, code_b, ft_tvd),
                        "Items A/B": f"{len(items_a)}/{len(items_b)}",
                        "min_pole": min(len(items_a), len(items_b)),
                        "spread": round(float(np.max(values) - np.min(values)), 1),
                        "Comp.": round(score * base_tvd, 1),
                        "conflict": dimension_conflict(dim_a, dim_b),
                    }
                )
        return (
            pd.DataFrame(rows)
            .sort_values("Comp.", ascending=False)
            .reset_index(drop=True)
        )

    def write_tension_table(table, model):
        """bare booktabs tabular (hidden index, like the
        divergent-questions tables); the report pulls it in with \\ctable."""
        top = table.copy()
        frame = pd.DataFrame(
            {
                "Schwartz value tension": [
                    f"{rank + 1}. {row.tension}" + ("$^{*}$" if row.conflict else "")
                    for rank, row in top.iterrows()
                ],
                "Opp.": [f"{value:.2f}" for value in top["Opp."]],
                "Base TVD": [f"{value:.1f}" for value in top["Base TVD"]],
                "Items A/B": top["Items A/B"].tolist(),
                "Comp.": [f"{value:.1f}" for value in top["Comp."]],
            }
        )
        tex = (
            frame.style.hide(axis="index")
            .format(na_rep="--")
            .to_latex(
                column_format="p{0.42\\textwidth}rrrr",
                hrules=True,
                convert_css=True,
            )
        )
        return write_tex(resizebox(tex), f"value-tensions-{model}.tex")

    tension_table = build_tension_table(TENSION_MODEL)
    print(f"tensions: {len(tension_table)} pairs ranked for {TENSION_MODEL}; ")
    print(
        tension_table.head(10)[
            ["tension", "Opp.", "Base TVD", FT_COLUMN, "Items A/B", "Comp."]
        ].to_string(index=False)
    )
    return (tension_table,)


@app.cell
def _(tension_table):
    # The full ranked candidate list behind the report table: every pair that
    # passes the candidate rule, with the columns used to curate the final
    # eleven dilemma tensions (code/dilemmas/tensions.json). "Min" is the
    # thinner pole's DIRECT-item count and "Spread" the item-TVD range behind
    # the pair mean - caveats on a row, not scores.

    def curation_table(table):
        full = table[
            [
                "tension",
                "a",
                "b",
                "Opp.",
                "Base TVD",
                "FT full string (val)",
                "Items A/B",
                "min_pole",
                "spread",
                "Comp.",
                "conflict",
            ]
        ].rename(
            columns={
                "tension": "Tension",
                "min_pole": "Min",
                "spread": "Spread",
                "conflict": "Higher-order",
            }
        )
        full.index = range(1, len(full) + 1)
        return full

    curation_df = curation_table(tension_table)
    curation_df
    return


if __name__ == "__main__":
    app.run()
