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
    import matplotlib.patheffects as patheffects
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
        patheffects,
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
def _(FIGS_DIR, pd):
    # Shared LaTeX table rendering, used by all sections below. Styler
    # .to_latex (convert_css) turns CSS font-weight into \bfseries, but
    # escapes cell values only, so headers with special chars (%, _, ...)
    # need escape_header. Tables are wrapped in \resizebox{\textwidth}{!}{...}
    # so they never overflow the page; the report pulls them in with
    # \ctable{...}{...}.

    def escape_header(label):
        """Escape LaTeX special chars in a column header."""

        return (
            label.replace("\\", "\\textbackslash{}")
            .replace("&", "\\&")
            .replace("%", "\\%")
            .replace("$", "\\$")
            .replace("#", "\\#")
            .replace("_", "\\_")
            .replace("{", "\\{")
            .replace("}", "\\}")
        )

    def escape_latex(text):
        """Escape LaTeX special chars in cell text (question/answer strings)."""

        return (
            text.replace("\\", "\\textbackslash{}")
            .replace("&", "\\&")
            .replace("%", "\\%")
            .replace("$", "\\$")
            .replace("#", "\\#")
            .replace("_", "\\_")
            .replace("{", "\\{")
            .replace("}", "\\}")
            .replace("~", "\\textasciitilde{}")
            .replace("^", "\\textasciicircum{}")
        )

    def highlight_best(df, senses=None):
        """Bold the best value per column: max for numeric columns by default,
        or per column via senses={label: "max"|"min"} (e.g. for MultiIndex
        columns keyed by metric)."""

        def highlight(s):
            if senses is None:
                if not pd.api.types.is_numeric_dtype(s):
                    return ["" for _ in s]  # e.g. CI columns: no highlighting
                best = s.max()
            else:
                best = s.max() if senses[s.name[0]] == "max" else s.min()
            return ["font-weight: bold;" if v == best else "" for v in s]

        return df.style.apply(highlight, axis=0)

    def write_tex(tex, out_name):
        """Write raw LaTeX to a file in FIGS_DIR."""
        out_path = FIGS_DIR / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(tex)
        return out_path

    def write_latex_table(df, fmt, n_cols, out_name, senses=None, column_format=None):
        """Style, format, and write a DataFrame to a resizebox-wrapped booktabs
        tabular in FIGS_DIR. The index name is suppressed (Styler would emit an
        extra header row), so the "Model" header is patched into the top-left
        corner of the first header row."""
        tex = (
            highlight_best(df, senses=senses)
            .format(fmt, na_rep="--")
            .to_latex(
                convert_css=True,
                column_format=column_format or ("l" + "r" * n_cols),
                hrules=True,
                multicol_align="l",
            )
        )
        tex = tex.replace(" & \\multicolumn", "Model & \\multicolumn", 1)
        # Scale the table to the text width so it never overflows the page.
        tex = "\\resizebox{\\textwidth}{!}{%\n" + tex + "}"
        return write_tex(tex, out_name)

    return escape_header, escape_latex, write_latex_table, write_tex


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
def _(escape_latex, model_frames, np, parse_list, pd, textwrap, write_tex):
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
def _(EVALS_ROOT, load_overall_per_question, math, np, parse_list, plt, textwrap):
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
    ## NZ value map (base models + fine-tuned adapters)

    Places models in a two-dimensional value space and compares them with
    the NZ population:

    - **OpenRouter base models:** one completed non-reasoning
      `full_string_distribution` run each (full-string logprob scoring).
    - **Fine-tuned LoRA adapters (vLLM):** the newest run on the adapter's
      own training format — `modal_response` for the SFT adapters,
      `first_token_distribution` for the first-token adapter — restricted
      to the same dataset revision as the base runs, so each adapter is
      scored on the prompt format it was trained on.

    The axes are the first two principal components of the **actual NZ
    respondents'** answers to the 251 WVS items
    (`wvs_value_survey.csv`), so PC1 is the dominant dimension of real
    human value variation in NZ and PC2 the next-largest contrast.
    Each model is a dot projected from its prompt-averaged answer
    distribution per item (an aggregate value profile, not a respondent).
    The NZ population star is the empirical `expected_distribution` of the
    same items projected the same way; it sits at the centre of the
    respondent cloud by construction, so a model's distance from the star
    is its distance from NZ's value profile.

    The figure is written to `code/figures/value-map.png`; the coordinate
    and item-loading tables go to `output/value_map/` (not the figure
    directory). Runs where reasoning could not be turned off are excluded.
    """)
    return


@app.cell
def _(EVALS_ROOT, Path, json, pd):
    # Value-map inputs (standard output/evals/<model>/<run>/ layout written
    # by evaluate.py): newest eligible run per model.
    VALUE_MAP_DATASET = "full_string_distribution"
    VALUE_MAP_ADAPTER_DATASETS = ("modal_response", "first_token_distribution")
    # Models kept out of the value map to keep the figure readable (matched
    # as substrings, lower-cased, against the base model id or adapter slug).
    VALUE_MAP_EXCLUDE = (
        "qwen3.5-9b",
        "qwen3.6-27b",
        "glm-4.7-flash",
        "mistral-nemo",
        "deepseek-v4-pro-0813",
        "gemma-4-26b-a4b-it",
    )
    VALUE_MAP_TRAINING = (
        Path(__file__).resolve().parents[1] / "training-dataset" / "output"
    )
    ADAPTER_METHOD_LABELS = {
        "modal_response": "modal",
        "sampled_response": "sampled",
        "full_string_distribution": "full-str",
        "first_token_distribution": "1st-tok",
    }

    def value_map_excluded(target):
        """Whether a base model id / adapter slug is excluded from the map."""
        lowered = str(target or "").lower()
        return any(slug in lowered for slug in VALUE_MAP_EXCLUDE)

    def value_map_eval_runs():
        """model label -> (run dir, config, per-question rows)."""
        base_candidates = {}  # model dir -> newest OpenRouter run
        adapter_candidates = {}  # adapter target -> {dataset: newest run}

        def newer(candidate, current):
            return current is None or candidate[0] > current[0]

        for model_dir in sorted(p for p in EVALS_ROOT.iterdir() if p.is_dir()):
            for run_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                cfg_path = run_dir / "config.json"
                csv_path = run_dir / "per_question_results.csv"
                if not (cfg_path.exists() and csv_path.exists()):
                    continue
                cfg = json.loads(cfg_path.read_text())
                if (
                    cfg.get("subpopulation") != "overall"
                    or cfg.get("use_logprobs") is not True
                    or cfg.get("reasoning")
                    or cfg.get("reasoning_mandatory")
                    or cfg.get("aborted")
                ):
                    continue
                provider = cfg.get("provider")
                dataset = cfg.get("dataset")
                entry = (cfg.get("timestamp", ""), run_dir, csv_path, cfg)
                if provider == "openrouter" and dataset == VALUE_MAP_DATASET:
                    if value_map_excluded(cfg.get("target")):
                        continue
                    if newer(entry, base_candidates.get(model_dir)):
                        base_candidates[model_dir] = entry
                elif provider == "vllm" and dataset in VALUE_MAP_ADAPTER_DATASETS:
                    target = cfg.get("target") or ""
                    if "-nz-wvs-" not in target:
                        continue  # self-served base model, not an adapter
                    if value_map_excluded(target):
                        continue
                    per_dataset = adapter_candidates.setdefault(target, {})
                    if newer(entry, per_dataset.get(dataset)):
                        per_dataset[dataset] = entry

        found = {}
        base_shas = []
        for model_dir, (_, run_dir, csv_path, cfg) in sorted(base_candidates.items()):
            label = (
                cfg.get("target", model_dir.name)
                .split("/")[-1]
                .replace(":free", " (free)")
            )
            found[label] = (run_dir, cfg, pd.read_csv(csv_path))
            base_shas.append((cfg.get("timestamp", ""), cfg.get("dataset_sha")))
        # Fine-tuned adapters must be on the same dataset revision as the
        # base runs (excludes adapters trained before the last rebuild).
        current_sha = sorted(base_shas)[-1][1] if base_shas else None

        for target, per_dataset in sorted(adapter_candidates.items()):
            eligible = {
                ds: entry
                for ds, entry in per_dataset.items()
                if current_sha is None or entry[3].get("dataset_sha") == current_sha
            }
            slug, _, suffix = target.partition("-nz-wvs-")
            method = suffix.rsplit("-", 1)[0]
            if method not in ADAPTER_METHOD_LABELS or not eligible:
                continue
            # Scored on the format the adapter was trained on: the
            # first-token adapter on lettered prompts, every other SFT
            # adapter on the standard option-text prompts.
            preferred_dataset = (
                "first_token_distribution"
                if method == "first_token_distribution"
                else "modal_response"
            )
            chosen = eligible.get(preferred_dataset) or sorted(eligible.values())[-1]
            _, run_dir, csv_path, cfg = chosen
            label = f"{slug} ({ADAPTER_METHOD_LABELS[method]})"
            found[label] = (run_dir, cfg, pd.read_csv(csv_path))
        return found

    value_map_runs = value_map_eval_runs()
    print(f"value map: {len(value_map_runs)} model runs")
    for label, (run_dir, _, df) in value_map_runs.items():
        if "is_correct" in df:
            acc_str = f"{df['is_correct'].mean() * 100:>5.1f}%"
        else:
            acc_str = "  n/a"
        mean_kl = df["kl_divergence"].mean() if "kl_divergence" in df else float("nan")
        print(f"  {label:<30} acc {acc_str}  mean KL {mean_kl:>6.3f}  ({run_dir.name})")
    return VALUE_MAP_DATASET, VALUE_MAP_TRAINING, value_map_runs


@app.cell
def _(VALUE_MAP_DATASET, VALUE_MAP_TRAINING, np, pd, parse_list, value_map_runs):
    # Item-level profiles: prompt-averaged model distributions and the
    # empirical NZ distribution (overall population) for every WVS item.
    #
    # Every model distribution is re-aligned to the canonical local category
    # order by normalised name before averaging; items whose option sets
    # genuinely differ are dropped and reported.

    def value_map_normalise(name):
        """Case/punctuation-insensitive category name for matching."""
        text = str(name).strip().lower()
        for char in ".,;:!?()[]{}\"'’":
            text = text.replace(char, "")
        return " ".join(text.split())

    def value_map_align_distribution(csv_categories, csv_dist, target_categories):
        """Reorder one model distribution to the canonical category order.

        Matches by normalised name (evals run on the corrected dataset, so
        the option sets are the same). Returns ``(aligned, unmatched_mass)``:
        categories that cannot be matched are dropped from the vector and
        their probability mass is reported so the caller can discard items
        whose option sets genuinely differ.
        """
        target_keys = [value_map_normalise(cat) for cat in target_categories]
        matched = {}
        unmatched_mass = 0.0
        for cat, prob in zip(csv_categories, csv_dist):
            key = value_map_normalise(cat)
            if key in target_keys:
                matched[key] = float(prob)
            else:
                unmatched_mass += float(prob)
        aligned = [matched.get(key, 0.0) for key in target_keys]
        total = sum(aligned)
        return ([p / total for p in aligned] if total > 0 else []), unmatched_mass

    def value_map_items():
        """Ordered items from the canonical validation parquet."""
        ref = pd.read_parquet(
            VALUE_MAP_TRAINING
            / "dataset"
            / VALUE_MAP_DATASET
            / "validation"
            / "overall.parquet"
        )
        items, seen = [], set()
        for _, row in ref.iterrows():
            key = (str(row["question_id"]), row["column_name"])
            if key in seen:
                continue
            seen.add(key)
            items.append(
                {
                    "key": key,
                    "categories": list(row["categories"]),
                    "question": str(row["question"]),
                    "sub_question": str(row["sub_question"]),
                }
            )
        return items

    def value_map_model_profiles():
        items = {item["key"]: item for item in value_map_items()}
        profiles = {}
        skipped = set()
        for label, (_, _, df) in value_map_runs.items():
            profile = {}
            for (qid, col), group in df.groupby(["question_id", "column_name"]):
                key = (str(qid), col)
                item = items.get(key)
                if item is None:
                    continue
                # Prefer reasoning-free rows; fall back to rows with
                # reasoning only when that is all the item has, so no item
                # drops out of the map entirely.
                if "reasoning_chars" in group.columns:
                    clean = group[group["reasoning_chars"].fillna(0) == 0]
                    group = clean if len(clean) else group
                dists = []
                for _, row in group.iterrows():
                    cats = parse_list(row.get("categories"))
                    dist = parse_list(row.get("model_distribution"))
                    if not cats or not dist or sum(dist) <= 0:
                        continue
                    aligned, unmatched = value_map_align_distribution(
                        cats, dist, item["categories"]
                    )
                    if unmatched > 0.02:
                        # The eval copy of this item offers different options
                        # (e.g. party lists); it cannot be compared cleanly.
                        skipped.add(key)
                        continue
                    if aligned:
                        dists.append(aligned)
                if dists:
                    profile[key] = np.mean(np.array(dists), axis=0).tolist()
            profiles[label] = profile
        if skipped:
            columns = ", ".join(sorted({key[1] for key in skipped}))
            print(
                f"value map: dropped {len(skipped)} items whose eval option set "
                f"differs from the canonical dataset ({columns})"
            )
        return profiles

    def value_map_nz_profile():
        df = pd.read_parquet(
            VALUE_MAP_TRAINING
            / "dataset"
            / VALUE_MAP_DATASET
            / "validation"
            / "overall.parquet"
        )
        profile = {}
        for _, row in df.iterrows():
            key = (str(row["question_id"]), row["column_name"])
            if key not in profile:
                profile[key] = list(row["expected_distribution"])
        return profile

    value_map_data = {
        "items": value_map_items(),
        "profiles": value_map_model_profiles(),
        "nz": value_map_nz_profile(),
    }
    print(
        f"value map: aligned {len(value_map_data['profiles'])} model profiles to "
        f"{len(value_map_data['items'])} canonical items"
    )
    return (value_map_data,)


@app.cell
def _(VALUE_MAP_TRAINING, json, np, pd, value_map_data):
    # Fitting the NZ value axes on the real respondents and projecting every
    # group (models + NZ population) onto them.

    def value_map_respondent_matrix(items):
        """One-hot respondent x item-category matrix.

        Answer codes are mapped to categories with question_mapping.json
        (the same mapping the dataset build uses); missing/refused codes
        contribute an all-zero item block.
        """
        survey = pd.read_csv(VALUE_MAP_TRAINING / "wvs_value_survey.csv")
        mapping = {}
        for entry in json.loads(
            (VALUE_MAP_TRAINING / "question_mapping.json").read_text()
        ):
            for col in entry["column_names"]:
                mapping[col] = entry

        blocks = []
        for item in items:
            entry = mapping[item["key"][1]]
            code_to_idx = {
                round(float(code), 3): idx
                for idx, code in enumerate(entry["numeric_response_types"])
            }
            values = survey[item["key"][1]].map(
                lambda v: code_to_idx.get(round(float(v), 3)) if pd.notna(v) else None
            )
            block = np.zeros((len(survey), len(item["categories"])))
            valid = values.notna().to_numpy()
            block[np.where(valid)[0], values[valid].astype(int).to_numpy()] = 1.0
            blocks.append(block)

        return np.hstack(blocks).astype(float)

    def value_map_build():
        groups = dict(value_map_data["profiles"])
        groups["NZ population"] = value_map_data["nz"]

        # Keep only items present in every group so all vectors align.
        keys = [
            item["key"]
            for item in value_map_data["items"]
            if all(item["key"] in p for p in groups.values())
        ]
        items = [item for item in value_map_data["items"] if item["key"] in set(keys)]
        block_bounds, offset = {}, 0
        for item in items:
            k = len(item["categories"])
            block_bounds[item["key"]] = (offset, offset + k)
            offset += k

        matrix = value_map_respondent_matrix(items)
        mean = matrix.mean(axis=0)
        centered = matrix - mean
        _, singular, components = np.linalg.svd(centered, full_matrices=False)
        explained = singular**2 / (singular**2).sum()

        # Fix the component signs so the axes read consistently across reruns
        # (SVD signs are arbitrary): PC1+ = traditional (homosexuality least
        # justifiable), PC2+ = secular (no belief in God).
        def _anchor_weight(axis_idx, column_name, category_name):
            for item in items:
                if item["key"][1] != column_name:
                    continue
                start, end = block_bounds[item["key"]]
                cats = item["categories"]
                if category_name in cats:
                    return components[axis_idx, start + cats.index(category_name)]
            return None

        for axis_idx, (column_name, category_name) in enumerate(
            [("Q182", "10"), ("Q165", "Yes")]
        ):
            weight = _anchor_weight(axis_idx, column_name, category_name)
            if weight is not None and weight > 0:
                components[axis_idx] *= -1

        coords = []
        for label, profile in groups.items():
            vec = np.concatenate(
                [np.asarray(profile[key], dtype=float) for key in keys]
            )
            xy = (vec - mean) @ components[:2].T
            coords.append({"group": label, "pc1": xy[0], "pc2": xy[1]})

        loading_rows = []
        axis_totals = {0: 0.0, 1: 0.0}
        for item in items:
            start, end = block_bounds[item["key"]]
            weights = components[:2, start:end]
            for axis_idx, axis_label in enumerate(["PC1", "PC2"]):
                w = weights[axis_idx]
                dominant = int(np.argmax(np.abs(w)))
                squared = float(np.sum(w**2))
                axis_totals[axis_idx] += squared
                loading_rows.append(
                    {
                        "item": f"{item['key'][0]}:{item['key'][1]}",
                        "column_name": item["key"][1],
                        "question": item["question"],
                        "sub_question": item["sub_question"],
                        "axis": axis_label,
                        "squared_loading": squared,
                        "loading_share": 0.0,  # filled below
                        "dominant_category": item["categories"][dominant],
                        "dominant_loading": float(w[dominant]),
                        "direction": "+" if w[dominant] >= 0 else "-",
                    }
                )
        loadings = pd.DataFrame(loading_rows)
        for axis_idx, axis_label in enumerate(["PC1", "PC2"]):
            mask = loadings["axis"] == axis_label
            loadings.loc[mask, "loading_share"] = (
                loadings.loc[mask, "squared_loading"] / axis_totals[axis_idx]
            )

        return {
            "coords": pd.DataFrame(coords),
            "respondent_xy": centered @ components[:2].T,
            "explained": explained,
            "loadings": loadings,
            "n_items": len(items),
        }

    value_map_result = value_map_build()
    print(
        f"value map: PCA on {value_map_result['respondent_xy'].shape[0]} NZ "
        f"respondents x {value_map_result['n_items']} items — PC1 "
        f"{value_map_result['explained'][0] * 100:.1f}%, PC2 "
        f"{value_map_result['explained'][1] * 100:.1f}%"
    )
    loadings_all = value_map_result["loadings"]
    for axis_label in ["PC1", "PC2"]:
        top = loadings_all[loadings_all["axis"] == axis_label].nlargest(
            10, "squared_loading"
        )
        print(f"\nItems specifying {axis_label} (largest share of the axis):")
        for row in top.itertuples():
            item_label = (
                row.sub_question
                if row.sub_question not in ("", "nan", "None")
                else row.question[:80]
            )
            print(
                f"  {row.column_name:<6} {item_label[:78]:<78} "
                f"-> {row.dominant_category} ({row.direction}, "
                f"{row.loading_share * 100:.1f}% of axis)"
            )
    return (value_map_result,)


@app.cell
def _(EVALS_ROOT, FIGS_DIR, patheffects, plt, value_map_result):
    # Drawing the map (NZ respondent cloud + numbered models + NZ population)
    # and writing the figure and coordinate tables
    # (figure -> code/figures, data -> output/value_map).
    MODEL_COLORS = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#bcbd22",
        "#17becf",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
        "#c49c94",
        "#f7b6d2",
        "#dbdb8d",
        "#9edae5",
        "#7f7f7f",
        "#c7c7c7",
    ]
    NZ_COLOR = "#111111"

    def value_map_draw():
        coords = value_map_result["coords"]
        respondent_xy = value_map_result["respondent_xy"]
        explained = value_map_result["explained"]
        loadings = value_map_result["loadings"]

        model_rows = coords[coords["group"] != "NZ population"].sort_values(
            "group", key=lambda col: col.str.lower()
        )
        nz = coords[coords["group"] == "NZ population"].iloc[0]
        # Number the models 1..N alphabetically; the numbers appear on the
        # dots and in the legend next to the model names.
        number_map = {group: idx + 1 for idx, group in enumerate(model_rows["group"])}
        color_map = {
            group: MODEL_COLORS[idx % len(MODEL_COLORS)]
            for idx, group in enumerate(model_rows["group"])
        }

        fig, ax = plt.subplots(figsize=(12.5, 8))
        legend_handles, legend_labels = [], []

        legend_handles.append(
            ax.scatter(
                respondent_xy[:, 0],
                respondent_xy[:, 1],
                s=9,
                alpha=0.2,
                color="#999999",
                linewidths=0,
            )
        )
        legend_labels.append("NZ respondents")

        for _, row in model_rows.iterrows():
            color = color_map[row["group"]]
            legend_handles.append(
                ax.scatter(
                    row["pc1"],
                    row["pc2"],
                    s=190,
                    color=color,
                    zorder=5,
                    edgecolors="white",
                    linewidths=0.8,
                )
            )
            legend_labels.append(f"{number_map[row['group']]}. {row['group']}")
            marker = ax.annotate(
                str(number_map[row["group"]]),
                (row["pc1"], row["pc2"]),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
                zorder=6,
            )
            marker.set_path_effects(
                [patheffects.withStroke(linewidth=1.8, foreground="#444444")]
            )

        legend_handles.append(
            ax.scatter(
                nz["pc1"],
                nz["pc2"],
                s=320,
                marker="*",
                color=NZ_COLOR,
                zorder=6,
            )
        )
        legend_labels.append("NZ population (WVS-7)")
        ax.annotate(
            "NZ population",
            (nz["pc1"], nz["pc2"]),
            xytext=(10, -16),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=NZ_COLOR,
        )

        # Pole names are read off the top item loadings (see
        # output/value_map/value-map-item-loadings.csv): PC1 separates
        # progressive answers (homosexuality, gender equality, trust; −) from
        # traditional ones (+); PC2 separates religious answers (God, church;
        # −) from secular ones (+).
        ax.set_xlabel(
            f"Value dimension 1 — PC1 ({explained[0] * 100:.0f}%): "
            "traditional (+) ↔ progressive (−)"
        )
        ax.set_ylabel(
            f"Value dimension 2 — PC2 ({explained[1] * 100:.0f}%): "
            "religious (−) ↔ secular (+)"
        )
        ax.set_title(
            "Where models sit in NZ value space\n"
            f"(PCA of {value_map_result['n_items']} WVS items answered by "
            "NZ respondents)"
        )
        ax.axhline(0, color="#dddddd", lw=0.8, zorder=0)
        ax.axvline(0, color="#dddddd", lw=0.8, zorder=0)

        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=9,
            frameon=False,
            title="Models (alphabetical)",
            title_fontsize=9,
        )

        def _axis_hint(axis, n=4):
            """Short 'Qn: label' list of the items loading most on one axis."""
            top = loadings[loadings["axis"] == axis].nlargest(n, "squared_loading")
            hints = []
            for row in top.itertuples():
                text = (
                    row.sub_question
                    if row.sub_question not in ("", "nan", "None")
                    else row.question
                )
                hints.append(f"{row.column_name}: {text[:32].strip()}")
            return "  ·  ".join(hints)

        fig.text(
            0.42,
            -0.02,
            f"PC1 ({explained[0] * 100:.0f}%)  {_axis_hint('PC1')}",
            ha="center",
            fontsize=7.5,
            color="#555555",
        )
        fig.text(
            0.42,
            -0.06,
            f"PC2 ({explained[1] * 100:.0f}%)  {_axis_hint('PC2')}",
            ha="center",
            fontsize=7.5,
            color="#555555",
        )

        FIGS_DIR.mkdir(parents=True, exist_ok=True)
        figure_path = FIGS_DIR / "value-map.png"
        fig.savefig(figure_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        # Coordinate / loading tables are data, not figures.
        data_dir = EVALS_ROOT.parent / "value_map"
        data_dir.mkdir(parents=True, exist_ok=True)
        coords.to_csv(data_dir / "value-map-coordinates.csv", index=False)
        loadings.to_csv(data_dir / "value-map-item-loadings.csv", index=False)
        return figure_path

    value_map_figure = value_map_draw()
    print(f"value map: wrote {value_map_figure}")
    print(
        value_map_result["coords"]
        .sort_values("group", key=lambda col: col.str.lower())[["group", "pc1", "pc2"]]
        .round(3)
        .to_string(index=False)
    )
    return


@app.cell
def _(FIGS_DIR, np, plt, textwrap, value_map_result):
    # Explains the two dimensions: the items loading most strongly on each
    # axis, plotted as signed bars for the item's most-loading answer. Full
    # question text is wrapped so the figure is readable on its own. PC1+ is
    # the traditional pole, PC2+ the secular pole.
    def value_map_loadings_figure():
        loadings = value_map_result["loadings"]
        explained = value_map_result["explained"]
        axis_info = {
            "PC1": (
                0,
                "traditional (+) ↔ progressive (−)",
                "right of zero = traditional answers, left = progressive answers",
            ),
            "PC2": (
                1,
                "secular (+) ↔ religious (−)",
                "right of zero = secular answers, left = religious answers",
            ),
        }

        def full_label(row):
            """Qn + full question text (sub-item first for matrix questions)."""
            stem = str(row.question).strip()
            sub = str(row.sub_question).strip()
            text = f"{sub} — {stem}" if sub and sub not in ("", "nan", "None") else stem
            return f"{row.column_name}. {textwrap.fill(text, width=85)}"

        fig, axes = plt.subplots(2, 1, figsize=(14, 17))
        for ax, axis_label in zip(axes, ["PC1", "PC2"]):
            axis_idx, poles, hint = axis_info[axis_label]
            top = (
                loadings[loadings["axis"] == axis_label]
                .nlargest(10, "squared_loading")
                .iloc[::-1]
            )
            labels, values, shares = [], [], []
            for row in top.itertuples():
                labels.append(
                    f"{full_label(row)}\nMost-loading answer: {row.dominant_category}"
                )
                values.append(
                    np.sign(row.dominant_loading) * np.sqrt(row.squared_loading)
                )
                shares.append(row.loading_share * 100)
            colors = ["#d62728" if value >= 0 else "#1f77b4" for value in values]
            ax.barh(range(len(values)), values, height=0.62, color=colors, alpha=0.85)
            for y, value, share in zip(range(len(values)), values, shares):
                ax.annotate(
                    f"{share:.1f}% of {axis_label}",
                    (value, y),
                    xytext=(3 if value >= 0 else -3, 0),
                    textcoords="offset points",
                    va="center",
                    ha="left" if value >= 0 else "right",
                    fontsize=7.5,
                    color="#555555",
                )
            ax.set_yticks(range(len(values)), labels, fontsize=7.5)
            ax.axvline(0, color="#888888", lw=0.8)
            ax.set_title(
                f"{axis_label} ({explained[axis_idx] * 100:.1f}% of NZ respondent "
                f"variance): {poles}\n{hint}",
                fontsize=11,
                linespacing=1.4,
            )
            ax.set_xlabel("signed loading of the item's most-loading answer")
            ax.margins(x=0.18)
        fig.suptitle(
            "What specifies the two value dimensions — top 10 WVS items per axis",
            fontsize=13,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.98], h_pad=5.5)
        FIGS_DIR.mkdir(parents=True, exist_ok=True)
        figure_path = FIGS_DIR / "value-map-loadings.png"
        fig.savefig(figure_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return figure_path

    value_map_loadings_path = value_map_loadings_figure()
    print(f"value map: wrote {value_map_loadings_path}")
    return


if __name__ == "__main__":
    app.run()
