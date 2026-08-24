import marimo

__generated_with = "0.24.0"
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
    ## Base models vs the population

    How far is each base model's response distribution from the NZ
    population, per split (mean over held-out validation questions and
    all system prompts).
    """)
    return


@app.cell
def _(EVALS_ROOT, json, np, pd):

    def collect_base_tvd():
        """Per (base model, split) mean TVD (%) from the newest
        non-reasoning modal-config eval run of each base model."""
        rows = []
        for model_dir in sorted(EVALS_ROOT.iterdir()):
            if not model_dir.is_dir() or "nz-wvs-" in model_dir.name:
                continue  # base models only
            runs = []
            for run_dir in model_dir.iterdir():
                cfg_path = run_dir / "config.json"
                if not cfg_path.exists():
                    continue
                cfg = json.loads(cfg_path.read_text())
                if cfg.get("dataset") != "modal_response" or cfg.get("reasoning"):
                    continue
                runs.append((cfg.get("timestamp", ""), run_dir))
            if not runs:
                continue
            _, run_dir = sorted(runs)[-1]
            df = pd.read_csv(run_dir / "per_question_results.csv")
            if df.empty or "true_distribution" not in df.columns:
                continue
            df = df.assign(
                subpopulation=df["subpopulation"]
                if "subpopulation" in df
                else "overall",
                split=df["split"] if "split" in df else "validation",
            )
            df = df[df["subpopulation"] == "overall"]

            def row_tvd(row):
                md = row.get("model_distribution")
                td = row.get("true_distribution")
                try:
                    md = json.loads(md) if isinstance(md, str) else md
                    td = json.loads(td) if isinstance(td, str) else td
                    return 0.5 * sum(abs(a - b) for a, b in zip(md, td)) * 100
                except Exception:
                    return float("nan")

            df["tvd"] = df.apply(row_tvd, axis=1)
            for (split, sp_id), h in df.groupby(
                ["split", "system_prompt_id"], dropna=False
            ):
                rows.append(
                    {
                        "base_model": model_dir.name,
                        "split": split,
                        "system_prompt_id": sp_id,
                        "n": len(h),
                        "tvd": h["tvd"].mean(),
                    }
                )
        per = pd.DataFrame(rows)
        if per.empty:
            return pd.DataFrame()
        # Mean across prompts per split.
        per_split = (
            per.groupby(["base_model", "split"], dropna=False)["tvd"].mean().unstack()
        )
        # "All": per prompt, train + validation pooled by question count,
        # then mean across prompts.
        all_rows = []
        for (bm, sp_id), h in per.groupby(
            ["base_model", "system_prompt_id"], dropna=False
        ):
            all_rows.append(
                {"base_model": bm, "tvd": np.average(h["tvd"], weights=h["n"])}
            )
        per_all = (
            pd.DataFrame(all_rows).groupby("base_model", dropna=False)["tvd"].mean()
        )
        out = per_split.reindex(columns=["validation", "train"])
        out["overall"] = per_all
        return out[["validation", "train", "overall"]]

    base_tvd = collect_base_tvd()
    print("\n=== How far is each base model from the NZ population? ===")
    print(
        "TVD (%) = 0.5 * sum |model - true| over the response categories; "
        "0 = answers exactly like NZ, 100 = completely different. "
        "Validation = held-out questions/prompts."
    )
    print(
        base_tvd.rename(
            columns={
                "validation": "TVD (%) validation",
                "train": "TVD (%) train",
                "overall": "TVD (%) all",
            }
        )
        .round(1)
        .to_string()
    )
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
    against the empirical response distribution. Above the table, a short
    section reports the mean TVD (%) between each base model's response
    distribution and the NZ population (0.5 * sum |model - true| over the
    response categories; 0 = answers exactly like NZ). Every model (base
    and fine-tuned) is
    scored on the overall population. The LaTeX
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
    METHOD_ORDER = [
        "modal_response",
        "sampled_response",
        "first_token_distribution",
        "full_string_distribution",
    ]
    SPLITS = ["train", "validation", "overall"]
    METRICS = ["accuracy", "cross_entropy", "kl_divergence"]

    def base_and_method(model_name):
        # "{base}-nz-wvs-{method}" -> (base, method); no suffix -> (name, "base")
        if "nz-wvs-" in model_name:
            base, method = model_name.split("nz-wvs-", 1)
            return base.rstrip("-"), method
        return model_name, "base"

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
            df = df[df["subpopulation"] == "overall"]
            base, method = base_and_method(model_dir.name)
            for (split, sp_id), h in df.groupby(
                ["split", "system_prompt_id"], dropna=False
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
            sub = metrics[metrics["base_model"] == base_model]
            rows = []
            base_row = sub[sub["method"] == "base"]
            if not base_row.empty:
                rows.append((0, "Base", base_row))
            adapters = []
            for method, g in sub[sub["method"] != "base"].groupby(
                "method", dropna=False
            ):
                stem = method.rsplit("-", 1)[0]
                adapters.append(
                    (
                        METHOD_ORDER.index(stem)
                        if stem in METHOD_ORDER
                        else len(METHOD_ORDER),
                        METHOD_PRETTY.get(stem, stem),
                        g,
                    )
                )
            rows += sorted(adapters, key=lambda r: (r[0], r[1]))

            # Collapse duplicates: average the per-split metrics across all
            # eval dirs that map to the same method label (e.g. reruns).
            by_label = {}
            for sort_key, label, g in rows:
                by_label.setdefault((sort_key, label), []).append(g)
            collapsed = []
            for (sort_key, label), gs in by_label.items():
                merged = (
                    pd.concat(gs)
                    .groupby("split", dropna=False)[METRICS]
                    .mean()
                    .reset_index()
                )
                collapsed.append((sort_key, label, merged))
            collapsed.sort(key=lambda r: r[0])

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
    return METHOD_ORDER, METHOD_PRETTY, base_and_method, base_refs, tables


@app.cell
def _(FIGS_DIR, pd):
    # Shared LaTeX table rendering, used by both the main results table and
    # the capability table below. Styler.to_latex (convert_css) turns CSS
    # font-weight into \bfseries, but escapes cell values only, so headers
    # with special chars (%, _, ...) need escape_header. Each file holds only
    # the tabular, wrapped in \resizebox{\textwidth}{!}{...} so it never
    # overflows the page; the report pulls it in with \ctable{...}{...}.

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

    def write_latex_table(df, fmt, n_cols, out_name, senses=None):
        """Style, format, and write a DataFrame to a resizebox-wrapped booktabs
        tabular in FIGS_DIR. The index name is suppressed (Styler would emit an
        extra header row), so the "Model" header is patched into the top-left
        corner of the first header row."""
        tex = (
            highlight_best(df, senses=senses)
            .format(fmt, na_rep="--")
            .to_latex(
                convert_css=True,
                column_format="l" + "r" * n_cols,
                hrules=True,
                multicol_align="l",
            )
        )
        tex = tex.replace(" & \\multicolumn", "Model & \\multicolumn", 1)
        # Scale the table to the text width so it never overflows the page.
        tex = "\\resizebox{\\textwidth}{!}{%\n" + tex + "}"
        out_path = FIGS_DIR / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(tex)
        return out_path

    return escape_header, write_latex_table


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
def _(EVALS_ROOT, METHOD_ORDER, METHOD_PRETTY, base_and_method, json, pd):
    # Building the capability table data from summary.json of the newest run
    # per model (same "pick newest run by config timestamp" rule as the main
    # table).

    CAPABILITY_ROOT = EVALS_ROOT.parent / "capability"
    TASK_LABELS = {
        "mmlu_pro": "MMLU-Pro",
        "gpqa_diamond": "GPQA Diamond",
        "swe_bench": "SWE-bench",
    }
    TASK_ORDER = ["mmlu_pro", "gpqa_diamond", "swe_bench"]

    def newest_run(model_dir):
        """Newest completed run dir for one model (by config timestamp)."""
        runs = []
        for run_dir in model_dir.iterdir():
            cfg_path = run_dir / "config.json"
            if not cfg_path.exists():
                continue
            cfg = json.loads(cfg_path.read_text())
            runs.append((cfg.get("timestamp", ""), run_dir))
        if not runs:
            return None
        return sorted(runs)[-1][1]

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

    def build_capability_tables():
        """dict base_model -> DataFrame, one row per fine-tuned version (base
        first), columns = accuracy/stderr per task. Adapters are labelled and
        ordered exactly like in the main table."""
        scores = collect_capability_scores()
        tables = {}
        if scores.empty:
            return tables
        for base_model in sorted(scores["base_model"].unique()):
            sub = scores[scores["base_model"] == base_model]
            rows = []
            base_row = sub[sub["method"] == "base"]
            if not base_row.empty:
                rows.append((0, "Base", base_row))
            adapters = []
            for method, g in sub[sub["method"] != "base"].groupby(
                "method", dropna=False
            ):
                stem = method.rsplit("-", 1)[0]
                adapters.append(
                    (
                        METHOD_ORDER.index(stem)
                        if stem in METHOD_ORDER
                        else len(METHOD_ORDER),
                        METHOD_PRETTY.get(stem, stem),
                        g,
                    )
                )
            rows += sorted(adapters, key=lambda r: (r[0], r[1]))

            table_rows = []
            for _, label, g in rows:
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

    capability_tables = build_capability_tables()
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
    return CAPABILITY_ROOT, TASK_LABELS, TASK_ORDER, capability_tables


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
def _(TASK_LABELS, TASK_ORDER, capability_tables, escape_header, pd, write_latex_table):
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


if __name__ == "__main__":
    app.run()
