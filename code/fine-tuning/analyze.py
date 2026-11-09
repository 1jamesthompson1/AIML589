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
        "prompted": "Prompted",
        "modal_response": "modal fine-tune",
        "sampled_response": "sampled fine-tune",
        "first_token_distribution": "first-token fine-tune",
        "full_string_distribution": "full-string fine-tune",
    }
    METHOD_ORDER = [
        # Prompted baseline first: the same base model with an added persona
        # instruction (not a fine-tune), directly under the Base row.
        "prompted",
        "modal_response",
        "sampled_response",
        "first_token_distribution",
        "full_string_distribution",
    ]
    # The validation split is reported twice because the eval holds out two
    # different things: a whole prompt (generic_answer*, all questions) and
    # individual questions under the seen prompts (6 per prompt). Keeping them
    # separate avoids the tiny 6-question slices dominating a pooled mean.
    SPLITS = ["validation_prompt", "validation_question", "train", "all"]
    METRICS = ["accuracy", "kl_divergence"]
    CAPABILITY_ROOT = EVALS_ROOT.parent / "capability"
    # Only this base model's adapters are in scope for the capability table.
    # Anything else in output/capability/ is ignored rather than rendered as a
    # mostly-empty row.
    TASK_LABELS = {
        "mmlu_pro_logprob": "MMLU-Pro (direct)",
        "mmlu_pro": "MMLU-Pro (CoT)",
        "ifeval": "IFEval",
    }
    TASK_ORDER = ["mmlu_pro_logprob", "mmlu_pro", "ifeval"]

    def base_and_method(model_name):
        # "{base}-nz-wvs-{method}" -> (base, method); a "{base}-prompted" dir
        # is the base model evaluated with an added persona instruction (not
        # a fine-tune); anything else is a plain base model.
        if "nz-wvs-" in model_name:
            base, method = model_name.split("nz-wvs-", 1)
            return base.rstrip("-"), method
        if model_name.endswith("-prompted"):
            return model_name[: -len("-prompted")], "prompted"
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
            base_val = (
                float(base_row["tvd_validation"].iloc[0])
                if not base_row.empty
                else float("nan")
            )
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
            # A prompted dir can exist without a base run (e.g. the prompted
            # eval used OpenRouter while the base used vLLM): no base row
            # means all deltas stay NaN and render as "--".
            base_val = (
                float(base_row["tvd_validation"].iloc[0])
                if not base_row.empty
                else float("nan")
            )
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
    top), columns = accuracy and KL divergence on two validation splits,
    train, and all. ``Validation prompt`` is the held-out generic prompt (all
    its questions); ``Validation question`` is the held-out questions under
    the seen prompts. Keeping them separate stops the tiny 6-question prompt
    slices from dominating the validation number. Metrics are pooled over
    rows, not averaged over prompts. CE is omitted because, on the same
    questions, it differs from KL only by the true distribution's entropy.

    All metrics come from the modal eval config (``modal_response``), so
    accuracy always means exact match against the modal response and KL is
    against the empirical response distribution. Every model is scored on the
    overall population. The LaTeX tables are written to ``code/figures/`` as
    ``ft-results-<model>.tex``. The first-token eval in the same layout
    (reasoning off, all system prompts) is written to
    ``ft-results-first_token_distribution-<model>.tex``.
    """)
    return


@app.cell
def _(
    EVALS_ROOT,
    METRICS,
    SPLITS,
    adapter_rows,
    base_and_method,
    collapse_versions,
    model_frames,
    newest_run,
    parse_list,
    pd,
):
    # Building the main table data.
    from scoring import leading_letter as _leading_letter

    _ALL_LETTERS = [chr(65 + i) for i in range(26)]

    def load_config_frames(eval_config, reasoning=False):
        """dict model dir name -> (base_model, method, overall per-question df)
        for the newest full run of one eval config and thinking mode."""
        frames = {}
        for model_dir in sorted(EVALS_ROOT.iterdir()):
            if not model_dir.is_dir():
                continue
            run_dir = newest_run(
                model_dir,
                keep=lambda cfg: (
                    cfg.get("dataset") == eval_config
                    and bool(cfg.get("reasoning")) is reasoning
                    and not cfg.get("num_test_examples")
                ),
            )
            if run_dir is None:
                continue
            csv_path = run_dir / "per_question_results.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            df = df.assign(
                subpopulation=df.get("subpopulation", "overall"),
                split=df.get("split", "validation"),
            )
            frames[model_dir.name] = (
                *base_and_method(model_dir.name),
                df[df["subpopulation"] == "overall"],
            )
        return frames

    def expected_letter(expected_text, categories):
        """Option letter for a gold answer stored as a letter, option text or
        numeric label."""
        text = str(expected_text).strip()
        if len(text) == 1 and text.upper() in _ALL_LETTERS:
            return text.upper()
        for i, category in enumerate(categories or []):
            if str(category).strip().lower() == text.lower():
                return _ALL_LETTERS[i]
        return None

    def collect_metrics(frames, system_prompt_id_filter=None, first_token=False):
        """Pooled metrics per (model, evaluation split).

        ``validation_prompt`` is the held-out generic prompt (all its
        questions); ``validation_question`` is the held-out questions under
        the seen prompts; ``train`` is the seen questions; ``all`` is every
        row. Metrics are pooled over rows, so a prompt with more items
        contributes proportionally (unlike an equal-prompt mean).
        ``first_token`` scores accuracy with the lenient leading-letter rule.
        """
        rows = []
        for model, (base, method, df) in frames.items():
            if not {"expected_text", "model_answer"}.issubset(df.columns):
                continue
            if system_prompt_id_filter is not None:
                if "system_prompt_id" not in df:
                    continue
                df = df[
                    df["system_prompt_id"]
                    .astype(str)
                    .str.startswith(system_prompt_id_filter)
                ]
            if df.empty:
                continue
            prompt = df["system_prompt_id"].astype(str)
            generic = prompt.str.startswith("generic_answer")
            split = df["split"].astype(str)
            subsets = {
                "validation_prompt": generic,
                "validation_question": split.eq("validation") & ~generic,
                "train": split.eq("train"),
                "all": pd.Series(True, index=df.index),
            }

            def accuracy(h):
                if not first_token:
                    return (
                        h["expected_text"].str.strip().str.lower()
                        == h["model_answer"].str.strip().str.lower()
                    ).mean() * 100
                categories = h["categories"].map(parse_list)
                letters = categories.map(
                    lambda cats: (
                        [chr(65 + i) for i in range(len(cats or []))] or _ALL_LETTERS
                    )
                )
                lead = pd.Series(
                    [
                        _leading_letter(answer, valid)
                        if isinstance(answer, str)
                        else None
                        for answer, valid in zip(h["model_answer"], letters)
                    ],
                    index=h.index,
                )
                expected = pd.Series(
                    [
                        expected_letter(gold, cats)
                        for gold, cats in zip(h["expected_text"], categories)
                    ],
                    index=h.index,
                )
                capped = (
                    h["truncated"].astype(str).str.lower().eq("true")
                    if "truncated" in h
                    else pd.Series(False, index=h.index)
                )
                return (lead.eq(expected) & ~capped).mean() * 100

            for split_label, mask in subsets.items():
                h = df[mask]
                rows.append(
                    {
                        "model": model,
                        "base_model": base,
                        "method": method,
                        "split": split_label,
                        "n": len(h),
                        "accuracy": accuracy(h) if not h.empty else float("nan"),
                        "kl_divergence": (
                            h["kl_divergence"].mean() if not h.empty else float("nan")
                        ),
                    }
                )
        return pd.DataFrame(rows)

    def collect_modal_metrics(system_prompt_id_filter=None):
        """Pooled metrics from the newest non-reasoning modal-config eval run,
        restricted to the overall population. Exact-match accuracy, as in the
        main table.

        With ``system_prompt_id_filter`` only prompts starting with that name
        are scored (e.g. ``"no_persona"`` for the prompted-vs-base
        comparison). Older runs without split/subpopulation tags are treated
        as validation-only, so their train columns render as "--".
        """
        return collect_metrics(
            model_frames, system_prompt_id_filter=system_prompt_id_filter
        )

    def build_main_tables(metrics):
        """dict base_model -> DataFrame, one row per fine-tuned version (base
        first), columns = metrics x splits. Metrics are already pooled per
        split. Duplicate entries (several eval dirs mapping to the same
        method) are averaged per split."""
        tables = {}
        for base_model in sorted(metrics["base_model"].unique()):
            collapsed = collapse_versions(
                adapter_rows(metrics[metrics["base_model"] == base_model])
            )
            table_rows = []
            for _, label, g in collapsed:
                agg = g.groupby("split", dropna=False)[METRICS].mean()
                row = {"label": label}
                for split in SPLITS:
                    for metric in METRICS:
                        row[f"{metric}_{split}"] = (
                            float("nan")
                            if split not in agg.index
                            else agg.loc[split, metric]
                        )
                table_rows.append(row)
            tables[base_model] = pd.DataFrame(table_rows)
        return tables

    def show_main_tables(tables, context="overall population"):
        """Console view: Accuracy and KL by split. ``validation_prompt`` is
        the held-out generic prompt; ``validation_question`` is the held-out
        questions under the seen prompts. ``context`` labels the data scope."""
        metric_labels = {"accuracy": "Acc (%)", "kl_divergence": "KL"}
        split_labels = ["Val prompt", "Val question", "Train", "All"]
        split_keys = ["validation_prompt", "validation_question", "train", "all"]
        for base_model, table in tables.items():
            print(
                f"\n=== {base_model} — {context} "
                "(val prompt = held-out generic prompt; val question = "
                "held-out questions under seen prompts) ==="
            )
            cols = pd.MultiIndex.from_product(
                [[metric_labels[m] for m in METRICS], split_labels]
            )
            out = pd.DataFrame(index=table["label"], columns=cols)
            for metric in METRICS:
                for split, label in zip(split_keys, split_labels):
                    out[(metric_labels[metric], label)] = (
                        table[f"{metric}_{split}"].round(3).values
                    )
            print(out.to_string())

    tables = build_main_tables(collect_modal_metrics())
    show_main_tables(tables)
    return (
        build_main_tables,
        collect_metrics,
        collect_modal_metrics,
        load_config_frames,
        show_main_tables,
        tables,
    )


@app.cell
def _(pd, tables, write_latex_table):
    # Columns: Accuracy and KL by split. The two validation columns separate
    # the held-out generic prompt from the held-out questions under the seen
    # prompts, so the tiny 6-question prompt slices cannot dominate the mean.
    # CE is omitted: on the same questions it differs from KL only by the true
    # distribution's entropy, so it adds no ranking information.

    def render_tables_to_tex(tables, name_template="ft-results-{base_model}.tex"):
        METRIC_LABELS = ["Accuracy", "KL"]
        METRIC_COLS = ["accuracy", "kl_divergence"]
        SPLIT_LABELS = ["Validation prompt", "Validation question", "Train", "All"]
        SPLIT_KEYS = ["validation_prompt", "validation_question", "train", "all"]
        SENSES = {"Accuracy": "max", "KL": "min"}
        FORMATS = {"Accuracy": "{:.1f}", "KL": "{:.3f}"}
        written = []
        for base_model, table in tables.items():
            columns = pd.MultiIndex.from_product([METRIC_LABELS, SPLIT_LABELS])
            out = pd.DataFrame(index=table["label"], columns=columns)
            # Index name suppressed (Styler would emit an extra header row);
            # the "Model" header is re-added by write_latex_table.
            out.index.name = None
            fmt = {}
            for mcol, label in zip(METRIC_COLS, METRIC_LABELS):
                for key, slabel in zip(SPLIT_KEYS, SPLIT_LABELS):
                    out[(label, slabel)] = table[f"{mcol}_{key}"].values
                    fmt[(label, slabel)] = FORMATS[label]
            written.append(
                write_latex_table(
                    out,
                    fmt,
                    out.shape[1],
                    name_template.format(base_model=base_model),
                    senses=SENSES,
                )
            )
        return written

    table_files = render_tables_to_tex(tables)
    table_files
    return (render_tables_to_tex,)


@app.cell
def _(
    build_main_tables,
    collect_metrics,
    load_config_frames,
    render_tables_to_tex,
):
    # First-token eval in the same layout as the main table: every model on
    # the first-token format, reasoning off and all system prompts. Only
    # base models with at least one adapter are rendered; the others would
    # be a single Base row. LaTeX -> code/figures/
    # ft-results-first_token_distribution-<model>.tex.
    first_token_metrics = collect_metrics(
        load_config_frames("first_token_distribution", reasoning=False),
        first_token=True,
    )
    if first_token_metrics.empty:
        first_token_tables = {}
    else:
        n_methods = first_token_metrics.groupby("base_model")["method"].nunique()
        first_token_metrics = first_token_metrics[
            first_token_metrics["base_model"].isin(n_methods[n_methods > 1].index)
        ]
        first_token_tables = build_main_tables(first_token_metrics)
    first_token_table_files = render_tables_to_tex(
        first_token_tables,
        name_template="ft-results-first_token_distribution-{base_model}.tex",
    )
    first_token_table_files
    return (first_token_table_files,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### No-persona prompted vs fine-tuned

    Same layout and metrics as the main table, computed only on the
    ``no_persona`` system prompt (no persona wording at all), so the
    **Prompted** row is exactly that prompt plus the added "respond as an
    average New Zealand person would" instruction. Comparing Prompted with
    Base shows the effect of prompting alone on the most neutral prompt;
    comparing it with the fine-tuned rows shows how much of fine-tuning's
    move towards the NZ population a single instruction reproduces. There is
    no held-out prompt in this single-prompt view, so ``Validation prompt`` is
    ``--``; ``Validation question`` is the six held-out questions under this
    prompt. LaTeX is written to
    ``code/figures/ft-results-nopersona-<model>.tex``.
    """)
    return


@app.cell
def _(
    build_main_tables,
    collect_modal_metrics,
    render_tables_to_tex,
    show_main_tables,
):
    # no_persona-only version of the main table: the prompted baseline and
    # every fine-tuned method scored on just this system prompt.
    nopersona_metrics = collect_modal_metrics(system_prompt_id_filter="no_persona")
    nopersona_tables = build_main_tables(nopersona_metrics)
    show_main_tables(nopersona_tables, context="no_persona prompt only")
    nopersona_table_files = render_tables_to_tex(
        nopersona_tables,
        name_template="ft-results-nopersona-{base_model}.tex",
    )
    nopersona_table_files
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reasoning (thinking) on vs off

    The same models evaluated twice on identical items: once with
    ``--reasoning`` (chain of thought, answer read after ``</think>``) and once
    without. Both passes use the ``no_persona`` system prompt and the overall
    population, with every other setting at the eval default, so the only
    difference is thinking. The question is whether fine-tuning's advantage
    survives when the model is allowed to reason.

    Scoring is the same in both passes. A first-token reply is correct when it
    starts with the right option letter (text may follow); a reply with no
    leading letter is counted wrong and given no distribution, so flagged and
    token-capped rows drop out of TVD. ``Δ Acc`` is on minus off on the same
    items, with an item-bootstrap 95% CI and an exact McNemar p-value.
    ``vs Base`` is the adapter's accuracy minus the base model's **in the
    same condition** (reasoning on), i.e. what fine-tuning still buys once the
    model thinks.

    Read ``Δ Acc`` carefully in the first-token table: without reasoning some
    adapters answer with the option text instead of a letter, which is a format
    effect, not a value effect (see the flagged column). Accuracy, not TVD, is
    the cleaner comparison across conditions because reasoning makes most
    distributions near one-hot. The report-style tables (reasoning on, same
    accuracy/CE/KL layout as the main table) are written to
    ``code/figures/ft-reasoning-<config>-<model>.tex``; the paired on/off
    diagnostic is written to
    ``code/figures/ft-reasoning-comparison-<config>-<model>.tex``.
    """)
    return


@app.cell
def _(
    EVALS_ROOT,
    adapter_rows,
    base_and_method,
    collapse_versions,
    math,
    newest_run,
    np,
    parse_list,
    pd,
):
    # Building the reasoning on/off table data from the persisted eval runs.
    from scoring import leading_letter

    REASONING_FORMATS = ("modal_response", "first_token_distribution")
    REASONING_KEYS = [
        "question_id",
        "sub_question",
        "column_name",
        "split",
        "system_prompt_id",
    ]
    ALL_LETTERS = [chr(65 + i) for i in range(26)]

    def newest_phase_run(model_dir, fmt, reasoning):
        """Newest full run of one model dir for a format and thinking mode."""
        return newest_run(
            model_dir,
            keep=lambda cfg: (
                cfg.get("dataset") == fmt
                and bool(cfg.get("reasoning")) is reasoning
                and not cfg.get("num_test_examples")
            ),
        )

    def scored_run(run_dir, first_token):
        """Per-question rows scored with the evaluate.py rule."""
        df = pd.read_csv(run_dir / "per_question_results.csv")
        if "sub_question" not in df:
            df["sub_question"] = ""
        df["sub_question"] = df["sub_question"].fillna("")
        if "split" not in df:
            df["split"] = "validation"
        if "system_prompt_id" in df:
            df = df[df["system_prompt_id"].astype(str).str.startswith("no_persona")]
        if "truncated" not in df:
            df["truncated"] = False
        df["truncated"] = df["truncated"].astype(str).str.lower().eq("true")
        if first_token:
            letters = df["categories"].map(
                lambda s: (
                    [chr(65 + i) for i in range(len(parse_list(s) or []))]
                    or ALL_LETTERS
                )
            )
            lead = pd.Series(
                [
                    leading_letter(a, ok) if isinstance(a, str) else None
                    for a, ok in zip(df["model_answer"], letters)
                ],
                index=df.index,
            )
            df["flagged"] = lead.isna() & ~df["truncated"]
            df["correct"] = lead.eq(df["expected_text"]) & ~df["truncated"]
        else:
            df["flagged"] = False
            df["correct"] = (
                df["is_correct"].astype(str).str.lower().eq("true")
                if "is_correct" in df
                else False
            )
        return df

    def tvd_frame(df):
        """Rows with a usable distribution, with TVD (%) to the NZ population."""
        ok = df[~df["truncated"] & ~df["flagged"]]
        rows = []
        for _, r in ok.iterrows():
            p, q = (
                parse_list(r.get("model_distribution")),
                parse_list(r.get("true_distribution")),
            )
            if not p or not q or len(p) != len(q) or sum(p) == 0:
                continue
            rows.append(
                {
                    **{k: r[k] for k in REASONING_KEYS},
                    "tvd": 0.5 * sum(abs(a - b) for a, b in zip(p, q)) * 100,
                }
            )
        return pd.DataFrame(rows, columns=REASONING_KEYS + ["tvd"])

    def bootstrap_ci(a, b, n=2000):
        """Item-bootstrap 95% CI on the paired difference (seeded per call so
        the interval does not depend on the order comparisons are run in)."""
        rng = np.random.default_rng(0)
        diff = np.asarray(a, float) - np.asarray(b, float)
        idx = rng.integers(0, len(diff), size=(n, len(diff)))
        means = diff[idx].mean(axis=1)
        return diff.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)

    def mcnemar_p(a, b):
        """Exact two-sided McNemar p over the discordant pairs."""
        a, b = np.asarray(a, bool), np.asarray(b, bool)
        n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
        n = n10 + n01
        if n == 0:
            return 1.0
        k = min(n01, n10)
        return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)

    def collect_reasoning_rows():
        rows = []
        for model_dir in sorted(EVALS_ROOT.iterdir()):
            if not model_dir.is_dir():
                continue
            base, method = base_and_method(model_dir.name)
            for fmt in REASONING_FORMATS:
                on_dir = newest_phase_run(model_dir, fmt, True)
                off_dir = newest_phase_run(model_dir, fmt, False)
                if on_dir is None or off_dir is None:
                    continue
                first = fmt == "first_token_distribution"
                on, off = scored_run(on_dir, first), scored_run(off_dir, first)
                paired = on[REASONING_KEYS + ["correct"]].merge(
                    off[REASONING_KEYS + ["correct"]],
                    on=REASONING_KEYS,
                    suffixes=("_on", "_off"),
                )
                if paired.empty:
                    continue
                delta, lo, hi = bootstrap_ci(
                    paired["correct_on"], paired["correct_off"]
                )
                tv = tvd_frame(on).merge(
                    tvd_frame(off), on=REASONING_KEYS, suffixes=("_on", "_off")
                )
                capped = (
                    int((on["finish_reason"] == "length").sum())
                    if "finish_reason" in on
                    else int(on["truncated"].sum())
                )
                nothink = (
                    int(
                        (
                            on["think_closed"].astype(str).str.lower().eq("false")
                            & ~on["truncated"]
                        ).sum()
                    )
                    if "think_closed" in on
                    else 0
                )
                rows.append(
                    {
                        "model": model_dir.name,
                        "base_model": base,
                        "method": method,
                        "config": fmt,
                        "n_items": len(paired),
                        "acc_off": paired["correct_off"].mean() * 100,
                        "acc_on": paired["correct_on"].mean() * 100,
                        "acc_delta": delta * 100,
                        "acc_lo": lo * 100,
                        "acc_hi": hi * 100,
                        "mcnemar_p": mcnemar_p(
                            paired["correct_on"], paired["correct_off"]
                        ),
                        "tvd_off": tv["tvd_off"].mean() if not tv.empty else np.nan,
                        "tvd_on": tv["tvd_on"].mean() if not tv.empty else np.nan,
                        "flagged_on": int(on["flagged"].sum()),
                        "capped_on": capped,
                        "nothink_on": nothink,
                    }
                )
        return pd.DataFrame(rows)

    def build_reasoning_tables(rows):
        """dict (config, base_model) -> one row per model version, Base first."""
        tables = {}
        if rows.empty:
            return tables
        for base_model in sorted(rows["base_model"].unique()):
            for fmt in REASONING_FORMATS:
                sub = rows[(rows["base_model"] == base_model) & (rows["config"] == fmt)]
                if sub.empty:
                    continue
                built = []
                for _, label, g in collapse_versions(adapter_rows(sub)):
                    built.append(
                        {
                            "label": label,
                            **{
                                col: g[col].mean()
                                for col in sub.columns
                                if col
                                not in {"model", "base_model", "method", "config"}
                            },
                        }
                    )
                table = pd.DataFrame(built)
                base_row = table[table["label"] == "Base"]
                base_on = (
                    float(base_row["acc_on"].iloc[0])
                    if not base_row.empty
                    else float("nan")
                )
                table["acc_vs_base_on"] = (table["acc_on"] - base_on).where(
                    table["label"] != "Base"
                )
                tables[(fmt, base_model)] = table
        return tables

    def show_reasoning_tables(tables):
        for (fmt, base_model), table in tables.items():
            print(f"\n=== {base_model} — {fmt}, reasoning off vs on ===")
            out = pd.DataFrame(index=table["label"])
            out["Acc off"] = table["acc_off"].round(1).values
            out["Acc on"] = table["acc_on"].round(1).values
            out["Δ Acc"] = [
                f"{v:+.1f}" if pd.notna(v) else "--" for v in table["acc_delta"]
            ]
            out["95% CI"] = [
                f"[{lo:+.1f}, {hi:+.1f}]" if pd.notna(lo) and pd.notna(hi) else "--"
                for lo, hi in zip(table["acc_lo"], table["acc_hi"])
            ]
            out["p"] = table["mcnemar_p"].round(4).values
            out["vs Base (on)"] = [
                f"{v:+.1f}" if pd.notna(v) else "--" for v in table["acc_vs_base_on"]
            ]
            out["TVD off"] = table["tvd_off"].round(1).values
            out["TVD on"] = table["tvd_on"].round(1).values
            out["Flag on"] = table["flagged_on"].astype(int).values
            out["Cap on"] = table["capped_on"].astype(int).values
            out["NoThink"] = table["nothink_on"].astype(int).values
            print(out.to_string())
        print(
            "\nΔ Acc = reasoning on minus off, same items (+CI, McNemar p). "
            "TVD (%) excludes flagged and capped rows. Flag on = replies with "
            "no leading option letter (counted wrong). NoThink = answered "
            "without closing </think>."
        )

    reasoning_rows = collect_reasoning_rows()
    reasoning_tables = build_reasoning_tables(reasoning_rows)
    show_reasoning_tables(reasoning_tables)
    return (REASONING_FORMATS, reasoning_tables)


@app.cell
def _(
    REASONING_FORMATS,
    build_main_tables,
    collect_metrics,
    load_config_frames,
    pd,
    reasoning_tables,
    render_tables_to_tex,
    write_latex_table,
):
    # The report uses ft-results-style tables for the reasoning runs: reasoning
    # on, accuracy/CE/KL by split (ft-reasoning-<config>-<model>.tex). The
    # paired on/off comparison is kept as a diagnostic table
    # (ft-reasoning-comparison-<config>-<model>.tex).

    def render_reasoning_comparison_tables(tables):
        columns = pd.MultiIndex.from_tuples(
            [
                ("Accuracy (%)", "Off"),
                ("Accuracy (%)", "On"),
                ("Δ Acc (pp)", "On−Off"),
                ("vs Base (pp)", "On"),
                ("p", "McNemar"),
                ("TVD (%)", "Off"),
                ("TVD (%)", "On"),
                ("Flagged", "On"),
            ]
        )
        senses = {"Accuracy (%)": "max", "TVD (%)": "min"}
        bold_below = {"p": 0.05}
        written = []
        for (fmt, base_model), table in tables.items():
            out = pd.DataFrame(index=table["label"], columns=columns)
            out.index.name = None
            out[("Accuracy (%)", "Off")] = table["acc_off"].values
            out[("Accuracy (%)", "On")] = table["acc_on"].values
            out[("Δ Acc (pp)", "On−Off")] = table["acc_delta"].values
            out[("vs Base (pp)", "On")] = table["acc_vs_base_on"].values
            out[("p", "McNemar")] = table["mcnemar_p"].values
            out[("TVD (%)", "Off")] = table["tvd_off"].values
            out[("TVD (%)", "On")] = table["tvd_on"].values
            out[("Flagged", "On")] = table["flagged_on"].values
            fmt_spec = {
                ("Accuracy (%)", "Off"): "{:.1f}",
                ("Accuracy (%)", "On"): "{:.1f}",
                ("Δ Acc (pp)", "On−Off"): "{:+.1f}",
                ("vs Base (pp)", "On"): "{:+.1f}",
                ("p", "McNemar"): "{:.3f}",
                ("TVD (%)", "Off"): "{:.1f}",
                ("TVD (%)", "On"): "{:.1f}",
                ("Flagged", "On"): "{:.0f}",
            }
            written.append(
                write_latex_table(
                    out,
                    fmt_spec,
                    out.shape[1],
                    f"ft-reasoning-comparison-{fmt}-{base_model}.tex",
                    senses=senses,
                    bold_below=bold_below,
                )
            )
        return written

    def render_main_style_reasoning_tables():
        written = []
        for fmt in REASONING_FORMATS:
            metrics = collect_metrics(
                load_config_frames(fmt, reasoning=True),
                first_token=(fmt == "first_token_distribution"),
            )
            if metrics.empty:
                continue
            tables = build_main_tables(metrics)
            written += render_tables_to_tex(
                tables,
                name_template=f"ft-reasoning-{fmt}-{{base_model}}.tex",
            )
        return written

    reasoning_tex_files = render_main_style_reasoning_tables()
    reasoning_comparison_tex_files = render_reasoning_comparison_tables(
        reasoning_tables
    )
    return (reasoning_tex_files, reasoning_comparison_tex_files)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Capability results table

    Same layout as the main table, but for the capability evals: one row per
    fine-tuned version (base model at the top), columns per benchmark. Scores
    are read from the newest completed run per model in
    ``output/capability/<model>/<run>/summary.json`` (written by
    ``capability_eval.py``). Three benchmarks: MMLU-Pro answered directly
    (first-token logprobs, no chain of thought), MMLU-Pro with chain-of-thought
    scored **leniently** (with the benchmark's own strict template shown as a
    separate column), and IFEval. The LaTeX (booktabs) tables are written to
    ``code/figures/ft-capability-<model>.tex`` and included in the report with
    ``\ctable{ft-capability-<model>.tex}{<caption>}``.

    Lenient scoring matters: the adapters were fine-tuned to answer with only
    the option text or bare letter, so the benchmark's ``ANSWER: $LETTER``
    template scores a correct bare ``D`` as a miss. The lenient column is the
    capability number; the strict column measures template compliance.
    """)
    return


@app.cell
def _(
    CAPABILITY_BASE_MODELS,
    CAPABILITY_ROOT,
    TASK_LABELS,
    TASK_ORDER,
    adapter_rows,
    base_and_method,
    json,
    pd,
):
    # Building the capability table data from summary.json of the newest run
    # per model.

    def newest_capability_summary(model_dir):
        """(run_dir, summary) of the newest run — never an older one.

        Raises when the newest run does not cover every task, so a failed or
        partial rerun surfaces instead of being silently replaced by an earlier
        grid run.
        """
        runs = [d for d in model_dir.iterdir() if (d / "config.json").exists()]
        if not runs:
            return None, {}

        def started(run_dir):
            return json.loads((run_dir / "config.json").read_text()).get(
                "timestamp", ""
            )

        run_dir = max(runs, key=started)
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            raise RuntimeError(
                f"{model_dir.name}: newest run {run_dir.name} has no "
                "summary.json. Refusing to fall back to an older run — rerun "
                "it or remove the directory."
            )
        summary = json.loads(summary_path.read_text())
        missing = [
            task
            for task in TASK_ORDER
            if (summary.get(task) or {}).get("accuracy") is None
        ]
        if missing:
            raise RuntimeError(
                f"{model_dir.name}: newest run {run_dir.name} is missing "
                f"{missing}. Refusing to fall back to an older run — rerun it "
                "or remove the directory."
            )
        return run_dir, summary

    def collect_capability_scores():
        """Per (model, task) accuracy/stderr from the newest run."""
        rows = []
        if not CAPABILITY_ROOT.exists():
            return pd.DataFrame(rows)
        for model_dir in sorted(CAPABILITY_ROOT.iterdir()):
            if not model_dir.is_dir():
                continue
            base, method = base_and_method(model_dir.name)
            # Scope filters come first so an out-of-scope directory can never
            # raise the missing-task error above.
            if base not in CAPABILITY_BASE_MODELS:
                continue
            # Only the overall population adapters (plus the base model):
            # cluster-trained adapters are out of scope.
            if method != "base" and not method.endswith("-overall"):
                continue
            run_dir, summary = newest_capability_summary(model_dir)
            if run_dir is None:
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
                        # None for tasks without a lenient reading (logprob,
                        # ifeval); populated for the generative MCQ task.
                        "lenient_accuracy": entry.get("lenient_accuracy"),
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
                    # Lenient accuracy exists only for the generative MCQ task;
                    # NaN elsewhere so the renderer can skip the column.
                    len_acc = h["lenient_accuracy"].iloc[0] if not h.empty else None
                    row[f"{task}_len"] = (
                        len_acc if len_acc is not None else float("nan")
                    )
                    row[f"{task}_n"] = h["n"].iloc[0] if not h.empty else float("nan")
                table_rows.append(row)
            tables[base_model] = pd.DataFrame(table_rows)
        return tables

    capability_scores = collect_capability_scores()
    capability_tables = build_capability_tables(capability_scores)
    for base_model, table in capability_tables.items():
        print(f"\n=== {base_model} — capability evals (accuracy in %) ===")
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
            disp[TASK_LABELS[task] + " (strict)"] = cells
            if not table[f"{task}_len"].isna().all():
                disp[TASK_LABELS[task] + " (lenient)"] = [
                    "--" if pd.isna(v) else f"{v * 100:.1f}"
                    for v in table[f"{task}_len"]
                ]
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
    pd,
    write_latex_table,
):
    # Rendering the capability tables to LaTeX (booktabs, best accuracy per
    # column bolded). Each benchmark gets two sub-columns: accuracy (%) and its
    # 95% confidence interval (acc +- 1.96 x s.e. from the inspect metrics).

    def render_capability_tables_to_tex():
        # For the generative MCQ task the headline number is the LENIENT
        # accuracy (the benchmark's own template understates adapters that were
        # fine-tuned to answer with a bare option), so that task gets an extra
        # "Strict (%)" column showing what the benchmark's template scores.
        #
        # write_latex_table escapes both the DataFrame columns and the fmt
        # keys, so both are given here RAW.
        subs = ["Acc (%)", "95% CI"]
        strict_label = "Strict (%)"
        written = []
        for base_model, table in capability_tables.items():
            columns = []
            for task in TASK_ORDER:
                header = TASK_LABELS[task]
                columns.append((header, subs[0]))
                columns.append((header, subs[1]))
                if not table[f"{task}_len"].isna().all():
                    columns.append((header, strict_label))
            out = pd.DataFrame(
                index=table["label"].values,
                columns=pd.MultiIndex.from_tuples(columns),
            )
            out.index.name = None
            fmt = {}
            for task in TASK_ORDER:
                header = TASK_LABELS[task]
                acc = table[f"{task}_acc"].astype(float)
                strict_se = table[f"{task}_se"].astype(float)
                n = table[f"{task}_n"].astype(float)
                has_lenient = not table[f"{task}_len"].isna().all()
                if has_lenient:
                    acc = table[f"{task}_len"].astype(float)
                    # Binomial s.e. for the lenient rate; summary.json only
                    # holds the strict stderr.
                    se = (acc * (1 - acc) / n).pow(0.5)
                else:
                    se = strict_se
                lo = (acc - 1.96 * se) * 100
                hi = (acc + 1.96 * se) * 100
                out[(header, subs[0])] = (acc * 100).round(1).values
                out[(header, subs[1])] = [
                    f"{lo_v:.1f}-{hi_v:.1f}"
                    if pd.notna(lo_v) and pd.notna(hi_v)
                    else "--"
                    for lo_v, hi_v in zip(lo, hi)
                ]
                fmt[(header, subs[0])] = "{:.1f}"
                if has_lenient:
                    out[(header, strict_label)] = (
                        table[f"{task}_acc"].astype(float).mul(100).round(1).values
                    )
                    fmt[(header, strict_label)] = "{:.1f}"
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Probability mass on the modal response

    One illustrative WVS item for all five models: the base model and the
    four fine-tuned adapters. Each panel shows the probability the model
    assigns to every response option, averaged over the eight system
    prompts, next to the NZ public distribution. The item is Q210
    ("Joining in boycotts"), chosen from the training-split questions as
    the one closest to the median across questions on all three metrics:
    80% accuracy, 34% TVD and KL 2.8 (medians 80%, 34% and 2.6). Its
    four-option scale is spread across the public. The modal fine-tune
    puts all of its probability on the modal answer, the sampled
    fine-tune on a single non-modal answer, while the base model and the
    distributional fine-tunes keep a spread. Written to
    ``code/figures/ft-modal-mass.png``.
    """)
    return


@app.cell
def _(
    EVALS_ROOT,
    FIGS_DIR,
    load_overall_per_question,
    np,
    parse_list,
    plt,
    textwrap,
):
    # Illustrative item: Q210, "Joining in boycotts", chosen from the
    # training-split questions because its per-question accuracy, TVD and KL
    # are all at the median across questions (averaged over the five models).
    # The public is spread over four options, the two SFT adapters are one-hot
    # (on the modal answer and on a non-modal one respectively), and the base
    # model and distributional adapters keep a spread.
    MASS_QID = "61"
    MASS_COL = "Q210"
    MASS_MODELS = [
        ("Qwen3.8-27B", "Base"),
        ("Qwen3.8-27B-nz-wvs-modal_response-overall", "modal fine-tune"),
        ("Qwen3.8-27B-nz-wvs-sampled_response-overall", "sampled fine-tune"),
        (
            "Qwen3.8-27B-nz-wvs-first_token_distribution-overall",
            "first-token fine-tune",
        ),
        (
            "Qwen3.8-27B-nz-wvs-full_string_distribution-overall",
            "full-string fine-tune",
        ),
    ]
    MASS_MODEL_COLOR = "#4C72B0"
    MASS_PUBLIC_COLOR = "#DD8452"

    def _load_mass_item(model_dir):
        """(categories, public distribution, prompt-averaged model
        distribution, question text) for one model, or None."""
        df = load_overall_per_question(EVALS_ROOT / model_dir)
        if df is None:
            return None
        g = df[
            (df["question_id"].astype(str) == MASS_QID)
            & (df["column_name"] == MASS_COL)
        ]
        if g.empty:
            return None
        cats = parse_list(g.iloc[0].get("categories"))
        true_dist = parse_list(g.iloc[0].get("true_distribution"))
        dists = [d for d in g["model_distribution"].map(parse_list) if d is not None]
        if not cats or not true_dist or not dists:
            return None
        if any(len(d) != len(cats) for d in dists):
            return None
        avg = [float(np.mean(vals)) for vals in zip(*dists)]
        question = str(g.iloc[0].get("question", ""))
        sub_question = g.iloc[0].get("sub_question")
        if isinstance(sub_question, str) and sub_question.strip():
            question = f"{question} — {sub_question.strip()}"
        return cats, true_dist, avg, question

    def _plot_mass_panel(ax, cats, dist, color, ylabel=False):
        x = np.arange(len(cats))
        bars = ax.bar(x, dist, color=color, width=0.78)
        for bar in bars:
            height = bar.get_height()
            if height >= 0.05:
                ax.annotate(
                    f"{height:.0%}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [textwrap.fill(str(cat), width=10) for cat in cats],
            fontsize=8,
            linespacing=0.95,
        )
        ax.set_ylim(0, 1.08)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        if ylabel:
            ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    def plot_modal_mass():
        items = []
        for model_dir, label in MASS_MODELS:
            item = _load_mass_item(model_dir)
            if item is None:
                print(f"[plots] missing {MASS_COL} for {model_dir}")
                continue
            items.append((label, item))
        if not items:
            return None

        reference = items[0][1]
        cats, true_dist, _, question = reference

        fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True, sharey=True)
        panels = items + [("Public (NZ)", (cats, true_dist, true_dist, question))]

        for index, (ax, (label, (_, _, dist, _))) in enumerate(zip(axes.flat, panels)):
            is_public = label.startswith("Public")
            _plot_mass_panel(
                ax,
                cats,
                dist,
                MASS_PUBLIC_COLOR if is_public else MASS_MODEL_COLOR,
                ylabel=index % 3 == 0,
            )
            ax.set_title(label, fontsize=10)

        for ax in axes.flat[len(panels) :]:
            ax.set_visible(False)

        fig.suptitle(textwrap.fill(question, width=100), fontsize=11, linespacing=1.3)
        fig.supxlabel("Response option", fontsize=9, y=0.045)
        fig.supylabel("Probability", fontsize=9)
        fig.tight_layout(rect=[0.02, 0.06, 1, 0.92])
        FIGS_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGS_DIR / "ft-modal-mass.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[plots] wrote {path}")
        return path

    modal_mass_figure = plot_modal_mass()
    modal_mass_figure
    return


@app.cell
def _(EVALS_ROOT, Path, csv, datetime, json, pd, timezone):
    """Write the webapp manifest (output/evals/index.json).

    The website fetches this at runtime so nothing is bundled at build time.
    It lists every model/run (with config + bucket-relative path) plus a
    question inventory keyed by (question_id, column_name) — the 251 distinct
    survey sub-items, each with in_training / in_eval flags and the answer
    options used by the viewer's fuzzy question/answer search.
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

    def _load_question_options(path: Path) -> dict[str, list[str]]:
        """column_name -> answer options from the canonical mapping.

        The viewer searches these alongside the question text, so a query
        like "honest" also finds the survey items whose answer scale contains
        that word.
        """
        mapping = json.loads(path.read_text())
        out: dict[str, list[str]] = {}
        for entry in mapping:
            if entry.get("question_type") != "value survey":
                continue
            options = [
                str(option)
                for option in (entry.get("word_response_types") or [])
                if str(option).strip()
            ]
            for column_name in entry.get("column_names", []):
                out[str(column_name)] = options
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
        q_options = (
            _load_question_options(QUESTION_MAPPING)
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
                    "answer_options": q_options.get(col_name, []),
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
    # passes the candidate rule, with the columns used to curate the dilemma
    # value tensions. "Min" is the
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
