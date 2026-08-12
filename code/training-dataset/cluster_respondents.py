# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "pandas>=2.0.0",
#     "numpy>=1.26.0",
#     "matplotlib>=3.8.0",
#     "stepmix>=1.0.0",
#     "scikit-learn>=1.4.0",
#     "joblib>=1.3.0",
#     "prince>=0.20.0",
#     "kmodes>=0.12.2",
# ]
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Clustering WVS Respondents into Value Subgroups

    This notebook partitions the 1,057 NZ WVS respondents into value-based
    subgroups (clusters). Downstream, `build_dataset.py` uses these clusters to
    build per-subgroup training datasets, and the public consultation assigns
    participants to the same subgroups.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Imports
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import json

    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from stepmix.stepmix import StepMix
    import joblib

    # Configuration for the clustering analysis. These are set to reasonable defaults, but can be overridden via CLI args.

    N_CLUSTERS = (
        None if mo.cli_args().get("run_sweep") else mo.cli_args().get("n_clusters", 2)
    )
    K_RANGE = range(1, 7)  # candidate cluster counts for the sweep
    N_INIT_SWEEP = 2  # EM restarts per k in the sweep (keeps sweep fast)
    N_INIT_FINAL = 10  # EM restarts for the final model (best log-likelihood)
    SEED = 42
    USE_SAVED_MODEL = not mo.cli_args().get(
        "run_model"
    )  # if True, load model from disk instead of fitting

    # Output directories for the clustering analysis. These are set to reasonable defaults which is in the output folder
    general_output_dir = Path("output")
    _analysis_dir = general_output_dir / "cluster_analysis" / f"k{N_CLUSTERS}_analysis"
    output_dir = _analysis_dir
    figures_dir = output_dir / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    (figures_dir / "cluster_profiles").mkdir(parents=True, exist_ok=True)
    print(f"Outputs → {output_dir}")
    print(f"Figures → {figures_dir}")
    return (
        K_RANGE,
        N_CLUSTERS,
        N_INIT_FINAL,
        N_INIT_SWEEP,
        Path,
        SEED,
        StepMix,
        USE_SAVED_MODEL,
        figures_dir,
        general_output_dir,
        joblib,
        json,
        mo,
        np,
        output_dir,
        pd,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Data Preparation

    Only the **value survey** columns are clustered on — respondent information
    columns are excluded so that subgroups are defined by values, not
    demographics (per the funding proposal).

    Preprocessing:
    - `-1` (Don't know) is kept as a valid response category — uncertainty
      about values is itself value-relevant for clustering.
    - `-5` (Missing / Refused / Not applicable / Not asked) becomes `NaN`
      and is treated as missing by the model.
    - Each column's observed codes are recoded to contiguous integers
      `0..K_j-1` (required by stepmix); the mapping is kept so results can be
      mapped back to the original WVS codes and word labels.
    """)
    return


@app.cell
def _(general_output_dir, json, np, pd):
    def load_wvs_data(input_dir):
        """Read the value survey responses and the question mapping."""
        wvs_df = pd.read_csv(input_dir / "wvs_value_survey.csv")
        question_mapping = json.load(open(input_dir / "question_mapping.json"))
        return wvs_df, question_mapping

    def recode_to_contiguous(X_raw, value_cols):
        """Recode each column to 0..K_j-1, remembering the original codes."""
        category_codes = {c: sorted(X_raw[c].dropna().unique()) for c in value_cols}
        X = pd.DataFrame(
            {
                c: X_raw[c].map({v: i for i, v in enumerate(category_codes[c])})
                for c in value_cols
            }
        )
        return X, category_codes

    def build_col2question(question_mapping):
        """column -> question text lookup for later display."""
        col2question = {}
        for entry in question_mapping:
            cols = entry["column_names"]
            sqs = entry.get("sub_questions") or [None] * len(cols)
            for i, col in enumerate(cols):
                sq = sqs[i] if i < len(sqs) else None
                text = entry["question"]
                col2question[col] = f"{text} — {sq}" if sq else text
        return col2question

    input_dir = general_output_dir
    wvs_df, question_mapping = load_wvs_data(input_dir)

    value_cols = [
        c for c in wvs_df.columns if c.startswith("Q") and not c.startswith("resp_info")
    ]
    X_raw = wvs_df[value_cols].mask(wvs_df[value_cols] == -5.0, np.nan)
    X, category_codes = recode_to_contiguous(X_raw, value_cols)
    col2question = build_col2question(question_mapping)

    f"{X.shape[0]} respondents × {X.shape[1]} value questions, "
    f"{X.isna().mean().mean():.1%} cells missing"
    return (
        X,
        X_raw,
        category_codes,
        col2question,
        question_mapping,
        value_cols,
        wvs_df,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Latent Class Analysis Model

    A categorical LCA (fit with `stepmix`) partitions respondents into $K$
    latent value profiles. The cells below select $K$, fit the final model and
    inspect its separation quality.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Model Selection: Number of Clusters

    Fit an LCA model for each candidate $K$ and record log-likelihood, AIC and
    BIC. **BIC is the selection criterion** — it penalises the (large) parameter
    count of categorical LCA and is the standard choice in the literature
    (AIC is known to over-extract classes). This cell takes a few minutes.
    """)
    return


@app.cell
def _(
    K_RANGE,
    N_CLUSTERS,
    N_INIT_SWEEP,
    Path,
    SEED,
    StepMix,
    X,
    output_dir,
    pd,
):
    def run_model_selection_sweep(K_RANGE, N_INIT_SWEEP, SEED, X, output_dir):
        """Fit an LCA for each candidate k and save loglik/AIC/BIC to disk."""
        print(f"Doing model selection sweep over k={list(K_RANGE)}")
        sweep_rows = []
        for k in K_RANGE:
            m = StepMix(
                n_components=k,
                measurement="categorical_nan",
                n_init=N_INIT_SWEEP,
                random_state=SEED,
                max_iter=300,
                progress_bar=0,
            )
            m.fit(X)
            sweep_rows.append(
                {
                    "k": k,
                    "loglik": m.score(X) * len(X),
                    "aic": m.aic(X),
                    "bic": m.bic(X),
                }
            )
            print(f"k={k} done")

        sweep_df = pd.DataFrame(sweep_rows).set_index("k").round(0)
        sweep_df.to_csv(output_dir / "model_selection_sweep.csv")
        print(f"Saved sweep results to {output_dir / 'model_selection_sweep.csv'}")

    if (
        N_CLUSTERS is None
        or not Path(output_dir / "model_selection_sweep.csv").exists()
    ):
        run_model_selection_sweep(K_RANGE, N_INIT_SWEEP, SEED, X, output_dir)
    else:
        print(f"Skipping model selection using k={N_CLUSTERS} (already done)")
    return


@app.cell
def _(N_CLUSTERS, figures_dir, output_dir, pd, plt):
    def plot_model_selection_sweep(sweep_df, K, figures_dir):
        """Plot BIC/AIC vs k; return the chosen k."""
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(sweep_df.index, sweep_df["bic"], "o-", label="BIC")
        ax.plot(sweep_df.index, sweep_df["aic"], "s--", label="AIC")
        ax.set_xlabel("Number of latent classes (k)")
        ax.set_ylabel("Information criterion (lower = better)")

        bic_best_k = int(sweep_df["bic"].idxmin())
        ax.axvline(K, color="grey", ls=":", label=f"chosen k={K}")
        ax.legend()
        print(f"BIC-optimal k = {bic_best_k}; using k = {K}")

        fig.savefig(
            figures_dir / "model_selection_sweep.png", dpi=300, bbox_inches="tight"
        )
        return fig

    sweep_df = pd.read_csv(output_dir / "model_selection_sweep.csv")
    _fig = plot_model_selection_sweep(sweep_df, N_CLUSTERS, figures_dir)
    _fig
    if f"k{N_CLUSTERS}_analysis" not in str(output_dir):
        print(
            f"Warning: outputs go to {output_dir}, which is not the "
            f"k{N_CLUSTERS}_analysis dir. Run with --n_clusters={N_CLUSTERS} for the standard layout."
        )
    return (sweep_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fit Final Model

    Refit at the chosen $K$ with more EM restarts and keep the best-likelihood
    solution. A reproducibility note: with ~260 indicators the likelihood surface
    is multimodal, so different seeds can yield different (near-equally likely)
    partitions at $K \geq 3$. The saved model below is internally consistent —
    all downstream artefacts (profiles, assignment of new participants) derive
    from this one fitted object.
    """)
    return


@app.cell
def _(
    N_CLUSTERS,
    N_INIT_FINAL,
    SEED,
    StepMix,
    USE_SAVED_MODEL,
    X,
    joblib,
    output_dir,
):
    def train_model(X, K, N_INIT_FINAL, SEED):
        model = StepMix(
            n_components=K,
            measurement="categorical_nan",
            n_init=N_INIT_FINAL,
            random_state=SEED,
            max_iter=500,
            progress_bar=1,
        )
        model.fit(X)
        print(f"avg log-likelihood per respondent: {model.score(X):.2f}")

        save_name = f"stepmix_model_k{K}.joblib"

        joblib.dump(model, output_dir / save_name)
        print(f"Saved model → {output_dir / save_name}")

    if not USE_SAVED_MODEL:
        train_model(X, N_CLUSTERS, N_INIT_FINAL, SEED)
    return


@app.cell
def _(N_CLUSTERS, X, joblib, output_dir):
    def load_fitted_model(K, output_dir, X):
        """Load the fitted LCA and predict cluster memberships."""
        model = joblib.load(output_dir / f"stepmix_model_k{K}.joblib")
        posteriors = model.predict_proba(X)
        assignments = posteriors.argmax(axis=1)
        return model, posteriors, assignments

    model, posteriors, assignments = load_fitted_model(N_CLUSTERS, output_dir, X)
    return assignments, model, posteriors


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Model Diagnostics

    Cluster sizes, posterior certainty and separation quality of the fitted
    model.
    """)
    return


@app.cell
def _(N_CLUSTERS, assignments, json, np, output_dir, pd, posteriors):
    def cluster_diagnostics(K, assignments, posteriors):
        """Cluster sizes/shares, posterior certainty and relative entropy."""
        n = len(assignments)
        cert = posteriors.max(axis=1)
        rel_entropy = 1 - (
            -(posteriors * np.log(posteriors + 1e-12)).sum(axis=1).sum()
        ) / (n * np.log(K))
        diagnostics = pd.DataFrame(
            {
                "cluster": range(K),
                "n_respondents": np.bincount(assignments, minlength=K),
                "share": np.bincount(assignments, minlength=K) / n,
            }
        )
        return rel_entropy, cert, diagnostics

    def save_model_diagnostics(K, cert, rel_entropy, diagnostics, output_dir, json):
        """Persist machine-readable diagnostics for downstream comparison.

        `cluster_diagnostics.csv` holds per-cluster sizes/shares;
        `model_diagnostics.json` holds overall quality metrics that
        `cluster_analysis.py` reads to compare solutions across k.
        """
        diagnostics.to_csv(output_dir / "cluster_diagnostics.csv", index=False)
        json.dump(
            {
                "k": K,
                "n_respondents": int(len(cert)),
                "mean_certainty": float(cert.mean()),
                "median_certainty": float(np.median(cert)),
                "frac_cert_gt_0.9": float((cert > 0.9).mean()),
                "relative_entropy": float(rel_entropy),
            },
            open(output_dir / "model_diagnostics.json", "w"),
            indent=2,
        )

    rel_entropy, cert, diagnostics = cluster_diagnostics(
        N_CLUSTERS, assignments, posteriors
    )
    save_model_diagnostics(N_CLUSTERS, cert, rel_entropy, diagnostics, output_dir, json)
    print(f"Mean posterior certainty: {cert.mean():.3f}")
    print(f"Relative entropy (1 = perfectly separated): {rel_entropy:.3f}")
    print(
        f"Saved diagnostics → {output_dir / 'cluster_diagnostics.csv'} and "
        f"{output_dir / 'model_diagnostics.json'}"
    )
    diagnostics
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Cluster Outputs

    Artefacts saved for downstream use: per-respondent cluster assignments and
    the per-cluster empirical response distributions consumed by
    `build_dataset.py`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Cluster Assignments

    Per-respondent hard assignments (posterior mode) and membership
    probabilities, keyed by respondent id.
    """)
    return


@app.cell
def _(N_CLUSTERS, assignments, output_dir, pd, posteriors, wvs_df):
    def save_cluster_assignments(K, wvs_df, assignments, posteriors, output_dir):
        """Save per-respondent cluster assignments and membership probabilities."""
        assign_df = pd.DataFrame(
            {"id": wvs_df["id"], "cluster": assignments}
            | {f"prob_cluster_{k}": posteriors[:, k] for k in range(K)}
        )
        assign_df.to_csv(output_dir / "cluster_assignments.csv", index=False)

    save_cluster_assignments(N_CLUSTERS, wvs_df, assignments, posteriors, output_dir)
    print(f"Saved clusters → {output_dir / 'cluster_assignments.csv'}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Cluster Response Distributions

    For each question and cluster, compute the **posterior-weighted empirical
    response distribution**: each respondent contributes to each cluster in
    proportion to their membership probability. This (not the model's smoothed
    emission parameters) is what `build_dataset.py` consumes, since it reflects
    the raw response patterns including each cluster's non-response rate.
    """)
    return


@app.cell
def _(N_CLUSTERS, X_raw, category_codes, np, posteriors, value_cols):
    def cluster_distributions(col, X_raw, category_codes, K, posteriors):
        """Per-cluster distribution over the column's original codes."""
        obs = X_raw[col]
        codes = category_codes[col]
        dist = np.zeros((K, len(codes)))
        for k in range(K):
            w = posteriors[:, k]
            for j, v in enumerate(codes):
                dist[k, j] = w[obs == v].sum()
            if dist[k].sum() > 0:
                dist[k] /= dist[k].sum()
        # non-response rate per cluster (structural missing only: refused/not asked/not applicable)
        nonresponse = np.array(
            [
                posteriors[obs.isna(), k].sum() / max(posteriors[:, k].sum(), 1e-12)
                for k in range(K)
            ]
        )
        return dist, nonresponse

    def compute_all_distributions(value_cols, X_raw, category_codes, K, posteriors):
        """Posterior-weighted response distributions for every value column."""
        return {
            c: cluster_distributions(c, X_raw, category_codes, N_CLUSTERS, posteriors)
            for c in value_cols
        }

    distributions = compute_all_distributions(
        value_cols, X_raw, category_codes, N_CLUSTERS, posteriors
    )
    f"Computed distributions for {len(distributions)} questions × {N_CLUSTERS} clusters"
    return (distributions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Consultation Survey Design

    The public consultation asks each participant a handful of WVS questions.
    The cells below pick which questions are most informative for cluster
    assignment and how many are needed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Question Selection for the Consultation Survey

    Rank questions by how differently the clusters answer them (mean pairwise
    total variation distance between cluster distributions). The public
    consultation only has time to ask each participant a handful of WVS
    questions — the top of this list is where those questions should come from.
    """)
    return


@app.cell
def _(
    N_CLUSTERS,
    col2question,
    distributions,
    np,
    output_dir,
    pd,
    question_mapping,
):
    def informativeness(col, distributions, K):
        """Mean pairwise TVD between cluster distributions for a column."""
        d = distributions[col][0]
        return float(
            np.mean(
                [
                    0.5 * np.abs(d[i] - d[j]).sum()
                    for i in range(K)
                    for j in range(i + 1, K)
                ]
            )
        )

    def build_lookups(question_mapping):
        """column -> word response types / sub-question text lookups."""
        word_responses = {}
        sub_questions = {}
        for entry in question_mapping:
            sqs = entry.get("sub_questions") or [None] * len(entry["column_names"])
            for i, col in enumerate(entry["column_names"]):
                word_responses[col] = entry["word_response_types"]
                sub_questions[col] = sqs[i] if i < len(sqs) else None
        return word_responses, sub_questions

    def find_controversial_columns(question_mapping):
        """Columns whose text matches an exclusion keyword, plus a manual list."""
        exclude_keywords = [
            "homosexual",
            "immig",
            "abortion",
            "prostitution",
            "divorce",
            "sex before marriage",
            "casual sex",
            "wife",
            "husband",
            "women",
            "gender",
        ]
        controversial = set()
        for entry in question_mapping:
            sqs = entry.get("sub_questions") or [None] * len(entry["column_names"])
            qtext = entry["question"].lower()
            for i, col in enumerate(entry["column_names"]):
                sq = (sqs[i] or "") if i < len(sqs) else ""
                combined = qtext + " " + sq.lower()
                if any(k in combined for k in exclude_keywords):
                    controversial.add(col)
        controversial.update(
            {"Q189", "Q190", "Q22", "Q36", "Q119", "Q263", "Q264", "Q265"}
        )
        return controversial

    def compute_informativeness(
        distributions, K, col2question, question_mapping, output_dir
    ):
        """Rank questions by how differently the clusters answer them; save to disk."""
        sorted_cols = sorted(
            distributions,
            key=lambda c: informativeness(c, distributions, K),
            reverse=True,
        )
        word_responses, sub_questions = build_lookups(question_mapping)
        controversial = find_controversial_columns(question_mapping)

        informativeness_df = pd.DataFrame(
            {
                "column": sorted_cols,
                "informativeness": [
                    informativeness(c, distributions, K) for c in sorted_cols
                ],
                "question": [col2question.get(c, "") for c in sorted_cols],
                "sub_question": [sub_questions.get(c, "") for c in sorted_cols],
                "response_options": [word_responses.get(c, []) for c in sorted_cols],
                "controversial": [c in controversial for c in sorted_cols],
            }
        ).round(3)

        informativeness_df.to_csv(
            output_dir / "question_informativeness.csv", index=False
        )
        return informativeness_df

    informativeness_df = compute_informativeness(
        distributions, N_CLUSTERS, col2question, question_mapping, output_dir
    )
    print(
        f"Saved informativeness rankings → {output_dir / 'question_informativeness.csv'}"
    )
    print()
    print(informativeness_df.head(15).to_string())
    return (informativeness_df,)


@app.cell
def _(
    N_CLUSTERS,
    category_codes,
    col2question,
    distributions,
    figures_dir,
    informativeness_df,
    np,
    plt,
    question_mapping,
):
    def build_word_labels(question_mapping):
        """column -> {numeric code: word label} for the value columns."""
        word_labels = {}
        for entry in question_mapping:
            codes = [float(x) for x in entry["numeric_response_types"]]
            for col in entry["column_names"]:
                labels = {}
                for w, c in zip(entry["word_response_types"], codes):
                    if c >= 0:
                        labels[c] = w
                word_labels[col] = labels
        return word_labels

    def plot_top_splitting_questions(
        informativeness_df,
        distributions,
        category_codes,
        col2question,
        word_labels,
        K,
        output_dir,
    ):
        """Multi-panel figure with the full per-cluster response distribution for
        each of the top-5 most splitting questions. Saved next to the LaTeX table
        so the table can reference it with a relative path."""
        import textwrap

        top_cols = informativeness_df["column"].head(5).tolist()
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        fig, axes = plt.subplots(len(top_cols), 1, figsize=(10, 14))
        for ax, col in zip(np.atleast_1d(axes), top_cols):
            dist, _ = distributions[col]
            labels = [
                word_labels.get(col, {}).get(v, str(v)) for v in category_codes[col]
            ]
            x = np.arange(len(labels))
            w = 0.8 / K
            for k in range(K):
                ax.bar(
                    x + (k - (K - 1) / 2) * w,
                    dist[k],
                    w,
                    label=f"Cluster {k}",
                    color=colors[k % len(colors)],
                    alpha=0.85,
                )
            ax.set_ylabel("Probability")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_title(
                textwrap.fill(col2question.get(col, col), width=110),
                fontsize=9,
                fontweight="bold",
            )
            ax.legend(fontsize=8, loc="upper right")

        fig.tight_layout(h_pad=2.5)
        fig.savefig(
            output_dir / "top5_splitting_questions.png", dpi=150, bbox_inches="tight"
        )
        return fig

    word_labels = build_word_labels(question_mapping)
    _fig = plot_top_splitting_questions(
        informativeness_df,
        distributions,
        category_codes,
        col2question,
        word_labels,
        N_CLUSTERS,
        figures_dir,
    )
    print(f"Saved distribution figure → {figures_dir / 'top5_splitting_questions.png'}")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Certainty vs Number of Questions

    The consultation will only ask each participant ~5–10 questions.
    This section evaluates which questions — **both value survey and demographic** —
    are most informative for cluster assignment, and how many are needed.
    Demographic questions are often available without asking (e.g. region from IP
    geolocation, sex from profile data), so they are "free" discriminators.
    """)
    return


@app.cell
def _(N_CLUSTERS, informativeness_df, np, posteriors, wvs_df):
    def tvd(a, b):
        return float(0.5 * np.abs(a - b).sum())

    def demographic_tvd(col, wvs_df, K, posteriors):
        """Mean pairwise TVD of a demographic column's distribution across clusters."""
        vals = wvs_df[col].astype(str)
        codes = sorted(vals.unique())
        dist = np.zeros((K, len(codes)))
        for k in range(K):
            w = posteriors[:, k]
            for j, v in enumerate(codes):
                dist[k, j] = w[vals == v].sum()
            if dist[k].sum() > 0:
                dist[k] /= dist[k].sum()
        return np.mean(
            [tvd(dist[i], dist[j]) for i in range(K) for j in range(i + 1, K)]
        )

    def rank_all_questions(wvs_df, informativeness_df, K, posteriors):
        """Value + demographic questions ranked by cluster informativeness."""
        demo_cols = [
            c
            for c in wvs_df.columns
            if not c.startswith("Q") and c not in ("id", "date")
        ]
        demo_items = [
            (col, demographic_tvd(col, wvs_df, K, posteriors), "demographic", col)
            for col in demo_cols
        ]
        value_items = [
            (
                c,
                float(informativeness_df.set_index("column").loc[c, "informativeness"]),
                "value",
                informativeness_df.set_index("column").loc[c, "question"],
            )
            for c in informativeness_df["column"]
        ]
        all_items = value_items + demo_items
        all_items.sort(key=lambda x: x[1], reverse=True)
        return all_items

    all_items = rank_all_questions(wvs_df, informativeness_df, N_CLUSTERS, posteriors)

    print(f"{'Rank':<5} {'Type':<13} {'Question':<40} {'Informativeness':<10}")
    print("-" * 68)
    for i, (col, info, typ, qtext) in enumerate(all_items[:25]):
        print(
            f"{i + 1:<5} {typ:<13} {(qtext[:38] if len(qtext) > 38 else qtext):<40} {info:<10.3f}"
        )
    return (all_items,)


@app.cell
def _(X, informativeness_df, model, np, question_mapping):
    def evaluate_question_group_certainty(
        X,
        informativeness_df,
        question_mapping,
        model,
        TOTAL_QUESTIONS=10,
        EXCLUDE_CONTROVERSIAL=True,
    ):
        """Expected assignment certainty from the top question groups.

        `TOTAL_QUESTIONS` = number of question groups to show;
        `EXCLUDE_CONTROVERSIAL` = filter out controversial questions.
        """
        from collections import defaultdict

        qid_by_col = {}
        for entry in question_mapping:
            for col in entry["column_names"]:
                qid_by_col[col] = entry["id"]

        pool = (
            informativeness_df[~informativeness_df["controversial"]][:TOTAL_QUESTIONS]
            if EXCLUDE_CONTROVERSIAL
            else informativeness_df[:TOTAL_QUESTIONS]
        )
        groups = defaultdict(list)
        for _, row in pool.iterrows():
            groups[qid_by_col.get(row["column"])].append(row)

        ranked = sorted(
            groups.items(),
            key=lambda x: max(i["informativeness"] for i in x[1]),
            reverse=True,
        )

        selected = []
        print(
            f"Top {TOTAL_QUESTIONS} question groups{' (non-controversial only)' if EXCLUDE_CONTROVERSIAL else ''}:\n"
        )
        for gid, items in ranked:
            entry = next((e for e in question_mapping if e["id"] == gid), None)
            qtext = entry["question"] if entry else "?"
            tops = sorted(items, key=lambda i: i["informativeness"], reverse=True)
            selected.extend(s["column"] for s in tops)

            print(f"  {qtext}  ({len(items)} sub-questions)")
            for s in tops:
                sq = s.get("sub_question", "") or ""
                tag = " [CONTROVERSIAL]" if s.get("controversial", False) else ""
                print(
                    f"    {s['column']:5s}  info={s['informativeness']:.3f}  {sq}{tag}"
                )
            print()

        Xp = X.copy()
        Xp[[c for c in X.columns if c not in selected]] = np.nan
        cert = model.predict_proba(Xp).max(axis=1)
        print(
            f"Expected certainty (all {len(selected)} sub-questions from {TOTAL_QUESTIONS} questions):"
        )
        print(
            f"  Mean: {cert.mean():.3f}  Median: {np.median(cert):.3f}  >0.9: {(cert > 0.9).mean():.1%}"
        )
        return float(cert.mean()), float(np.median(cert)), float((cert > 0.9).mean())

    evaluate_question_group_certainty(X, informativeness_df, question_mapping, model)
    return


@app.cell
def _(X, all_items, figures_dir, informativeness_df, model, np, pd, plt):
    def plot_certainty_curves(
        X, all_items, informativeness_df, model, figures_dir, max_questions=15
    ):
        """Assignment certainty vs number of questions for value vs non-controversial pools."""
        value_only = [c for c, _, t, _ in all_items if t == "value"]
        clean_only = informativeness_df[~informativeness_df["controversial"]][
            "column"
        ].tolist()
        n_list = range(1, max_questions + 1)

        def compute_certs(cols):
            rows = []
            for n_q in n_list:
                top_n = cols[:n_q]
                Xp = X.copy()
                Xp[[c for c in X.columns if c not in top_n]] = np.nan
                cert = model.predict_proba(Xp).max(axis=1)
                rows.append(
                    {
                        "n_questions": n_q,
                        "avg_certainty": round(cert.mean(), 3),
                        "med_certainty": round(float(np.median(cert)), 3),
                        "frac_gt_0.9": (cert > 0.9).mean(),
                    }
                )
            return pd.DataFrame(rows)

        all_certs = compute_certs(value_only)
        clean_certs = compute_certs(clean_only)

        # Print top 5 comparison table
        print(
            "Metric" + " " * 12 + "Top 5 (all)" + " " * 5 + "Top 5 (non-controversial)"
        )
        r_all = all_certs[all_certs["n_questions"] == 5].iloc[0]
        r_clean = clean_certs[clean_certs["n_questions"] == 5].iloc[0]
        print(
            f"Mean certainty       {r_all['avg_certainty']:<15.3f} {r_clean['avg_certainty']:.3f}"
        )
        print(
            f"Median certainty     {r_all['med_certainty']:<15.3f} {r_clean['med_certainty']:.3f}"
        )
        print(
            f"p(cert > 0.9)        {r_all['frac_gt_0.9']:<15.1%} {r_clean['frac_gt_0.9']:.1%}"
        )

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            all_certs["n_questions"],
            all_certs["avg_certainty"],
            "o-",
            color="#1f77b4",
            label="avg — top value questions",
        )
        ax.plot(
            all_certs["n_questions"],
            all_certs["med_certainty"],
            "s--",
            color="#1f77b4",
            label="median — top value questions",
        )
        ax.plot(
            clean_certs["n_questions"],
            clean_certs["avg_certainty"],
            "o-",
            color="#ff7f0e",
            label="avg — top non-controversial",
        )
        ax.plot(
            clean_certs["n_questions"],
            clean_certs["med_certainty"],
            "s--",
            color="#ff7f0e",
            label="median — top non-controversial",
        )

        ax.fill_between(
            all_certs["n_questions"],
            0,
            all_certs["frac_gt_0.9"],
            alpha=0.08,
            color="#1f77b4",
            label="frac > 0.9 — all",
        )
        ax.fill_between(
            clean_certs["n_questions"],
            0,
            clean_certs["frac_gt_0.9"],
            alpha=0.08,
            color="#ff7f0e",
            label="frac > 0.9 — non-controversial",
        )

        ax.axhline(0.9, color="grey", ls=":", alpha=0.5)
        ax.set_xlabel("Number of questions asked")
        ax.set_ylabel("Posterior certainty")
        ax.set_title("Assignment confidence: top value questions vs non-controversial")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xticks(list(n_list))
        fig.tight_layout()
        fig.savefig(
            figures_dir / "certainty_by_n_questions.png", dpi=150, bbox_inches="tight"
        )
        return fig

    _fig = plot_certainty_curves(X, all_items, informativeness_df, model, figures_dir)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Understanding cluster

    Basic demographic summary of what each cluster looks like, using the
    demographic columns in `wvs_value_survey.csv` (the columns that follow the
    value questions). The cell first ranks demographics by how differently
    their distribution varies across clusters (mean pairwise total variation
    distance, posterior-weighted) — this tells us which demographics
    distinguish the subgroups — then shows the dominant response per cluster.
    Non-response categories (Missing, Don't know, No answer, Not applicable)
    are excluded from the per-cluster shares. The full per-cluster
    distributions are saved to `cluster_demographics.csv` for the report.
    """)
    return


@app.cell
def _(N_CLUSTERS, assignments, np, output_dir, pd, posteriors, wvs_df):
    # Columns of interest: the demographic columns that come after the value
    # questions (Q1..Q258) in wvs_value_survey.csv, restricted to those that
    # give an interpretable "who is in each cluster" picture.
    DEMOGRAPHIC_COLS = [
        "age",  # numeric
        "household_size",  # numeric
        "sex",
        "region",
        "townsize",
        "settlement_type",
        "immigrant",
        "children",
        "marital_status",
        "education_respondent_cs",
        "employment_respondent",
        "savings",
        "social_class",
        "income_scale",
        "religion",
        "home_language",
    ]

    def clean_categories(col):
        """Drop junk/non-response categories (Missing, Don't know, etc.)."""
        junk = "missing|no answer|don't know|don´t know|not applicable|other missing|unknown|none of these"
        mask = ~wvs_df[col].astype(str).str.lower().str.contains(junk, regex=True)
        return wvs_df.loc[mask, col]

    def _tvd_pair(a, b):
        return float(0.5 * np.abs(a - b).sum())

    def demographic_tvd_rank(col):
        """Mean pairwise TVD of a demographic column's distribution across
        clusters, weighted by posterior membership probabilities."""
        vals = wvs_df[col].astype(str)
        codes = sorted(vals.unique())
        dist = np.zeros((N_CLUSTERS, len(codes)))
        for k in range(N_CLUSTERS):
            w = posteriors[:, k]
            for j, v in enumerate(codes):
                dist[k, j] = w[vals == v].sum()
            if dist[k].sum() > 0:
                dist[k] /= dist[k].sum()
        return np.mean(
            [
                _tvd_pair(dist[i], dist[j])
                for i in range(N_CLUSTERS)
                for j in range(i + 1, N_CLUSTERS)
            ]
        )

    def build_wide_summary():
        """One row per demographic; one column per cluster. Categorical cells
        show the dominant response (share %); numeric cells show mean / median.
        Sorted by how differently the demographic's distribution varies across
        clusters (mean pairwise TVD)."""
        rows = []
        for _col in DEMOGRAPHIC_COLS:
            row = {"demographic": _col, "tvd": demographic_tvd_rank(_col)}
            if _col in ("age", "household_size"):
                vals = pd.to_numeric(wvs_df[_col], errors="coerce")
                for k in range(N_CLUSTERS):
                    row[f"cluster_{k}"] = (
                        f"{vals[assignments == k].mean():.1f} / "
                        f"{vals[assignments == k].median():.0f}"
                    )
            else:
                s = clean_categories(_col)
                tab = (
                    pd.crosstab(
                        s, pd.Series(assignments, name="cluster"), normalize="columns"
                    )
                    * 100
                )
                for k in range(N_CLUSTERS):
                    if k not in tab.columns:
                        row[f"cluster_{k}"] = "—"
                        continue
                    top = tab[k].idxmax()
                    row[f"cluster_{k}"] = f"{top} ({tab[k].max():.0f}%)"
            rows.append(row)
        return pd.DataFrame(rows)

    summary_df = build_wide_summary().sort_values("tvd", ascending=False).round(3)
    summary_df.to_csv(output_dir / "cluster_demographics.csv", index=False)
    print(f"Saved demographic summary → {output_dir / 'cluster_demographics.csv'}")

    summary_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Cluster Demographic Infographic

    A visual profile of each cluster across the five most readable demographics —
    education, social class, income, religion and age (binned into decades).
    Each cluster panel stacks one 100%-stacked horizontal bar per demographic,
    so the composition of every cluster can be compared at a glance.
    Non-response categories are excluded and segments ≥ 8% are labelled.
    """)
    return


@app.cell
def _(N_CLUSTERS, assignments, figures_dir, np, pd, plt, wvs_df):
    def cluster_demographic_infographic(K, wvs_df, assignments, figures_dir):
        """Per-cluster 100%-stacked bar infographic of 5 key demographics.

        One panel per cluster; inside each panel one horizontal stacked bar per
        demographic: education, social class, income, religion and age (binned
        into decades). Segment width = that category's share of the cluster.
        Non-response categories are excluded; segments >= 8% are labelled.
        """
        demos = [
            ("education_respondent_cs", "Education"),
            ("social_class", "Social class"),
            ("income_scale", "Income"),
            ("religion", "Religion"),
            ("age", "Age"),
        ]
        cmaps = [
            plt.cm.Blues,
            plt.cm.Oranges,
            plt.cm.Greens,
            plt.cm.Purples,
            plt.cm.Greys,
        ]
        junk = (
            "missing|no answer|don't know|don´t know|not applicable|"
            "other missing|unknown|none of these"
        )

        def short_labels(col, cat):
            """Condense WVS category strings for readability inside the bars."""
            if col == "age":
                return cat
            if col == "education_respondent_cs":
                return {
                    "NZ: Completed university or polytechnic degree": "University degree",
                    "NZ: Some university, wananga, polytechnic or other tertiary": "Some tertiary",
                    "NZ: Secondary school for 4 years or more": "Sec. school 4+ yrs",
                    "NZ: Secondary school for up to 3 years": "Sec. school ≤3 yrs",
                    "NZ: Kura kaupapa/primary school (including intermediate)": "Primary school",
                    "NZ: No formal schooling": "No schooling",
                }.get(cat, cat)
            if col == "social_class":
                return {
                    "Upper middle class": "Upper middle",
                    "Lower middle class": "Lower middle",
                }.get(cat, cat)
            if col == "income_scale":
                # NZ WVS income question: "Which of the following categories best
                # describes your own total yearly income from all sources before
                # tax?" Raw codes are steps 1-10; map to compact NZ$ yearly bands.
                return {
                    "1": "≤$10K",
                    "2": "$10-15K",
                    "3": "$15-20K",
                    "4": "$20-25K",
                    "5": "$25-30K",
                    "6": "$30-40K",
                    "7": "$40-50K",
                    "8": "$50-70K",
                    "9": "$70-100K",
                    "10": "$100K+",
                }.get(cat.split("{")[1].rstrip("}") if "{" in cat else cat, cat)
            if col == "religion":
                return {
                    "Do not belong to a denomination{No religion}": "No religion",
                    "Other Christian (Jehova withness...){Other Christian}": "Other Christian",
                    "Catholic (Roman/Greek/etc)": "Catholic",
                }.get(cat, cat)
            return cat

        def cleaned_values(col):
            """Per-respondent cleaned category strings for the column."""
            if col == "age":
                age_n = pd.to_numeric(wvs_df["age"], errors="coerce")
                return pd.cut(
                    age_n,
                    bins=[17, 24, 34, 44, 54, 64, 100],
                    labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
                ).astype(str)
            keep = ~wvs_df[col].astype(str).str.lower().str.contains(junk, regex=True)
            return wvs_df.loc[keep, col].astype(str)

        fig, axes = plt.subplots(1, K, figsize=(12.0 * K, 9.5))
        axes = np.atleast_1d(axes)
        n = len(wvs_df)

        cluster_membership = np.asarray(assignments)

        for k, ax in enumerate(axes):
            k_sel = cluster_membership == k
            for y_i, ((col, label), cmap) in enumerate(zip(demos, cmaps)):
                vals = cleaned_values(col)
                vals = vals[k_sel[vals.index.to_numpy()]]
                cats = np.array([c for c in vals.unique() if c != "nan"])
                counts = np.array([(vals == c).sum() for c in cats])
                order = np.argsort(-counts)  # most common category first (darkest)
                cats, counts = cats[order], counts[order]
                shares = counts / counts.sum()
                left = 0.0
                for i, (c, s) in enumerate(zip(cats, shares)):
                    color = cmap(0.25 + 0.6 * i / max(len(cats), 1))
                    ax.barh(
                        y_i,
                        s,
                        left=left,
                        color=color,
                        edgecolor="white",
                        linewidth=0.7,
                    )
                    if s >= 0.08:
                        ax.text(
                            left + s / 2,
                            y_i,
                            f"{short_labels(col, c)} {s:.0%}",
                            ha="center",
                            va="center",
                            fontsize=12,
                            fontweight="bold",
                            color="white" if np.mean(color[:3]) < 0.55 else "black",
                        )
                    left += s
            ax.set_yticks(range(len(demos)))
            if k == 0:
                ax.set_yticklabels([d[1] for d in demos], fontsize=9, fontweight="bold")
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", left=False)
            ax.set_ylim(-0.5, len(demos) - 0.5)
            ax.set_xlim(0, 1)
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
            n_k = int((assignments == k).sum())
            ax.set_title(
                f"Cluster {k}  ·  n={n_k} ({n_k / n:.0%})",
                fontsize=9,
                fontweight="bold",
            )

        fig.suptitle(
            "Who is in each cluster? Composition by key demographics",
            fontsize=13,
            fontweight="bold",
            y=1.0,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(
            figures_dir / "cluster_demographics_infographic.png",
            dpi=400,
            bbox_inches="tight",
        )
        return fig

    _fig = cluster_demographic_infographic(N_CLUSTERS, wvs_df, assignments, figures_dir)
    print(f"Saved infographic → {figures_dir / 'cluster_demographics_infographic.png'}")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Sanity Checks

    Independent, assumption-free checks that the LCA partition reflects real
    structure: PCA and t-SNE projections, a plain K-Means comparison, and an
    MCA + K-Means pipeline that makes none of the LCA's local-independence
    assumptions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### PCA 2D Reduction

    A 2D PCA projection of the one-hot-encoded response data, coloured by
    cluster assignment, to visually confirm that the clusters separate in a
    simple linear subspace.
    """)
    return


@app.cell
def _(X_raw, assignments, figures_dir, pd, plt):
    def pca_sanity_check(X_raw, assignments, figures_dir):
        """2D PCA of the one-hot-encoded responses coloured by LCA cluster."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        X_str = X_raw.astype(str).replace("nan", "Missing")
        X_encoded = pd.get_dummies(X_str, prefix_sep="=")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)

        pca = PCA(n_components=5, random_state=42)
        coords = pca.fit_transform(X_scaled)

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.tab10.colors  # one distinct colour per cluster (works for K>2)
        for k in sorted(set(assignments)):
            mask = assignments == k
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=colors[k % len(colors)],
                label=f"Cluster {k}",
                alpha=0.6,
                s=15,
                edgecolors="none",
            )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        ax.set_title("PCA projection of WVS responses coloured by cluster")
        ax.legend(markerscale=3)
        fig.tight_layout()
        fig.savefig(
            figures_dir / "pca_cluster_sanity_check.png", dpi=150, bbox_inches="tight"
        )

        print(
            "PCA: {:.1%} + {:.1%} = {:.1%} variance in 2 components".format(
                pca.explained_variance_ratio_[0],
                pca.explained_variance_ratio_[1],
                pca.explained_variance_ratio_[:2].sum(),
            )
        )
        return fig, pca.explained_variance_ratio_

    _fig, explained_variance = pca_sanity_check(X_raw, assignments, figures_dir)
    explained_variance
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### t-SNE 2D Reduction

    t-SNE is a non-linear method that often reveals more structure than PCA
    for high-dimensional categorical data. Here it is applied to the same
    one-hot-encoded responses.
    """)
    return


@app.cell
def _(X_raw, assignments, figures_dir, pd, plt):
    def tsne_sanity_check(X_raw, assignments, figures_dir):
        """2D t-SNE of the one-hot-encoded responses coloured by LCA cluster."""
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler

        X_str = X_raw.astype(str).replace("nan", "Missing")
        X_encoded = pd.get_dummies(X_str, prefix_sep="=")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)

        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=30,
            learning_rate="auto",
            init="random",
        )
        coords = tsne.fit_transform(X_scaled)

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.tab10.colors  # one distinct colour per cluster (works for K>2)
        for k in sorted(set(assignments)):
            mask = assignments == k
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=colors[k % len(colors)],
                label=f"Cluster {k}",
                alpha=0.6,
                s=15,
                edgecolors="none",
            )
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
        ax.set_title("t-SNE projection of WVS responses coloured by cluster")
        ax.legend(markerscale=3)
        fig.tight_layout()
        fig.savefig(
            figures_dir / "tsne_cluster_sanity_check.png", dpi=150, bbox_inches="tight"
        )
        return fig

    _fig = tsne_sanity_check(X_raw, assignments, figures_dir)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### LCA vs K-Modes Robustness Check

    LCA assumes local independence — that all association between items is
    explained by the latent classes. K-Modes makes no such assumption: it is a
    centroid-based clustering of the raw categorical responses themselves (the
    categorical analogue of K-Means, no embedding involved). If this
    assumption-free pipeline reproduces essentially the same partition, the LCA
    solution is not an artefact of its assumptions.

    Workflow:
    1. Fit K-Modes on the raw categorical responses (shared across the method
       comparison below).
    2. Compare the K-Modes partition with the LCA reference (ARI, cross-tab).
    3. Plot both partitions side by side in the shared MCA embedding.
    """)
    return


@app.cell
def _(
    assignments,
    figures_dir,
    kmodes_labels,
    mca_coords,
    mca_inertia,
    np,
    pd,
    plt,
):
    def lca_vs_kmodes_check(assignments, kmodes_labels, coords, inertia, figures_dir):
        """K-Modes (raw categorical responses, no embedding) vs the LCA reference:
        ARI, cross-tab and a side-by-side projection on the shared MCA embedding."""
        from sklearn.metrics import adjusted_rand_score

        cum = np.cumsum(inertia) / inertia.sum()
        print(
            f"MCA: {coords.shape[1]} components explain {cum[coords.shape[1] - 1]:.1%} of inertia "
            f"(dim 1 alone: {inertia[0] / inertia.sum():.1%})"
        )

        ari = adjusted_rand_score(assignments, kmodes_labels)
        ct = pd.crosstab(
            pd.Series(assignments, name="LCA cluster"),
            pd.Series(kmodes_labels, name="K-Modes cluster"),
        )
        print(f"Adjusted Rand Index (LCA vs K-Modes): {ari:.3f}")
        print("Cross-tabulation:")
        print(ct.to_string())
        print()
        print(
            "Cluster sizes — LCA:",
            dict(zip(*np.unique(assignments, return_counts=True))),
        )
        print(
            "Cluster sizes — K-Modes:",
            dict(zip(*np.unique(kmodes_labels, return_counts=True))),
        )

        # Side-by-side projection of the two partitions on the first two MCA dims,
        # with the ARI between LCA and K-Modes shown in the figure title
        colors = plt.cm.tab10.colors[
            2:6
        ]  # one distinct colour per cluster (works for K>2)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, labels, title in [
            (axes[0], assignments, "LCA"),
            (axes[1], kmodes_labels, "K-Modes"),
        ]:
            for k in sorted(set(labels)):
                mask = labels == k
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    c=colors[k % len(colors)],
                    label=f"Cluster {k}",
                    alpha=0.6,
                    s=15,
                    edgecolors="none",
                )
            ax.set_xlabel(f"MCA dim 1 ({inertia[0] / inertia.sum():.1%} inertia)")
            ax.set_ylabel(f"MCA dim 2 ({inertia[1] / inertia.sum():.1%} inertia)")
            ax.set_title(title)
            ax.legend(markerscale=3)
        fig.suptitle(f"Adjusted Rand Index (LCA vs K-Modes) = {ari:.3f}", fontsize=12)
        fig.tight_layout()
        fig.savefig(
            figures_dir / "kmodes_cluster_stability_check.png",
            dpi=150,
            bbox_inches="tight",
        )
        return fig

    _fig = lca_vs_kmodes_check(
        assignments, kmodes_labels, mca_coords, mca_inertia, figures_dir
    )
    _fig
    return


@app.cell
def _(
    N_CLUSTERS,
    X_raw,
    assignments,
    category_codes,
    distributions,
    kmeans_labels,
    mca_coords,
    np,
    pd,
):
    def distribution_ablation(
        X_raw, category_codes, K, distributions, assignments, kmeans_labels, coords
    ):
        """How much do the downstream per-cluster response distributions change?

        `build_dataset.py` consumes per-cluster empirical response distributions.
        Compare the LCA posterior-weighted distributions against hard assignments
        from MCA+K-Means and MCA+Ward, question by question, using total
        variation distance.
        """
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import adjusted_rand_score

        def hard_distributions(col, labels):
            """Per-cluster response distribution over original codes, hard-assigned."""
            obs = X_raw[col]
            codes = category_codes[col]
            dist = np.zeros((K, len(codes)))
            for k in range(K):
                m = labels == k
                for j, v in enumerate(codes):
                    dist[k, j] = (obs[m] == v).sum()
                if dist[k].sum() > 0:
                    dist[k] /= dist[k].sum()
            return dist

        def mean_pairwise_tvd(dist):
            """Mean total variation distance across cluster pairs (informativeness)."""
            return float(
                np.mean(
                    [
                        0.5 * np.abs(dist[i] - dist[j]).sum()
                        for i in range(K)
                        for j in range(i + 1, K)
                    ]
                )
            )

        # Also try Ward hierarchical clustering on the MCA coordinates
        agg_labels = AgglomerativeClustering(n_clusters=K).fit_predict(coords)
        print(
            f"Adjusted Rand Index (LCA vs MCA+Ward): "
            f"{adjusted_rand_score(assignments, agg_labels):.3f}"
        )

        rows = []
        for col in X_raw.columns:
            lca_dist = distributions[col][0]
            km_dist = hard_distributions(col, kmeans_labels)
            agg_dist = hard_distributions(col, agg_labels)
            rows.append(
                {
                    "column": col,
                    "lca_vs_kmeans_tvd": float(
                        0.5 * np.abs(lca_dist - km_dist).sum(axis=1).mean()
                    ),
                    "lca_vs_ward_tvd": float(
                        0.5 * np.abs(lca_dist - agg_dist).sum(axis=1).mean()
                    ),
                    "lca_informativeness": mean_pairwise_tvd(lca_dist),
                    "kmeans_informativeness": mean_pairwise_tvd(km_dist),
                }
            )
        return pd.DataFrame(rows).round(4)

    ablation_df = distribution_ablation(
        X_raw,
        category_codes,
        N_CLUSTERS,
        distributions,
        assignments,
        kmeans_labels,
        mca_coords,
    )

    l2k = ablation_df["lca_vs_kmeans_tvd"]
    l2a = ablation_df["lca_vs_ward_tvd"]
    print(
        "\nDistribution ablation (mean TVD between LCA and alternative "
        "per-cluster distributions, averaged over clusters and questions):"
    )
    print(
        f"  LCA vs MCA+KMeans: mean={l2k.mean():.4f}  median={l2k.median():.4f}  "
        f"p95={l2k.quantile(0.95):.4f}"
    )
    print(
        f"  LCA vs MCA+Ward:   mean={l2a.mean():.4f}  median={l2a.median():.4f}  "
        f"p95={l2a.quantile(0.95):.4f}"
    )
    print(
        f"\nInformativeness (mean pairwise TVD across clusters):\n"
        f"  LCA: {ablation_df['lca_informativeness'].mean():.4f}   "
        f"MCA+KMeans: {ablation_df['kmeans_informativeness'].mean():.4f}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Method Comparison: Which Clustering Produces the Best Split?

    A compact table comparing clustering methods on four basic metrics,
    all computed on the same data (k = $K$, $K$ = chosen for LCA):

    - **Silhouette** (higher = better): mean silhouette over respondents,
      computed on a shared MCA embedding so every method is scored in the
      same space.
    - **BIC** (lower = better): information criterion for model fit. Only
      meaningful for generative models (LCA, MCA+GMM); hard-clustering
      methods (K-Modes, K-Means, Ward, K-Medoids, Spectral, SOM) have no
      likelihood so their BIC is blank.
    - **Cluster size entropy** (higher = more balanced): normalised entropy
      of the cluster-size distribution (1 = perfectly balanced).
    - **Mean pairwise TVD** (higher = more informative): how different the
      per-cluster response distributions are, averaged over cluster pairs
      and questions.
    - **Noise fraction** (lower = better): share of respondents that a
      density-based method (HDBSCAN) could not confidently assign; these
      are re-assigned to the nearest cluster medoid so they can still be
      scored. All other methods assign everyone, so their fraction is 0.

    Each method is fitted by its own cell below (`fit_*_labels`). All
    MCA-based methods share the same embedding so the comparison is fair.
    """)
    return


@app.cell
def _(SEED, X_raw, np):
    def fit_shared_mca(X_raw, SEED):
        """Shared MCA embedding (≥ 70% of inertia kept) so that every
        MCA-based method below is fitted in the same space."""
        import prince

        X_str = X_raw.fillna("Missing").astype(str)
        mca = prince.MCA(n_components=50, random_state=SEED).fit(X_str)
        inertia = mca.eigenvalues_
        cum = np.cumsum(inertia) / inertia.sum()
        n_comp = int(np.searchsorted(cum, 0.70) + 1)
        coords = mca.row_coordinates(X_str).iloc[:, :n_comp].to_numpy()
        return coords, X_str, mca.eigenvalues_

    mca_coords, X_str, mca_inertia = fit_shared_mca(X_raw, SEED)
    print(
        f"Shared MCA embedding: {mca_coords.shape[1]} components "
        f"for {mca_coords.shape[0]} respondents"
    )
    return X_str, mca_coords, mca_inertia


@app.cell
def _(X, assignments, model, np):
    def fit_lca_labels(X, assignments, model):
        """LCA is the reference solution (generative → has a real BIC)."""
        labels = np.asarray(assignments).astype(int)
        return labels, float(model.bic(X))

    lca_labels, lca_bic = fit_lca_labels(X, assignments, model)
    return lca_bic, lca_labels


@app.cell
def _(N_CLUSTERS, SEED, X_str):
    def fit_kmodes_labels(K, SEED, X_str):
        """K-Modes on the raw categorical codes (no embedding)."""
        from kmodes.kmodes import KModes

        km = KModes(n_clusters=K, init="Huang", n_init=10, random_state=SEED, verbose=0)
        return km.fit_predict(X_str.to_numpy())

    kmodes_labels = fit_kmodes_labels(N_CLUSTERS, SEED, X_str)
    return (kmodes_labels,)


@app.cell
def _(N_CLUSTERS, SEED, mca_coords):
    def fit_kmeans_labels(K, SEED, coords):
        """K-Means on the shared MCA embedding."""
        from sklearn.cluster import KMeans

        return KMeans(n_clusters=K, random_state=SEED, n_init=10).fit_predict(coords)

    kmeans_labels = fit_kmeans_labels(N_CLUSTERS, SEED, mca_coords)
    return (kmeans_labels,)


@app.cell
def _(N_CLUSTERS, mca_coords):
    def fit_ward_labels(K, coords):
        """Agglomerative (Ward) on the shared MCA embedding."""
        from sklearn.cluster import AgglomerativeClustering

        return AgglomerativeClustering(n_clusters=K).fit_predict(coords)

    ward_labels = fit_ward_labels(N_CLUSTERS, mca_coords)
    return (ward_labels,)


@app.cell
def _(N_CLUSTERS, SEED, mca_coords, np):
    def fit_kmedoids_labels(K, SEED, coords):
        """PAM (k-medoids) on the embedding with Euclidean distance."""
        rng = np.random.default_rng(SEED)
        n = len(coords)
        D = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
        best_score, best_labels = np.inf, None
        for _ in range(10):  # n_init random k-means++-style starts
            medoids = [int(rng.integers(n))]
            d2m = D[medoids[0]].copy()
            for _ in range(1, K):
                probs = d2m / d2m.sum()
                m = int(rng.choice(n, p=probs))
                medoids.append(m)
                d2m = np.minimum(d2m, D[m])
            medoids = np.array(medoids)
            for _ in range(200):  # alternate assign / update-medoid
                labels = D[:, medoids].argmin(axis=1)
                changed = False
                for j in range(K):
                    members = np.nonzero(labels == j)[0]
                    if len(members) == 0:
                        continue
                    within = D[np.ix_(members, members)].sum(axis=1)
                    new_med = members[within.argmin()]
                    if new_med != medoids[j]:
                        medoids[j] = new_med
                        changed = True
                if not changed:
                    break
            score = D[np.arange(n), medoids[labels]].sum()
            if score < best_score:
                best_score, best_labels = score, labels.copy()
        return best_labels

    kmedoids_labels = fit_kmedoids_labels(N_CLUSTERS, SEED, mca_coords)
    return (kmedoids_labels,)


@app.cell
def _(N_CLUSTERS, SEED, mca_coords):
    def fit_spectral_labels(K, SEED, coords):
        """Spectral clustering on the shared MCA embedding."""
        from sklearn.cluster import SpectralClustering

        spec = SpectralClustering(
            n_clusters=K, random_state=SEED, affinity="nearest_neighbors"
        )
        return spec.fit_predict(coords)

    spectral_labels = fit_spectral_labels(N_CLUSTERS, SEED, mca_coords)
    return (spectral_labels,)


@app.cell
def _(N_CLUSTERS, SEED, mca_coords):
    def fit_gmm_labels(K, SEED, coords):
        """Gaussian mixture on the MCA embedding (generative → real BIC)."""
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(
            n_components=K, covariance_type="diag", random_state=SEED, n_init=10
        )
        return gmm.fit_predict(coords), float(gmm.bic(coords))

    gmm_labels, gmm_bic = fit_gmm_labels(N_CLUSTERS, SEED, mca_coords)
    return gmm_bic, gmm_labels


@app.cell
def _(SEED, mca_coords, np):
    def fit_hdbscan_labels(coords, SEED):
        """Density-based clustering on the MCA embedding.

        HDBSCAN takes no fixed k: `min_cluster_size` starts at ~2% of
        respondents and is relaxed until at least one cluster is found (the
        default can return all-noise on MCA embeddings). Respondents in
        low-density regions are labelled noise (-1); we re-assign them to
        the nearest cluster medoid so the row can still be scored, and
        report the noise fraction alongside.
        """
        from sklearn.cluster import HDBSCAN

        n = len(coords)
        raw = None
        for min_size in range(max(2, int(0.02 * n)), 4, -1):
            hdb = HDBSCAN(
                min_cluster_size=min_size,
                min_samples=5,
                cluster_selection_epsilon=0.0,
                copy=True,
            )
            raw = hdb.fit_predict(coords)
            if (raw != -1).any():
                break
        noise_frac = float((raw == -1).mean())

        labels = raw.copy()
        if (raw == -1).any():
            cluster_ids = np.unique(raw[raw != -1])
            medoid = {
                c: coords[raw == c][
                    np.sum(
                        (coords[raw == c] - coords[raw == c].mean(0)) ** 2, 1
                    ).argmin()
                ]
                for c in cluster_ids
            }
            for i in np.nonzero(raw == -1)[0]:
                labels[i] = min(
                    cluster_ids,
                    key=lambda c: float(np.sum((coords[i] - medoid[c]) ** 2)),
                )
        return labels, noise_frac

    hdbscan_labels, hdbscan_noise = fit_hdbscan_labels(mca_coords, SEED)
    print(f"HDBSCAN noise fraction: {hdbscan_noise:.3f}")
    return hdbscan_labels, hdbscan_noise


@app.cell
def _(N_CLUSTERS, SEED, mca_coords, np):
    def fit_som_labels(K, SEED, coords):
        """Self-organising map (Kohonen) on the MCA embedding.

        Pure-numpy rectangular SOM: a grid of prototypes is trained with a
        shrinking Gaussian neighbourhood; the node prototypes are then
        k-means'd into K and each respondent is assigned via its best
        matching unit.
        """
        from sklearn.cluster import KMeans
        import time

        t0 = time.time()

        def progress(msg):
            print(f"SOM [{time.time() - t0:6.1f}s]: {msg}", flush=True)

        rng = np.random.default_rng(SEED)
        n, d = coords.shape
        grid_side = max(5, int(np.ceil(np.sqrt(5 * np.sqrt(n)))))
        n_nodes = grid_side * grid_side

        # Standardise so grid distances are comparable across dimensions
        mu, sd = coords.mean(0), coords.std(0)
        std = (coords - mu) / (sd + 1e-12)

        # Prototypes initialised on random respondents
        proto = std[rng.integers(0, n, n_nodes)].copy()

        # Grid positions of each node (for the Gaussian neighbourhood)
        ys, xs = np.meshgrid(np.arange(grid_side), np.arange(grid_side))
        pos = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(float)

        n_iter = max(1000, 10 * n_nodes)
        sigma0 = grid_side / 2.0
        tau = n_iter / np.log(sigma0)
        alpha0 = 0.5
        progress(f"grid {grid_side}x{grid_side} ({n_nodes} nodes), {n_iter} iters")
        report_every = max(1, n_iter // 10)
        for t in range(1, n_iter + 1):
            s = std[rng.integers(0, n)]  # stochastic SOM: one sample per step
            bmu = int(np.argmin(np.sum((proto - s) ** 2, axis=1)))
            sigma = sigma0 * np.exp(-t / tau)
            alpha = alpha0 * np.exp(-t / n_iter)
            g = np.exp(-np.sum((pos - pos[bmu]) ** 2, axis=1) / (2 * sigma**2))
            proto += (alpha * g)[:, None] * (s - proto)
            if t % report_every == 0:
                progress(f"{t}/{n_iter} iters done ({100 * t // n_iter}%)")

        progress(f"training done, k-means'ing {n_nodes} prototypes into {K} clusters")

        # Collapse the grid into K clusters, then map respondents to them
        node_k = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit_predict(proto)
        progress("k-means on prototypes done, finding each respondent's BMU")
        bmu_all = np.argmin(
            np.sum((std[:, None, :] - proto[None, :, :]) ** 2, axis=2), axis=1
        )
        progress("BMU assignment done, merging tiny clusters")
        labels = node_k[bmu_all].astype(int)

        print(
            f"SOM: initial cluster sizes: {dict(zip(*np.unique(labels, return_counts=True)))}"
        )

        # K-Means on the node grid can leave tiny spurious clusters; merge any
        # cluster with < 2 members into the nearest remaining one so the
        # comparison metrics stay meaningful.
        counts = np.bincount(labels, minlength=K)
        while (counts < 2).any():
            # Only 1-member clusters need merging; 0-member clusters (already
            # emptied by a previous merge) would otherwise make this loop a
            # no-op forever, since argmax picks the first index with count < 2.
            smalls = np.nonzero((counts >= 1) & (counts < 2))[0]
            if len(smalls) == 0:
                break
            small = int(smalls[0])
            others = np.nonzero(counts >= 2)[0]
            if len(others) == 0:
                break
            centroids = {c: std[labels == c].mean(0) for c in others}
            for i in np.nonzero(labels == small)[0]:
                labels[i] = min(
                    others, key=lambda c: float(np.sum((std[i] - centroids[c]) ** 2))
                )
            counts = np.bincount(labels, minlength=K)
        progress(
            f"done, cluster sizes: {dict(zip(*np.unique(labels, return_counts=True)))}"
        )
        return labels

    som_labels = fit_som_labels(N_CLUSTERS, SEED, mca_coords)
    return (som_labels,)


@app.cell
def _(
    gmm_bic,
    gmm_labels,
    hdbscan_labels,
    hdbscan_noise,
    kmeans_labels,
    kmedoids_labels,
    kmodes_labels,
    lca_bic,
    lca_labels,
    som_labels,
    spectral_labels,
    ward_labels,
):
    """Group the per-method fits into the dicts the comparison consumes."""
    method_labels = {
        "LCA": lca_labels,
        "K-Modes": kmodes_labels,
        "MCA+KMeans": kmeans_labels,
        "MCA+Ward": ward_labels,
        "MCA+KMedoids": kmedoids_labels,
        "MCA+Spectral": spectral_labels,
        "MCA+HDBSCAN": hdbscan_labels,
        "MCA+SOM": som_labels,
        "MCA+GMM": gmm_labels,
    }
    method_bics = {"LCA": lca_bic, "MCA+GMM": gmm_bic}
    method_noise = {"MCA+HDBSCAN": hdbscan_noise}
    return method_bics, method_labels, method_noise


@app.cell
def _(method_labels, np, output_dir, pd):
    def pairwise_ari(labels):
        """Pairwise adjusted Rand Index between every pair of methods.

        ARI is chance-corrected agreement between two partitions (1 = identical,
        0 = what random assignment would give). It is label-invariant, so it
        directly measures how much the methods agree on who belongs with whom.
        """
        from sklearn.metrics import adjusted_rand_score

        names = list(labels)
        mat = np.eye(len(names))
        for i, a in enumerate(names):
            for j in range(i + 1, len(names)):
                b = names[j]
                ari = adjusted_rand_score(labels[a], labels[b])
                mat[i, j] = ari
                mat[j, i] = ari
        return pd.DataFrame(mat, index=names, columns=names).round(3)

    ari_matrix = pairwise_ari(method_labels)
    ari_matrix
    print(f"Saved ARI matrix → {output_dir / 'method_ari.csv'}")
    ari_matrix.to_csv(output_dir / "method_ari.csv")
    return


@app.cell
def _(
    N_CLUSTERS,
    X_raw,
    category_codes,
    figures_dir,
    mca_coords,
    method_bics,
    method_labels,
    method_noise,
    np,
    output_dir,
    pd,
):
    def evaluate_methods(
        K, X_raw, category_codes, labels, bics, noise, coords, output_dir, figures_dir
    ):
        """Score every method and assemble the comparison table."""
        from sklearn.metrics import silhouette_score

        def hard_distributions(col, labels, n_clusters):
            """Per-cluster response distribution over original codes, hard-assigned."""
            obs = X_raw[col]
            codes = category_codes[col]
            dist = np.zeros((n_clusters, len(codes)))
            for k in range(n_clusters):
                m = labels == k
                for j, v in enumerate(codes):
                    dist[k, j] = (obs[m] == v).sum()
                if dist[k].sum() > 0:
                    dist[k] /= dist[k].sum()
            return dist

        def highlight_best(df):
            """Highlight the best value in each column (lowest for BIC, highest otherwise)."""

            def highlight(s):
                if s.name == "bic":
                    return ["font-weight: bold;" if v == s.min() else "" for v in s]
                if not pd.api.types.is_numeric_dtype(s):
                    return ["" for _ in s]  # label columns: no highlighting
                return ["font-weight: bold;" if v == s.max() else "" for v in s]

            return df.style.apply(highlight, axis=0)

        def mean_pairwise_tvd(dist, present):
            """Mean total variation distance across present cluster pairs."""
            pairs = [(i, j) for i in present for j in present if i < j]
            if not pairs:
                return np.nan
            return float(
                np.mean([0.5 * np.abs(dist[i] - dist[j]).sum() for i, j in pairs])
            )

        def within_cluster_coherence(col_dists, labels):
            """Mean over questions of the size-weighted within-cluster response
            concentration, measured as 1 - normalised entropy: 1 = every member
            gives the same answer, 0 = responses spread uniformly over the
            categories."""
            weights = np.bincount(labels).astype(float)
            keep = weights > 0
            w = weights[keep] / weights.sum()
            coherence = []
            for d in col_dists:
                j = d.shape[1]
                if j < 2:
                    continue  # single-category question is trivially "narrow"
                p = d[keep]
                # A small cluster can have NO valid response on a filtered
                # question (all members missing) -> its row is all zeros.
                # Exclude it for that question and renormalise the weights.
                valid = p.sum(axis=1) > 0
                if valid.sum() < 2:
                    continue
                wq = w * valid
                wq = wq / wq.sum()
                ent = -(p[valid] * np.log(p[valid] + 1e-12)).sum(axis=1) / np.log(j)
                coherence.append(float(np.average(1.0 - ent, weights=wq[valid])))
            return float(np.mean(coherence)) if coherence else float("nan")

        def size_entropy(labels):
            """Normalised entropy of the cluster-size distribution (1 = balanced)."""
            p = np.bincount(labels).astype(float)
            p = p[p > 0] / p.sum()
            if len(p) < 2:
                return float("nan")
            return float(-np.sum(p * np.log(p)) / np.log(len(p)))

        rows = []
        for name, lab in labels.items():
            # Size everything by the clusters the method actually produced:
            # HDBSCAN can find more (or fewer) clusters than K.
            counts = np.bincount(lab)
            n_clusters = len(counts)
            present = [k for k in range(n_clusters) if counts[k] > 0]
            sil = (
                float(silhouette_score(coords, lab))
                if len(present) >= 2 and counts.min() >= 2
                else np.nan
            )
            informativeness = np.nan
            coherence = np.nan
            if len(present) >= 2:
                col_dists = [
                    hard_distributions(col, lab, n_clusters) for col in X_raw.columns
                ]
                informativeness = float(
                    np.mean([mean_pairwise_tvd(d, present) for d in col_dists])
                )
                coherence = within_cluster_coherence(col_dists, lab)
            rows.append(
                {
                    "method": name,
                    "silhouette": round(sil, 3),
                    "bic": round(bics.get(name, np.nan), 0),
                    "cluster_size_entropy": round(size_entropy(lab), 3),
                    "mean_pairwise_tvd": round(informativeness, 4),
                    "within_cluster_coherence": round(coherence, 3),
                    "noise_frac": round(noise.get(name, 0.0), 3),
                }
            )

        df = pd.DataFrame(rows)
        # LaTeX-friendly headers: Styler.to_latex escapes cell values but not
        # column labels, so underscores (e.g. cluster_size_entropy) would be
        # swallowed by LaTeX. Rename to plain words instead.
        show = df.rename(
            columns={
                "method": "Method",
                "cluster_size_entropy": "Cluster size entropy",
                "mean_pairwise_tvd": "Mean pairwise TVD",
                "within_cluster_coherence": "Within-cluster coherence",
            }
        )[
            [
                "Method",
                "Cluster size entropy",
                "Mean pairwise TVD",
                "Within-cluster coherence",
            ]
        ]
        (
            highlight_best(show)
            .hide(axis="index")
            .to_latex(figures_dir / "method_comparison.tex", convert_css=True)
        )
        return df

    comparison_df = evaluate_methods(
        N_CLUSTERS,
        X_raw,
        category_codes,
        method_labels,
        method_bics,
        method_noise,
        mca_coords,
        output_dir,
        figures_dir,
    )
    comparison_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Robustness: Does the LCA Two-Cluster Solution Hold Up?

    The k=2 LCA solution drives every downstream artefact (subpopulation
    labels, per-cluster response distributions, fine-tuning targets). The
    checks below decide whether that structure is real, correctly sized and
    robust to the clustering choice:

    1. **Bootstrap stability** — are the same respondents grouped together when
       the model is refit on resamples of the population?
    2. **BLRT** — is k=3 genuinely better than k=2, or is the extra class just
       overfitting noise? (BIC alone under-picks in LCA.)
    3. **Distribution ablation** — the MCA + K-Means section above quantifies how
       much the per-cluster empirical response distributions (the actual `q`
       targets consumed by `build_dataset.py`) depend on the clustering choice,
       question by question, via total variation distance.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1. Bootstrap Stability

    Resample the 1,057 respondents with replacement and refit the k=2 LCA on
    each resample, then re-predict clusters for the full sample. The Adjusted
    Rand Index (ARI) between the refit partition and the original partition
    quantifies reproducibility: ~0.9+ means the subgroups are stable structure,
    ~0.3 means they are substantially sampling noise. Class labels are
    unidentifiable across fits, so ARI (label-invariant) is the right metric.

    Adjust `N_BOOTSTRAP` if you want more precision (each refit is a few
    seconds; 50 runs ≈ 5–15 min).
    """)
    return


@app.cell
def _(
    N_CLUSTERS,
    SEED,
    StepMix,
    X,
    assignments,
    figures_dir,
    np,
    output_dir,
    pd,
    plt,
):
    N_BOOTSTRAP = 50  # configurable — increase for tighter confidence bounds

    def bootstrap_stability(n_runs, seed, K, X, assignments):
        """Refit the k=K LCA on bootstrap resamples and return ARIs vs the original partition."""
        from sklearn.metrics import adjusted_rand_score

        rng = np.random.default_rng(seed)
        n = len(X)
        aris = []
        for i in range(n_runs):
            idx = rng.integers(0, n, size=n)
            X_boot = X.iloc[idx].reset_index(drop=True)
            m = StepMix(
                n_components=K,
                measurement="categorical_nan",
                n_init=3,
                random_state=int(rng.integers(0, 2**31)),
                max_iter=300,
                progress_bar=0,
            )
            m.fit(X_boot)
            lab = m.predict_proba(X).argmax(axis=1)
            aris.append(adjusted_rand_score(assignments, lab))
            if (i + 1) % 10 == 0:
                print(f"  bootstrap {i + 1}/{n_runs} (ARI={aris[-1]:.3f})")
        return np.array(aris)

    def save_bootstrap_stability(aris, K, output_dir, pd):
        """Persist the bootstrap ARI summary for downstream comparison."""
        lo, hi = np.percentile(aris, [2.5, 97.5])
        interpretation = (
            "stable structure"
            if aris.mean() >= 0.7
            else "partly unstable - treat subgroups cautiously"
        )
        pd.DataFrame(
            [
                {
                    "k": K,
                    "n_resamples": len(aris),
                    "mean_ari": aris.mean(),
                    "median_ari": np.median(aris),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "interpretation": interpretation,
                }
            ]
        ).round(3).to_csv(output_dir / "bootstrap_stability.csv", index=False)
        print(f"Saved bootstrap stability → {output_dir / 'bootstrap_stability.csv'}")

    def plot_bootstrap_stability(aris, K, figures_dir):
        """Histogram of bootstrap ARIs with 95% CI summary."""
        lo, hi = np.percentile(aris, [2.5, 97.5])
        print(f"\nBootstrap stability (k={K}, n={len(aris)} resamples):")
        print(f"  mean ARI = {aris.mean():.3f}   median = {np.median(aris):.3f}")
        print(f"  95% CI   = [{lo:.3f}, {hi:.3f}]")
        print(
            "  interpretation: "
            + (
                "stable structure"
                if aris.mean() >= 0.7
                else "partly unstable — treat subgroups cautiously"
            )
        )

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(aris, bins=15, color="#1f77b4", alpha=0.8)
        ax.axvline(aris.mean(), color="black", ls="--", label=f"mean={aris.mean():.3f}")
        ax.set_xlabel("Adjusted Rand Index vs original partition")
        ax.set_ylabel("Bootstrap runs")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            figures_dir / "bootstrap_stability.png", dpi=150, bbox_inches="tight"
        )
        return fig

    aris = bootstrap_stability(N_BOOTSTRAP, SEED, N_CLUSTERS, X, assignments)
    save_bootstrap_stability(aris, N_CLUSTERS, output_dir, pd)
    _fig = plot_bootstrap_stability(aris, N_CLUSTERS, figures_dir)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. Bootstrap Likelihood Ratio Test (BLRT)

    BIC prefers k=2, but BIC under-picks in LCA. The BLRT asks: *is the
    k=3 log-likelihood gain real?* The observed statistic is
    $LR = 2(\log L_{k+1} - \log L_k)$ on the full data. We then simulate
    datasets from the fitted k=2 model (`StepMix.sample`), refit **both** k=2
    and k=3 on each simulation, and count how often the null (k=2) can
    produce an LR that large by chance. A small p-value rejects k=2 in favour
    of k=3.

    Each bootstrap round refits two models, so this is the slowest cell
    (≈1–2 min per round; `N_BLRT_BOOTSTRAPS` is configurable).
    """)
    return


@app.cell
def _(
    N_CLUSTERS,
    SEED,
    StepMix,
    X,
    figures_dir,
    model,
    np,
    output_dir,
    pd,
    plt,
    sweep_df,
):
    N_BLRT_BOOTSTRAPS = 30  # configurable — each round refits k=K and k=K+1

    def blrt_test(K, SEED, n_bootstraps, X, model, np, pd, StepMix, sweep_df):
        """Bootstrap likelihood ratio test of k=K+1 vs k=K."""
        # sweep_df loglik is the TOTAL log-likelihood (score * n_resp was saved),
        # so the LR statistic needs no further n_resp scaling.
        ll = sweep_df.set_index("k")["loglik"]
        lr_obs = 2 * (ll[K + 1] - ll[K])

        rng = np.random.default_rng(SEED + 1)
        lr_boot = []
        for i in range(n_bootstraps):
            out = model.sample(len(X))
            Xb = pd.DataFrame(
                out[0] if isinstance(out, tuple) else out, columns=X.columns
            )
            scores = {}
            for kk in (K, K + 1):
                m = StepMix(
                    n_components=kk,
                    measurement="categorical_nan",
                    n_init=2,
                    random_state=int(rng.integers(0, 2**31)),
                    max_iter=250,
                    progress_bar=0,
                )
                m.fit(Xb)
                scores[kk] = m.score(Xb)
            lr_boot.append(2 * len(Xb) * (scores[K + 1] - scores[K]))
            if (i + 1) % 10 == 0:
                print(f"  BLRT round {i + 1}/{n_bootstraps}")
        return lr_obs, np.array(lr_boot)

    def save_blrt_outputs(lr_obs, lr_boot, p_value, K, output_dir, figures_dir):
        """Persist BLRT results as a CSV (for the report table) and a histogram
        figure showing the bootstrap LR distribution against the observed LR."""
        conclusion = (
            f"reject k={K}: k={K + 1} is a real improvement"
            if p_value <= 0.05
            else f"fail to reject k={K}: extra class is noise"
        )
        pd.DataFrame(
            [
                {
                    "k_null": K,
                    "k_alt": K + 1,
                    "lr_observed": lr_obs,
                    "lr_boot_mean": lr_boot.mean(),
                    "lr_boot_max": lr_boot.max(),
                    "n_bootstraps": len(lr_boot),
                    "p_value": p_value,
                    "conclusion": conclusion,
                }
            ]
        ).round(3).to_csv(output_dir / "blrt_results.csv", index=False)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(lr_boot, bins=15, color="#1f77b4", alpha=0.8, label="Bootstrap LRs")
        ax.axvline(
            lr_obs,
            color="red",
            ls="--",
            label=f"Observed LR = {lr_obs:,.0f} (p={p_value:.3f})",
        )
        ax.set_xlabel("Likelihood ratio statistic  $2(\\log L_{k+1} - \\log L_k)$")
        ax.set_ylabel("Bootstrap rounds")
        ax.set_title(f"BLRT: k={K} vs k={K + 1}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "blrt.png", dpi=150, bbox_inches="tight")
        return fig

    lr_obs, lr_boot = blrt_test(
        N_CLUSTERS, SEED, N_BLRT_BOOTSTRAPS, X, model, np, pd, StepMix, sweep_df
    )
    p_value = (1 + int((lr_boot >= lr_obs).sum())) / (1 + len(lr_boot))
    print(f"Observed LR (k={N_CLUSTERS + 1} vs k={N_CLUSTERS}): {lr_obs:,.0f}")
    print(f"Bootstrap LRs: mean={lr_boot.mean():,.0f}  max={lr_boot.max():,.0f}")
    print(
        f"p-value = {p_value:.3f}  -> "
        + (
            "reject k=2: k=3 is a real improvement"
            if p_value <= 0.05
            else "fail to reject k=2: extra class is noise"
        )
    )
    print(
        f"Saved BLRT results → {output_dir / 'blrt_results.csv'} and "
        f"{figures_dir / 'blrt.png'}"
    )
    _fig = save_blrt_outputs(
        lr_obs, lr_boot, p_value, N_CLUSTERS, output_dir, figures_dir
    )
    _fig
    return


if __name__ == "__main__":
    app.run()
