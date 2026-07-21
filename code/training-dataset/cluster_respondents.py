import marimo

__generated_with = "0.23.14"
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

    output_dir = Path("output")
    figures_dir = Path("../figures")

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    (figures_dir / "cluster_profiles").mkdir(parents=True, exist_ok=True)
    return (
        Path,
        StepMix,
        figures_dir,
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
    ## Configuration

    `N_CLUSTERS = None` means "use the BIC-optimal $K$ from the model-selection
    sweep".
    """)
    return


@app.cell
def _(mo):
    N_CLUSTERS = None if mo.cli_args().get("run_sweep") else 2  # None -> BIC-optimal
    K_RANGE = range(1, 7)  # candidate cluster counts for the sweep
    N_INIT_SWEEP = 2  # EM restarts per k in the sweep (keeps sweep fast)
    N_INIT_FINAL = 10  # EM restarts for the final model (best log-likelihood)
    SEED = 42
    USE_SAVED_MODEL = not mo.cli_args().get(
        "run_model"
    )  # if True, load model from disk instead of fitting
    return (
        K_RANGE,
        N_CLUSTERS,
        N_INIT_FINAL,
        N_INIT_SWEEP,
        SEED,
        USE_SAVED_MODEL,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Load and Prepare Data

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
def _(Path, json, np, pd):
    input_dir = Path("output")
    wvs_df = pd.read_csv(input_dir / "wvs_value_survey.csv")
    question_mapping = json.load(open(input_dir / "question_mapping.json"))

    value_cols = [
        c for c in wvs_df.columns if c.startswith("Q") and not c.startswith("resp_info")
    ]

    X_raw = wvs_df[value_cols].mask(wvs_df[value_cols] == -5.0, np.nan)

    # Recode each column to 0..K_j-1, remembering the original codes
    category_codes = {c: sorted(X_raw[c].dropna().unique()) for c in value_cols}
    X = pd.DataFrame(
        {
            c: X_raw[c].map({v: i for i, v in enumerate(category_codes[c])})
            for c in value_cols
        }
    )

    # column -> question text lookup for later display
    col2question = {}
    for _entry in question_mapping:
        _cols = _entry["column_names"]
        _sqs = _entry.get("sub_questions") or [None] * len(_cols)
        for _i, _col in enumerate(_cols):
            _sq = _sqs[_i] if _i < len(_sqs) else None
            _text = _entry["question"]
            col2question[_col] = f"{_text} — {_sq}" if _sq else _text

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
    ## Model Selection: Number of Clusters

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

    if (
        N_CLUSTERS is None
        or not Path(output_dir / "model_selection_sweep.csv").exists()
    ):
        print(f"Doing model selection sweep over k={list(K_RANGE)}")
        sweep_rows = []
        for _k in K_RANGE:
            _m = StepMix(
                n_components=_k,
                measurement="categorical_nan",
                n_init=N_INIT_SWEEP,
                random_state=SEED,
                max_iter=300,
                progress_bar=0,
            )
            _m.fit(X)
            sweep_rows.append(
                {
                    "k": _k,
                    "loglik": _m.score(X) * len(X),
                    "aic": _m.aic(X),
                    "bic": _m.bic(X),
                }
            )
            print(f"k={_k} done")

        sweep_df_temp = pd.DataFrame(sweep_rows).set_index("k").round(0)

        sweep_df_temp.to_csv(output_dir / "model_selection_sweep.csv")
        print(f"Saved sweep results to {output_dir / 'model_selection_sweep.csv'}")
    else:
        print(f"Skipping model selection using k={N_CLUSTERS} (already done)")
    return


@app.cell
def _(N_CLUSTERS, output_dir, pd, plt):
    sweep_df = pd.read_csv(output_dir / "model_selection_sweep.csv")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(sweep_df.index, sweep_df["bic"], "o-", label="BIC")
    ax.plot(sweep_df.index, sweep_df["aic"], "s--", label="AIC")
    ax.set_xlabel("Number of latent classes (k)")
    ax.set_ylabel("Information criterion (lower = better)")
    ax.legend()

    bic_best_k = int(sweep_df["bic"].idxmin())
    K = N_CLUSTERS if N_CLUSTERS is not None else bic_best_k
    ax.axvline(K, color="grey", ls=":", label=f"chosen k={K}")
    ax.legend()
    print(f"BIC-optimal k = {bic_best_k}; using k = {K}")
    fig
    return (K,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fit Final Model

    Refit at the chosen $K$ with more EM restarts and keep the best-likelihood
    solution. A reproducibility note: with ~260 indicators the likelihood surface
    is multimodal, so different seeds can yield different (near-equally likely)
    partitions at $K \geq 3$. The saved model below is internally consistent —
    all downstream artefacts (profiles, assignment of new participants) derive
    from this one fitted object.
    """)
    return


@app.cell
def _(K, N_INIT_FINAL, SEED, StepMix, USE_SAVED_MODEL, X, joblib, output_dir):
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
        train_model(X, K, N_INIT_FINAL, SEED)
    return


@app.cell
def _(K, X, joblib, output_dir):
    model = joblib.load(output_dir / f"stepmix_model_k{K}.joblib")

    posteriors = model.predict_proba(X)
    assignments = posteriors.argmax(axis=1)
    return assignments, model, posteriors


@app.cell
def _(K, assignments, np, pd, posteriors):
    n = len(assignments)
    rel_entropy = 1 - (-(posteriors * np.log(posteriors + 1e-12)).sum(axis=1).sum()) / (
        n * np.log(K)
    )
    diagnostics = pd.DataFrame(
        {
            "cluster": range(K),
            "n_respondents": np.bincount(assignments, minlength=K),
            "share": np.bincount(assignments, minlength=K) / n,
        }
    )
    print(f"Mean posterior certainty: {posteriors.max(axis=1).mean():.3f}")
    print(f"Relative entropy (1 = perfectly separated): {rel_entropy:.3f}")
    diagnostics
    return


@app.cell
def _(K, assignments, output_dir, pd, posteriors, wvs_df):
    assign_df = pd.DataFrame(
        {"id": wvs_df["id"], "cluster": assignments}
        | {f"prob_cluster_{_k}": posteriors[:, _k] for _k in range(K)}
    )

    assign_df.to_csv(output_dir / "cluster_assignments.csv", index=False)
    print(f"Saved clusters → {output_dir / 'cluster_assignments.csv'}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Cluster Response Distributions

    For each question and cluster, compute the **posterior-weighted empirical
    response distribution**: each respondent contributes to each cluster in
    proportion to their membership probability. This (not the model's smoothed
    emission parameters) is what `build_dataset.py` consumes, since it reflects
    the raw response patterns including each cluster's non-response rate.
    """)
    return


@app.cell
def _(K, X_raw, category_codes, np, posteriors, value_cols):
    def cluster_distributions(col):
        """Per-cluster distribution over the column's original codes."""
        obs = X_raw[col]
        codes = category_codes[col]
        dist = np.zeros((K, len(codes)))
        for _k in range(K):
            w = posteriors[:, _k]
            for j, v in enumerate(codes):
                dist[_k, j] = w[obs == v].sum()
            if dist[_k].sum() > 0:
                dist[_k] /= dist[_k].sum()
        # non-response rate per cluster (structural missing only: refused/not asked/not applicable)
        nonresponse = np.array(
            [
                posteriors[obs.isna(), _k].sum() / max(posteriors[:, _k].sum(), 1e-12)
                for _k in range(K)
            ]
        )
        return dist, nonresponse

    distributions = {c: cluster_distributions(c) for c in value_cols}
    f"Computed distributions for {len(distributions)} questions × {K} clusters"
    return (distributions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Question Selection for the Consultation Survey

    Rank questions by how differently the clusters answer them (mean pairwise
    total variation distance between cluster distributions). The public
    consultation only has time to ask each participant a handful of WVS
    questions — the top of this list is where those questions should come from.
    """)
    return


@app.cell
def _(K, col2question, distributions, np, pd):
    def informativeness(col):
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

    sorted_cols = sorted(distributions, key=informativeness, reverse=True)
    informativeness_df = pd.DataFrame(
        {
            "column": sorted_cols,
            "informativeness": [informativeness(c) for c in sorted_cols],
            "question": [col2question.get(c, "") for c in sorted_cols],
        }
    ).round(3)

    informativeness_df.head(15)
    return (informativeness_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The consultation will only ask each participant ~5–10 questions.
    This section evaluates which questions — **both value survey and demographic** —
    are most informative for cluster assignment, and how many are needed.
    Demographic questions are often available without asking (e.g. region from IP
    geolocation, sex from profile data), so they are "free" discriminators.
    """)
    return


@app.cell
def _(K, informativeness_df, np, posteriors, wvs_df):
    survey_question = dict(
        zip(informativeness_df["column"], informativeness_df["question"])
    )

    demo_cols = [
        c for c in wvs_df.columns if not c.startswith("Q") and c not in ("id", "date")
    ]

    demo_items = []
    for col in demo_cols:
        vals = wvs_df[col].astype(str)
        codes = sorted(vals.unique())
        dist = np.zeros((K, len(codes)))
        for _k in range(K):
            _w = posteriors[:, _k]
            for j, v in enumerate(codes):
                dist[_k, j] = _w[vals == v].sum()
            if dist[_k].sum() > 0:
                dist[_k] /= dist[_k].sum()
        tvd = float(
            np.mean(
                [
                    0.5 * np.abs(dist[_i] - dist[_j]).sum()
                    for _i in range(K)
                    for _j in range(_i + 1, K)
                ]
            )
        )
        demo_items.append((col, tvd, "demographic", col))

    all_items = [
        (
            c,
            float(informativeness_df.set_index("column").loc[c, "informativeness"]),
            "value",
            survey_question.get(c, c),
        )
        for c in informativeness_df["column"]
    ]
    all_items += demo_items
    all_items.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Rank':<5} {'Type':<13} {'Question':<40} {'Informativeness':<10}")
    print("-" * 68)
    for i, (col, info, typ, qtext) in enumerate(all_items[:25]):
        print(
            f"{i + 1:<5} {typ:<13} {(qtext[:38] if len(qtext) > 38 else qtext):<40} {info:<10.3f}"
        )
    return (all_items,)


@app.cell
def _(X, all_items, figures_dir, model, np, pd, plt):
    value_only = [c for c, _, t, _ in all_items if t == "value"]
    n_list = range(1, 16)

    rows = []
    for n_q in n_list:
        top_n = value_only[:n_q]
        X_partial = X.copy()
        X_partial[[c for c in X.columns if c not in top_n]] = np.nan
        partial_post = model.predict_proba(X_partial)
        cert = partial_post.max(axis=1)
        rows.append(
            {
                "n_questions": n_q,
                "avg_certainty": round(cert.mean(), 3),
                "med_certainty": round(float(np.median(cert)), 3),
                "n_>0.9": int((cert > 0.9).sum()),
            }
        )

    _results = pd.DataFrame(rows)
    print(
        "Assignment confidence for NEW respondents (using top-N value survey questions):"
    )
    print(_results.to_string(index=False))

    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.plot(
        _results["n_questions"], _results["avg_certainty"], "o-", label="avg certainty"
    )
    _ax.plot(
        _results["n_questions"],
        _results["med_certainty"],
        "s--",
        label="median certainty",
    )
    _ax.fill_between(
        _results["n_questions"],
        0,
        _results["n_>0.9"] / len(X),
        alpha=0.15,
        label="frac > 0.9",
    )
    _ax.axhline(0.9, color="grey", ls=":", alpha=0.5)
    _ax.set_xlabel("Number of value survey questions asked")
    _ax.set_ylabel("Posterior certainty")
    _ax.set_title("Assignment confidence for new respondents")
    _ax.legend(fontsize=9)
    _ax.set_xticks(list(n_list))
    _fig.savefig(
        figures_dir / "survey_questions_cluster_assignment_confidence.png",
        dpi=150,
        bbox_inches="tight",
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Example Cluster Profiles

    A peek at the top discriminating questions so the clusters are
    interpretable, not just statistical objects.
    """)
    return


@app.cell
def _(
    K,
    category_codes,
    col2question,
    distributions,
    figures_dir,
    informativeness_df,
    np,
    plt,
    question_mapping,
):
    import textwrap

    _word_labels = {}
    for _entry in question_mapping:
        _codes = [float(x) for x in _entry["numeric_response_types"]]
        for _col in _entry["column_names"]:
            _labels = {}
            for _w, _c in zip(_entry["word_response_types"], _codes):
                if _c >= 0:
                    _labels[_c] = _w
            _word_labels[_col] = _labels

    _top_cols = informativeness_df["column"].head(5)

    for _col in _top_cols:
        _dist, _ = distributions[_col]
        _lbl = _word_labels.get(_col, {})
        _labels = [_lbl.get(v, str(v)) for v in category_codes[_col]]
        _title = textwrap.fill(col2question.get(_col, ""), width=60)

        _fig, _ax = plt.subplots(figsize=(9, 3.5))

        x = np.arange(len(_labels))
        w = 0.7 / K
        for _k in range(K):
            _ax.bar(
                x + _k * w - 0.35 + w / 2,
                _dist[_k],
                w,
                label=f"cluster {_k}",
                alpha=0.85,
            )

        _ax.set_xticks(x)
        _ax.set_xticklabels(_labels, fontsize=9)
        _ax.set_ylabel("Probability", fontsize=10)
        _ax.set_title(f"Q: {_title}", fontsize=10, fontweight="bold")
        _ax.legend(fontsize=9)
        _ax.set_ylim(0, 1)
        _fig.tight_layout()
        _fig.savefig(
            figures_dir / "cluster_profiles" / f"question_distribution_{_col}.png",
            dpi=150,
            bbox_inches="tight",
        )
        _fig
    return


if __name__ == "__main__":
    app.run()
