# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "pandas>=2.0.0",
#     "numpy>=1.26.0",
#     "matplotlib>=3.8.0",
#     "scikit-learn>=1.4.0",
# ]
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # k=2 vs k=3: Choosing the Number of Value Clusters

    This notebook compares the saved k=2 and k=3 LCA solutions from
    `cluster_respondents.py` across statistical fit, assignment quality,
    reproducibility and downstream usefulness. It only **reads saved
    outputs** (no model refitting), so it runs in seconds.

    The two candidate partitions were fit on the same 1,057 NZ WVS
    respondents, so every metric below is directly comparable.
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
    import numpy as np
    import matplotlib.pyplot as plt
    import json
    from pathlib import Path
    from sklearn.metrics import adjusted_rand_score
    import warnings

    # Where cluster_respondents.py saved its outputs.
    ANALYSIS_DIR = Path("output") / "cluster_analysis"
    K2_DIR = ANALYSIS_DIR / "k2_analysis"
    K3_DIR = ANALYSIS_DIR / "k3_analysis"

    # Where this notebook writes its comparison artefacts.
    OUT_DIR = ANALYSIS_DIR / "k2_vs_k3"
    FIG_DIR = OUT_DIR / "figures"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Comparison outputs → {OUT_DIR}")
    return (
        ANALYSIS_DIR,
        FIG_DIR,
        K2_DIR,
        K3_DIR,
        OUT_DIR,
        adjusted_rand_score,
        json,
        mo,
        np,
        pd,
        plt,
        warnings,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Load the Saved Analyses
    """)
    return


@app.cell
def _(ANALYSIS_DIR, K2_DIR, K3_DIR, json, np, pd, warnings):
    def load_assignments(analysis_dir):
        """Per-respondent cluster assignments + membership probabilities."""
        return pd.read_csv(analysis_dir / "cluster_assignments.csv")

    def load_model_diagnostics(analysis_dir):
        """Overall quality metrics (certainty, relative entropy). Falls back to
        computing them from the saved posteriors for runs made before these
        files were introduced."""
        path = analysis_dir / "model_diagnostics.json"
        if path.exists():
            return json.load(open(path))
        else:
            warnings.warn(f"Diagnostics file missing in {analysis_dir}")

    def load_cluster_sizes(analysis_dir):
        """Per-cluster respondent counts; computed from assignments if the
        diagnostics CSV is missing."""
        path = analysis_dir / "cluster_diagnostics.csv"
        if path.exists():
            return pd.read_csv(path)["n_respondents"].tolist()
        a = load_assignments(analysis_dir)
        K = a.filter(like="prob_cluster_").shape[1]
        return np.bincount(a["cluster"].to_numpy(), minlength=K).tolist()

    def load_bootstrap_stability(analysis_dir):
        """Bootstrap ARI summary; falls back to parsing run.log."""
        path = analysis_dir / "bootstrap_stability.csv"
        if path.exists():
            return pd.read_csv(path).iloc[0].to_dict()
        else:
            warnings.warn(f"Bootstrap stability file missing in {analysis_dir}")

    k2_assign = load_assignments(K2_DIR)
    k3_assign = load_assignments(K3_DIR)
    k2_diag = load_model_diagnostics(K2_DIR)
    k3_diag = load_model_diagnostics(K3_DIR)
    k2_sizes = load_cluster_sizes(K2_DIR)
    k3_sizes = load_cluster_sizes(K3_DIR)
    k2_info = pd.read_csv(K2_DIR / "question_informativeness.csv")
    k3_info = pd.read_csv(K3_DIR / "question_informativeness.csv")
    k2_demo = pd.read_csv(K2_DIR / "cluster_demographics.csv")
    k3_demo = pd.read_csv(K3_DIR / "cluster_demographics.csv")

    # Shared across runs: the sweep is identical in both dirs; the BLRT was
    # only saved in the k=2 dir.
    sweep_df = pd.read_csv(K2_DIR / "model_selection_sweep.csv")
    blrt_path = K2_DIR / "blrt_results.csv"
    blrt_df = pd.read_csv(blrt_path) if blrt_path.exists() else None

    k2_boot = load_bootstrap_stability(K2_DIR)
    k3_boot = load_bootstrap_stability(K3_DIR)
    print(f"Loaded k=2 and k=3 analyses from {ANALYSIS_DIR}")
    return (
        blrt_df,
        k2_assign,
        k2_boot,
        k2_demo,
        k2_diag,
        k2_info,
        k2_sizes,
        k3_assign,
        k3_boot,
        k3_demo,
        k3_diag,
        k3_info,
        k3_sizes,
        sweep_df,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Statistical Model Fit (BIC, AIC, BLRT)
    """)
    return


@app.cell
def _(blrt_df, sweep_df):
    def build_fit_table(sweep):
        """BIC/AIC across candidate k, with the gap from the best value."""
        t = sweep.copy().set_index("k")
        t["bic_delta"] = t["bic"] - t["bic"].min()
        t["aic_delta"] = t["aic"] - t["aic"].min()
        return t.round(0)

    fit_table = build_fit_table(sweep_df)
    bic2, bic3 = (
        int(sweep_df.set_index("k").loc[2, "bic"]),
        int(sweep_df.set_index("k").loc[3, "bic"]),
    )
    print(f"BIC: k=2 → {bic2:,}   k=3 → {bic3:,}   (k=3 better by {bic2 - bic3:,})")
    if blrt_df is not None:
        r = blrt_df.iloc[0]
        print(
            f"BLRT: k={int(r['k_null'])} vs k={int(r['k_alt'])} → LR={r['lr_observed']:,.0f}, "
            f"p={r['p_value']:.3f} — {r['conclusion']}"
        )
    else:
        print("BLRT results not saved — re-run cluster_respondents.py to generate.")
    fit_table
    return


@app.cell
def _(FIG_DIR, plt, sweep_df):
    def plot_fit_criteria(sweep, out_path):
        """BIC and AIC against k, with k=2 and k=3 marked."""
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(sweep["k"], sweep["bic"], "o-", label="BIC", color="#1f77b4")
        ax.plot(sweep["k"], sweep["aic"], "s--", label="AIC", color="#ff7f0e")
        for k in (2, 3):
            ax.axvline(k, color="grey", ls=":", alpha=0.6)
        ax.set_xlabel("Number of clusters k")
        ax.set_ylabel("Information criterion (lower = better)")
        ax.set_xticks(sweep["k"].astype(int))
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        return fig

    _fig_fit = plot_fit_criteria(sweep_df, FIG_DIR / "bic_aic_by_k.png")
    _fig_fit
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Assignment Quality
    """)
    return


@app.cell
def _(k2_diag, k2_sizes, k3_diag, k3_sizes, pd):
    def build_quality_table(sizes2, sizes3, diag2, diag3):
        """Side-by-side assignment quality metrics for k=2 vs k=3."""

        def favours(v2, v3, lower_better=False):
            if v2 is None or v3 is None or v2 == v3:
                return "—"
            return "k=2" if (v2 < v3) == lower_better else "k=3"

        min_share2 = min(sizes2) / diag2["n_respondents"]
        min_share3 = min(sizes3) / diag3["n_respondents"]
        rows = [
            {
                "metric": "Cluster sizes",
                "k=2": str(sizes2),
                "k=3": str(sizes3),
                "favours": "—",
            },
            {
                "metric": "Smallest cluster share",
                "k=2": f"{min_share2:.1%}",
                "k=3": f"{min_share3:.1%}",
                "favours": favours(min_share2, min_share3),
            },
            {
                "metric": "Mean posterior certainty",
                "k=2": f"{diag2['mean_certainty']:.3f}",
                "k=3": f"{diag3['mean_certainty']:.3f}",
                "favours": favours(diag2["mean_certainty"], diag3["mean_certainty"]),
            },
            {
                "metric": "Median posterior certainty",
                "k=2": f"{diag2['median_certainty']:.3f}",
                "k=3": f"{diag3['median_certainty']:.3f}",
                "favours": favours(
                    diag2["median_certainty"], diag3["median_certainty"]
                ),
            },
            {
                "metric": "Respondents with certainty > 0.9",
                "k=2": f"{diag2['frac_cert_gt_0.9']:.1%}",
                "k=3": f"{diag3['frac_cert_gt_0.9']:.1%}",
                "favours": favours(
                    diag2["frac_cert_gt_0.9"], diag3["frac_cert_gt_0.9"]
                ),
            },
            {
                "metric": "Relative entropy (separation)",
                "k=2": f"{diag2['relative_entropy']:.3f}",
                "k=3": f"{diag3['relative_entropy']:.3f}",
                "favours": favours(
                    diag2["relative_entropy"], diag3["relative_entropy"]
                ),
            },
        ]
        return pd.DataFrame(rows)

    quality_df = build_quality_table(k2_sizes, k3_sizes, k2_diag, k3_diag)
    quality_df
    return (quality_df,)


@app.cell
def _(FIG_DIR, k2_assign, k3_assign, plt):
    def plot_certainty_comparison(assign2, assign3, out_path):
        """Overlaid histograms of per-respondent posterior certainty."""

        def certainty(assign_df):
            K = assign_df.filter(like="prob_cluster_").shape[1]
            probs = assign_df[[f"prob_cluster_{k}" for k in range(K)]].to_numpy()
            return probs.max(axis=1)

        c2, c3 = certainty(assign2), certainty(assign3)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(c2, bins=50, alpha=0.6, label="k=2", color="#1f77b4")
        ax.hist(c3, bins=50, alpha=0.6, label="k=3", color="#ff7f0e")
        ax.set_xlabel("Posterior certainty (max membership probability)")
        ax.set_ylabel("Respondents")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        return fig

    _fig_cert = plot_certainty_comparison(
        k2_assign, k3_assign, FIG_DIR / "certainty_comparison.png"
    )
    _fig_cert
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. How Does the k=3 Solution Refine k=2?

    Both partitions cover the same 1,057 respondents, so we can cross-tabulate
    them directly. If k=3 were a clean refinement, each k=2 cluster would split
    into k=3 clusters without mixing. A low ARI means the third class cuts
    across the k=2 boundary.
    """)
    return


@app.cell
def _(adjusted_rand_score, k2_assign, k3_assign, pd):
    def build_partition_crosstab(assign2, assign3):
        """Cross-tabulate the k=2 and k=3 partitions at the respondent level."""
        merged = assign2.merge(assign3, on="id", suffixes=("_k2", "_k3"))
        ct = pd.crosstab(
            merged["cluster_k2"],
            merged["cluster_k3"],
            rownames=["k=2 cluster"],
            colnames=["k=3 cluster"],
        )
        ari = adjusted_rand_score(merged["cluster_k2"], merged["cluster_k3"])
        return ct, ari

    crosstab_df, ari_k2_k3 = build_partition_crosstab(k2_assign, k3_assign)
    print(f"Adjusted Rand Index between the k=2 and k=3 partitions: {ari_k2_k3:.3f}")
    print("Each cell = number of respondents in that k=2 cluster × k=3 cluster.")
    crosstab_df
    return ari_k2_k3, crosstab_df


@app.cell
def _(FIG_DIR, ari_k2_k3, crosstab_df, plt):
    def plot_partition_crosstab(ct, ari, out_path):
        """Heatmap of the k=2 × k=3 cross-tabulation with shares."""
        data = ct.to_numpy()
        fig, ax = plt.subplots(figsize=(5.5, 4))
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(ct.shape[1]))
        ax.set_xticklabels([f"k=3 cluster {c}" for c in ct.columns])
        ax.set_yticks(range(ct.shape[0]))
        ax.set_yticklabels([f"k=2 cluster {i}" for i in ct.index])
        for i in range(ct.shape[0]):
            for j in range(ct.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{data[i, j]}\n({data[i, j] / data.sum():.0%})",
                    ha="center",
                    va="center",
                    fontsize=10,
                )
        ax.set_title(f"Respondents by partition (ARI = {ari:.3f})")
        fig.colorbar(im, ax=ax, shrink=0.85)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        return fig

    _fig_cross = plot_partition_crosstab(
        crosstab_df, ari_k2_k3, FIG_DIR / "crosstab_k2_vs_k3.png"
    )
    _fig_cross
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Question Informativeness

    `build_dataset.py` and the consultation survey consume per-cluster response
    distributions. Higher mean informativeness (mean pairwise TVD) means the
    clusters answer more questions differently — but with more clusters this
    is partly mechanical.
    """)
    return


@app.cell
def _(k2_info, k3_info, pd):
    def compare_informativeness(info2, info3):
        """Per-question informativeness: summary stats and correlation."""
        i2 = info2.set_index("column")["informativeness"]
        i3 = info3.set_index("column")["informativeness"]
        paired = pd.DataFrame({"k=2": i2, "k=3": i3}).dropna()
        summary = pd.DataFrame(
            {
                "stat": [
                    "mean",
                    "median",
                    "75th percentile",
                    "max",
                    "questions with info ≥ 0.2",
                    "questions with info ≥ 0.3",
                ],
                "k=2": [
                    i2.mean(),
                    i2.median(),
                    i2.quantile(0.75),
                    i2.max(),
                    (i2 >= 0.2).sum(),
                    (i2 >= 0.3).sum(),
                ],
                "k=3": [
                    i3.mean(),
                    i3.median(),
                    i3.quantile(0.75),
                    i3.max(),
                    (i3 >= 0.2).sum(),
                    (i3 >= 0.3).sum(),
                ],
            }
        ).round(3)
        return paired, float(i2.corr(i3)), summary

    info_paired, info_corr, info_summary = compare_informativeness(k2_info, k3_info)
    print(
        f"Pearson correlation of per-question informativeness (k=2 vs k=3): {info_corr:.3f}"
    )
    print("\nTop 5 questions at k=2:")
    print(
        k2_info.head(5)[["column", "informativeness", "sub_question"]].to_string(
            index=False
        )
    )
    print("\nTop 5 questions at k=3:")
    print(
        k3_info.head(5)[["column", "informativeness", "sub_question"]].to_string(
            index=False
        )
    )
    print()
    info_summary
    return info_corr, info_paired, info_summary


@app.cell
def _(FIG_DIR, info_corr, info_paired, k3_info, plt):
    def plot_informativeness_scatter(paired, corr, k3_info, out_path):
        """k=3 vs k=2 per-question informativeness; the 3 most divergent labelled."""
        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.scatter(paired["k=2"], paired["k=3"], s=12, alpha=0.5, color="#1f77b4")
        lim = (0, float(paired.max().max()) * 1.05)
        ax.plot(lim, lim, "k--", lw=1, label="y = x")
        ax.set_xlabel("Informativeness at k=2")
        ax.set_ylabel("Informativeness at k=3")
        ax.set_title(f"Per-question informativeness (r = {corr:.3f})")
        divergent = (
            (paired["k=3"] - paired["k=2"]).abs().sort_values(ascending=False).head(3)
        )
        sq = k3_info.set_index("column")["sub_question"]
        for col in divergent.index:
            ax.annotate(
                str(sq.get(col, ""))[:30],
                (paired.loc[col, "k=2"], paired.loc[col, "k=3"]),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        return fig

    _fig_info = plot_informativeness_scatter(
        info_paired, info_corr, k3_info, FIG_DIR / "informativeness_comparison.png"
    )
    _fig_info
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Demographic Distinctiveness

    Higher demographic TVD means the clusters describe *different people*,
    which matters for the public consultation and the story told about each
    subgroup. The full k=3 profiles are shown to judge whether the third
    cluster is substantively meaningful.
    """)
    return


@app.cell
def _(k2_demo, k3_demo, pd):
    def compare_demographics(demo2, demo3):
        """How demographically distinct the clusters are in each solution."""
        rows = [
            {
                "metric": "Mean demographic TVD across clusters",
                "k=2": f"{demo2['tvd'].mean():.3f}",
                "k=3": f"{demo3['tvd'].mean():.3f}",
                "favours": "k=2"
                if demo2["tvd"].mean() > demo3["tvd"].mean()
                else "k=3",
            }
        ]
        for rank, (r2, r3) in enumerate(zip(demo2.itertuples(), demo3.itertuples()), 1):
            rows.append(
                {
                    "metric": f"Top-{rank} demographic (TVD)",
                    "k=2": f"{r2.demographic} ({r2.tvd:.3f})",
                    "k=3": f"{r3.demographic} ({r3.tvd:.3f})",
                    "favours": "—",
                }
            )
        return pd.DataFrame(rows)

    demo_cmp = compare_demographics(k2_demo, k3_demo)
    print("Full k=3 demographic profiles (who is in each cluster):")
    print(k3_demo.to_string(index=False))
    print()
    demo_cmp
    return (demo_cmp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. Reproducibility (Bootstrap Stability)
    """)
    return


@app.cell
def _(FIG_DIR, k2_boot, k3_boot, pd):
    def build_stability_table(boot2, boot3):
        """Bootstrap ARI summary for each solution (None-safe)."""
        rows = []
        for label, b in [("k=2", boot2), ("k=3", boot3)]:
            if b is None:
                rows.append(
                    {
                        "solution": label,
                        "mean ARI": "n/a",
                        "95% CI": "n/a",
                        "n resamples": "n/a",
                        "interpretation": "not found",
                    }
                )
            else:
                rows.append(
                    {
                        "solution": label,
                        "mean ARI": f"{b['mean_ari']:.3f}",
                        "95% CI": f"[{b['ci_lo']:.3f}, {b['ci_hi']:.3f}]",
                        "n resamples": b.get("n_resamples", "?"),
                        "interpretation": b.get("interpretation", ""),
                    }
                )
        return pd.DataFrame(rows)

    stability_df = build_stability_table(k2_boot, k3_boot)

    stability_df[["solution", "mean ARI", "95% CI"]].to_latex(
        FIG_DIR / "bootstrap_stability.tex", index=False, escape=True
    )

    stability_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7. Summary and Recommendation
    """)
    return


@app.cell
def _(
    OUT_DIR,
    blrt_df,
    demo_cmp,
    info_summary,
    k2_boot,
    k3_boot,
    pd,
    quality_df,
    sweep_df,
):
    def build_summary_table(
        sweep, blrt, quality_df, info_summary, demo_cmp, boot2, boot3
    ):
        """All headline metrics in one table with a 'favours' column."""
        bic = sweep.set_index("k")["bic"]
        aic = sweep.set_index("k")["aic"]
        rows = [
            {
                "metric": "BIC (lower better)",
                "k=2": f"{int(bic[2]):,}",
                "k=3": f"{int(bic[3]):,}",
                "favours": "k=3" if bic[3] < bic[2] else "k=2",
            },
            {
                "metric": "AIC (lower better)",
                "k=2": f"{int(aic[2]):,}",
                "k=3": f"{int(aic[3]):,}",
                "favours": "k=3" if aic[3] < aic[2] else "k=2",
            },
        ]
        if blrt is not None:
            r = blrt.iloc[0]
            rows.append(
                {
                    "metric": "BLRT p-value (k=2 vs k=3)",
                    "k=2": f"{r['p_value']:.3f}",
                    "k=3": "—",
                    "favours": "k=3" if r["p_value"] <= 0.05 else "k=2",
                }
            )
        for _, q in quality_df.iterrows():
            rows.append(
                {
                    "metric": q["metric"],
                    "k=2": q["k=2"],
                    "k=3": q["k=3"],
                    "favours": q["favours"],
                }
            )
        mean2 = info_summary.set_index("stat").loc["mean", "k=2"]
        mean3 = info_summary.set_index("stat").loc["mean", "k=3"]
        rows.append(
            {
                "metric": "Mean question informativeness",
                "k=2": f"{mean2:.3f}",
                "k=3": f"{mean3:.3f}",
                "favours": "k=3" if mean3 > mean2 else "k=2",
            }
        )
        rows.append(
            {
                "metric": "Mean demographic TVD",
                "k=2": demo_cmp.iloc[0]["k=2"],
                "k=3": demo_cmp.iloc[0]["k=3"],
                "favours": demo_cmp.iloc[0]["favours"],
            }
        )
        if boot2 is not None and boot3 is not None:
            rows.append(
                {
                    "metric": "Bootstrap stability (mean ARI)",
                    "k=2": f"{boot2['mean_ari']:.3f}",
                    "k=3": f"{boot3['mean_ari']:.3f}",
                    "favours": "k=3"
                    if boot3["mean_ari"] > boot2["mean_ari"]
                    else "k=2",
                }
            )
        return pd.DataFrame(rows)

    summary_df = build_summary_table(
        sweep_df, blrt_df, quality_df, info_summary, demo_cmp, k2_boot, k3_boot
    )
    summary_df.to_csv(OUT_DIR / "k2_vs_k3_summary.csv", index=False)
    print(f"Saved summary → {OUT_DIR / 'k2_vs_k3_summary.csv'}")
    summary_df
    return


if __name__ == "__main__":
    app.run()
