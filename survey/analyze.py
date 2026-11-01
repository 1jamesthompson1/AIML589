import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import re
    import sys
    from math import erfc, sqrt
    from pathlib import Path
    from statistics import NormalDist

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.linalg import cho_factor, cho_solve
    from scipy.optimize import minimize
    from scipy.stats import chi2, trim_mean

    DIR = Path(__file__).resolve().parent
    REPO = DIR.parent
    sys.path.insert(0, str(REPO / "code"))
    from latex_tables import write_latex_table

    RESPONSES = DIR / "output" / "responses.csv"
    QUESTION_MAP = DIR / "output" / "responses-question-ids.csv"
    COMPARISONS = REPO / "code" / "dilemmas" / "output" / "comparisons"
    POLICY_JSON = DIR / "policy_questions.json"
    FIGURES = REPO / "code" / "figures"
    return (
        COMPARISONS,
        FIGURES,
        NormalDist,
        POLICY_JSON,
        QUESTION_MAP,
        RESPONSES,
        chi2,
        cho_factor,
        cho_solve,
        erfc,
        json,
        minimize,
        mo,
        np,
        pd,
        plt,
        re,
        sqrt,
        trim_mean,
        write_latex_table,
    )


@app.cell
def _(COMPARISONS, QUESTION_MAP, RESPONSES, json, np, pd, re):
    PREF_SCORES = {
        "Strongly prefer Agent 1": -2,
        "Somewhat prefer Agent 1": -1,
        "About the same": 0,
        "Neither agree nor disagree": 0,
        "Somewhat prefer Agent 2": 1,
        "Strongly prefer Agent 2": 2,
    }
    AIUSE_SCORES = {
        "Strongly disagree": 1,
        "Disagree": 2,
        "Neither agree nor disagree": 3,
        "Agree": 4,
        "Strongly agree": 5,
    }
    ATTENTION = {
        "A_attention": "Somewhat disagree",
        "B_attention": "Neither agree nor disagree",
        "C_attention": "Strongly agree",
    }
    BOOTSTRAP_REPS = 10_000
    H2_SEED = 20261015

    def column_name(column, qmap):
        """Restore full DataExportTags from Qualtrics' JSON ImportIds."""
        try:
            meta = json.loads(column)
        except (json.JSONDecodeError, TypeError):
            meta = {}
        qid = meta.get("ImportId", column)
        tag = qmap.get(qid, qid)
        if "choiceId" in meta:
            return f"{tag}_{meta['choiceId']}"
        match = re.fullmatch(r"(QID\d+)_([0-9]+)", qid)
        if match and match.group(1) in qmap:
            return f"{qmap[match.group(1)]}_{match.group(2)}"
        match = re.fullmatch(r"(QID\d+)_([A-Z_]+)", qid)
        if match and match.group(1) in qmap:
            return f"{qmap[match.group(1)]}_{match.group(2)}"
        return tag

    def load_responses():
        qmap = pd.read_csv(QUESTION_MAP).set_index("question_id")["data_export_tag"]
        qmap = qmap.to_dict()
        data = pd.read_csv(RESPONSES, header=2, dtype=str)
        data.columns = [column_name(c, qmap) for c in data.columns]
        return data

    def load_manifest(comparison_ids):
        """The survey subset covering the most response comparisons."""
        best = None
        best_overlap = 0
        for path in sorted(COMPARISONS.glob("*/survey/*/comparisons.json")):
            manifest = json.loads(path.read_text())
            ids = {c["comparison_id"] for c in manifest["comparisons"]}
            overlap = len(comparison_ids & ids)
            if overlap > best_overlap:
                best, best_overlap = manifest, overlap
        if best is None:
            raise SystemExit("Could not match the responses to a survey subset.")
        return best

    def age_band(age):
        if pd.isna(age):
            return None
        age = float(age)
        if age < 25:
            return "18-24"
        if age >= 65:
            return "65 and older"
        low = int(age // 5) * 5
        return f"{low}-{low + 4}"

    PANEL_REGIONS = {
        "21167544": "West Coast",
        "21167545": "Waikato",
        "21167546": "Auckland",
        "21167547": "Canterbury",
        "21167548": "Northland",
        "21167549": "Southland",
        "21167550": "Tasman",
        "21167551": "Chatham Islands",
        "21167552": "Nelson",
        "21167553": "Manawatu-Wanganui",
        "21167554": "Wellington",
        "21167555": "Gisborne",
        "21167556": "Otago",
        "21167557": "Taranaki",
        "21167558": "Bay of Plenty",
        "21167559": "Marlborough",
        "21167560": "Hawke's Bay",
        "25064601": "Niue",
    }

    def add_demographics(data):
        """Panel codes to analysis labels, with text answers as fallback."""
        out = data.copy()
        ages = (
            pd.to_numeric(out["age"], errors="coerce")
            if "age" in out
            else pd.Series(np.nan, index=out.index)
        )
        text_age = out["Z1-1"] if "Z1-1" in out else pd.Series(None, index=out.index)
        out["age_band"] = [
            age_band(a) if pd.notna(a) else t for a, t in zip(ages, text_age)
        ]
        gender = out.get("gender", pd.Series(None, index=out.index))
        text_gender = out["Z2-1"] if "Z2-1" in out else pd.Series(None, index=out.index)
        out["gender_label"] = [
            {"1": "Male", "2": "Female"}.get(str(g), t)
            for g, t in zip(gender, text_gender)
        ]
        region = out.get("region", pd.Series(None, index=out.index))
        text_region = out["Z5-1"] if "Z5-1" in out else pd.Series(None, index=out.index)
        out["region_label"] = [
            PANEL_REGIONS.get(str(r), t) for r, t in zip(region, text_region)
        ]
        return out

    return (
        AIUSE_SCORES,
        ATTENTION,
        BOOTSTRAP_REPS,
        H2_SEED,
        PREF_SCORES,
        add_demographics,
        load_manifest,
        load_responses,
    )


@app.cell
def _(ATTENTION, np, pd):
    CUTOFF = pd.Timestamp("2026-10-29 14:30", tz="Pacific/Auckland")
    MIN_DURATION_SECONDS = 240  # 4 minutes, matching the live screenout
    MAX_DURATION_SECONDS = 2 * 60 * 60  # 2 hours: tab-left-open cap

    def exclusion_masks(data):
        """Requested exclusions: dev tests, incompletes, attention, speed
        (both tails) and duplicate panelists.

        ``Q_TerminateFlag == "PoorQuality"`` is deliberately *not* an
        exclusion; it is kept as a flag for manual review. Straightlining is
        likewise diagnostic only.
        """
        finished = data["finished"].eq("True")
        progress = pd.to_numeric(data["progress"], errors="coerce").ge(100)
        status = data["status"].fillna("").str.lower()
        test = status.str.contains("test|preview", regex=True)
        started = pd.to_datetime(data["startDate"], errors="coerce", utc=True)
        started = started.dt.tz_convert("Pacific/Auckland")
        after_cutoff = started >= CUTOFF
        dev_test = ~after_cutoff | test
        incomplete = ~(finished & progress)

        duration = pd.to_numeric(data["duration"], errors="coerce")
        too_fast = duration < MIN_DURATION_SECONDS
        too_long = duration > MAX_DURATION_SECONDS

        checks = {tag: data[tag].eq(value) for tag, value in ATTENTION.items()}
        correct = sum(checks.values())
        attention = correct >= 2

        likert_cols = [c for c in data if c.endswith(("_pref", "_aiuse"))]
        likert_cols += [c for c in ("C8-1", "C1-1", "Q943") if c in data]
        likert = data[likert_cols]
        items = likert.notna().sum(axis=1)
        max_same = likert.apply(
            lambda row: (
                row.value_counts(normalize=True).max() if row.notna().any() else np.nan
            ),
            axis=1,
        )
        straight = (items >= 5) & (max_same > 0.90)

        # Duplicates: psid is the panel's stable respondent id; ResponseId and
        # _recordId are unique per submission. Keep the earliest valid row.
        psid = (
            data["psid"].fillna("").astype(str)
            if "psid" in data
            else pd.Series("", index=data.index)
        )
        psid = psid.replace({"nan": "", "None": ""})
        eligible = ~dev_test & ~incomplete & attention & ~too_fast & ~too_long
        duplicate = pd.Series(False, index=data.index)
        seen_psid, seen_rid = set(), set()
        for idx in started.sort_values(na_position="last").index:
            if not eligible.loc[idx]:
                continue
            rid = data.at[idx, "_recordId"]
            pid = psid.at[idx]
            if rid in seen_rid or (pid and pid in seen_psid):
                duplicate.at[idx] = True
                continue
            seen_rid.add(rid)
            if pid:
                seen_psid.add(pid)

        base = eligible & ~duplicate
        three = data[[f"C4-3_{i}" for i in range(1, 10)]].notna().sum(axis=1).eq(3)
        policy = base & three

        flag = data.get("Q_TerminateFlag", pd.Series("", index=data.index))
        poor_quality = pd.Series(flag, index=data.index).eq("PoorQuality")

        diagnostics = pd.DataFrame(
            {
                "_recordId": data["_recordId"],
                "duration": duration,
                "started": started,
                "dev_test": dev_test,
                "incomplete": incomplete,
                "completed": ~dev_test & ~incomplete,
                "attention_correct": correct,
                "attention_pass": attention,
                "too_fast": too_fast,
                "too_long": too_long,
                "speed_pass": ~too_fast & ~too_long,
                "likert_items": items,
                "max_same_share": max_same,
                "straightline": straight,
                "duplicate": duplicate,
                "poor_quality": poor_quality,
                "base": base,
                "three_policies": three,
                "policy": policy,
            }
        )
        for tag, passed in checks.items():
            diagnostics[f"{tag}_pass"] = passed

        diagnostics["reason"] = ""
        diagnostics.loc[dev_test, "reason"] = "Development test"
        diagnostics.loc[~dev_test & incomplete, "reason"] = "Incomplete"
        diagnostics.loc[~dev_test & ~incomplete & ~attention, "reason"] = (
            "Attention checks"
        )
        diagnostics.loc[~dev_test & ~incomplete & attention & too_fast, "reason"] = (
            "Too fast"
        )
        diagnostics.loc[
            ~dev_test & ~incomplete & attention & ~too_fast & too_long, "reason"
        ] = "Too long"
        diagnostics.loc[
            ~dev_test & ~incomplete & attention & ~too_fast & ~too_long & duplicate,
            "reason",
        ] = "Duplicate"
        # PoorQuality and straightlining are flagged, not excluded.

        steps = []
        before = pd.Series(True, index=data.index)

        def add_step(label, after):
            nonlocal before
            removed = before & ~after
            steps.append(
                {
                    "Step": label,
                    "Excluded": int(removed.sum()),
                    "Remaining": int(after.sum()),
                    "Removed median duration (s)": (
                        f"{duration[removed].median():.1f}" if removed.any() else "--"
                    ),
                    "Remaining median duration (s)": f"{duration[after].median():.1f}",
                }
            )
            before = after

        add_step("All downloaded responses", before)
        add_step(
            "Development tests (before cutoff / test / preview)",
            ~dev_test,
        )
        add_step("Incomplete", ~dev_test & ~incomplete)
        add_step("Attention checks", ~dev_test & ~incomplete & attention)
        add_step(
            "Too fast (under 4 min)",
            ~dev_test & ~incomplete & attention & ~too_fast,
        )
        add_step(
            "Too long (over 2 h)",
            ~dev_test & ~incomplete & attention & ~too_fast & ~too_long,
        )
        add_step("Duplicate", base)
        add_step("Policy sample: not exactly three policies", policy)
        steps = pd.DataFrame(steps)
        return base, policy, steps, diagnostics

    return (exclusion_masks,)


@app.cell
def _(AIUSE_SCORES, PREF_SCORES, load_manifest, np, pd):
    def preference_data(data):
        """One row per preference rating, with AI-use support and page time."""
        prefs = data.melt(
            id_vars=["_recordId"],
            value_vars=[c for c in data if c.endswith("_pref")],
            var_name="tag",
            value_name="preference",
        ).dropna(subset=["preference"])
        prefs = prefs[prefs["preference"].str.strip() != ""].copy()
        prefs["score"] = prefs["preference"].map(PREF_SCORES)
        prefs["comparison_id"] = prefs["tag"].str.removesuffix("_pref")
        manifest = load_manifest(set(prefs["comparison_id"]))
        meta = {
            c["comparison_id"]: (
                c["dilemma_id"],
                c.get("situation_id") or c["dilemma_id"],
            )
            for c in manifest["comparisons"]
        }
        prefs["dilemma"] = prefs["comparison_id"].map(lambda c: meta[c][0])
        prefs["situation"] = prefs["comparison_id"].map(lambda c: meta[c][1])
        prefs = prefs.dropna(subset=["dilemma"]).copy()
        fields = {
            d: f"ord_{i}"
            for i, d in enumerate(
                sorted({c["dilemma_id"] for c in manifest["comparisons"]}), 1
            )
        }

        ai = data.melt(
            id_vars=["_recordId"],
            value_vars=[c for c in data if c.endswith("_aiuse")],
            var_name="atag",
            value_name="aiuse",
        ).dropna(subset=["aiuse"])
        ai["comparison_id"] = ai["atag"].str.removesuffix("_aiuse")
        ai["aiuse_score"] = ai["aiuse"].map(AIUSE_SCORES)
        prefs = prefs.merge(
            ai[["_recordId", "comparison_id", "aiuse_score"]],
            on=["_recordId", "comparison_id"],
            how="left",
        )

        by_id = data.set_index("_recordId")
        prefs["seconds"] = [
            pd.to_numeric(by_id.at[rid, f"{cid}_timing_PAGE_SUBMIT"], errors="coerce")
            if f"{cid}_timing_PAGE_SUBMIT" in by_id.columns
            else np.nan
            for rid, cid in zip(prefs["_recordId"], prefs["comparison_id"])
        ]
        prefs["order"] = [
            by_id.at[rid, fields[d]]
            for rid, d in zip(prefs["_recordId"], prefs["dilemma"])
        ]
        ft_first = prefs["order"].eq("ft")
        prefs["ft_score"] = prefs["score"] * np.where(ft_first, 1, -1)
        prefs["win"] = np.where(
            prefs["ft_score"] > 0, 1.0, np.where(prefs["ft_score"] < 0, 0.0, 0.5)
        )
        prefs["order_effect"] = np.where(ft_first, 0.5, -0.5)
        return prefs

    return (preference_data,)


@app.cell
def _(add_demographics, exclusion_masks, load_responses, preference_data):
    responses = add_demographics(load_responses())
    base_keep, policy_keep, exclusion_steps, exclusion_diagnostics = exclusion_masks(
        responses
    )
    h2_data = preference_data(responses[base_keep])
    policy_sample = responses[policy_keep].copy()
    return (
        base_keep,
        exclusion_diagnostics,
        exclusion_steps,
        h2_data,
        policy_sample,
        responses,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Wrangling

    ### Exclusions

    Each step is applied in order. The table shows how many responses it
    removes, the median duration of those removed, and the median duration
    of those retained.
    """)
    return


@app.cell
def _(exclusion_steps, write_latex_table):
    exclusion_table = exclusion_steps.set_index("Step")
    exclusion_table.index.name = None
    write_latex_table(
        exclusion_table,
        "{}",
        4,
        "survey-exclusions.tex",
        stub_label="",
        highlight=False,
    )
    exclusion_steps
    return


@app.cell
def _(mo):
    mo.md("""
    ### Attention checks
    """)
    return


@app.cell
def _(exclusion_diagnostics, pd):
    def attention_table(diag):
        rows = []
        labels = {
            "A_attention": "A (Section A)",
            "B_attention": "B (Section B)",
            "C_attention": "C (Section C)",
        }
        for tag, label in labels.items():
            passed = diag[f"{tag}_pass"]
            rows.append(
                {
                    "Check": label,
                    "Failed (all)": int((~passed).sum()),
                    "Failed after earlier steps": int(
                        (diag["completed"] & ~passed).sum()
                    ),
                    "Median duration failed (s)": round(
                        diag.loc[~passed, "duration"].median(), 1
                    ),
                    "Passed": int(passed.sum()),
                    "Median duration passed (s)": round(
                        diag.loc[passed, "duration"].median(), 1
                    ),
                }
            )
        failed_two = diag["attention_correct"] < 2
        rows.append(
            {
                "Check": "Any 2+ failures",
                "Failed (all)": int(failed_two.sum()),
                "Failed after earlier steps": int(
                    (diag["completed"] & failed_two).sum()
                ),
                "Median duration failed (s)": round(
                    diag.loc[failed_two, "duration"].median(), 1
                ),
                "Passed": int((~failed_two).sum()),
                "Median duration passed (s)": round(
                    diag.loc[~failed_two, "duration"].median(), 1
                ),
            }
        )
        return pd.DataFrame(rows)

    attention_table(exclusion_diagnostics)
    return


@app.cell
def _(exclusion_steps, mo):
    attention_removed = int(
        exclusion_steps.loc[
            exclusion_steps["Step"] == "Attention checks", "Excluded"
        ].iloc[0]
    )
    note = (
        "No respondent fails two or more of the three checks after earlier "
        "exclusions, so the attention step itself removes none."
        if attention_removed == 0
        else f"The attention step removes {attention_removed} responses."
    )
    mo.md(note)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Straightlining

    This is calculated per respondent across all non-policy Likert items:
    the five preference ratings, five AI-use ratings, AI usage frequency
    (C8-1), AI outlook (C1-1), and perceived risk (Q943). For each
    respondent we take the largest share of non-missing answers that use
    the same label. A respondent is excluded only if they answered at
    least five items and that share is greater than 90%. This is an
    overall per-respondent check, not a pairwise comparison between items.
    Because the item scales differ, only identical labels count as the
    same response, so this is a conservative check.
    """)
    return


@app.cell
def _(exclusion_diagnostics, mo, pd):
    def straightline_diagnostics(diag):
        metrics = pd.DataFrame(
            [
                {
                    "Metric": "Respondents with >= 5 non-policy Likert items",
                    "Value": int((diag["likert_items"] >= 5).sum()),
                },
                {
                    "Metric": "Median non-policy Likert items answered",
                    "Value": f"{diag['likert_items'].median():.0f}",
                },
                {
                    "Metric": "Median largest same-answer share",
                    "Value": f"{diag['max_same_share'].median():.2f}",
                },
                {
                    "Metric": "Respondents with > 90% same answer",
                    "Value": int(diag["straightline"].sum()),
                },
            ]
        )
        bins = pd.cut(
            diag["max_same_share"],
            [0, 0.60, 0.70, 0.80, 0.90, 1.001],
            right=False,
            labels=["<60%", "60-70%", "70-80%", "80-90%", ">90%"],
        )
        counts = (
            bins.value_counts()
            .sort_index()
            .rename_axis("Largest same-answer share")
            .reset_index(name="Respondents")
        )
        return metrics, counts

    metrics, counts = straightline_diagnostics(exclusion_diagnostics)
    mo.vstack(
        [
            mo.ui.table(metrics, selection=None),
            mo.ui.table(counts, selection=None),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### Excluded responses
    """)
    return


@app.cell
def _(exclusion_diagnostics, mo):
    excluded = exclusion_diagnostics[~exclusion_diagnostics["policy"]].copy()
    excluded["duration (min)"] = (excluded["duration"] / 60).round(1)
    excluded["max same share"] = excluded["max_same_share"].round(2)
    excluded = excluded[
        [
            "_recordId",
            "reason",
            "duration (min)",
            "attention_correct",
            "max same share",
            "likert_items",
        ]
    ].rename(columns={"_recordId": "Response ID"})
    mo.accordion({"Excluded responses": mo.ui.table(excluded, selection=None)})
    return


@app.cell
def _(mo):
    mo.md("""
    ## H2 analysis

    Public preference between the fine-tuned model and the base model.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Linear mixed model
    """)
    return


@app.cell
def _(
    NormalDist,
    cho_factor,
    cho_solve,
    erfc,
    h2_data,
    minimize,
    np,
    pd,
    sqrt,
):
    def fit_lmm(data):
        """REML fit of ft_score ~ order + (1|respondent) + (1|dilemma)."""
        y = data["ft_score"].to_numpy(float)
        X = np.column_stack([np.ones(len(data)), data["order_effect"].to_numpy(float)])
        respondent, _ = pd.factorize(data["_recordId"])
        dilemma, _ = pd.factorize(data["dilemma"])
        n_resp, n_dil = respondent.max() + 1, dilemma.max() + 1
        Z = np.zeros((len(data), n_resp + n_dil))
        Z[np.arange(len(data)), respondent] = 1.0
        Z[np.arange(len(data)), n_resp + dilemma] = 1.0

        XtX, XtZ, ZtZ = X.T @ X, X.T @ Z, Z.T @ Z
        Xty, Zty, yty = X.T @ y, Z.T @ y, y @ y
        n, p = X.shape
        q = Z.shape[1]

        def terms(theta):
            var = np.exp(theta)
            scale = np.sqrt(np.r_[np.full(n_resp, var[0]), np.full(n_dil, var[1])])
            A = np.eye(q) + (scale[:, None] * ZtZ) * scale[None, :]
            chol = cho_factor(A, lower=True)
            B = scale[:, None] * XtZ.T
            C = scale * Zty
            AinvB, AinvC = cho_solve(chol, B), cho_solve(chol, C)
            XtVX = XtX - B.T @ AinvB
            XtVy = Xty - B.T @ AinvC
            ytVy = yty - C @ AinvC
            return 2 * np.log(np.diag(chol[0])).sum(), XtVX, XtVy, ytVy

        def neg_reml(theta):
            logdet, XtVX, XtVy, ytVy = terms(theta)
            beta = np.linalg.solve(XtVX, XtVy)
            sigma2 = (ytVy - beta @ XtVy) / (n - p)
            return 0.5 * (
                logdet + np.linalg.slogdet(XtVX)[1] + (n - p) * np.log(sigma2)
            )

        fit = minimize(
            neg_reml,
            np.log([0.5, 0.1]),
            method="Nelder-Mead",
            options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 1000},
        )
        _, XtVX, XtVy, ytVy = terms(fit.x)
        beta = np.linalg.solve(XtVX, XtVy)
        sigma2 = (ytVy - beta @ XtVy) / (n - p)
        se = np.sqrt(np.diag(sigma2 * np.linalg.inv(XtVX)))
        z = NormalDist().inv_cdf(0.975)
        estimate, std_error = float(beta[0]), float(se[0])
        return {
            "estimate": estimate,
            "ci": (estimate - z * std_error, estimate + z * std_error),
            "p": erfc(abs(estimate / std_error) / sqrt(2)),
            "n": int(n),
            "respondents": int(data["_recordId"].nunique()),
        }

    h2_lmm = fit_lmm(h2_data)
    return (h2_lmm,)


@app.cell
def _(mo):
    mo.md("""
    ### Bradley--Terry
    """)
    return


@app.cell
def _(BOOTSTRAP_REPS, H2_SEED, h2_data, np):
    def fit_bt(data):
        """Win score with a respondent-clustered bootstrap interval."""
        grouped = data.groupby("_recordId")["win"]
        sums = grouped.sum().to_numpy(float)
        counts = grouped.size().to_numpy(float)
        estimate = sums.sum() / counts.sum()
        rng = np.random.default_rng(H2_SEED)
        draw = rng.integers(0, len(sums), size=(BOOTSTRAP_REPS, len(sums)))
        boot = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
        low, high = np.quantile(boot, [0.025, 0.975])
        p = min(1.0, 2 * min((boot <= 0.5).mean(), (boot >= 0.5).mean()))
        return {
            "estimate": float(estimate),
            "ci": (float(low), float(high)),
            "p": float(p),
            "n": len(data),
            "respondents": len(sums),
        }

    h2_bt = fit_bt(h2_data)
    return (h2_bt,)


@app.cell
def _(h2_bt, h2_lmm, pd, write_latex_table):
    def fmt_p(p):
        return "<0.001" if p < 0.001 else f"{p:.3f}"

    def tex_p(p):
        return "$<0.001$" if p < 0.001 else f"{p:.3f}"

    h2_table = pd.DataFrame(
        {
            "Estimate": [
                f"{h2_lmm['estimate']:.3f}",
                f"{h2_bt['estimate']:.3f}",
            ],
            "95% CI": [
                f"[{h2_lmm['ci'][0]:.3f}, {h2_lmm['ci'][1]:.3f}]",
                f"[{h2_bt['ci'][0]:.3f}, {h2_bt['ci'][1]:.3f}]",
            ],
            "p": [tex_p(h2_lmm["p"]), tex_p(h2_bt["p"])],
            "N": [f"{h2_lmm['n']:,}", f"{h2_bt['n']:,}"],
        },
        index=["LMM (points)", "Bradley--Terry (win rate)"],
    )
    write_latex_table(h2_table, "{}", 4, "survey-h2-results.tex", stub_label="")
    h2_table
    return


@app.cell
def _(h2_bt, h2_lmm, mo):
    h2_verdict = (
        "supported" if h2_lmm["ci"][0] > 0 and h2_bt["ci"][0] > 0.5 else "not supported"
    )
    mo.md(
        f"""
        **LMM:** {h2_lmm["estimate"]:+.3f} points, 95% CI
        [{h2_lmm["ci"][0]:+.3f}, {h2_lmm["ci"][1]:+.3f}],
        p = {h2_lmm["p"]:.3f}.

        **Bradley--Terry:** {h2_bt["estimate"]:.3f} win rate, 95% CI
        [{h2_bt["ci"][0]:.3f}, {h2_bt["ci"][1]:.3f}],
        p = {h2_bt["p"]:.3f}.

        H2 is **{h2_verdict}** under the pre-registered decision rule.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### Per-dilemma preference
    """)
    return


@app.cell
def _(FIGURES, dilemma_stats, h2_data, np, plt):
    def plot_dilemma_preference(data):
        stats = dilemma_stats(data).sort_values("preference")
        overall = data["ft_score"].mean()
        y = np.arange(len(stats))
        colours = [
            "tab:green" if value >= 0 else "tab:red" for value in stats["preference"]
        ]
        fig, ax = plt.subplots(figsize=(9, 6.5))
        ax.barh(y, stats["preference"], color=colours, alpha=0.85)
        ax.errorbar(
            stats["preference"],
            y,
            xerr=[
                stats["preference"] - stats["pref_low"],
                stats["pref_high"] - stats["preference"],
            ],
            fmt="none",
            ecolor="0.25",
            capsize=3,
            lw=1,
        )
        for yi, value in zip(y, stats["preference"]):
            offset = 0.03 if value >= 0 else -0.03
            ax.text(
                value + offset,
                yi,
                f"{value:+.2f}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=8,
            )
        ax.axvline(0, color="0.3", lw=0.8)
        ax.axvline(
            overall,
            color="black",
            ls="--",
            lw=1,
            label=f"Overall mean {overall:+.2f}",
        )
        low = min(stats["pref_low"].min(), 0)
        high = max(stats["pref_high"].max(), 0)
        pad = max((high - low) * 0.25, 0.1)
        ax.set_xlim(low - pad, high + pad)
        ax.set_yticks(y)
        ax.set_yticklabels(stats["dilemma"])
        ax.set_xlabel("Mean preference (points; positive = fine-tune)")
        ax.set_title("Fine-tune preference per dilemma")
        ax.legend(frameon=False, loc="lower right")
        ax.grid(axis="x", alpha=0.3, lw=0.5)
        fig.tight_layout()
        fig.savefig(
            FIGURES / "survey-dilemma-preference.png",
            dpi=200,
            bbox_inches="tight",
        )
        return fig

    plot_dilemma_preference(h2_data)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Policy analysis

    Primary outcome: census-weighted probability of inclusion in a
    respondent's top three. Weights rake age x gender and region to the
    2023 Census; intervals use 10,000 respondent-level bootstraps.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Demographic comparison
    """)
    return


@app.cell
def _(mo, policy_sample):
    mo.md(
        f"Analysis sample: **N = {len(policy_sample):,}**. "
        "Missing and prefer-not-to-say categories are left out of each denominator."
    )
    return


@app.cell
def _(
    AGE_CENSUS,
    EMPLOYMENT_CENSUS,
    FIGURES,
    GENDER_CENSUS,
    INCOME_CENSUS,
    REGION_POP,
    REGION_TOTAL,
    plt,
    policy_sample,
):
    EMPLOYMENT_MAP = {
        "Full time (30+ hrs/wk)": "Employed full-time",
        "Part time (<30 hrs)": "Employed part-time",
        "Self employed": "Employed full-time",
        "Retired/pensioned": "Not in the labour force",
        "Stay at home parent/guardian": "Not in the labour force",
        "Student": "Not in the labour force",
        "Unemployed": "Unemployed",
        "Other work/arrangement": "Not in the labour force",
    }

    def pct_map(series):
        counts = series.value_counts()
        total = counts.sum()
        return {k: 100 * v / total for k, v in counts.items()} if total else {}

    income = policy_sample["Z6-1"]
    low = (income == "Up to $30,000").sum() + (2 / 3) * (
        income == "$30,001 - $60,000"
    ).sum()
    medium = (1 / 3) * (income == "$30,001 - $60,000").sum() + (
        income == "$60,001 - $100,000"
    ).sum()
    high = income.isin(
        ["$100,001 - $150,000", "$150,001 - $200,000", "$200,001 or more"]
    ).sum()
    income_total = low + medium + high
    income_pct = {
        "Low": 100 * low / income_total,
        "Medium": 100 * medium / income_total,
        "High": 100 * high / income_total,
    }

    groups = [
        ("Age", AGE_CENSUS, pct_map(policy_sample["age_band"])),
        ("Gender", GENDER_CENSUS, pct_map(policy_sample["gender_label"])),
        (
            "Region",
            {k: 100 * v / REGION_TOTAL for k, v in REGION_POP.items()},
            pct_map(policy_sample["region_label"]),
        ),
        (
            "Employment",
            EMPLOYMENT_CENSUS,
            pct_map(policy_sample["Z4-1"].map(EMPLOYMENT_MAP)),
        ),
        ("Income (3 bands)", INCOME_CENSUS, income_pct),
    ]

    bars, heads, seps = [], [], []
    y = 0.0
    for group_index, (name, census, sample) in enumerate(groups):
        if group_index:
            seps.append(y - 0.9)
            y -= 1.8
        heads.append((y, name))
        y -= 1.0
        for category, target in census.items():
            bars.append((y, category, sample.get(category, 0.0), target))
            y -= 1.0

    fig, ax = plt.subplots(figsize=(11, 17))
    ypos = [b[0] for b in bars]
    sample_vals = [b[2] for b in bars]
    census_vals = [b[3] for b in bars]
    height = 0.38
    ax.barh(
        [p + height / 2 for p in ypos],
        sample_vals,
        height=height,
        label=f"Survey sample (n={len(policy_sample):,})",
        color="#4c72b0",
    )
    ax.barh(
        [p - height / 2 for p in ypos],
        census_vals,
        height=height,
        label="2023 NZ Census",
        color="#c44e52",
    )
    ax.set_yticks(ypos)
    ax.set_yticklabels([b[1] for b in bars], fontsize=9)
    for heading_y, name in heads:
        ax.text(
            -0.02,
            heading_y,
            name,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
    for separator_y in seps:
        ax.axhline(separator_y, color="0.75", lw=0.8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Percent of respondents / population")
    ax.set_title("Demographics: survey sample vs 2023 Census")
    ax.legend(fontsize=9, loc="lower right")
    for p, sample_value, census_value in zip(ypos, sample_vals, census_vals):
        ax.text(
            sample_value + 0.6,
            p + height / 2,
            f"{sample_value:.1f}",
            va="center",
            fontsize=7,
        )
        ax.text(
            census_value + 0.6,
            p - height / 2,
            f"{census_value:.1f}",
            va="center",
            fontsize=7,
        )
    ax.set_xlim(0, max(sample_vals + census_vals) * 1.25)
    ax.set_ylim(min(ypos) - 1.0, 0.8)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "survey-demographics-vs-census.png",
        dpi=200,
        bbox_inches="tight",
    )
    return


@app.cell
def _(POLICY_JSON, json):
    POLICY = json.loads(POLICY_JSON.read_text())
    POLICY_TITLES = [o["title"] for o in POLICY["options"]]
    POLICY_SHORT = [
        "Office for AI",
        "AI detectors",
        "Harden infrastructure",
        "Jobseeker 2.0",
        "Tax AI usage",
        "Public AI infrastructure",
        "Human control",
        "Protect children",
        "Build AI hardware here",
    ]
    POLICY_COLS = [f"C4-3_{i}" for i in range(1, 10)]
    RANK_COLS = [f"C4-4_{i}" for i in range(1, 10)]
    POLICY_SEED = 20261029
    AGE_CENSUS = {
        "18-24": 11.4,
        "25-29": 8.7,
        "30-34": 9.7,
        "35-39": 8.9,
        "40-44": 8.2,
        "45-49": 7.8,
        "50-54": 8.3,
        "55-59": 7.9,
        "60-64": 7.7,
        "65 and older": 21.4,
    }
    GENDER_CENSUS = {"Male": 49.5, "Female": 50.5}
    EMPLOYMENT_CENSUS = {
        "Employed full-time": 51.2,
        "Employed part-time": 13.4,
        "Unemployed": 3.0,
        "Not in the labour force": 32.4,
    }
    INCOME_CENSUS = {
        "Low": 57.02755036631119,
        "Medium": 30.895181735713,
        "High": 12.077193963067002,
    }
    REGION_POP = {
        "Northland": 194_007,
        "Auckland": 1_656_486,
        "Waikato": 498_771,
        "Bay of Plenty": 334_140,
        "Gisborne": 51_135,
        "Hawke's Bay": 175_074,
        "Taranaki": 126_015,
        "Manawatu-Wanganui": 251_412,
        "Wellington": 520_971,
        "Tasman": 57_807,
        "Nelson": 52_584,
        "Marlborough": 49_431,
        "West Coast": 33_390,
        "Canterbury": 651_027,
        "Otago": 240_900,
        "Southland": 100_143,
        "Chatham Islands": 633,
    }
    REGION_TOTAL = sum(REGION_POP.values())
    return (
        AGE_CENSUS,
        EMPLOYMENT_CENSUS,
        GENDER_CENSUS,
        INCOME_CENSUS,
        POLICY_COLS,
        POLICY_SEED,
        POLICY_SHORT,
        POLICY_TITLES,
        RANK_COLS,
        REGION_POP,
        REGION_TOTAL,
    )


@app.cell
def _(mo):
    mo.md("""
    ### Shares and weights
    """)
    return


@app.cell
def _(
    AGE_CENSUS,
    GENDER_CENSUS,
    POLICY_COLS,
    POLICY_TITLES,
    REGION_POP,
    REGION_TOTAL,
    np,
    policy_sample,
):
    def policy_matrix(responses):
        for i, column in enumerate(POLICY_COLS):
            values = responses[column].dropna().unique()
            if any(not str(v).startswith(POLICY_TITLES[i]) for v in values):
                raise SystemExit(f"Policy wording mismatch in {column}")
        return responses[POLICY_COLS].notna().to_numpy(int)

    def rake_weights(responses):
        weights = np.ones(len(responses))
        for _ in range(500):
            previous = weights.copy()
            for age, age_share in AGE_CENSUS.items():
                for gender, gender_share in GENDER_CENSUS.items():
                    mask = (
                        (responses["age_band"] == age)
                        & (responses["gender_label"] == gender)
                    ).to_numpy()
                    if mask.any():
                        target = age_share / 100 * gender_share / 100
                        weights[mask] *= target / weights[mask].sum()
            for region, population in REGION_POP.items():
                mask = (responses["region_label"] == region).to_numpy()
                if mask.any():
                    weights[mask] *= population / REGION_TOTAL / weights[mask].sum()
            if np.max(np.abs(weights - previous)) / np.mean(weights) < 1e-10:
                break

        known_age_gender = responses["age_band"].isin(AGE_CENSUS) & responses[
            "gender_label"
        ].isin(GENDER_CENSUS)
        known_region = responses["region_label"].isin(REGION_POP)
        mean_weight = weights.mean()
        for i in np.where(~known_age_gender)[0]:
            cell = (
                (responses["age_band"] == responses["age_band"].iloc[i])
                & responses["gender_label"].isin(GENDER_CENSUS)
                & (responses["region_label"] == responses["region_label"].iloc[i])
            ).to_numpy()
            weights[i] = weights[cell].mean() if cell.any() else mean_weight
        for i in np.where(~known_region)[0]:
            weights[i] = (
                weights[known_region].mean() if known_region.any() else mean_weight
            )

        low, high = np.quantile(weights, [0.01, 0.99])
        weights = np.clip(weights, low, high)
        weights *= len(weights) / weights.sum()
        kish = weights.sum() ** 2 / (weights**2).sum()
        return weights, kish

    policy_Y = policy_matrix(policy_sample)
    policy_weights, policy_kish = rake_weights(policy_sample)
    policy_shares = (policy_Y * policy_weights[:, None]).sum(
        axis=0
    ) / policy_weights.sum()
    return policy_Y, policy_kish, policy_shares, policy_weights


@app.cell
def _(BOOTSTRAP_REPS, POLICY_SEED, np, policy_Y, policy_weights):
    def bootstrap_shares(Y, weights):
        rng = np.random.default_rng(POLICY_SEED)
        draw = rng.integers(0, len(Y), size=(BOOTSTRAP_REPS, len(Y)))
        denominator = weights[draw].sum(axis=1)
        return np.column_stack(
            [
                (weights[draw] * Y[draw, k]).sum(axis=1) / denominator
                for k in range(Y.shape[1])
            ]
        )

    policy_boot = bootstrap_shares(policy_Y, policy_weights)
    policy_low, policy_high = np.quantile(policy_boot, [0.025, 0.975], axis=0)
    return policy_boot, policy_high, policy_low


@app.cell
def _(mo):
    mo.md("""
    ### Policy support
    """)
    return


@app.cell
def _(BOOTSTRAP_REPS, np, pd, policy_boot):
    def bh_qvalues(pvalues):
        pvalues = np.asarray(pvalues, float)
        order = np.argsort(pvalues)
        qvalues = np.empty_like(pvalues)
        previous = 1.0
        m = len(pvalues)
        for rank, index in enumerate(order[::-1], 1):
            value = pvalues[index] * m / (m - rank + 1)
            previous = min(previous, value)
            qvalues[index] = previous
        return qvalues

    def pairwise_table(boot):
        rows = []
        for i in range(9):
            for j in range(i + 1, 9):
                difference = boot[:, i] - boot[:, j]
                low_p = ((difference <= 0).sum() + 1) / (BOOTSTRAP_REPS + 1)
                high_p = ((difference >= 0).sum() + 1) / (BOOTSTRAP_REPS + 1)
                rows.append(
                    {
                        "i": i,
                        "j": j,
                        "difference": difference.mean(),
                        "low": np.quantile(difference, 0.025),
                        "high": np.quantile(difference, 0.975),
                        "p": min(1.0, 2 * min(low_p, high_p)),
                    }
                )
        table = pd.DataFrame(rows)
        table["q"] = bh_qvalues(table["p"].to_numpy())
        return table

    policy_pairwise = pairwise_table(policy_boot)
    return (policy_pairwise,)


@app.cell
def _(policy_pairwise, policy_shares):
    def preference_tiers(shares, pairwise):
        lookup = {}
        for row in pairwise.itertuples():
            lookup[(row.i, row.j)] = (row.q, row.low, row.high)

        def reliably_greater(i, j):
            if (i, j) in lookup:
                q, low, _ = lookup[(i, j)]
                return q < 0.05 and low > 0
            q, _, high = lookup[(j, i)]
            return q < 0.05 and high < 0

        remaining = set(range(len(shares)))
        tiers = []
        while remaining:
            top = max(remaining, key=lambda k: shares[k])
            tier = [k for k in remaining if k == top or not reliably_greater(top, k)]
            tier = sorted(tier, key=lambda k: -shares[k])
            tiers.append(tier)
            remaining -= set(tier)
        return tiers

    policy_tiers = preference_tiers(policy_shares, policy_pairwise)
    policy_tier_of = {
        policy: tier_index + 1
        for tier_index, tier in enumerate(policy_tiers)
        for policy in tier
    }
    return policy_tier_of, policy_tiers


@app.cell
def _(chi2, np, policy_boot, policy_shares):
    contrasts = np.hstack([np.eye(8), -np.ones((8, 1))])
    difference = contrasts @ policy_shares
    covariance = np.cov(policy_boot, rowvar=False, ddof=1)
    policy_wald = float(
        difference @ np.linalg.pinv(contrasts @ covariance @ contrasts.T) @ difference
    )
    policy_omnibus_p = float(chi2.sf(policy_wald, 8))
    return policy_omnibus_p, policy_wald


@app.cell
def _(mo, policy_Y, policy_kish, policy_omnibus_p, policy_tiers, policy_wald):
    tiers = " | ".join(
        "Tier {}: {}".format(i + 1, ", ".join(f"P{k + 1}" for k in tier))
        for i, tier in enumerate(policy_tiers)
    )
    top_tier = policy_tiers[0]
    winner = (
        f"P{top_tier[0] + 1} is the single most preferred policy."
        if len(top_tier) == 1
        else "No single policy is reliably ahead of the rest of the top tier."
    )
    mo.md(
        f"""
        **N = {len(policy_Y):,} valid respondents**;
        Kish effective n = {policy_kish:.1f}.

        Omnibus Wald test: $\\chi^2(8)$ = {policy_wald:.2f},
        p = {policy_omnibus_p:.3g}.

        **Tiers:** {tiers}

        **Most preferred:** {winner}
        """
    )
    return


@app.cell
def _(
    FIGURES,
    POLICY_SHORT,
    np,
    plt,
    policy_high,
    policy_low,
    policy_shares,
    policy_tier_of,
    policy_tiers,
):
    def plot_policy():
        order = np.argsort(policy_shares)
        # Up to nine policies can each be their own tier; use the default
        # matplotlib cycle (C0..C9) so every tier gets a distinct colour.
        colours = [f"C{i}" for i in range(10)]
        tier_colours = [colours[(policy_tier_of[k] - 1) % len(colours)] for k in order]
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        y = np.arange(len(order))
        ax.errorbar(
            100 * policy_shares[order],
            y,
            xerr=[
                100 * (policy_shares[order] - policy_low[order]),
                100 * (policy_high[order] - policy_shares[order]),
            ],
            fmt="o",
            color="black",
            ecolor="0.45",
            capsize=3,
            zorder=2,
        )
        for yi, k, colour in zip(y, order, tier_colours):
            ax.scatter(100 * policy_shares[k], yi, color=colour, s=45, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([f"P{k + 1}  {POLICY_SHORT[k]}" for k in order])
        ax.set_xlabel("Weighted % selecting policy in top three")
        ax.set_title("Policy support (2023 Census weighted)")
        ax.grid(axis="x", alpha=0.3, lw=0.5)
        handles = [
            plt.Line2D(
                [],
                [],
                marker="o",
                ls="",
                color=colours[i % len(colours)],
                label=f"Tier {i + 1}",
            )
            for i in range(len(policy_tiers))
        ]
        ax.legend(handles=handles, frameon=False, loc="lower right")
        fig.tight_layout()
        fig.savefig(FIGURES / "survey-policy-shares.png", dpi=200, bbox_inches="tight")
        return fig

    plot_policy()
    return


@app.cell
def _(mo):
    mo.md("""
    ### Pairwise comparisons
    """)
    return


@app.cell
def _(mo, pd, policy_pairwise):
    pairwise_display = pd.DataFrame(
        [
            {
                "Difference (pp)": f"{100 * row.difference:.1f}",
                "95% CI": f"[{100 * row.low:.1f}, {100 * row.high:.1f}]",
                "q": f"{row.q:.3f}",
                "Reliable": "yes" if row.q < 0.05 and row.low > 0 else "no",
            }
            for row in policy_pairwise.itertuples()
        ],
        index=[f"P{row.i + 1} - P{row.j + 1}" for row in policy_pairwise.itertuples()],
    )
    mo.accordion({"All 36 comparisons": mo.ui.table(pairwise_display, selection=None)})
    return


@app.cell
def _(mo):
    mo.md("""
    ### Rank analysis (secondary)
    """)
    return


@app.cell
def _(
    BOOTSTRAP_REPS,
    POLICY_SEED,
    POLICY_TITLES,
    RANK_COLS,
    mo,
    np,
    pd,
    policy_Y,
    policy_sample,
    policy_weights,
):
    def rank_summary(responses, Y, weights):
        ranks = responses[RANK_COLS].apply(pd.to_numeric, errors="coerce").to_numpy()
        selected = np.where(Y == 1, ranks, np.nan)
        valid = np.all(np.sort(selected, axis=1)[:, :3] == [1, 2, 3], axis=1)
        if not valid.any():
            return pd.DataFrame(), 0
        subset, subset_ranks, subset_weights = Y[valid], ranks[valid], weights[valid]
        borda = np.where(subset == 1, 4 - subset_ranks, 0)
        first = np.where((subset == 1) & (subset_ranks == 1), 1, 0)
        first_share = (first * subset_weights[:, None]).sum(
            axis=0
        ) / subset_weights.sum()
        borda_mean = (borda * subset_weights[:, None]).sum(
            axis=0
        ) / subset_weights.sum()

        rng = np.random.default_rng(POLICY_SEED)
        draw = rng.integers(0, len(subset), size=(BOOTSTRAP_REPS, len(subset)))
        denominator = subset_weights[draw].sum(axis=1)
        first_boot = np.column_stack(
            [
                (subset_weights[draw] * first[draw, k]).sum(axis=1) / denominator
                for k in range(9)
            ]
        )
        borda_boot = np.column_stack(
            [
                (subset_weights[draw] * borda[draw, k]).sum(axis=1) / denominator
                for k in range(9)
            ]
        )
        first_low, first_high = np.quantile(first_boot, [0.025, 0.975], axis=0)
        borda_low, borda_high = np.quantile(borda_boot, [0.025, 0.975], axis=0)
        order = np.argsort(-first_share)
        table = pd.DataFrame(
            [
                {
                    "First choice (%)": f"{100 * first_share[k]:.1f}",
                    "95% CI": f"[{100 * first_low[k]:.1f}, {100 * first_high[k]:.1f}]",
                    "Mean Borda": f"{borda_mean[k]:.2f}",
                    "Borda 95% CI": f"[{borda_low[k]:.2f}, {borda_high[k]:.2f}]",
                }
                for k in order
            ],
            index=[f"P{k + 1} {POLICY_TITLES[k]}" for k in order],
        )
        return table, int(valid.sum())

    rank_table, ranked_n = rank_summary(policy_sample, policy_Y, policy_weights)
    mo.accordion({f"C4-4 (n = {ranked_n})": mo.ui.table(rank_table, selection=None)})
    return


@app.cell
def _(mo):
    mo.md("""
    ## Timing analysis

    How long respondents spent on the survey overall and on each page.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Timing

    Shares use stage medians; medians need not add exactly to the total median.
    """)
    return


@app.cell
def _(base_keep, h2_data, pd, responses, write_latex_table):
    def timing_allocation(responses, data):
        by_id = responses.set_index("_recordId")
        frame = pd.DataFrame(
            {
                "first": pd.to_numeric(by_id["Z_timing_PAGE_SUBMIT"], errors="coerce"),
                "comparison": data.groupby("_recordId")["seconds"].sum(),
                "last": pd.to_numeric(by_id["C_timing_PAGE_SUBMIT"], errors="coerce"),
                "total": pd.to_numeric(by_id["duration"], errors="coerce"),
            }
        )
        frame["other"] = (
            frame["total"] - frame[["first", "comparison", "last"]].sum(axis=1)
        ).clip(lower=0)
        stages = [
            ("First page (Section A)", "first"),
            ("Five comparison pages", "comparison"),
            ("Last page (Section C)", "last"),
            ("Other/unmeasured pages (info sheet, loading time etc.)", "other"),
            ("Total survey", "total"),
        ]
        stage_total = sum(frame[column].median() for _, column in stages[:-1])
        rows = [
            {
                "Stage": stage,
                "Median (s)": f"{frame[column].median():.1f}",
                "IQR (s)": (
                    f"{frame[column].quantile(0.25):.1f}-"
                    f"{frame[column].quantile(0.75):.1f}"
                ),
                "Median (min)": f"{frame[column].median() / 60:.1f}",
                # NB: the percent sign is escaped here because
                # write_latex_table only escapes headers/index, not cells.
                "Share of stages": (
                    "100.0\\%"
                    if column == "total"
                    else f"{100 * frame[column].median() / stage_total:.1f}\\%"
                ),
            }
            for stage, column in stages
        ]
        per_page = data["seconds"].dropna()
        rows.insert(
            1,
            {
                "Stage": "Comparison page (per page)",
                "Median (s)": f"{per_page.median():.1f}",
                "IQR (s)": (
                    f"{per_page.quantile(0.25):.1f}-{per_page.quantile(0.75):.1f}"
                ),
                "Median (min)": f"{per_page.median() / 60:.1f}",
                "Share of stages": "—",
            },
        )
        table = pd.DataFrame(rows)
        table.index = table.pop("Stage")
        table.index.name = None
        write_latex_table(
            table,
            "{}",
            4,
            "survey-timing-allocation.tex",
            stub_label="",
        )
        return table

    timing_allocation(responses[base_keep], h2_data)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Overall response time
    """)
    return


@app.cell
def _(FIGURES, exclusion_diagnostics, np, pd, plt, responses):
    def plot_response_time(data):
        """Raw response-time distribution, excluding only development tests.

        The histogram is capped at 60 minutes for readability; the count of
        longer responses is reported in the corner. The 4-minute line marks
        the live lower cutoff.
        """
        minutes = pd.to_numeric(data["duration"], errors="coerce").dropna() / 60
        shown = minutes[minutes <= 60]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(
            shown,
            bins=np.arange(0, 61, 1),
            color="tab:blue",
            alpha=0.85,
            edgecolor="white",
        )
        ax.axvline(
            minutes.median(),
            color="black",
            ls="--",
            lw=1,
            label=f"Median {minutes.median():.1f} min",
        )
        ax.axvline(
            4,
            color="tab:red",
            ls=":",
            lw=1,
            label="4-minute cutoff",
        )
        ax.set_xlim(0, 60)
        ax.set_xlabel("Total response time (minutes; 1-minute bins)")
        ax.set_ylabel("Respondents")
        ax.set_title("Overall response time (all non-test responses)")
        ax.text(
            0.99,
            0.95,
            f"N = {len(minutes):,} non-test responses\n"
            f"{len(minutes) - len(shown)} over 60 min not shown",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="0.35",
        )
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.3, lw=0.5)
        fig.tight_layout()
        fig.savefig(
            FIGURES / "survey-response-time-histogram.png",
            dpi=200,
            bbox_inches="tight",
        )
        return fig

    plot_response_time(responses[~exclusion_diagnostics["dev_test"]])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Dilemma analysis

    ### Support for AI in each dilemma
    """)
    return


@app.cell
def _(np, pd, trim_mean):
    def dilemma_stats(data):
        rows = []
        for (_, situation), group in data.groupby(["dilemma", "situation"], sort=False):
            support = group["aiuse_score"].dropna()
            preference = group["ft_score"].dropna()
            seconds = group["seconds"].dropna()
            pref_margin = 1.96 * preference.std(ddof=1) / np.sqrt(len(preference))
            rows.append(
                {
                    "dilemma": situation,
                    "preference": preference.mean(),
                    "pref_low": preference.mean() - pref_margin,
                    "pref_high": preference.mean() + pref_margin,
                    "support": support.mean(),
                    "support_q1": support.quantile(0.25),
                    "support_q3": support.quantile(0.75),
                    "seconds": trim_mean(seconds.to_numpy(), 0.05),
                    "time_q1": seconds.quantile(0.25),
                    "time_q3": seconds.quantile(0.75),
                    "n": len(group),
                }
            )
        return pd.DataFrame(rows)

    return (dilemma_stats,)


@app.cell
def _(FIGURES, dilemma_stats, h2_data, np, plt):
    def plot_ai_support(data):
        stats = dilemma_stats(data).sort_values("support")
        support = data["aiuse_score"].dropna()
        y = np.arange(len(stats))
        fig, ax = plt.subplots(figsize=(9, 6.5))
        ax.hlines(
            y,
            stats["support_q1"],
            stats["support_q3"],
            color="0.55",
            lw=2,
        )
        ax.plot(stats["support"], y, "o", color="tab:blue", ms=5)
        ax.axvline(
            support.mean(),
            color="black",
            ls="--",
            lw=1,
            label=f"Overall mean {support.mean():.2f}",
        )
        ax.set_yticks(y)
        ax.set_yticklabels(stats["dilemma"])
        ax.set_xlim(1, 5)
        ax.set_xlabel("Mean agreement that AI should be used (1-5)")
        ax.set_title("Support for AI in each dilemma")
        ax.legend(frameon=False, loc="lower right")
        ax.grid(axis="x", alpha=0.3, lw=0.5)
        fig.tight_layout()
        fig.savefig(
            FIGURES / "survey-dilemma-ai-support.png",
            dpi=200,
            bbox_inches="tight",
        )
        return fig

    plot_ai_support(h2_data)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Mean time per dilemma (5% trimmed)
    """)
    return


@app.cell
def _(FIGURES, dilemma_stats, h2_data, np, plt, trim_mean):
    def plot_mean_time(data):
        stats = dilemma_stats(data).sort_values("seconds")
        seconds = data["seconds"].dropna()
        y = np.arange(len(stats))
        fig, ax = plt.subplots(figsize=(9, 6.5))
        ax.hlines(
            y,
            stats["time_q1"],
            stats["time_q3"],
            color="0.55",
            lw=2,
        )
        ax.plot(stats["seconds"], y, "o", color="tab:orange", ms=5)
        ax.axvline(
            trim_mean(seconds.to_numpy(), 0.05),
            color="black",
            ls="--",
            lw=1,
            label=f"Overall 5% trimmed mean {trim_mean(seconds.to_numpy(), 0.05):.0f}s",
        )
        ax.set_yticks(y)
        ax.set_yticklabels(stats["dilemma"])
        ax.set_xlabel("5% trimmed mean page time (seconds)")
        ax.set_title("Mean time per dilemma (5% trimmed)")
        ax.legend(frameon=False, loc="lower right")
        ax.grid(axis="x", alpha=0.3, lw=0.5)
        fig.tight_layout()
        fig.savefig(
            FIGURES / "survey-dilemma-mean-time.png",
            dpi=200,
            bbox_inches="tight",
        )
        return fig

    plot_mean_time(h2_data)
    return


if __name__ == "__main__":
    app.run()
