import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import re
    import sys
    import textwrap
    from math import erfc, sqrt
    from pathlib import Path
    from statistics import NormalDist

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from scipy.linalg import cho_factor, cho_solve
    from scipy.optimize import minimize
    from scipy.sparse.linalg import splu
    from scipy.special import expit, logsumexp
    from scipy.stats import chi2, f_oneway, trim_mean

    DIR = Path(__file__).resolve().parent
    REPO = DIR.parent
    sys.path.insert(0, str(REPO / "code"))
    sys.path.insert(0, str(DIR))
    from latex_tables import write_latex_table
    from r_models import fit_bt_glmm, fit_lmm

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
        expit,
        f_oneway,
        fit_bt_glmm,
        fit_lmm,
        json,
        logsumexp,
        minimize,
        mo,
        np,
        pd,
        plt,
        re,
        sparse,
        splu,
        sqrt,
        textwrap,
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
def _(NormalDist, np, pd):
    # --- Analysis-ready demographic groups and AI-view scores ---------------
    #
    # The raw export has the census-style questions (Z*) and the three
    # Section C AI items. This cell turns them into the categories used by
    # every demographic breakdown below. The groupings keep enough
    # respondents per cell for a stable interval while still matching the
    # census/WVS categories used in the representativeness analysis.
    AIUSE_FREQ_SCORES = {
        "Never": 1,
        "Monthly or less often": 2,
        "Weekly": 3,
        "A couple of times a week": 4,
        "Daily": 5,
    }
    OUTLOOK_SCORES = {
        "Very negative": 1,
        "Somewhat negative": 2,
        "Neither negative nor positive": 3,
        "Somewhat positive": 4,
        "Very positive": 5,
    }
    RISK_SCORES = {
        "Strongly disagree": 1,
        "Disagree": 2,
        "Neither agree nor disagree": 3,
        "Agree": 4,
        "Strongly agree": 5,
    }
    AGE_ORDER = [
        "18-24",
        "25-29",
        "30-34",
        "35-39",
        "40-44",
        "45-49",
        "50-54",
        "55-59",
        "60-64",
        "65 and older",
    ]
    GENDER_ORDER = ["Male", "Female", "Non-binary or another gender identity"]
    ETHNICITY_ORDER = [
        "New Zealand European / Pākehā",
        "Māori",
        "Pacific peoples",
        "Asian",
        "Middle Eastern / Latin American / African",
        "Other",
    ]
    EMPLOYMENT_ORDER = [
        "Employed full-time",
        "Employed part-time",
        "Unemployed",
        "Not in the labour force",
    ]
    # Region has 16 populated categories in the sample, several with fewer
    # than ten respondents. The model uses four broad bands so each
    # coefficient is estimated from a stable cell; the census figure still
    # shows every region.
    REGION_ORDER = [
        "Auckland",
        "Wellington",
        "Rest of North Island",
        "South Island",
        "Other / overseas",
    ]
    REGION_GROUPS = {
        "Auckland": "Auckland",
        "Wellington": "Wellington",
        "Northland": "Rest of North Island",
        "Waikato": "Rest of North Island",
        "Bay of Plenty": "Rest of North Island",
        "Gisborne": "Rest of North Island",
        "Hawke's Bay": "Rest of North Island",
        "Taranaki": "Rest of North Island",
        "Manawatu-Wanganui": "Rest of North Island",
        "Tasman": "South Island",
        "Nelson": "South Island",
        "Marlborough": "South Island",
        "West Coast": "South Island",
        "Canterbury": "South Island",
        "Otago": "South Island",
        "Southland": "South Island",
        "Chatham Islands": "South Island",
        "Other": "Other / overseas",
        "I live overseas": "Other / overseas",
    }
    # ANZSCO major groups, with the two smallest cells (machinery operators
    # and never worked) pooled into "Other" for the joint model.
    OCCUPATION_ORDER = [
        "Professionals",
        "Managers",
        "Technicians and Trades Workers",
        "Community and Personal Service Workers",
        "Clerical and Administrative Workers",
        "Sales Workers",
        "Labourers",
        "Other",
    ]
    OCCUPATION_GROUPS = {
        "Managers (e.g. business owners, chief executives, general managers, department managers)": (
            "Managers"
        ),
        "Professionals (e.g. doctors, teachers, engineers, lawyers, scientists, accountants)": (
            "Professionals"
        ),
        "Technicians and Trades Workers (e.g. electricians, plumbers, mechanics, engineering technicians)": (
            "Technicians and Trades Workers"
        ),
        "Community and Personal Service Workers (e.g. caregivers, police officers, childcare workers, fitness instructors)": (
            "Community and Personal Service Workers"
        ),
        "Clerical and Administrative Workers (e.g. receptionists, office administrators, payroll clerks, bank tellers)": (
            "Clerical and Administrative Workers"
        ),
        "Sales Workers (e.g. retail sales assistants, real estate agents, sales representatives)": (
            "Sales Workers"
        ),
        "Machinery Operators and Drivers (e.g. truck drivers, bus drivers, forklift operators, production machinery operators)": (
            "Other"
        ),
        "Labourers (e.g. construction labourers, cleaners, farm labourers, fruit pickers)": (
            "Labourers"
        ),
        "I have never had a job": "Other",
    }
    INCOME_ORDER = ["Low", "Medium", "High"]
    EDUCATION_ORDER = [
        "No formal / primary",
        "Secondary",
        "Tertiary <2 years",
        "Tertiary 3-4 years",
        "Tertiary 4+ years",
    ]
    AIUSE_ORDER = [
        "Never",
        "Monthly or less often",
        "Weekly",
        "A couple of times a week",
        "Daily",
    ]
    OUTLOOK_ORDER = [
        "Very negative",
        "Somewhat negative",
        "Neither negative nor positive",
        "Somewhat positive",
        "Very positive",
    ]
    RISK_ORDER = [
        "Strongly disagree",
        "Disagree",
        "Neither agree nor disagree",
        "Agree",
        "Strongly agree",
    ]
    EMPLOYMENT_GROUPS = {
        "Full time (30+ hrs/wk)": "Employed full-time",
        "Self employed": "Employed full-time",
        "Part time (<30 hrs)": "Employed part-time",
        "Retired/pensioned": "Not in the labour force",
        "Stay at home parent/guardian": "Not in the labour force",
        "Student": "Not in the labour force",
        "Other work/arrangement": "Not in the labour force",
        "Unemployed": "Unemployed",
    }
    INCOME_GROUPS = {
        "Up to $30,000": "Low",
        "$30,001 - $60,000": "Medium",
        "$60,001 - $100,000": "Medium",
        "$100,001 - $150,000": "High",
        "$150,001 - $200,000": "High",
        "$200,001 or more": "High",
    }
    EDUCATION_GROUPS = {
        "No formal schooling": "No formal / primary",
        "Primary school (including intermediate) or kura kaupapa (e.g. Years 1-8)": (
            "No formal / primary"
        ),
        "Secondary school for up to 3 years (e.g. NCEA Level 1 or less)": "Secondary",
        "Secondary school for 4 years or more (e.g. NCEA Level 2 or 3)": "Secondary",
        "Tertiary for less than 2 years (trade apprenticeship, certificates or diploma)": (
            "Tertiary <2 years"
        ),
        "Tertiary for 3-4 years (a bachelor's or honours degree, e.g. BA or BSc)": (
            "Tertiary 3-4 years"
        ),
        "Tertiary for more than 4 years (a master's or doctorate, e.g. MA or PhD)": (
            "Tertiary 4+ years"
        ),
    }
    ETHNICITY_COLUMNS = [
        ("Z3-1_1", "New Zealand European / Pākehā"),
        ("Z3-1_2", "Māori"),
        ("Z3-1_3", "Pacific peoples"),
        ("Z3-1_4", "Asian"),
        ("Z3-1_5", "Middle Eastern / Latin American / African"),
        ("Z3-1_6", "Other"),
        ("Z3-1_7", "Prefer not to say"),
    ]
    # Priority order for the multi-select ethnicity item: a respondent who
    # ticks Māori and NZ European is counted as Māori, and so on. This keeps
    # one row per respondent and follows the usual small-sample convention.
    ETHNICITY_PRIORITY = [
        "Māori",
        "Pacific peoples",
        "Asian",
        "Middle Eastern / Latin American / African",
        "Other",
        "New Zealand European / Pākehā",
    ]

    def analysis_demographics(data):
        """Add the grouped demographics and 1-5 AI-view scores to responses."""
        out = data.copy()
        out["employment_group"] = out["Z4-1"].map(EMPLOYMENT_GROUPS)
        out["region_group"] = out["region_label"].map(REGION_GROUPS)
        out["occupation_group"] = out["Z7-1"].map(OCCUPATION_GROUPS)
        out["income_band"] = out["Z6-1"].map(INCOME_GROUPS)
        out["education_group"] = out["Z8-1"].map(EDUCATION_GROUPS)
        selected = pd.DataFrame(
            {
                label: out[column].eq(label)
                for column, label in ETHNICITY_COLUMNS
                if column in out
            },
            index=out.index,
        )
        out["ethnicity_group"] = selected.apply(
            lambda row: next(
                (label for label in ETHNICITY_PRIORITY if row.get(label, False)),
                None,
            ),
            axis=1,
        )
        out["ai_use_score"] = out["C8-1"].map(AIUSE_FREQ_SCORES)
        out["ai_outlook_score"] = out["C1-1"].map(OUTLOOK_SCORES)
        out["ai_risk_score"] = out["Q943"].map(RISK_SCORES)
        return out

    def mean_ci(values):
        """Mean, normal 95% interval and n for a 1-5 or -2..2 outcome."""
        values = pd.Series(values).dropna().astype(float)
        n = len(values)
        if not n:
            return np.nan, np.nan, np.nan, 0
        mean = float(values.mean())
        if n == 1:
            return mean, np.nan, np.nan, 1
        half = NormalDist().inv_cdf(0.975) * values.std(ddof=1) / np.sqrt(n)
        return mean, mean - half, mean + half, n

    def survey_bh_qvalues(pvalues):
        """Benjamini-Hochberg adjusted q-values."""
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

    return (
        AGE_ORDER,
        AIUSE_FREQ_SCORES,
        AIUSE_ORDER,
        EDUCATION_ORDER,
        EMPLOYMENT_ORDER,
        ETHNICITY_ORDER,
        GENDER_ORDER,
        INCOME_ORDER,
        OCCUPATION_ORDER,
        OUTLOOK_ORDER,
        OUTLOOK_SCORES,
        REGION_ORDER,
        RISK_ORDER,
        RISK_SCORES,
        analysis_demographics,
        mean_ci,
        survey_bh_qvalues,
    )


@app.cell
def _(ATTENTION, np, pd):
    CUTOFF = pd.Timestamp("2026-10-29 14:30", tz="Pacific/Auckland")
    MIN_DURATION_SECONDS = 240  # 4 minutes, matching the live screenout
    MAX_DURATION_SECONDS = 2 * 60 * 60  # 2 hours: tab-left-open cap

    def exclusion_masks(data):
        """Requested exclusions: dev tests, incompletes, attention, speed
        (both tails), straightlining and duplicate panelists.

        ``Q_TerminateFlag == "PoorQuality"`` is deliberately *not* an
        exclusion; it is kept as a flag for manual review.
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
        eligible = (
            ~dev_test & ~incomplete & attention & ~too_fast & ~too_long & ~straight
        )
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
            ~dev_test & ~incomplete & attention & ~too_fast & ~too_long & straight,
            "reason",
        ] = "Same Likert answers"
        diagnostics.loc[
            ~dev_test
            & ~incomplete
            & attention
            & ~too_fast
            & ~too_long
            & ~straight
            & duplicate,
            "reason",
        ] = "Duplicate"
        # PoorQuality is flagged, not excluded.

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
        add_step(
            "Same Likert answers (>90%)",
            ~dev_test & ~incomplete & attention & ~too_fast & ~too_long & ~straight,
        )
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
        # PREF_SCORES is negative for "prefer Agent 1" and positive for
        # "prefer Agent 2". When the fine-tune is Agent 1 (order == "ft") the
        # sign must be flipped so positive always means "prefers fine-tune";
        # when the base is Agent 1 it must not be. Getting this backwards
        # inverts every preference in the survey.
        prefs["ft_score"] = prefs["score"] * np.where(ft_first, -1, 1)
        # Order-coding guard: the four order x extreme-preference combinations
        # must map to the expected sign. A silent inversion here flips every
        # result in the chapter, so fail loudly instead.
        expected_sign = {
            ("ft", "Strongly prefer Agent 1"): 2,
            ("ft", "Strongly prefer Agent 2"): -2,
            ("base", "Strongly prefer Agent 1"): -2,
            ("base", "Strongly prefer Agent 2"): 2,
        }
        for (order_value, label), expected in expected_sign.items():
            rows = prefs[
                (prefs["order"] == order_value) & (prefs["preference"] == label)
            ]
            if len(rows) and not (rows["ft_score"] == expected).all():
                raise SystemExit(
                    "Preference order coding is inverted: "
                    f"{order_value}/{label} should be {expected:+d}."
                )
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
def _(analysis_demographics, h2_data, responses):
    # Rating-level analysis frame: every preference rating plus the
    # respondent's grouped demographics and Section C AI views. The
    # respondent-level frame is one row per respondent and is used by the
    # AI-view breakdowns; ``ai_support_mean`` is the respondent's mean
    # agreement that AI should be used across their five dilemmas.
    def preference_frame(ratings, data):
        respondents = analysis_demographics(data).drop_duplicates("_recordId")
        respondents = respondents.set_index("_recordId")
        support = ratings.groupby("_recordId")["aiuse_score"].mean()
        frame = ratings.copy()
        columns = [
            "age_band",
            "gender_label",
            "region_label",
            "region_group",
            "employment_group",
            "occupation_group",
            "income_band",
            "education_group",
            "ethnicity_group",
            "ai_use_score",
            "ai_outlook_score",
            "ai_risk_score",
            "C8-1",
            "C1-1",
            "Q943",
        ]
        for column in columns:
            frame[column] = frame["_recordId"].map(respondents[column])
        frame["ai_support_mean"] = frame["_recordId"].map(support)
        return frame

    def respondent_rows(frame):
        """One row per respondent from the rating-level analysis frame."""
        return frame.drop_duplicates("_recordId").set_index("_recordId")

    analysis_frame = preference_frame(h2_data, responses)
    respondents = respondent_rows(analysis_frame)
    respondents["preference_mean"] = analysis_frame.groupby("_recordId")[
        "ft_score"
    ].mean()
    return (analysis_frame, respondents)


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
def _(
    AGE_ORDER,
    NormalDist,
    exclusion_diagnostics,
    np,
    pd,
    responses,
    write_latex_table,
):
    def speed_age_table(diag, data):
        """Speed exclusions by age band, with Wilson 95% intervals."""
        stage = (
            ~diag["dev_test"]
            & ~diag["incomplete"]
            & diag["attention_pass"]
            & ~diag["too_long"]
        )
        sample = data.loc[stage].copy()
        sample["speed"] = diag.loc[stage, "too_fast"]
        z = NormalDist().inv_cdf(0.975)

        def interval(k, n):
            p = k / n
            den = 1 + z * z / n
            centre = (p + z * z / (2 * n)) / den
            half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
            return centre - half, centre + half

        rows = []
        for band in [*AGE_ORDER, "All ages"]:
            subset = (
                sample if band == "All ages" else sample[sample["age_band"] == band]
            )
            n = len(subset)
            k = int(subset["speed"].sum())
            low, high = interval(k, n)
            rows.append(
                {
                    "Age": band,
                    "Respondents": n,
                    "Excluded for speeding": k,
                    # write_latex_table escapes headers and the index only.
                    "Speeding % (95% CI)": (
                        f"{100 * k / n:.1f}\\% ({100 * low:.1f}-{100 * high:.1f})"
                    ),
                }
            )
        table = pd.DataFrame(rows).set_index("Age")
        table.index.name = None
        write_latex_table(
            table,
            "{}",
            3,
            "survey-speed-by-age.tex",
            stub_label="",
            highlight=False,
        )
        return table

    speed_age_table(exclusion_diagnostics, responses)
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
    same response, so this is a conservative check. In the current export
    the largest observed share is 0.46, so no respondent reaches the 90%
    threshold and the step removes none; the row stays in the exclusion
    table because the rule is in the pre-registration.
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
def _(fit_lmm, h2_data):
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
        index=["LMM (points)", "Respondent-clustered win rate"],
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
    ### Sensitivity to the random-effects structure

    The same five structures are fitted to both H2 outcomes. The comparison
    term is nested in the dilemma (17 pairs per dilemma), so it separates
    pair-to-pair variance from scenario variance; it does not replace the
    dilemma term. The LMM row labelled ``Pre-registered'' is the respondent +
    dilemma model from the pre-registration. The BT table puts the registered
    respondent-clustered win rate first (all ratings, ties split evenly), then
    the proper Bradley--Terry model: a binomial GLMM with a logit link fitted
    by ``lme4::glmer`` on decided comparisons, so its N excludes ties.
    p-values test the preference intercept against zero and the BT intercept
    against no preference.
    """)
    return


@app.cell
def _(fit_bt_glmm, fit_lmm, h2_bt, h2_data, pd, write_latex_table):
    H2_STRUCTURES = (
        ("No random effects", ()),
        ("Respondent", ("_recordId",)),
        ("Respondent + comparison", ("_recordId", "comparison_id")),
        ("Pre-registered (respondent + dilemma)", ("_recordId", "dilemma")),
        (
            "Respondent + comparison + dilemma",
            ("_recordId", "comparison_id", "dilemma"),
        ),
    )
    H2_SHORT = {"_recordId": "resp", "comparison_id": "comp", "dilemma": "dil"}

    def sigma_label(sigma):
        """'resp 0.000; comp 0.223' for the random SDs in a fit."""
        if not sigma:
            return "--"
        return "; ".join(
            f"{H2_SHORT[name]} {value:.3f}" for name, value in sigma.items()
        )

    def _format_p(value):
        return "<0.001" if value < 0.001 else f"{value:.3f}"

    def lmm_variant_table():
        """One row per random-effects structure for the preference rating.

        The respondent + dilemma row is the pre-registered H2 model; it is
        labelled as such so the table always carries the registered estimate.
        """
        rows = {}
        for label, groups in H2_STRUCTURES:
            fit = fit_lmm(h2_data, groups=groups, y_col="ft_score")
            rows[label] = {
                "Estimate": f"{fit['estimate']:+.3f}",
                "95% CI": f"[{fit['ci'][0]:+.3f}, {fit['ci'][1]:+.3f}]",
                "p": _format_p(fit["p"]),
                "N": f"{fit['n']:,}",
                "Random SDs": sigma_label(fit["sigma"]),
            }
        return pd.DataFrame.from_dict(rows, orient="index")

    def bt_variant_table():
        """Pre-registered win rate plus the proper BT GLMM variants.

        The first row is the registered estimand: the pooled win rate with
        ties split evenly and a respondent-clustered bootstrap. The remaining
        rows are the binomial GLMM fitted by lme4, on decided comparisons only
        (the Bradley--Terry likelihood has no tie category).
        """
        decided = h2_data[h2_data["ft_score"] != 0]
        rows = {
            "Pre-registered (respondent-clustered win rate)": {
                "Win rate": f"{h2_bt['estimate']:.3f}",
                "95% CI": f"[{h2_bt['ci'][0]:.3f}, {h2_bt['ci'][1]:.3f}]",
                "p": _format_p(h2_bt["p"]),
                "N": f"{h2_bt['n']:,}",
                "Random SDs": "--",
            }
        }
        for label, groups in H2_STRUCTURES:
            fit = fit_bt_glmm(decided, groups)
            # The LMM table marks respondent + dilemma as pre-registered; in
            # the BT table the registered estimand is the bootstrap row above,
            # so keep this row as the plain structure label.
            bt_label = (
                "Respondent + dilemma" if label.startswith("Pre-registered") else label
            )
            rows[bt_label] = {
                "Win rate": f"{fit['estimate']:.3f}",
                "95% CI": f"[{fit['ci'][0]:.3f}, {fit['ci'][1]:.3f}]",
                "p": _format_p(fit["p"]),
                "N": f"{fit['n']:,}",
                "Random SDs": sigma_label(fit["sigma"]),
            }
        return pd.DataFrame.from_dict(rows, orient="index")

    h2_lmm_variants = lmm_variant_table()
    h2_bt_variants = bt_variant_table()
    write_latex_table(
        h2_lmm_variants,
        "{}",
        5,
        "survey-h2-lmm-variants.tex",
        stub_label="Structure",
        highlight=False,
    )
    write_latex_table(
        h2_bt_variants,
        "{}",
        5,
        "survey-h2-bt-variants.tex",
        stub_label="Structure",
        highlight=False,
    )
    h2_lmm_variants, h2_bt_variants


@app.cell
def _(mo):
    mo.md("""
    ### Does the order term change the estimate?

    Order was randomised and balanced (49.3% fine-tune shown first). Including
    it barely moves the population estimate, because the order effect averages
    out across the design; but per rating it is large for the preference scale
    and small for the win outcome. The table fits the pre-registered structure
    (respondent + dilemma random intercepts) with and without the order
    covariate. The order effect is in points for the LMM and log-odds for the
    BT.
    """)
    return


@app.cell
def _(fit_bt_glmm, fit_lmm, h2_data, pd, write_latex_table):
    def order_effect_table():
        """With/without order for the pre-registered structure."""
        groups = ("_recordId", "dilemma")
        lmm_no = fit_lmm(h2_data, groups=groups, include_order=False)
        lmm_yes = fit_lmm(h2_data, groups=groups, include_order=True)
        decided = h2_data[h2_data["ft_score"] != 0]
        bt_no = fit_bt_glmm(decided, groups, include_order=False)
        bt_yes = fit_bt_glmm(decided, groups, include_order=True)
        rows = {
            "LMM (points)": {
                "Without order": f"{lmm_no['estimate']:+.3f}",
                "With order": f"{lmm_yes['estimate']:+.3f}",
                "Difference": f"{lmm_yes['estimate'] - lmm_no['estimate']:+.3f}",
                "Order effect (SE)": (
                    f"{lmm_yes['beta'][1]:+.3f} ({lmm_yes['se'][1]:.3f})"
                ),
            },
            "BT (win rate)": {
                "Without order": f"{bt_no['estimate']:.3f}",
                "With order": f"{bt_yes['estimate']:.3f}",
                "Difference": f"{bt_yes['estimate'] - bt_no['estimate']:+.3f}",
                "Order effect (SE)": (
                    f"{bt_yes['beta'][1]:+.3f} ({bt_yes['se'][1]:.3f})"
                ),
            },
        }
        return pd.DataFrame.from_dict(rows, orient="index")

    h2_order_effect = order_effect_table()
    write_latex_table(
        h2_order_effect,
        "{}",
        4,
        "survey-h2-order-effect.tex",
        stub_label="Model",
        highlight=False,
    )
    h2_order_effect


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
    ### Distribution of preference ratings

    The 5-point preference scale is signed so that negative values mean the
    base model was preferred and positive values mean the fine-tune was
    preferred. "About the same" is the neutral middle category. The overall
    bar chart shows how many ratings fall in each category; the per-dilemma
    chart below decomposes the same ratings by dilemma and sorts the dilemmas
    by how decisive they were.
    """)
    return


@app.cell
def _(h2_data, pd):
    LIKERT_ORDER = [-2, -1, 0, 1, 2]
    LIKERT_LABELS = {
        -2: "Strongly prefer base",
        -1: "Somewhat prefer base",
        0: "About the same",
        1: "Somewhat prefer fine-tune",
        2: "Strongly prefer fine-tune",
    }
    LIKERT_COLOURS = {
        -2: "#b2182b",
        -1: "#ef8a62",
        0: "#c7c7c7",
        1: "#67a9cf",
        2: "#2166ac",
    }

    def likert_counts(data):
        """Overall count and share of each signed preference category."""
        counts = data["ft_score"].value_counts().reindex(LIKERT_ORDER, fill_value=0)
        table = pd.DataFrame(
            {
                "score": LIKERT_ORDER,
                "label": [LIKERT_LABELS[score] for score in LIKERT_ORDER],
                "count": counts.to_numpy(),
            }
        )
        table["share"] = table["count"] / table["count"].sum()
        return table

    preference_likert = likert_counts(h2_data)
    return LIKERT_COLOURS, LIKERT_LABELS, LIKERT_ORDER, preference_likert


@app.cell
def _(FIGURES, LIKERT_COLOURS, preference_likert, np, plt):
    def plot_likert_overall(table):
        """Overall signed preference distribution as a horizontal bar chart."""
        y = np.arange(len(table))
        fig, ax = plt.subplots(figsize=(8.5, 4.0))
        ax.barh(
            y,
            table["count"],
            color=[LIKERT_COLOURS[score] for score in table["score"]],
            alpha=0.9,
        )
        for position, row in zip(y, table.itertuples()):
            ax.text(
                row.count + max(table["count"]) * 0.015,
                position,
                f"{row.count:,}  ({row.share:.1%})",
                va="center",
                fontsize=8,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(table["label"])
        ax.invert_yaxis()
        ax.set_xlim(0, max(table["count"]) * 1.28)
        ax.set_xlabel("Number of ratings")
        ax.set_title(
            f"Overall distribution of preference ratings "
            f"(N = {int(table['count'].sum()):,})"
        )
        ax.grid(axis="x", alpha=0.3, lw=0.5)
        fig.tight_layout()
        fig.savefig(
            FIGURES / "survey-preference-likert-overall.png",
            dpi=200,
            bbox_inches="tight",
        )
        return fig

    plot_likert_overall(preference_likert)
    return


@app.cell
def _(LIKERT_ORDER, h2_data, np, pd):
    def dilemma_likert_table(data):
        """Per-dilemma shares of each rating, plus decisiveness measures.

        ``decided`` is 1 minus the "about the same" share, i.e. the share of
        ratings that preferred one model. ``strong`` is the share that chose
        the strongest option in either direction (|score| = 2). Rows are
        sorted by ``decided`` so the most decisive dilemma is first.
        """
        rows = []
        for (dilemma, situation), group in data.groupby(
            ["dilemma", "situation"], sort=False
        ):
            counts = (
                group["ft_score"].value_counts().reindex(LIKERT_ORDER, fill_value=0)
            )
            n = len(group)
            row = {
                "dilemma": dilemma,
                "situation": situation,
                "n": n,
                "mean": group["ft_score"].mean(),
                "decided": 1 - counts.get(0, 0) / n,
                "strong": (counts.get(-2, 0) + counts.get(2, 0)) / n,
            }
            for score in LIKERT_ORDER:
                row[f"share_{score}"] = counts.get(score, 0) / n
            rows.append(row)
        return pd.DataFrame(rows).sort_values(["strong", "decided"], ascending=False)

    preference_likert_by_dilemma = dilemma_likert_table(h2_data)
    return (preference_likert_by_dilemma,)


@app.cell
def _(
    FIGURES,
    LIKERT_COLOURS,
    LIKERT_LABELS,
    LIKERT_ORDER,
    np,
    plt,
    preference_likert_by_dilemma,
):
    def plot_likert_by_dilemma(table):
        """100% stacked preference distribution, sorted by strong share."""
        y = np.arange(len(table))
        fig, ax = plt.subplots(figsize=(10.5, 0.34 * len(table) + 2.2))
        left = np.zeros(len(table))
        for score in LIKERT_ORDER:
            values = 100 * table[f"share_{score}"].to_numpy()
            ax.barh(
                y,
                values,
                left=left,
                color=LIKERT_COLOURS[score],
                label=LIKERT_LABELS[score],
            )
            left += values
        for position, row in zip(y, table.itertuples()):
            ax.text(
                101.5,
                position,
                f"{100 * row.strong:.0f}% strong  ·  mean {row.mean:+.2f}",
                va="center",
                fontsize=7,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(table["situation"], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, 132)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_xlabel("Share of ratings (%)")
        ax.set_title(
            "Preference rating distribution by dilemma "
            "(sorted by strong-preference share)"
        )
        ax.legend(
            ncol=3,
            fontsize=7,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
        )
        ax.grid(axis="x", alpha=0.3, lw=0.5)
        fig.tight_layout()
        fig.savefig(
            FIGURES / "survey-preference-likert-by-dilemma.png",
            dpi=200,
            bbox_inches="tight",
        )
        return fig

    plot_likert_by_dilemma(preference_likert_by_dilemma)
    return


@app.cell
def _(FIGURES, np, plt, preference_likert_by_dilemma):
    def plot_decisiveness(table):
        """Decided share and strong-preference share per dilemma, with CIs."""
        y = np.arange(len(table))
        n = table["n"].to_numpy(float)
        overall_decided = float((table["decided"] * n).sum() / n.sum())
        overall_strong = float((table["strong"] * n).sum() / n.sum())
        fig, ax = plt.subplots(figsize=(8.5, 0.34 * len(table) + 2.0))
        for column, colour, marker, offset, label in (
            ("decided", "#4c72b0", "o", 0.14, "Decided (not 'about the same')"),
            ("strong", "#c44e52", "s", -0.14, "Strongly prefer either model"),
        ):
            values = table[column].to_numpy(float)
            half = 1.96 * np.sqrt(values * (1 - values) / n)
            ax.errorbar(
                values,
                y + offset,
                xerr=half,
                fmt=marker,
                color=colour,
                ecolor=colour,
                alpha=0.85,
                capsize=2,
                ms=4,
                label=label,
            )
        ax.axvline(
            overall_decided,
            color="#4c72b0",
            ls="--",
            lw=1,
            label=f"Overall decided {overall_decided:.1%}",
        )
        ax.axvline(
            overall_strong,
            color="#c44e52",
            ls=":",
            lw=1,
            label=f"Overall strong {overall_strong:.1%}",
        )
        ax.set_yticks(y)
        ax.set_yticklabels(table["situation"], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Share of ratings")
        ax.set_title("How decisive each dilemma was (sorted by strong share)")
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        ax.grid(axis="x", alpha=0.3, lw=0.5)
        fig.tight_layout()
        fig.savefig(
            FIGURES / "survey-dilemma-decisiveness.png",
            dpi=200,
            bbox_inches="tight",
        )
        return fig

    plot_decisiveness(preference_likert_by_dilemma)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Preference by demographics

    Does the public's preference between the base and fine-tuned model change
    with who the respondent is? The outcome is the same signed 5-point
    preference as H2 (positive = fine-tune preferred). The table and forest
    plot below come from **one joint linear mixed model** with respondent and
    dilemma random effects, an order control, and every Section A demographic
    item (age, gender, ethnicity, employment, region, income, occupation and
    education) plus the three AI-view items entered together. Region is
    grouped into Auckland, Wellington, the rest of the North Island and the
    South Island because several regions have fewer than ten respondents;
    occupation keeps the ANZSCO major groups, with machinery operators and
    "never had a job" pooled into "Other". A coefficient is therefore the
    adjusted difference in preference points between that level and the
    variable's reference level, holding the other variables fixed. These are
    associations, not causal effects: the AI-view items are measured after
    the comparisons.

    The raw panels show the unadjusted group means, so the pattern is
    readable without the model. The model is unweighted, matching the
    pre-registered H2 analysis; the survey's age x gender x region quotas
    already make the sample close to the census on those variables. The joint
    model is fitted on the complete-case sample; respondents missing any
    modelled item (including "prefer not to say") are excluded, and the table
    reports the number of respondents and ratings behind each level. The
    forest plot prints respondent n beside every level, the reference level
    and its n in each header, and marks Benjamini-Hochberg q < 0.05 in red
    with two stars; orange with one star marks a nominally p < 0.05 effect
    that does not survive the correction.
    """)
    return


@app.cell
def _(
    AGE_ORDER,
    AIUSE_ORDER,
    EDUCATION_ORDER,
    EMPLOYMENT_ORDER,
    ETHNICITY_ORDER,
    GENDER_ORDER,
    INCOME_ORDER,
    OCCUPATION_ORDER,
    OUTLOOK_ORDER,
    REGION_ORDER,
    RISK_ORDER,
):
    # (label, column, category order, reference level) for every variable in
    # the preference model and the raw breakdown panels. The reference is the
    # level a coefficient is compared with. Region and occupation are grouped
    # in the demographics cell; the order follows Section A of the survey.
    PREFERENCE_VARIABLES = [
        ("Age", "age_band", AGE_ORDER, "18-24"),
        ("Gender", "gender_label", GENDER_ORDER, "Male"),
        (
            "Ethnicity",
            "ethnicity_group",
            ETHNICITY_ORDER,
            "New Zealand European / Pākehā",
        ),
        ("Employment", "employment_group", EMPLOYMENT_ORDER, "Employed full-time"),
        ("Region", "region_group", REGION_ORDER, "Auckland"),
        ("Income", "income_band", INCOME_ORDER, "Low"),
        ("Occupation", "occupation_group", OCCUPATION_ORDER, "Professionals"),
        ("Education", "education_group", EDUCATION_ORDER, "Secondary"),
        ("AI use", "C8-1", AIUSE_ORDER, "Never"),
        ("AI outlook", "C1-1", OUTLOOK_ORDER, "Neither negative nor positive"),
        ("AI risk", "Q943", RISK_ORDER, "Neither agree nor disagree"),
    ]
    return (PREFERENCE_VARIABLES,)


@app.cell
def _(
    FIGURES,
    NormalDist,
    PREFERENCE_VARIABLES,
    analysis_frame,
    chi2,
    fit_lmm,
    np,
    pd,
    plt,
    survey_bh_qvalues,
    write_latex_table,
):
    def preference_model(frame, variables):
        """Joint LMM and the level-by-level coefficient table.

        The complete-case sample is the set of ratings with every modelled
        variable present. Each non-reference level becomes one dummy column;
        the coefficient is its adjusted difference from the reference level
        in preference points.
        """
        columns = [spec[1] for spec in variables]
        complete = frame.dropna(subset=columns).copy()
        covariates, lookup = [], []
        for label, column, order, reference in variables:
            levels = [level for level in order if level in set(complete[column])]
            if reference not in levels:
                raise SystemExit(f"Reference {reference!r} missing from {column}")
            for level in levels:
                if level == reference:
                    continue
                name = f"{label}: {level}"
                covariates.append((name, (complete[column] == level).to_numpy(float)))
                lookup.append((label, column, level, reference, name))
        fit = fit_lmm(complete, covariates)
        position = {name: i for i, name in enumerate(fit["names"])}
        rows = []
        for label, column, order, reference in variables:
            levels = [level for level in order if level in set(complete[column])]
            for level in levels:
                mask = complete[column] == level
                values = complete.loc[mask, "ft_score"]
                respondents = int(complete.loc[mask, "_recordId"].nunique())
                if level == reference:
                    rows.append(
                        {
                            "Variable": label,
                            "Level": f"{level} (ref)",
                            "Respondents": respondents,
                            "Ratings": int(mask.sum()),
                            "Mean": float(values.mean()),
                            "Difference": 0.0,
                            "CI": "--",
                            "p": np.nan,
                            "q": np.nan,
                        }
                    )
                    continue
                name = f"{label}: {level}"
                index = position[name]
                beta = float(fit["beta"][index])
                se = float(fit["se"][index])
                rows.append(
                    {
                        "Variable": label,
                        "Level": level,
                        "Respondents": respondents,
                        "Ratings": int(mask.sum()),
                        "Mean": float(values.mean()),
                        "Difference": beta,
                        "CI": (
                            f"[{beta - NormalDist().inv_cdf(0.975) * se:+.2f}, "
                            f"{beta + NormalDist().inv_cdf(0.975) * se:+.2f}]"
                        ),
                        "p": 2 * NormalDist().cdf(-abs(beta / se)),
                        "q": np.nan,
                    }
                )
        table = pd.DataFrame(rows)
        tested = table["p"].notna()
        table.loc[tested, "q"] = survey_bh_qvalues(table.loc[tested, "p"].to_numpy())
        return complete, fit, table

    def variable_tests(fit, variables):
        """Wald test that all levels of a variable are jointly zero."""
        rows = []
        for label, column, order, reference in variables:
            indices = [
                i
                for i, name in enumerate(fit["names"])
                if name.startswith(f"{label}: ")
            ]
            beta = fit["beta"][indices]
            covariance = fit["cov"][np.ix_(indices, indices)]
            statistic = float(beta @ np.linalg.pinv(covariance) @ beta)
            df = len(indices)
            rows.append(
                {
                    "Variable": label,
                    "df": df,
                    "chi2": statistic,
                    "p": float(chi2.sf(statistic, df)),
                }
            )
        table = pd.DataFrame(rows)
        table["q"] = survey_bh_qvalues(table["p"].to_numpy())
        return table

    def format_p(value):
        return "<0.001" if value < 0.001 else f"{value:.3f}"

    preference_complete, preference_fit, preference_effects = preference_model(
        analysis_frame, PREFERENCE_VARIABLES
    )
    preference_tests = variable_tests(preference_fit, PREFERENCE_VARIABLES)
    preference_effects_display = preference_effects.copy()
    preference_effects_display["Respondents"] = preference_effects_display[
        "Respondents"
    ].map(lambda value: f"{value:,}")
    preference_effects_display["Ratings"] = preference_effects_display["Ratings"].map(
        lambda value: f"{value:,}"
    )
    preference_effects_display["Mean"] = preference_effects_display["Mean"].map(
        lambda value: f"{value:+.2f}"
    )
    preference_effects_display["Difference"] = preference_effects_display[
        "Difference"
    ].map(lambda value: f"{value:+.2f}")
    preference_effects_display["p"] = preference_effects_display["p"].map(
        lambda value: "--" if pd.isna(value) else format_p(value)
    )
    preference_effects_display["q"] = preference_effects_display["q"].map(
        lambda value: "--" if pd.isna(value) else format_p(value)
    )
    preference_tests_display = preference_tests.copy()
    preference_tests_display["chi2"] = preference_tests_display["chi2"].map(
        lambda value: f"{value:.2f}"
    )
    preference_tests_display["p"] = preference_tests_display["p"].map(format_p)
    preference_tests_display["q"] = preference_tests_display["q"].map(format_p)

    def tex_table(table, variables, filename):
        """Write the level table with the variable name on its first row only."""
        subset = table[table["Variable"].isin(variables)].copy()
        index = []
        previous = None
        for variable in subset["Variable"]:
            index.append(variable if variable != previous else "")
            previous = variable
        out = subset.drop(columns="Variable")
        out.index = pd.Index(index, name=None)
        write_latex_table(
            out,
            "{}",
            len(out.columns),
            filename,
            stub_label="Variable",
            highlight=False,
        )

    ai_labels = [spec[0] for spec in PREFERENCE_VARIABLES if spec[0].startswith("AI ")]
    demographic_labels = [
        spec[0] for spec in PREFERENCE_VARIABLES if spec[0] not in ai_labels
    ]
    tex_table(
        preference_effects_display,
        demographic_labels,
        "survey-preference-demographics.tex",
    )
    tex_table(
        preference_effects_display,
        ai_labels,
        "survey-preference-ai-views.tex",
    )
    preference_tests_out = preference_tests_display.set_index("Variable")
    preference_tests_out.index.name = None
    write_latex_table(
        preference_tests_out,
        "{}",
        len(preference_tests_out.columns),
        "survey-preference-variable-tests.tex",
        stub_label="",
        highlight=False,
    )

    def plot_effects(table, tests):
        """Forest plot of adjusted differences, grouped by variable.

        Each level label carries the number of respondents behind it, and the
        variable header carries the reference level and its n. Red levels (and
        a double star) mark Benjamini-Hochberg q < 0.05; orange levels (single
        star) are nominally p < 0.05 but do not survive the correction. The
        header reports the joint Wald p and q so a variable with an
        interesting overall pattern is visible before reading every level.
        """
        rows = table[table["p"].notna()].copy()
        labels = []
        positions = []
        y = 0.0
        for label, _, _, _ in PREFERENCE_VARIABLES:
            group = rows[rows["Variable"] == label]
            if group.empty:
                continue
            test = tests.loc[tests["Variable"] == label].iloc[0]
            p_value, q_value = test["p"], test["q"]
            reference_row = table[
                (table["Variable"] == label) & table["Level"].str.endswith("(ref)")
            ].iloc[0]
            reference_level = reference_row["Level"].removesuffix(" (ref)")
            reference_n = reference_row["Respondents"]
            labels.append(
                (
                    y,
                    f"{label}  (p = {format_p(p_value)}, q = {format_p(q_value)}; "
                    f"ref {reference_level}, n={reference_n:,})",
                    q_value < 0.05,
                )
            )
            y -= 1.0
            for _, row in group.iterrows():
                positions.append((y, row))
                y -= 1.0
            y -= 0.7
        fig, ax = plt.subplots(figsize=(12.5, 0.30 * len(positions) + 2.4))
        for position, row in positions:
            beta = row["Difference"]
            low, high = [float(part) for part in row["CI"].strip("[]").split(",")]
            corrected = row["q"] < 0.05
            nominal = (row["p"] < 0.05) and not corrected
            colour = "tab:red" if corrected else ("tab:orange" if nominal else "0.35")
            marker = " **" if corrected else (" *" if nominal else "")
            ax.plot([low, high], [position, position], color=colour, lw=1.4)
            ax.plot(beta, position, "o", color=colour, ms=5)
            ax.text(
                high + 0.02,
                position,
                f"{beta:+.2f}{marker}",
                va="center",
                fontsize=7,
                color=colour,
            )
        for position, label, significant in labels:
            ax.text(
                -0.02,
                position,
                label,
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="tab:red" if significant else "black",
            )
        ax.axvline(0, color="black", lw=0.9)
        ax.set_yticks([position for position, _ in positions])
        ax.set_yticklabels(
            [f"{row['Level']}  (n={row['Respondents']})" for _, row in positions],
            fontsize=8,
        )
        ax.set_ylim(min(position for position, _ in positions) - 0.8, 0.8)
        low = min(float(row["CI"].strip("[]").split(",")[0]) for _, row in positions)
        high = max(float(row["CI"].strip("[]").split(",")[1]) for _, row in positions)
        span = high - low
        ax.set_xlim(low - 0.08 * span, high + 0.22 * span)
        ax.set_xlabel("Adjusted difference in preference points (vs reference)")
        ax.set_title(
            "Preference by demographics and AI views "
            "(adjusted LMM; n = respondents per level)"
        )
        ax.grid(axis="x", alpha=0.3, lw=0.5)
        ax.plot(
            [],
            [],
            color="tab:red",
            lw=1.4,
            marker="o",
            ms=4,
            label="q < 0.05 (**)",
        )
        ax.plot(
            [],
            [],
            color="tab:orange",
            lw=1.4,
            marker="o",
            ms=4,
            label="p < 0.05, q ≥ 0.05 (*)",
        )
        ax.plot(
            [],
            [],
            color="0.35",
            lw=1.4,
            marker="o",
            ms=4,
            label="p ≥ 0.05",
        )
        ax.legend(
            fontsize=8,
            loc="lower right",
            frameon=True,
            title="Level-level test",
            title_fontsize=8,
        )
        fig.tight_layout()
        fig.savefig(
            FIGURES / "survey-preference-effects.png",
            dpi=200,
            bbox_inches="tight",
        )
        return fig

    preference_forest = plot_effects(preference_effects, preference_tests)

    preference_effects
    return (
        preference_complete,
        preference_effects,
        preference_forest,
        preference_fit,
        preference_tests,
    )


@app.cell
def _(mo, preference_complete):
    mo.md(
        f"Complete-case model sample: **{preference_complete['_recordId'].nunique():,} "
        f"respondents**, **{len(preference_complete):,} ratings**. "
        "The table's `Respondents` and `Ratings` columns are the number of "
        "people and preference ratings behind each level; the forest plot "
        "labels show the respondent count."
    )
    return


@app.cell
def _(
    FIGURES,
    PREFERENCE_VARIABLES,
    mean_ci,
    np,
    plt,
    respondents,
):
    def plot_preference_groups(frame, variables, filename, ncols, figsize):
        """Raw (unadjusted) respondent-mean preference with 95% CI.

        The unit is the respondent (one mean of their five ratings), so the
        n in each label is a count of people, not ratings. The model below
        uses the rating-level data with respondent random effects.
        """
        overall = frame["preference_mean"].mean()
        nrows = int(np.ceil(len(variables) / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize, squeeze=False, sharex=True
        )
        for ax, (label, column, order, _) in zip(axes.flat, variables):
            rows = []
            for level in order:
                values = frame.loc[frame[column] == level, "preference_mean"]
                mean, low, high, n = mean_ci(values)
                if n:
                    rows.append((level, mean, low, high, n))
            y = np.arange(len(rows))
            means = np.array([row[1] for row in rows])
            low = np.array([row[2] for row in rows])
            high = np.array([row[3] for row in rows])
            ax.errorbar(
                means,
                y,
                xerr=[means - low, high - means],
                fmt="o",
                color="tab:blue",
                ecolor="0.5",
                capsize=2,
                ms=4,
            )
            ax.axvline(0, color="0.3", lw=0.8)
            ax.axvline(overall, color="black", ls="--", lw=1)
            ax.set_yticks(y)
            ax.set_yticklabels([f"{row[0]} (n={row[4]})" for row in rows], fontsize=7)
            ax.set_title(label, fontsize=9)
            ax.grid(axis="x", alpha=0.3, lw=0.5)
            ax.invert_yaxis()
        for ax in axes.flat[len(variables) :]:
            ax.set_visible(False)
        for ax in axes[-1, :]:
            ax.set_xlabel("Mean preference (positive = fine-tune)")
        fig.suptitle(
            "Raw respondent-mean preference by group (dashed line = overall mean)",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(FIGURES / filename, dpi=200, bbox_inches="tight")
        return fig

    demographic_variables = [
        variable
        for variable in PREFERENCE_VARIABLES
        if not variable[0].startswith("AI ")
    ]
    ai_variables = [
        variable for variable in PREFERENCE_VARIABLES if variable[0].startswith("AI ")
    ]
    preference_demographics_figure = plot_preference_groups(
        respondents,
        demographic_variables,
        "survey-preference-demographics.png",
        3,
        (13, 13),
    )
    preference_ai_figure = plot_preference_groups(
        respondents,
        ai_variables,
        "survey-preference-ai-views.png",
        3,
        (13, 4.8),
    )
    return (preference_ai_figure, preference_demographics_figure)


@app.cell
def _(mo):
    mo.md("""
    ## Policy analysis

    Primary outcome: unweighted probability of inclusion in a respondent's
    top three. Intervals use 10,000 respondent-level bootstraps.
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
    ETHNICITY_CENSUS,
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

    # Ethnicity is a multi-select item: compare the share of respondents who
    # ticked each group with the census total-response share. The denominator
    # is respondents who ticked at least one group, so "prefer not to say"
    # does not inflate the shares.
    ethnicity_columns = [
        ("Z3-1_1", "New Zealand European / Pākehā"),
        ("Z3-1_2", "Māori"),
        ("Z3-1_3", "Pacific peoples"),
        ("Z3-1_4", "Asian"),
        ("Z3-1_5", "Middle Eastern / Latin American / African"),
        ("Z3-1_6", "Other"),
    ]
    ethnicity_answered = (
        policy_sample[[column for column, _ in ethnicity_columns]].notna().any(axis=1)
    )
    ethnicity_total = int(ethnicity_answered.sum())
    ethnicity_pct = {
        label: (
            100
            * policy_sample.loc[ethnicity_answered, column].eq(label).sum()
            / ethnicity_total
        )
        for column, label in ethnicity_columns
    }

    groups = [
        ("Age", AGE_CENSUS, pct_map(policy_sample["age_band"])),
        ("Gender", GENDER_CENSUS, pct_map(policy_sample["gender_label"])),
        (
            "Ethnicity (multi-select)",
            ETHNICITY_CENSUS,
            ethnicity_pct,
        ),
        (
            "Employment",
            EMPLOYMENT_CENSUS,
            pct_map(policy_sample["Z4-1"].map(EMPLOYMENT_MAP)),
        ),
        (
            "Region",
            {k: 100 * v / REGION_TOTAL for k, v in REGION_POP.items()},
            pct_map(policy_sample["region_label"]),
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

    fig, ax = plt.subplots(figsize=(11.5, 20))
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
    fig.text(
        0.01,
        0.002,
        "Ethnicity is multi-select, so its shares sum to more than 100%; "
        "the census bars are the 2023 total-response ethnic groups.",
        fontsize=8,
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.015, 1, 1))
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
    PL_BOOTSTRAP_REPS = 1_000
    PL_SEED = 20261104
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
    # Stats NZ 2023 Census total-response ethnic groups. The census asks a
    # multi-select question, so the shares sum to more than 100%; the survey
    # comparison uses the same multi-select denominator. "New Zealand
    # European / Pākehā" is compared with the census European total-response
    # group (67.8%), which includes NZ European and other European.
    # https://www.stats.govt.nz/information-releases/2023-census-population-counts-by-ethnic-group-age-and-maori-descent-and-dwelling-counts
    ETHNICITY_CENSUS = {
        "New Zealand European / Pākehā": 67.8,
        "Māori": 17.8,
        "Pacific peoples": 8.9,
        "Asian": 17.3,
        "Middle Eastern / Latin American / African": 1.9,
        "Other": 3.0,
    }
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
        ETHNICITY_CENSUS,
        GENDER_CENSUS,
        INCOME_CENSUS,
        PL_BOOTSTRAP_REPS,
        PL_SEED,
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
    ### Policy support shares
    """)
    return


@app.cell
def _(POLICY_COLS, POLICY_TITLES, np, policy_sample):
    def policy_matrix(responses):
        for i, column in enumerate(POLICY_COLS):
            values = responses[column].dropna().unique()
            if any(not str(v).startswith(POLICY_TITLES[i]) for v in values):
                raise SystemExit(f"Policy wording mismatch in {column}")
        return responses[POLICY_COLS].notna().to_numpy(int)

    policy_Y = policy_matrix(policy_sample)
    policy_shares = policy_Y.mean(axis=0)
    return policy_Y, policy_shares


@app.cell
def _(BOOTSTRAP_REPS, POLICY_SEED, np, policy_Y):
    def bootstrap_shares(Y):
        rng = np.random.default_rng(POLICY_SEED)
        draw = rng.integers(0, len(Y), size=(BOOTSTRAP_REPS, len(Y)))
        return np.column_stack([Y[draw, k].mean(axis=1) for k in range(Y.shape[1])])

    policy_boot = bootstrap_shares(policy_Y)
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
def _(mo, policy_Y, policy_omnibus_p, policy_tiers, policy_wald):
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
        **N = {len(policy_Y):,} valid respondents**.

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
        ax.set_xlabel("Percentage selecting policy in top three")
        ax.set_title("Policy support")
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
    ### Rank analysis (descriptive check)

    A simple Borda-style summary of the rank data: the first-choice share and
    the mean Borda score (4 - rank) for each policy. The Plackett–Luce model
    below is the model-based version that uses the same ranking information
    with a proper likelihood.
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
):
    def rank_summary(responses, Y):
        ranks = responses[RANK_COLS].apply(pd.to_numeric, errors="coerce").to_numpy()
        selected = np.where(Y == 1, ranks, np.nan)
        valid = np.all(np.sort(selected, axis=1)[:, :3] == [1, 2, 3], axis=1)
        if not valid.any():
            return pd.DataFrame(), 0
        subset, subset_ranks = Y[valid], ranks[valid]
        borda = np.where(subset == 1, 4 - subset_ranks, 0)
        first = np.where((subset == 1) & (subset_ranks == 1), 1, 0)
        first_share = first.mean(axis=0)
        borda_mean = borda.mean(axis=0)

        rng = np.random.default_rng(POLICY_SEED)
        draw = rng.integers(0, len(subset), size=(BOOTSTRAP_REPS, len(subset)))
        first_boot = np.column_stack([first[draw, k].mean(axis=1) for k in range(9)])
        borda_boot = np.column_stack([borda[draw, k].mean(axis=1) for k in range(9)])
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

    rank_table, ranked_n = rank_summary(policy_sample, policy_Y)
    mo.accordion({f"C4-4 (n = {ranked_n})": mo.ui.table(rank_table, selection=None)})
    return


@app.cell
def _(mo):
    mo.md("""
    ### Where each policy sits in the ranking

    The simple outcome counts a policy once when it is anywhere in the top
    three. That hides whether a policy is a broad second/third choice or a
    polarising first choice. This figure splits the same top-three share into
    the share of respondents who ranked the policy first, second and
    third, plus a grey "selected, not ranked" segment for respondents who
    picked the policy but did not give a usable 1-2-3 ranking. The segments
    therefore add up to the policy's simple top-three share.
    """)
    return


@app.cell
def _(
    FIGURES,
    POLICY_COLS,
    POLICY_SHORT,
    RANK_COLS,
    np,
    pd,
    plt,
    policy_sample,
):
    def rank_composition(responses):
        """First/second/third and selected-not-ranked shares."""
        ranks = responses[RANK_COLS].apply(pd.to_numeric, errors="coerce").to_numpy()
        selected = responses[POLICY_COLS].notna().to_numpy()
        n = len(responses)
        rows = []
        for k, short in enumerate(POLICY_SHORT):
            values = ranks[:, k]
            first = (values == 1).sum() / n
            second = (values == 2).sum() / n
            third = (values == 3).sum() / n
            chosen = selected[:, k].sum() / n
            rows.append(
                {
                    "Policy": f"P{k + 1}  {short}",
                    "First": first,
                    "Second": second,
                    "Third": third,
                    "Selected, not ranked": max(chosen - first - second - third, 0.0),
                    "Top three": chosen,
                }
            )
        return pd.DataFrame(rows).sort_values("Top three")

    policy_rank_composition = rank_composition(policy_sample)

    def plot_rank_composition(table):
        y = np.arange(len(table))
        fig, ax = plt.subplots(figsize=(9, 5.2))
        left = np.zeros(len(table))
        for column, colour, label in (
            ("First", "#1b4f72", "Ranked 1st"),
            ("Second", "#2e86c1", "Ranked 2nd"),
            ("Third", "#85c1e9", "Ranked 3rd"),
            ("Selected, not ranked", "#b0b0b0", "Selected, not ranked"),
        ):
            values = 100 * table[column].to_numpy()
            ax.barh(y, values, left=left, color=colour, label=label)
            left += values
        for position, total in zip(y, 100 * table["Top three"].to_numpy()):
            ax.text(total + 0.8, position, f"{total:.1f}%", va="center", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(table["Policy"])
        ax.set_xlabel("Share of respondents (%)")
        ax.set_xlim(0, max(left) * 1.18)
        ax.set_title("Policy ranking: first, second and third choices")
        ax.legend(frameon=False, loc="lower right")
        ax.grid(axis="x", alpha=0.3, lw=0.5)
        fig.tight_layout()
        fig.savefig(
            FIGURES / "survey-policy-rank-composition.png",
            dpi=200,
            bbox_inches="tight",
        )
        return fig

    plot_rank_composition(policy_rank_composition)
    return (policy_rank_composition,)


@app.cell
def _(mo):
    mo.md("""
    ### Plackett–Luce ranking model

    The second method uses the full order of the three ranked policies, not
    just whether a policy was selected. Each respondent is treated as making
    three sequential choices: first from all nine policies, then from the
    remaining eight, then from the remaining seven. The probability of a
    policy being picked from a set is proportional to its **worth**
    parameter, so the likelihood is the product of three conditional logits.
    The six unranked policies are treated as censored: we know they were not
    in the top three, but not their order. This is the rank-ordered logit
    (Plackett–Luce) model for a partial ranking (Plackett 1975; Luce 1959;
    Beggs, Cardell and Hausman 1981); see Train (2009) for the econometric
    treatment.

    The worths are estimated by maximum likelihood and normalised to sum to
    one, so a worth can be read as the
    model-implied probability that the policy is chosen first from the full
    set. Intervals are percentile bootstraps over respondents (1,000
    replicates); the last policy is the reference in the log-worth
    parameterisation, which does not affect the normalised worths.
    """)
    return


@app.cell
def _(
    FIGURES,
    PL_BOOTSTRAP_REPS,
    PL_SEED,
    POLICY_SHORT,
    RANK_COLS,
    logsumexp,
    minimize,
    np,
    pd,
    plt,
    policy_high,
    policy_low,
    policy_sample,
    policy_shares,
    policy_tier_of,
    policy_Y,
    write_latex_table,
):
    def fit_plackett_luce(Y, ranks):
        """Plackett–Luce fit for a censored top-three ranking.

        ``Y`` is the n x k selection matrix and ``ranks`` the n x k matrix with
        1/2/3 on the selected policies and NaN elsewhere. The log-likelihood
        and its gradient are analytic, so the 1,000-replicate bootstrap is
        cheap.
        """
        n, k = Y.shape
        order = np.argsort(np.where(Y == 1, ranks, np.inf), axis=1)[:, :3]
        a1, a2, a3 = order[:, 0], order[:, 1], order[:, 2]
        idx = np.arange(n)
        selected = Y.astype(float)

        def negll_grad(theta):
            full = np.r_[theta, 0.0]
            p = np.exp(full - logsumexp(full))
            l0 = logsumexp(full)
            l1 = np.array([logsumexp(np.delete(full, j)) for j in range(k)])
            l2 = np.array(
                [
                    [logsumexp(np.delete(full, [j, m])) for m in range(k)]
                    for j in range(k)
                ]
            )
            ll = np.sum(full[a1] + full[a2] + full[a3] - l0 - l1[a1] - l2[a1, a2])
            gradient = selected.sum(0)
            gradient -= n * p
            q2 = p[None, :] / (1 - p[a1])[:, None]
            q2[idx, a1] = 0.0
            gradient -= q2.sum(0)
            q3 = p[None, :] / (1 - p[a1] - p[a2])[:, None]
            q3[idx, a1] = 0.0
            q3[idx, a2] = 0.0
            gradient -= q3.sum(0)
            return -ll, -gradient[:-1]

        fit = minimize(
            negll_grad,
            np.zeros(k - 1),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
        )
        full = np.r_[fit.x, 0.0]
        worth = np.exp(full - logsumexp(full))
        return worth, fit

    def plackett_luce_summary(responses, Y):
        ranks = responses[RANK_COLS].apply(pd.to_numeric, errors="coerce").to_numpy()
        selected = np.where(Y == 1, ranks, np.nan)
        valid = np.all(np.sort(selected, axis=1)[:, :3] == [1, 2, 3], axis=1)
        subset_y, subset_ranks = Y[valid], ranks[valid]
        worth, fit = fit_plackett_luce(subset_y, subset_ranks)
        rng = np.random.default_rng(PL_SEED)
        draws = rng.integers(0, len(subset_y), size=(PL_BOOTSTRAP_REPS, len(subset_y)))
        boot = np.empty((PL_BOOTSTRAP_REPS, Y.shape[1]))
        for b, draw in enumerate(draws):
            boot[b], _ = fit_plackett_luce(subset_y[draw], subset_ranks[draw])
        low, high = np.quantile(boot, [0.025, 0.975], axis=0)
        rank_one = (np.argmax(boot, axis=1)[:, None] == np.arange(Y.shape[1])).mean(
            axis=0
        )
        return worth, low, high, rank_one, int(valid.sum()), fit

    (
        policy_worth,
        policy_worth_low,
        policy_worth_high,
        policy_worth_first,
        policy_ranked_n,
        policy_pl_fit,
    ) = plackett_luce_summary(policy_sample, policy_Y)

    def worth_table():
        order = np.argsort(-policy_worth)
        rows = []
        for k in order:
            rows.append(
                {
                    "Top-3 share (%)": f"{100 * policy_shares[k]:.1f}",
                    "PL worth (%)": f"{100 * policy_worth[k]:.1f}",
                    "95% CI": (
                        f"[{100 * policy_worth_low[k]:.1f}, "
                        f"{100 * policy_worth_high[k]:.1f}]"
                    ),
                    "P(top)": f"{policy_worth_first[k]:.2f}",
                    "Tier": policy_tier_of[k],
                }
            )
        table = pd.DataFrame(
            rows, index=[f"P{k + 1}  {POLICY_SHORT[k]}" for k in order]
        )
        table.index.name = None
        return table

    policy_worth_table = worth_table()
    write_latex_table(
        policy_worth_table,
        "{}",
        len(policy_worth_table.columns),
        "survey-policy-plackett-luce.tex",
        stub_label="",
        highlight=False,
    )

    def plot_policy_methods():
        order = np.argsort(-policy_shares)
        y = np.arange(len(order))
        fig, axes = plt.subplots(
            1, 2, figsize=(12, 5.6), sharey=True, gridspec_kw={"wspace": 0.08}
        )
        panels = (
            (
                axes[0],
                100 * policy_shares[order],
                100 * (policy_shares[order] - policy_low[order]),
                100 * (policy_high[order] - policy_shares[order]),
                "Simple: share selected in top three",
                "#4c72b0",
            ),
            (
                axes[1],
                100 * policy_worth[order],
                100 * (policy_worth[order] - policy_worth_low[order]),
                100 * (policy_worth_high[order] - policy_worth[order]),
                "Plackett–Luce worth",
                "#c44e52",
            ),
        )
        for ax, values, low, high, title, colour in panels:
            ax.errorbar(
                values,
                y,
                xerr=[low, high],
                fmt="o",
                color=colour,
                ecolor="0.45",
                capsize=3,
                ms=5,
            )
            for position, (value, upper) in enumerate(zip(values, high)):
                ax.text(
                    value + upper + 0.8,
                    position,
                    f"{value:.1f}",
                    va="center",
                    fontsize=7,
                )
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Percent")
            ax.grid(axis="x", alpha=0.3, lw=0.5)
            ax.set_xlim(0, max(values + high) * 1.18)
        axes[0].set_yticks(y)
        axes[0].set_yticklabels([f"P{k + 1}  {POLICY_SHORT[k]}" for k in order])
        axes[0].invert_yaxis()
        fig.suptitle(
            "Policy preference: simple top-three share vs rank-based Plackett–Luce",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(
            FIGURES / "survey-policy-methods.png",
            dpi=200,
            bbox_inches="tight",
        )
        return fig

    plot_policy_methods()
    policy_worth_table
    return (
        policy_pl_fit,
        policy_ranked_n,
        policy_worth,
        policy_worth_first,
        policy_worth_high,
        policy_worth_low,
        policy_worth_table,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Views on AI by demographics

    Four 1-5 outcomes are broken down by the same demographic groups:

    - **AI use frequency** (1 = never, 5 = daily).
    - **AI outlook** (1 = very negative, 5 = very positive).
    - **AI risk belief** (1 = strongly disagree, 5 = strongly agree that
      advanced AI is a serious risk).
    - **AI support in the dilemmas** - the respondent's mean agreement that
      an AI agent should be used in the work situations they saw.

    Each point is an unweighted sample mean with a normal 95% interval; the
    dashed line is the overall mean. The panels show the descriptive pattern,
    not a causal effect. Region is omitted because most respondents are in a
    handful of large regions; the representativeness figure above covers it.
    The accompanying table gives Benjamini--Hochberg q-values from one-way
    ANOVAs across the groups for each outcome; it is a descriptive screen for
    which demographic splits matter, not a causal test.
    """)
    return


@app.cell
def _(
    AGE_ORDER,
    EDUCATION_ORDER,
    EMPLOYMENT_ORDER,
    ETHNICITY_ORDER,
    FIGURES,
    GENDER_ORDER,
    INCOME_ORDER,
    mean_ci,
    np,
    plt,
    respondents,
):
    AI_VIEW_OUTCOMES = [
        (
            "ai_use_score",
            "AI use frequency",
            "Mean AI use frequency (1 = never, 5 = daily)",
            "survey-ai-use-demographics.png",
        ),
        (
            "ai_outlook_score",
            "AI outlook",
            "Mean AI outlook (1 = very negative, 5 = very positive)",
            "survey-ai-outlook-demographics.png",
        ),
        (
            "ai_risk_score",
            "AI risk belief",
            "Mean agreement that AI is a serious risk (1-5)",
            "survey-ai-risk-demographics.png",
        ),
        (
            "ai_support_mean",
            "AI support in the dilemmas",
            "Mean agreement that AI should be used (1-5)",
            "survey-ai-support-demographics.png",
        ),
    ]
    DEMOGRAPHIC_SECTIONS = [
        ("Age", "age_band", AGE_ORDER),
        ("Gender", "gender_label", GENDER_ORDER),
        ("Ethnicity", "ethnicity_group", ETHNICITY_ORDER),
        ("Employment", "employment_group", EMPLOYMENT_ORDER),
        ("Income", "income_band", INCOME_ORDER),
        ("Education", "education_group", EDUCATION_ORDER),
    ]

    def plot_ai_view(frame, outcome, title, xlabel, filename):
        """One AI-view outcome broken down into labelled demographic sections."""
        overall = frame[outcome].mean()
        rows, heads, separators = [], [], []
        y = 0.0
        for section_index, (label, column, order) in enumerate(DEMOGRAPHIC_SECTIONS):
            if section_index:
                separators.append(y + 0.45)
                y -= 0.9
            heads.append((y, label))
            y -= 0.8
            for level in order:
                values = frame.loc[frame[column] == level, outcome]
                mean, low, high, n = mean_ci(values)
                if not n:
                    continue
                rows.append((y, f"{level} (n={n})", mean, low, high))
                y -= 1.0
        fig, ax = plt.subplots(figsize=(9, 0.30 * len(rows) + 1.8))
        positions = [row[0] for row in rows]
        means = np.array([row[2] for row in rows])
        low = np.array([row[3] for row in rows])
        high = np.array([row[4] for row in rows])
        ax.errorbar(
            means,
            positions,
            xerr=[means - low, high - means],
            fmt="o",
            color="tab:blue",
            ecolor="0.5",
            capsize=2,
            ms=4,
        )
        ax.axvline(
            overall, color="black", ls="--", lw=1, label=f"Overall {overall:.2f}"
        )
        for heading_y, label in heads:
            ax.text(
                -0.02,
                heading_y,
                label,
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=9,
                fontweight="bold",
            )
        for separator_y in separators:
            ax.axhline(separator_y, color="0.8", lw=0.7)
        ax.set_yticks(positions)
        ax.set_yticklabels([row[1] for row in rows], fontsize=7)
        ax.set_xlim(1, 5)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.legend(frameon=False, loc="lower right")
        ax.grid(axis="x", alpha=0.3, lw=0.5)
        ax.tick_params(axis="y", length=0)
        fig.tight_layout()
        fig.savefig(FIGURES / filename, dpi=200, bbox_inches="tight")
        return fig

    ai_view_figures = [
        plot_ai_view(respondents, *outcome) for outcome in AI_VIEW_OUTCOMES
    ]
    return (AI_VIEW_OUTCOMES, DEMOGRAPHIC_SECTIONS, ai_view_figures)


@app.cell
def _(
    AI_VIEW_OUTCOMES,
    DEMOGRAPHIC_SECTIONS,
    FIGURES,
    f_oneway,
    np,
    pd,
    respondents,
    survey_bh_qvalues,
    write_latex_table,
):
    def ai_view_tests(frame, outcomes, sections):
        """One-way ANOVA p-values for each AI outcome x demographic variable.

        This is a descriptive screen: it asks whether the group means differ,
        not whether the demographic causes the difference. The q-values are
        Benjamini-Hochberg adjusted across the whole 6 x 4 table.
        """
        rows = []
        for label, column, order in sections:
            row = {"Variable": label}
            for outcome, title, _, _ in outcomes:
                groups = [
                    frame.loc[frame[column] == level, outcome].dropna()
                    for level in order
                ]
                groups = [group for group in groups if len(group) > 1]
                if len(groups) < 2:
                    row[title] = np.nan
                else:
                    _, p_value = f_oneway(*groups)
                    row[title] = float(p_value)
            rows.append(row)
        pvalues = pd.DataFrame(rows).set_index("Variable")
        flat = survey_bh_qvalues(pvalues.to_numpy().ravel())
        qvalues = pd.DataFrame(
            flat.reshape(pvalues.shape),
            index=pvalues.index,
            columns=pvalues.columns,
        )
        qvalues.index.name = None
        return qvalues

    ai_view_qvalues = ai_view_tests(respondents, AI_VIEW_OUTCOMES, DEMOGRAPHIC_SECTIONS)
    ai_view_qvalues.columns = ["AI use", "AI outlook", "AI risk", "AI support"]
    write_latex_table(
        ai_view_qvalues,
        "{:.3f}",
        len(ai_view_qvalues.columns),
        "survey-ai-views-tests.tex",
        stub_label="",
        bold_below={column: 0.05 for column in ai_view_qvalues.columns},
    )
    ai_view_qvalues
    return (ai_view_qvalues,)


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


@app.cell
def _(mo):
    mo.md("""
    ### Which answer option the public prefers

    The survey shows two responses, not the answer options themselves, but
    each response was judged onto one of the dilemma's options. A preference
    rating therefore also says which option the respondent endorsed: the
    option of the response they preferred. "About the same" splits half a
    vote to each response's option.

    Each surveyed dilemma's figure compares three shares over its answer
    options:

    - **Public** - the ratings' votes for the option.
    - **Base model** and **fine-tuned model** - the judged options over all
      50 runs of the run the survey subset came from.

    The public saw only the 17 selected pairs per dilemma, chosen where the
    two models diverge, so its shares describe those comparisons rather than
    every run. **TVD** is the total variation distance between the public's
    shares and a model's shares: 0 means the same distribution, 1 means no
    shared option. The model with the smaller TVD is closer to the public.
    When a run export is unavailable, the model shares fall back to the 17
    surveyed pairs and the figure says so.

    The payload written for the website also carries the direct H2 preference
    results (overall and per dilemma), so the Survey Results tab can show the
    head-to-head vote next to this option-share comparison.
    """)
    return


@app.cell
def _(
    COMPARISONS,
    FIGURES,
    RESPONSES,
    h2_bt,
    h2_data,
    h2_lmm,
    json,
    load_manifest,
    mo,
    np,
    pd,
    plt,
    textwrap,
    write_latex_table,
):
    OPTION_PUBLIC_COLOUR = "#4c4c4c"
    OPTION_BASE_COLOUR = "#d1642f"
    OPTION_FT_COLOUR = "#3b7dd8"
    OPTION_FIGURES = RESPONSES.parent / "dilemma-option-distributions"

    def option_context(data):
        """The survey subset, ratings with each side's judged option, and
        dilemma -> answer options.

        The manifest is the same subset the H2 analysis reads, so the ratings
        and the options always describe the same comparisons.
        """
        manifest = load_manifest(set(data["comparison_id"]))
        choices = {
            record["comparison_id"]: (
                (record["agents"].get("base") or {}).get("choice"),
                (record["agents"].get("ft") or {}).get("choice"),
            )
            for record in manifest["comparisons"]
        }
        options = {
            dilemma["dilemma_id"]: dilemma["answer_options"]
            for dilemma in manifest["dilemmas"]
        }
        ratings = data.copy()
        ratings["base_choice"] = ratings["comparison_id"].map(
            lambda comparison_id: choices[comparison_id][0]
        )
        ratings["ft_choice"] = ratings["comparison_id"].map(
            lambda comparison_id: choices[comparison_id][1]
        )
        return manifest, ratings, options

    def public_option_shares(ratings):
        """dilemma -> option -> share of the public's preference votes.

        A rating votes for the answer option of the response it preferred;
        "About the same" splits half a vote to each. Shares sum to one per
        dilemma.
        """
        shares = {}
        for dilemma, group in ratings.groupby("dilemma"):
            votes = {}
            for base_choice, ft_choice, score in zip(
                group["base_choice"], group["ft_choice"], group["ft_score"]
            ):
                if not base_choice or not ft_choice:
                    continue
                if score > 0:
                    votes[ft_choice] = votes.get(ft_choice, 0.0) + 1.0
                elif score < 0:
                    votes[base_choice] = votes.get(base_choice, 0.0) + 1.0
                else:
                    votes[base_choice] = votes.get(base_choice, 0.0) + 0.5
                    votes[ft_choice] = votes.get(ft_choice, 0.0) + 0.5
            total = sum(votes.values())
            shares[dilemma] = (
                {option: weight / total for option, weight in votes.items()}
                if total
                else {}
            )
        return shares

    def model_option_counts(manifest):
        """dilemma -> model -> judged option -> runs, and the source label.

        The exported run manifest holds every run of each model, so its
        choice counts are the model's option distribution. If the manifest is
        not on disk, fall back to the agents in the survey subset: the figures
        then describe only the pairs the public saw, and say so.
        """
        runs_dir = COMPARISONS.parent / "runs"
        run_id = manifest.get("run_id")
        active = runs_dir / "index.json"
        for path in (runs_dir / "full" / str(run_id) / "index.json", active):
            if not path.exists():
                continue
            run_manifest = json.loads(path.read_text(encoding="utf-8"))
            if path == active and run_manifest.get("active_run_id") != run_id:
                continue
            counts = {
                f"{dilemma['profile_id']}-{dilemma['situation_id']}": dilemma.get(
                    "choice_counts", {}
                )
                for dilemma in run_manifest.get("dilemmas", [])
            }
            return counts, "all exported runs"
        counts, seen = {}, set()
        for record in manifest.get("comparisons", []):
            for agent in record.get("agents", {}).values():
                key = (record["dilemma_id"], agent.get("model"), agent.get("run_id"))
                if key in seen:
                    continue
                seen.add(key)
                choice = agent.get("choice")
                if not choice:
                    continue
                model_counts = counts.setdefault(record["dilemma_id"], {}).setdefault(
                    agent.get("model", ""), {}
                )
                model_counts[choice] = model_counts.get(choice, 0) + 1
        return counts, "the surveyed pairs only"

    def normalised(counts):
        """A count mapping as shares that sum to one."""
        total = sum(counts.values())
        return (
            {option: count / total for option, count in counts.items()} if total else {}
        )

    def option_tvd(left, right):
        """Total variation distance between two option distributions."""
        return 0.5 * sum(
            abs(left.get(option, 0.0) - right.get(option, 0.0))
            for option in set(left) | set(right)
        )

    manifest, ratings, options = option_context(h2_data)
    public_shares = public_option_shares(ratings)
    option_counts, count_source = model_option_counts(manifest)
    base_model = manifest["models"]["base"]
    ft_model = manifest["models"]["ft"]
    questions = {
        dilemma["dilemma_id"]: dilemma["question"] for dilemma in manifest["dilemmas"]
    }
    situations = {
        dilemma["dilemma_id"]: dilemma.get("situation", "")
        for dilemma in manifest["dilemmas"]
    }
    situation_ids = {
        dilemma["dilemma_id"]: dilemma["situation_id"]
        for dilemma in manifest["dilemmas"]
    }
    profiles = {
        dilemma["dilemma_id"]: dilemma["profile_id"] for dilemma in manifest["dilemmas"]
    }
    dilemma_order = sorted(
        public_shares, key=lambda dilemma: (profiles[dilemma], dilemma)
    )

    share_rows, alignment_rows = [], []
    for dilemma in dilemma_order:
        dilemma_counts = option_counts.get(dilemma, {})
        base_share = normalised(dilemma_counts.get(base_model, {}))
        ft_share = normalised(dilemma_counts.get(ft_model, {}))
        public_share = public_shares[dilemma]
        for option in options[dilemma]:
            share_rows.append(
                {
                    "Dilemma": dilemma,
                    "Option": option,
                    "Public": public_share.get(option, 0.0),
                    "Base": base_share.get(option, 0.0),
                    "Fine-tune": ft_share.get(option, 0.0),
                }
            )
        group = ratings[ratings["dilemma"] == dilemma]
        distance_base = option_tvd(public_share, base_share)
        distance_ft = option_tvd(public_share, ft_share)
        alignment_rows.append(
            {
                "Dilemma": dilemma,
                "Ratings": len(group),
                "Tie share": float((group["ft_score"] == 0).mean())
                if len(group)
                else np.nan,
                "TVD vs base": distance_base,
                "TVD vs fine-tune": distance_ft,
                "Gap (base - fine-tune)": distance_base - distance_ft,
                "Closer to public": (
                    "Fine-tune"
                    if distance_ft < distance_base
                    else "Base"
                    if distance_base < distance_ft
                    else "Tie"
                ),
                "Public top": max(public_share, key=public_share.get)
                if public_share
                else "",
                "Base top": max(base_share, key=base_share.get) if base_share else "",
                "Fine-tune top": max(ft_share, key=ft_share.get) if ft_share else "",
            }
        )
    option_shares = pd.DataFrame(share_rows)
    # Most favourable to the fine-tune first: the base's distance minus the
    # fine-tune's, so a positive gap means the fine-tune is closer.
    option_alignment = pd.DataFrame(alignment_rows).sort_values(
        "Gap (base - fine-tune)", ascending=False, kind="stable"
    )
    alignment_by_dilemma = option_alignment.set_index("Dilemma")

    def plot_dilemma_options(dilemma, path):
        """One dilemma's public/base/fine-tune shares as grouped bars."""
        subset = option_shares[option_shares["Dilemma"] == dilemma]
        x = np.arange(len(subset))
        width = 0.26
        figure, axis = plt.subplots(figsize=(max(6.0, 2.2 * len(subset)), 4.4))
        for offset, (column, colour, label) in enumerate(
            (
                ("Public", OPTION_PUBLIC_COLOUR, "Public"),
                ("Base", OPTION_BASE_COLOUR, "Base model"),
                ("Fine-tune", OPTION_FT_COLOUR, "Fine-tuned model"),
            )
        ):
            values = subset[column].to_numpy(dtype=float)
            positions = x + (offset - 1) * width
            axis.bar(positions, values, width, color=colour, label=label)
            for position, value in zip(positions, values):
                axis.text(
                    position,
                    value + 0.015,
                    f"{value:.0%}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        axis.set_xticks(x)
        axis.set_xticklabels(
            [textwrap.fill(str(option), 26) for option in subset["Option"]],
            fontsize=8,
        )
        axis.set_ylim(0, 1.08)
        axis.set_ylabel("share of answers")
        axis.set_title(
            textwrap.fill(questions[dilemma], 64) + f"\n{dilemma}",
            fontsize=9,
            loc="left",
        )
        row = alignment_by_dilemma.loc[dilemma]
        figure.text(
            0.5,
            0.01,
            f"public ratings: {int(row['Ratings'])} "
            f"({row['Tie share']:.0%} tied); model shares: {count_source}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#555",
        )
        axis.legend(
            fontsize=8,
            frameon=False,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
        )
        axis.grid(axis="y", alpha=0.25, lw=0.6)
        axis.set_axisbelow(True)
        figure.tight_layout(rect=(0, 0.04, 1, 1))
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        return path

    def plot_alignment(path):
        """The per-dilemma TVD of each model against the public, ranked."""
        ranked = option_alignment.sort_values("Gap (base - fine-tune)", ascending=True)
        y = np.arange(len(ranked))
        figure, axis = plt.subplots(figsize=(8.5, 7))
        for position, (_, row) in zip(y, ranked.iterrows()):
            axis.plot(
                [row["TVD vs base"], row["TVD vs fine-tune"]],
                [position, position],
                color="0.75",
                lw=1.2,
                zorder=1,
            )
        axis.scatter(
            ranked["TVD vs base"],
            y,
            s=42,
            color=OPTION_BASE_COLOUR,
            label="Base model",
            zorder=2,
        )
        axis.scatter(
            ranked["TVD vs fine-tune"],
            y,
            s=42,
            color=OPTION_FT_COLOUR,
            label="Fine-tuned model",
            zorder=2,
        )
        axis.set_yticks(y)
        axis.set_yticklabels(
            [dilemma.split("-", 1)[-1] for dilemma in ranked["Dilemma"]], fontsize=7
        )
        axis.set_xlim(0, 1)
        axis.set_xlabel("TVD between the public's option shares and the model's")
        axis.set_title(
            "How close each model's answer options are to the public's",
            fontsize=10,
        )
        axis.grid(axis="x", alpha=0.3, lw=0.6)
        axis.legend(frameon=False, loc="lower right")
        figure.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        return path

    option_figure_paths = {
        dilemma: plot_dilemma_options(dilemma, OPTION_FIGURES / f"{dilemma}.png")
        for dilemma in dilemma_order
    }
    option_alignment_figure = plot_alignment(
        FIGURES / "survey-dilemma-option-alignment.png"
    )

    alignment_columns = [
        "Ratings",
        "Tie share",
        "TVD vs base",
        "TVD vs fine-tune",
        "Closer to public",
    ]
    write_latex_table(
        alignment_by_dilemma[alignment_columns],
        {
            "Ratings": "{:.0f}",
            "Tie share": "{:.0%}",
            "TVD vs base": "{:.2f}",
            "TVD vs fine-tune": "{:.2f}",
        },
        len(alignment_columns),
        "survey-dilemma-option-alignment.tex",
        senses={"TVD vs base": "min", "TVD vs fine-tune": "min"},
        column_format="lrrrrl",
        stub_label="Dilemma",
    )

    def preference_stats(frame):
        """Head-to-head preference counts and mean for one set of ratings.

        ``ft_score`` is the 5-point preference score signed so positive means
        the fine-tune was preferred (negative = base). The decided shares leave
        ties out; the win rate includes ties at 0.5, matching the H2 test.
        """
        scores = frame["ft_score"].to_numpy(dtype=float)
        n = len(scores)
        base_wins = int((scores < 0).sum())
        fine_tune_wins = int((scores > 0).sum())
        ties = int((scores == 0).sum())
        decided = base_wins + fine_tune_wins
        mean = float(scores.mean()) if n else np.nan
        sd = float(scores.std(ddof=1)) if n > 1 else np.nan
        margin = 1.96 * sd / np.sqrt(n) if n > 1 else np.nan
        return {
            "ratings": n,
            "mean": round(mean, 4),
            "ci_low": round(mean - margin, 4),
            "ci_high": round(mean + margin, 4),
            "base_wins": base_wins,
            "fine_tune_wins": fine_tune_wins,
            "ties": ties,
            "decided": decided,
            "base_share": round(base_wins / decided, 4) if decided else np.nan,
            "fine_tune_share": (
                round(fine_tune_wins / decided, 4) if decided else np.nan
            ),
            "win_rate": round(float(frame["win"].mean()), 4) if n else np.nan,
        }

    def website_option_payload(
        manifest, alignment, shares, ratings, count_source, lmm, bradley_terry
    ):
        """The compact, non-identifying JSON the website reads.

        It keeps only aggregate option shares and preference counts: no
        respondent ids, free text or individual ratings. The same payload is
        written beside the survey subset (provenance) and to
        ``survey-results/index.json``, which is the stable path the website
        fetches.
        """
        base_name = str(manifest["models"]["base"]).rsplit("/", 1)[-1]
        adapter = (
            str(manifest.get("subset", ""))
            .removesuffix("_response-overall")
            .removesuffix("_distribution-overall")
            .replace("_", " ")
        )
        closer_names = {"Fine-tune": "fine_tune", "Base": "base", "Tie": "tie"}

        def top_index(values):
            return int(max(range(len(values)), key=values.__getitem__))

        dilemma_rows = []
        for _, row in alignment.iterrows():
            dilemma = row["Dilemma"]
            subset = shares[shares["Dilemma"] == dilemma]
            option_list = options[dilemma]

            def aligned(column):
                by_option = dict(zip(subset["Option"], subset[column]))
                return [
                    round(float(by_option.get(option, 0.0)), 4)
                    for option in option_list
                ]

            public_values = aligned("Public")
            base_values = aligned("Base")
            fine_tune_values = aligned("Fine-tune")
            group = ratings[ratings["dilemma"] == dilemma]
            preference = preference_stats(group)
            dilemma_rows.append(
                {
                    "dilemma_id": dilemma,
                    "profile_id": profiles[dilemma],
                    "situation_id": situation_ids[dilemma],
                    "question": questions[dilemma],
                    "situation": situations[dilemma],
                    "answer_options": option_list,
                    "ratings": int(preference["ratings"]),
                    "tie_share": round(float(row["Tie share"]), 4),
                    "preference": preference,
                    "shares": {
                        "public": public_values,
                        "base": base_values,
                        "fine_tune": fine_tune_values,
                    },
                    "tvd": {
                        "base": round(float(row["TVD vs base"]), 4),
                        "fine_tune": round(float(row["TVD vs fine-tune"]), 4),
                    },
                    "closer": closer_names[row["Closer to public"]],
                    "top": {
                        "public": top_index(public_values),
                        "base": top_index(base_values),
                        "fine_tune": top_index(fine_tune_values),
                    },
                }
            )

        closer_counts = alignment["Closer to public"].value_counts()
        overall_preference = preference_stats(ratings)
        selection = manifest.get("selection") or {}
        payload = {
            "schema": "wvs-survey-option-distributions/v1",
            "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "run_id": manifest.get("run_id"),
            "subset": manifest.get("subset"),
            "models": {
                "base": manifest["models"]["base"],
                "fine_tune": manifest["models"]["ft"],
                "base_label": f"{base_name} base",
                "fine_tune_label": (
                    f"{base_name} fine-tuned" + (f" ({adapter})" if adapter else "")
                ),
            },
            "summary": {
                "ratings": int(len(ratings)),
                "respondents": int(ratings["_recordId"].nunique()),
                "dilemmas": int(len(alignment)),
                "tie_share": round(float((ratings["ft_score"] == 0).mean()), 4),
                "mean_tvd_base": round(float(alignment["TVD vs base"].mean()), 4),
                "mean_tvd_fine_tune": round(
                    float(alignment["TVD vs fine-tune"].mean()), 4
                ),
                "base_closer": int(closer_counts.get("Base", 0)),
                "fine_tune_closer": int(closer_counts.get("Fine-tune", 0)),
                "ties": int(closer_counts.get("Tie", 0)),
            },
            "preference": {
                "lmm": {
                    key: lmm[key] for key in ("estimate", "ci", "p", "n", "respondents")
                },
                "bradley_terry": bradley_terry,
                "h2_supported": bool(lmm["ci"][0] > 0 and bradley_terry["ci"][0] > 0.5),
                **overall_preference,
            },
            "method": (
                "A preference rating votes for the answer option of the response "
                "it preferred; 'About the same' splits half a vote to each. Model "
                "shares are the judged options over every exported run. TVD is the "
                "total variation distance between the public shares and a model's "
                "shares (0 = identical, 1 = no shared option)."
            ),
            "model_share_source": count_source,
            "selection": {
                "rule": selection.get("rule", ""),
                "per_dilemma": selection.get("per_dilemma"),
                "p_threshold": selection.get("p_threshold"),
            },
            "dilemmas": dilemma_rows,
        }
        run_id = str(manifest.get("run_id"))
        subset_name = str(manifest.get("subset"))
        subset_path = (
            COMPARISONS / run_id / "survey" / subset_name / "option-distributions.json"
        )
        website_path = COMPARISONS.parent / "survey-results" / "index.json"
        for path in (subset_path, website_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return website_path

    website_payload_path = website_option_payload(
        manifest,
        option_alignment,
        option_shares,
        ratings,
        count_source,
        h2_lmm,
        h2_bt,
    )

    closer_counts = option_alignment["Closer to public"].value_counts()
    fine_tune_closer = int(closer_counts.get("Fine-tune", 0))
    base_closer = int(closer_counts.get("Base", 0))
    ties = int(closer_counts.get("Tie", 0))
    tie_note = f", and {ties} tie{'s' if ties != 1 else ''}" if ties else ""
    option_note = mo.md(
        f"The public's option shares pool {len(ratings):,} ratings from "
        f"{ratings['_recordId'].nunique():,} respondents across "
        f"{len(option_alignment)} dilemmas. The fine-tuned model is closer to "
        f"the public on {fine_tune_closer} dilemmas and the base model on "
        f"{base_closer}{tie_note}. Mean TVD: "
        f"{option_alignment['TVD vs fine-tune'].mean():.2f} for the fine-tune "
        f"and {option_alignment['TVD vs base'].mean():.2f} for the base "
        f"(lower is closer). The per-dilemma figures are written to "
        f"`{OPTION_FIGURES.relative_to(RESPONSES.parent.parent.parent)}` and the "
        f"website payload to "
        f"`{website_payload_path.relative_to(COMPARISONS.parents[3])}`."
    )

    mo.vstack(
        [
            option_alignment,
            option_note,
            mo.image(str(option_alignment_figure)),
            mo.accordion(
                {
                    f"{dilemma} - closer: {alignment_by_dilemma.loc[dilemma, 'Closer to public']}": mo.image(
                        str(path)
                    )
                    for dilemma, path in option_figure_paths.items()
                }
            ),
        ]
    )


@app.cell
def _(mo):
    mo.md("""
    ### Which fine-tune is closest to the public?

    The survey was fielded on the modal fine-tune, but every adapter
    answered the same dilemmas. This section compares **all exported
    fine-tunes and the base model** with the public's answer-option shares on
    the 18 surveyed dilemmas. For each model and dilemma, the model's option
    distribution is its judged answer counts over every exported run; the
    public's distribution is the preference-vote share used above. The
    distance is the total variation distance (TVD): 0 means identical, 1
    means no shared option. The mean TVD across dilemmas is the headline
    closeness measure, and the bootstrap over dilemmas gives its interval and
    the probability that each model is the closest.

    The model distributions come from the survey run's full-epoch export,
    which now holds all four adapters and the base model at 50 runs per
    dilemma. The notebook deliberately does not fall back to an older,
    shorter run: if any adapter is missing from the export it stops with the
    `export_results.py` command instead, so the TVD ranking always describes
    the same full run as the survey results. The public saw only the 17
    selected divergent pairs per dilemma, while the model shares use all runs,
    so the comparison is about the direction of the divergence rather than
    the full response distribution.
    """)
    return


@app.cell
def _(
    BOOTSTRAP_REPS,
    COMPARISONS,
    FIGURES,
    h2_data,
    json,
    load_manifest,
    mo,
    np,
    pd,
    plt,
    write_latex_table,
):
    MODEL_PRETTY = {
        "modal_response": "modal fine-tune",
        "sampled_response": "sampled fine-tune",
        "first_token_distribution": "first-token fine-tune",
        "full_string_distribution": "full-string fine-tune",
    }

    def model_label(model):
        """Readable label for a model id in the run manifest."""
        if "nz-wvs-" in model:
            method = model.split("nz-wvs-", 1)[1].removesuffix("-overall")
            return MODEL_PRETTY.get(method, method.replace("_", " "))
        return "Base (Qwen3.8-27B)"

    def _public_option_shares(ratings, manifest):
        """dilemma -> option -> share of the public's preference votes."""
        choices = {
            record["comparison_id"]: (
                (record["agents"].get("base") or {}).get("choice"),
                (record["agents"].get("ft") or {}).get("choice"),
            )
            for record in manifest["comparisons"]
        }
        shares = {}
        for dilemma, group in ratings.groupby("dilemma"):
            votes = {}
            for comparison_id, score in zip(group["comparison_id"], group["ft_score"]):
                base_choice, ft_choice = choices.get(comparison_id, (None, None))
                if not base_choice or not ft_choice:
                    continue
                if score > 0:
                    votes[ft_choice] = votes.get(ft_choice, 0.0) + 1.0
                elif score < 0:
                    votes[base_choice] = votes.get(base_choice, 0.0) + 1.0
                else:
                    votes[base_choice] = votes.get(base_choice, 0.0) + 0.5
                    votes[ft_choice] = votes.get(ft_choice, 0.0) + 0.5
            total = sum(votes.values())
            shares[dilemma] = (
                {option: weight / total for option, weight in votes.items()}
                if total
                else {}
            )
        return shares

    def _alignment_option_tvd(left, right):
        return 0.5 * sum(
            abs(left.get(option, 0.0) - right.get(option, 0.0))
            for option in set(left) | set(right)
        )

    def load_full_manifest(run_id):
        """The manifest for one exported run, with its directory."""
        runs_dir = COMPARISONS.parent / "runs"
        bundle = runs_dir / "full" / str(run_id) / "index.json"
        if bundle.exists():
            return json.loads(bundle.read_text(encoding="utf-8")), bundle
        active = runs_dir / "index.json"
        if active.exists():
            manifest = json.loads(active.read_text(encoding="utf-8"))
            if manifest.get("active_run_id") == run_id:
                return manifest, active
        return None, None

    def full_run_manifest(run_id):
        """The survey run's full-epoch manifest, with all four adapters.

        The analysis deliberately does not fall back to an older run: mixing a
        50-epoch survey run with a 10-epoch all-model run would make the TVD
        ranking depend on the export history rather than the models. A missing
        adapter stops the notebook with the export command instead.
        """
        manifest, path = load_full_manifest(run_id)
        if manifest is None:
            raise SystemExit(
                f"No exported run manifest for {run_id!r}. Run "
                "`uv run python code/dilemmas/export_results.py` first."
            )
        fine = [model for model in manifest.get("models", []) if "nz-wvs-" in model]
        if len(fine) < 4:
            raise SystemExit(
                f"Run {run_id!r} has only {len(fine)} fine-tunes exported; "
                "expected all four adapters at the full epoch count. Run "
                "`uv run python code/dilemmas/export_results.py` after the "
                "refit completes."
            )
        return manifest, path

    def epochs_per_model(manifest):
        epochs = {}
        for run in manifest.get("runs", []):
            epochs[run["model"]] = max(
                epochs.get(run["model"], 0),
                int(run.get("epoch") or run.get("version") or 0),
            )
        return epochs

    survey_manifest = load_manifest(set(h2_data["comparison_id"]))
    _public_shares = _public_option_shares(h2_data, survey_manifest)
    alignment_manifest, _alignment_path = full_run_manifest(survey_manifest["run_id"])
    _alignment_counts = {
        f"{dilemma['profile_id']}-{dilemma['situation_id']}": dilemma.get(
            "choice_counts", {}
        )
        for dilemma in alignment_manifest.get("dilemmas", [])
    }
    epochs = epochs_per_model(alignment_manifest)
    _survey_profiles = {
        dilemma["dilemma_id"]: dilemma["profile_id"]
        for dilemma in survey_manifest["dilemmas"]
    }
    surveyed = sorted(_public_shares, key=lambda d: (_survey_profiles.get(d, ""), d))

    def tvd_matrix():
        rows = {}
        for model in alignment_manifest.get("models", []):
            values = []
            for dilemma in surveyed:
                model_counts = _alignment_counts.get(dilemma, {}).get(model, {})
                total = sum(model_counts.values())
                if not total:
                    values.append(np.nan)
                    continue
                model_shares = {
                    option: count / total for option, count in model_counts.items()
                }
                values.append(
                    _alignment_option_tvd(_public_shares[dilemma], model_shares)
                )
            rows[model] = values
        return pd.DataFrame(rows, index=surveyed).T

    tvd_by_model = tvd_matrix()
    tvd_by_model = tvd_by_model.dropna(axis=0, how="all")
    _alignment_base_model = next(
        (model for model in tvd_by_model.index if "nz-wvs-" not in model), None
    )
    _alignment_base_values = (
        tvd_by_model.loc[_alignment_base_model].to_numpy(float)
        if _alignment_base_model
        else None
    )

    def summarise(matrix):
        rng = np.random.default_rng(20261104)
        n_dilemmas = matrix.shape[1]
        draws = rng.integers(0, n_dilemmas, size=(BOOTSTRAP_REPS, n_dilemmas))
        boot = np.empty((BOOTSTRAP_REPS, len(matrix)))
        for position, (_, row) in enumerate(matrix.iterrows()):
            values = row.to_numpy(float)
            boot[:, position] = np.nanmean(values[draws], axis=1)
        closest = (np.argmin(boot, axis=1)[:, None] == np.arange(len(matrix))).mean(
            axis=0
        )
        rows = []
        for position, (model, row) in enumerate(matrix.iterrows()):
            values = row.to_numpy(float)
            mean = float(np.nanmean(values))
            low, high = np.quantile(boot[:, position], [0.025, 0.975])
            closer = (
                int(np.nansum(values < _alignment_base_values))
                if _alignment_base_values is not None and model != _alignment_base_model
                else np.nan
            )
            rows.append(
                {
                    "Model": model_label(model),
                    "Runs": epochs.get(model, 0),
                    "Mean TVD": mean,
                    "95% CI": f"[{low:.3f}, {high:.3f}]",
                    "Closer than base": closer,
                    "P(closest)": float(closest[position]),
                    "_model": model,
                    "_values": values,
                }
            )
        table = pd.DataFrame(rows).sort_values("Mean TVD", kind="stable")
        table["Rank"] = np.arange(1, len(table) + 1)
        return table

    alignment_table = summarise(tvd_by_model)
    alignment_display = alignment_table.drop(columns=["_model", "_values"]).copy()
    alignment_display["Mean TVD"] = alignment_display["Mean TVD"].map(
        lambda value: f"{value:.3f}"
    )
    alignment_display["Closer than base"] = alignment_display["Closer than base"].map(
        lambda value: "--" if pd.isna(value) else f"{int(value)}"
    )
    alignment_display["P(closest)"] = alignment_display["P(closest)"].map(
        lambda value: f"{value:.2f}"
    )
    alignment_display = alignment_display.set_index("Model")
    alignment_display.index.name = None
    write_latex_table(
        alignment_display,
        "{}",
        len(alignment_display.columns),
        "survey-model-public-alignment.tex",
        stub_label="",
        highlight=False,
    )

    def _plot_alignment(table, matrix):
        order = table["_model"].tolist()
        labels = table["Model"].tolist()
        means = table["Mean TVD"].to_numpy(float)
        low = np.array([float(ci.strip("[]").split(",")[0]) for ci in table["95% CI"]])
        high = np.array([float(ci.strip("[]").split(",")[1]) for ci in table["95% CI"]])
        y = np.arange(len(order))
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(14, 5.2),
            gridspec_kw={"width_ratios": [1, 2.2], "wspace": 0.06},
        )
        colours = [
            "0.35" if "nz-wvs-" not in model else f"C{position % 10}"
            for position, model in enumerate(order)
        ]
        axes[0].errorbar(
            means,
            y,
            xerr=[means - low, high - means],
            fmt="o",
            color="black",
            ecolor="0.5",
            capsize=3,
            ms=5,
        )
        for position, (mean, colour) in enumerate(zip(means, colours)):
            axes[0].scatter(mean, position, color=colour, s=45, zorder=3)
            axes[0].text(
                high[position] + 0.008,
                position,
                f"{mean:.3f}",
                va="center",
                fontsize=8,
            )
        if _alignment_base_values is not None:
            axes[0].axvline(
                np.nanmean(_alignment_base_values),
                color="0.35",
                ls="--",
                lw=1,
                label="Base mean",
            )
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(labels)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Mean TVD vs public (lower = closer)")
        axes[0].set_title("Mean distance across the 18 dilemmas")
        axes[0].grid(axis="x", alpha=0.3, lw=0.5)
        axes[0].legend(frameon=False, loc="lower left")

        image = axes[1].imshow(
            matrix.loc[order].to_numpy(float),
            aspect="auto",
            cmap="viridis_r",
            vmin=0,
            vmax=1,
        )
        axes[1].set_xticks(np.arange(len(matrix.columns)))
        axes[1].set_xticklabels(
            [dilemma.split("-", 1)[-1] for dilemma in matrix.columns],
            rotation=90,
            fontsize=6,
        )
        axes[1].set_yticks(np.arange(len(order)))
        axes[1].set_yticklabels([])
        axes[1].set_title("Per-dilemma TVD")
        colorbar = fig.colorbar(image, ax=axes[1], fraction=0.025, pad=0.01)
        colorbar.set_label("TVD (lower = closer)")
        subtitle = (
            f"run {alignment_manifest.get('run_id')}, "
            f"{max(epochs.values()) if epochs else '?'} epochs per model"
        )
        fig.suptitle(
            f"Which model's answer distribution is closest to the public?\n{subtitle}",
            fontsize=11,
            y=0.99,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        fig.savefig(
            FIGURES / "survey-model-public-alignment.png",
            dpi=200,
            bbox_inches="tight",
        )
        return fig

    alignment_figure = _plot_alignment(alignment_table, tvd_by_model)
    alignment_note = mo.md(
        f"Model distributions: run `{alignment_manifest.get('run_id')}` "
        f"({max(epochs.values()) if epochs else '?'} epochs; "
        f"{len(tvd_by_model)} models). "
        f"Survey run: `{survey_manifest['run_id']}` "
        f"({len([m for m in survey_manifest.get('models', []) if 'nz-wvs-' in m])} "
        "fine-tunes exported). Lower mean TVD is closer to the public. "
        f"Closest: **{alignment_table.iloc[0]['Model']}** "
        f"(mean TVD {alignment_table.iloc[0]['Mean TVD']:.3f})."
    )
    mo.vstack([alignment_note, alignment_display, mo.image(str(alignment_figure))])
    return (alignment_table, tvd_by_model)


if __name__ == "__main__":
    app.run()
