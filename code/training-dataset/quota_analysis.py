# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "pandas>=2.0.0",
#     "numpy>=1.26.0",
#     "matplotlib>=3.8.0",
#     "scikit-learn>=1.5.0",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Quota analysis: WVS NZ demographics vs the 2023 NZ Census

    How representative is the WVS Wave 7 NZ sample? Which demographics are
    available for quota design, how do their distributions compare with the
    census, and where is the sample skewed (notably: too old, more female)?
    Section 4 adds held-out log-loss modelling of which demographics best
    predict people's value-survey answers. Each section below runs on its own.
    """)
    return


@app.cell(hide_code=True)
def _():
    import csv
    from pathlib import Path
    import marimo as mo
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import sklearn

    DIR = Path(__file__).resolve().parent
    INPUT = DIR / "input" / "WVS_Wave_7_New_Zealand_Csv_v5.1.csv"
    VALUE_SURVEY = DIR / "output" / "wvs_value_survey.csv"
    OUT = DIR / "output" / "quota_analysis"
    FIGURE_DIR = DIR.parent / "figures"
    return FIGURE_DIR, INPUT, OUT, VALUE_SURVEY, csv, mo, np, pd, plt, sklearn


@app.cell
def _(INPUT, csv, np, pd):
    MISSING_CODES = {-5, -4, -3, -2, -1}

    DEMOGRAPHICS = {
        "Q260": ("Sex", {1.0: "Male", 2.0: "Female"}),
        "Q273": (
            "Marital status",
            {
                1.0: "Married",
                2.0: "Living as married",
                3.0: "Divorced",
                4.0: "Separated",
                5.0: "Widowed",
                6.0: "Never married",
            },
        ),
        "Q269": (
            "NZ citizen",
            {1.0: "Citizen", 2.0: "Not a citizen", 3.0: "Something else"},
        ),
        "Q275R": ("Education (3-level)", {1.0: "Lower", 2.0: "Middle", 3.0: "Higher"}),
        "Q287": (
            "Subjective social class (self-assessed)",
            {
                1.0: "Upper class",
                2.0: "Upper middle class",
                3.0: "Lower middle class",
                4.0: "Working class",
                5.0: "Lower class",
            },
        ),
        # Q288 in the NZ questionnaire shows real dollar bands (not
        # percentile deciles):
        # 1 "Up to $10,000", 2 "$10,001-$15,000", 3 "$15,001-$20,000",
        # 4 "$20,001-$25,000", 5 "$25,001-$30,000", 6 "$30,001-$40,000",
        # 7 "$40,001-$50,000", 8 "$50,001-$70,000", 9 "$70,001-$100,000",
        # 10 "$100,001 and over".
        "Q288": (
            "Personal income band",
            {
                1.0: "Up to $10,000",
                2.0: "$10,001-$15,000",
                3.0: "$15,001-$20,000",
                4.0: "$20,001-$25,000",
                5.0: "$25,001-$30,000",
                6.0: "$30,001-$40,000",
                7.0: "$40,001-$50,000",
                8.0: "$50,001-$70,000",
                9.0: "$70,001-$100,000",
                10.0: "$100,001 and over",
            },
        ),
        "Q281": ("Occupational group", {i: f"Group {i}" for i in range(0, 11)}),
        "Q279": (
            "Employment status",
            {
                1.0: "Full time (30+ hrs/wk)",
                2.0: "Part time (<30 hrs)",
                3.0: "Self employed",
                4.0: "Retired/pensioned",
                5.0: "Homemaker",
                6.0: "Student",
                7.0: "Unemployed",
                8.0: "Other",
            },
        ),
        "N_REGION_ISO": ("Region (ISO recode)", {}),
    }

    def age_band(years):
        """Band the WVS age in years into the survey age bands."""
        if years is None or np.isnan(years):
            return np.nan
        for cut, label in [
            (25, "16-24"),
            (35, "25-34"),
            (45, "35-44"),
            (55, "45-54"),
            (65, "55-64"),
        ]:
            if int(years) < cut:
                return label
        return "65+"

    def income_band(code):
        """Band the WVS NZ income bands (Q288 codes 1-10) into the three
        bands used against the census, which share the same thresholds:
        Low = under $50k (codes 1-7), Medium = $50,001-$100,000
        (codes 8-9), High = over $100k (code 10)."""
        if code is None or np.isnan(code):
            return np.nan
        if code <= 7:
            return "Low"
        if code <= 9:
            return "Medium"
        return "High"

    def load_demo():
        """Labelled demographic columns, keyed by the wrangler's id.

        The raw WVS CSV's data rows have one more field than the header,
        so plain pandas read_csv mis-aligns some columns; this parses
        through the csv reader instead (same as wrangle_response_data.py).
        """
        with open(INPUT, encoding="utf-8-sig") as f:
            rows = list(csv.reader(f, delimiter=";"))
        names = [x.strip().split(" ", 1)[0] for x in rows[0]]

        def to_num(s: str):
            try:
                v = float(s)
            except ValueError:
                return np.nan
            return np.nan if int(v) in MISSING_CODES else v

        def labelled(row, col: str):
            v = to_num(row[names.index(col)])
            if np.isnan(v):
                return np.nan
            mapping = DEMOGRAPHICS[col][1]
            if mapping:
                return mapping.get(v, f"Code {int(v)}")
            return int(v)

        record = {
            "id": [],
            "Sex": [],
            "Age group": [],
            "NZ citizen": [],
            "Education (3-level)": [],
            "Subjective social class": [],
            "Personal income band": [],
            "Personal income (3-level)": [],
            "Occupational group": [],
            "Region (ISO recode)": [],
            "Employment status": [],
            "Marital status": [],
        }
        for row in rows[1:]:
            try:
                record["id"].append(
                    str(int(float(row[names.index("D_INTERVIEW")]) - 554070000))
                )
            except (ValueError, TypeError):
                continue
            record["Sex"].append(labelled(row, "Q260"))
            record["Age group"].append(age_band(to_num(row[names.index("Q262")])))
            record["NZ citizen"].append(labelled(row, "Q269"))
            # Merge formal education (Q275, ISCED) with the national
            # qualification recode (Q275R): use the 3-level recode where
            # present, else map ISCED - no qualification covers the two
            # lower ISCED levels (0-2 -> Lower), 3-4 -> Middle, 5+ ->
            # Higher.
            edu_fallback = {
                0.0: "Lower",
                1.0: "Lower",
                2.0: "Lower",
                3.0: "Middle",
                4.0: "Middle",
            }
            record["Education (3-level)"].append(
                labelled(row, "Q275R")
                if not np.isnan(to_num(row[names.index("Q275R")]))
                else edu_fallback.get(to_num(row[names.index("Q275")]), np.nan)
            )
            record["Subjective social class"].append(labelled(row, "Q287"))
            record["Personal income band"].append(labelled(row, "Q288"))
            # Three-band income coded here from the Q288 dollar bands
            # (not the official Q288R recode), so the thresholds are the
            # census/survey $50k / $100k cuts (see income_band).
            record["Personal income (3-level)"].append(
                income_band(to_num(row[names.index("Q288")]))
            )
            record["Occupational group"].append(labelled(row, "Q281"))
            # Employment status in census-comparable buckets (census
            # 'labour force status'): self employed counted with full
            # time; retired/homemaker/student/other -> not in the labour
            # force.
            emp = {
                1.0: "Employed full-time",
                3.0: "Employed full-time",
                2.0: "Employed part-time",
                7.0: "Unemployed",
                4.0: "Not in the labour force",
                5.0: "Not in the labour force",
                6.0: "Not in the labour force",
                8.0: "Not in the labour force",
            }
            record["Employment status"].append(
                emp.get(to_num(row[names.index("Q279")]), np.nan)
            )
            record["Region (ISO recode)"].append(labelled(row, "N_REGION_ISO"))
            record["Marital status"].append(labelled(row, "Q273"))
        return pd.DataFrame(record).set_index("id")

    demo = load_demo()
    demo
    return (demo,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Demographic distributions (WVS)

    Writes `output/quota_analysis/demographic_distributions.csv`.
    """)
    return


@app.cell
def _(OUT, demo, pd):
    def save_distributions() -> None:
        records = []
        for label in demo.columns:
            counts = demo[label].value_counts(dropna=True)
            pct = demo[label].value_counts(normalize=True, dropna=True) * 100
            for value in counts.index:
                records.append(
                    {
                        "demographic": label,
                        "value": value,
                        "wvs_pct": round(float(pct[value]), 1),
                        "n": int(counts[value]),
                    }
                )
        (
            pd.DataFrame(records)
            .sort_values(["demographic", "wvs_pct"], ascending=[True, False])
            .to_csv(OUT / "demographic_distributions.csv", index=False)
        )

    OUT.mkdir(parents=True, exist_ok=True)
    save_distributions()
    (OUT / "demographic_distributions.csv")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Top-5 values vs the 2023 NZ Census

    The most common value of each available demographic, WVS share vs the
    hard-coded 2023 Census share. Writes
    `output/quota_analysis/top5_vs_census.csv`.
    """)
    return


@app.cell
def _(OUT, demo, pd):
    # --- 2023 NZ Census hard-coded reference values ------------------------
    # All figures: 2023 Census of the usually resident population,
    # retrieved 2026-09-18. One source link per variable, in line:
    # Personal income (2023 Census personal income, % of people 15+ where
    # information available; source: Figure.NZ, Stats NZ 2023 Census)
    # https://figure.nz/chart/qzNrLY6yWSlzl1aF
    # Bands: under $50k / $50,001-$100,000 / over $100k, computed from the
    # raw census bands below.
    # WVS Q288 dollar bands banded into three census bands (see
    # income_band): Low = under $50k, Medium = $50,001-$100,000,
    # High = over $100k.
    income_census = {
        "Low": (
            0.4788764259272
            + 6.773546975655
            + 7.1489884595
            + 4.186416530072
            + 6.467382504486
            + 8.063119713192
            + 6.553812416777
            + 4.435946858543
            + 4.628103695193
            + 8.291356786966
        ),
        "Medium": (9.021611914167 + 8.101639802416 + 13.77193001913),
        "High": (7.542396126994 + 2.43149745553 + 2.103300380543),
    }
    # Education: the WVS 3-level recode (Lower/Middle/Higher) is compared
    # against census qualification bands grouped the same way - not a
    # precise 1:1; the postal-mode WVS sample leans towards higher
    # education. Stats NZ 2023 Census highest-qualification table (15+,
    # % of total stated), via
    # https://regions.infometrics.co.nz/new-zealand/census/indicator/highest-qualification
    # NZ structure: Level 1 certificate = some secondary school; no
    # qualification covers the two lower WVS bands.
    education_census = {
        "Lower": 15.7 + 10.3 + 9.8,  # no qualification + L1 + L2
        "Middle": 12.7 + 8.8 + 5.1 + 4.7 + 5.8,  # L3-L6 + overseas second.
        "Higher": 15.5 + 6.2 + 4.4 + 1.0,  # Bachelor/L7 + post-grad etc.
    }
    CENSUS = {
        # https://regions.infometrics.co.nz/new-zealand/census/indicator/five-year-age-group
        # 16-24 approximates 18-24 in the census (2/5 of the 15-19 band);
        # 65+ is the sum of the six 65+ bands.
        "Age group": {
            "16-24": 6.2 + 0.4 * 6.4,
            "25-34": 6.7 + 7.5,
            "35-44": 6.9 + 6.3,
            "45-54": 6.1 + 6.5,
            "55-64": 6.1 + 5.9,
            "65+": 5.1 + 4.3 + 3.3 + 2.2 + 1.2 + 0.7,
        },
        # https://regions.infometrics.co.nz/new-zealand/census/indicator/gender
        "Sex": {"Female": 50.3, "Male": 49.3},
        "Education (3-level)": education_census,
        # Labour force status (2023 Census, % of people 15+ where stated;
        # source: https://regions.infometrics.co.nz/new-zealand/census/
        # indicator/labour-force-status). WVS Q279 is bucketed into the
        # census categories (see the loader).
        "Employment status": {
            "Employed full-time": 51.2,
            "Employed part-time": 13.4,
            "Unemployed": 3.0,
            "Not in the labour force": 32.4,
        },
        # Personal income (3-level): % of people 15+ where income
        # information is available (Figure.NZ / Stats NZ 2023 Census,
        # link above). WVS deciles are banded to the same $50k/$100k
        # thresholds (see income_band).
        "Personal income (3-level)": income_census,
        # Ethnicity is NOT asked in the released WVS NZ file; census
        # multi-select shares for quota design (European 67.8, Maori 17.8,
        # Asian 17.3, Pacific 8.9, MELAA 1.9) come from the Stats NZ release
        # https://www.stats.govt.nz/information-releases/
        # 2023-census-population-counts-by-ethnic-group-age-and-maori-descent-
        # and-dwelling-counts/
    }

    def save_top5():
        top5 = []
        for demographic, census_share in CENSUS.items():
            share = demo[demographic].value_counts(normalize=True, dropna=True) * 100
            order = {}
            if demographic.startswith("Personal income"):
                order = {"Low": 0, "Medium": 1, "High": 2}
            elif demographic.startswith("Education"):
                order = {"Lower": 0, "Middle": 1, "Higher": 2}
            if order:
                share = share.reindex(
                    sorted(share.index, key=lambda v: order.get(v, 999))
                )
            for value, pct in share.head(5).items():
                census = census_share.get(value)
                top5.append(
                    {
                        "demographic": demographic,
                        "value": value,
                        "wvs_pct": round(pct, 1),
                        "census_pct": round(census, 1) if census is not None else None,
                        "gap_pct_wvs_minus_census": round(pct - census, 1)
                        if census is not None
                        else None,
                    }
                )
        return pd.DataFrame(top5)

    top5 = save_top5()
    top5.to_csv(OUT / "top5_vs_census.csv", index=False)
    return (top5,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Figure: WVS vs census side by side

    The one figure this analysis produces; written to `code/figures/quota/`.
    The mismatch pattern (WVS skews to 65+, female, higher education) is
    the target of the quota design.
    """)
    return


@app.cell
def _(FIGURE_DIR, np, plt, top5):
    # The four demographics the quota design targets (education is not
    # used as a quota question).
    FOCUS = ["Age group", "Sex", "Employment status", "Personal income (3-level)"]
    # Display order of values within each group (others: as in top5,
    # i.e. by WVS share descending).
    VALUE_ORDER = {
        "Employment status": [
            "Employed full-time",
            "Employed part-time",
            "Unemployed",
            "Not in the labour force",
        ],
    }

    def make_figure() -> None:
        # Group the rows by demographic, in display order.
        groups = {name: [] for name in FOCUS}
        for _, row in top5.iterrows():
            if row["demographic"] not in FOCUS:
                continue
            if row["census_pct"] is None or (
                isinstance(row["census_pct"], float) and np.isnan(row["census_pct"])
            ):
                continue
            groups[row["demographic"]].append(
                (row["value"], row["wvs_pct"], row["census_pct"])
            )
        ordered = []
        for name in FOCUS:
            rows = groups[name]
            if name in VALUE_ORDER:
                order = {v: i for i, v in enumerate(VALUE_ORDER[name])}
                rows = sorted(rows, key=lambda r: order.get(r[0], 999))
            ordered.append((name, rows))

        # Lay out rows top-down: heading slot per group, one slot per
        # value pair, and a separator in the gap between groups.
        bars, heads, seps = [], [], []  # (y, ...) tuples
        y = 0.0
        for g_i, (name, rows) in enumerate(ordered):
            if g_i:
                seps.append(y - 0.9)  # halfway through the 1.8 gap
                y -= 1.8
            heads.append((y, name))
            y -= 1.0
            for value, w, c in rows:
                bars.append((y, value, w, c))
                y -= 1.0

        fig, ax = plt.subplots(figsize=(10, 8))
        ypos = [b[0] for b in bars]
        wvs_vals = [b[2] for b in bars]
        census_vals = [b[3] for b in bars]
        h = 0.38
        ax.barh(
            [p + h / 2 for p in ypos],
            wvs_vals,
            height=h,
            label="WVS Wave 7 NZ (n=1057)",
            color="#4c72b0",
        )
        ax.barh(
            [p - h / 2 for p in ypos],
            census_vals,
            height=h,
            label="2023 NZ Census",
            color="#c44e52",
        )
        ax.set_yticks(ypos)
        ax.set_yticklabels([b[1] for b in bars], fontsize=9)
        # Demographic name as a bold subheading to the left of its group.
        for hy, name in heads:
            ax.text(
                -0.02,
                hy,
                name,
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=10,
                fontweight="bold",
            )
        # Horizontal separator lines between groups.
        for sy in seps:
            ax.axhline(sy, color="0.75", lw=0.8)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Percent of population / respondents")
        ax.set_title(
            "Most common demographic values: WVS vs 2023 Census\n"
            "(broadly representative, but the WVS sample skews old)"
        )
        ax.legend(fontsize=9, loc="lower right")
        for p, w, c in zip(ypos, wvs_vals, census_vals):
            ax.text(w + 0.6, p + h / 2, f"{w:.1f}", va="center", fontsize=7)
            ax.text(c + 0.6, p - h / 2, f"{c:.1f}", va="center", fontsize=7)
        ax.set_xlim(0, max(wvs_vals + census_vals) * 1.25)
        ax.set_ylim(min(ypos) - 1.0, 0.8)
        plt.tight_layout()
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(FIGURE_DIR / "top5-wvs-vs-census.png", dpi=150)
        plt.close()

    make_figure()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Which demographics predict value answers?

    Held-out CV log-loss (multinomial ridge, 5-fold x 2) where the
    prediction targets are the value-survey outcomes in
    `output/wvs_value_survey.csv` and the predictors are this notebook's
    labelled demographics. For each outcome, compare log-loss of
    age alone, predictor X alone, and age + X. `gain_added_to_age` is
    the mean log-loss reduction from adding the predictor to age;
    positive = the predictor carries extra information about how people
    answer the value questions. Probabilities are smoothed with 1% mass
    per class and outcomes with a class of <10 respondents are dropped.
    Religiosity (Q6, the importance of religion) is used as a predictor
    and removed from the outcome set while it is in use. Writes
    `output/quota_analysis/value_predictiveness.csv`. Runs for a few
    minutes.
    """)
    return


@app.cell
def _(OUT, VALUE_SURVEY, demo, np, pd, sklearn):
    def run_predictiveness() -> "pd.DataFrame":
        """Cross-validated log-loss gain of each demographic over age.

        Returns one row per predictor with mean gains across outcomes.
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import OneHotEncoder

        PREDICTORS = [
            "Age group",
            "Sex",
            "Education (3-level)",
            "Personal income (3-level)",
            "Religiosity",
            "Subjective social class",
            "Employment status",
            "Marital status",
        ]
        MIN_CLASS_N = 10
        SMOOTH = 0.01
        REG_C = 0.03
        FOLDS = 5
        REPEATS = 2

        values = pd.read_csv(VALUE_SURVEY, index_col=0)
        values.index = values.index.astype(float)
        joined = demo.copy()
        joined.index = joined.index.astype(float)
        joined["Religiosity"] = values["Q6"]
        joined = joined.dropna(subset=["Age group"])

        # Code -1/'Don't know' stays a class; other negatives missing.
        X_all = joined[PREDICTORS].astype(str).replace("nan", "Missing")

        def onehot(frame: pd.DataFrame) -> np.ndarray:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            return enc.fit_transform(frame)

        def cv_loss(X: np.ndarray, y: np.ndarray, fold_seeds) -> float:
            losses = []
            for seed in fold_seeds:
                for tr, te in StratifiedKFold(
                    n_splits=FOLDS, shuffle=True, random_state=seed
                ).split(X, y):
                    sk = sklearn.linear_model.LogisticRegression
                    model = sk(C=REG_C, max_iter=500)
                    model.fit(X[tr], y[tr])
                    proba = model.predict_proba(X[te])
                    classes = model.classes_
                    # 1% label smoothing over the known classes.
                    proba = (1 - SMOOTH) * proba + SMOOTH / len(classes)
                    ys = (y[te][:, None] == classes[None, :]).astype(float)
                    losses.append(
                        float(np.mean(-np.sum(ys * np.log(proba + 1e-12), axis=1)))
                    )
            return float(np.mean(losses))

        joined_index = joined.index
        records = []
        # Only Q-prefixed columns are value outcomes ('date' etc. excluded).
        outcome_cols = [c for c in values.columns if c.startswith("Q") and c != "Q6"]
        for outcome in outcome_cols:
            y_raw = values.loc[joined_index, outcome]
            counts = y_raw.value_counts()
            keep = y_raw.isin(counts[counts >= MIN_CLASS_N].index) & ~y_raw.isin(
                [-5.0, -4.0, -3.0]
            )
            y_raw = y_raw[keep]
            if y_raw.nunique() < 2:
                continue
            Xc = X_all.loc[y_raw.index]
            y = y_raw.to_numpy().astype(float)
            fold_seeds = [17 * r + 3 for r in range(REPEATS)]
            base = cv_loss(onehot(Xc[["Age group"]]), y, fold_seeds)
            for pred in PREDICTORS[1:]:
                cols = ["Age group", pred]
                plus = cv_loss(onehot(Xc[cols]), y, fold_seeds)
                alone = cv_loss(onehot(Xc[[pred]]), y, fold_seeds)
                records.append(
                    {
                        "outcome": outcome,
                        "predictor": pred,
                        "logloss_age": base,
                        "logloss_age_plus_pred": plus,
                        "logloss_pred_alone": alone,
                        "gain_added_to_age": base - plus,
                    }
                )
        per_outcome = pd.DataFrame(records)
        per_outcome.to_csv(OUT / "value_predictiveness_per_outcome.csv", index=False)
        summary = (
            per_outcome.groupby("predictor")
            .agg(
                gain_added_to_age=("gain_added_to_age", "mean"),
                alone_minus_age=("logloss_pred_alone", "mean"),
                n_outcomes=("outcome", "nunique"),
            )
            .sort_values("gain_added_to_age", ascending=True)
            .round(4)
        )
        return summary

    summary = run_predictiveness()
    summary.to_csv(OUT / "value_predictiveness.csv")
    summary
    return


if __name__ == "__main__":
    app.run()
