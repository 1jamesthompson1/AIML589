import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This notebook wrangles the raw WVS Wave 7 New Zealand dataset into a clean format.

    **Input:** `input/WVS_Wave_7_New_Zealand_Csv_v5.1.csv`
    **Output:** `output/wvs_value_survey.csv` — metadata + value survey and respondent information (for understanding clusters later downstream).
    """)
    return


@app.cell
def _():
    import marimo as mo

    import pandas as pd
    import json
    import hashlib
    from pathlib import Path

    input_dir = Path("input")
    output_dir = Path("output")
    return hashlib, input_dir, json, mo, output_dir, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load data
    """)
    return


@app.cell
def _(json, output_dir, pd):
    # Question mapping used to get the value survey columns and the respondent info columns
    question_mapping = json.load(open(output_dir / "question_mapping.json", "r"))

    rows = []
    for question in question_mapping:
        sub_qs = question.get("sub_questions") or [None] * len(question["column_names"])
        for i, survey_col in enumerate(question["column_names"]):
            rows.append(
                {
                    "column_name": survey_col,
                    "question": question["question"],
                    "sub_question": sub_qs[i] if i < len(sub_qs) else None,
                    "question_type": question["question_type"],
                    "question_format": question["question_format"],
                    "survey_section": question.get("survey_section"),
                    "id": question["id"],
                }
            )
    questions_df = pd.DataFrame(rows)

    value_survey_columns = questions_df[
        questions_df["question_type"] == "value survey"
    ]["column_name"].tolist()
    print(f"Found {len(value_survey_columns)} value survey columns in mapping")

    resp_info_cols = questions_df[
        questions_df["question_type"] == "respondent information"
    ]["column_name"].tolist()
    print(f"Found {len(resp_info_cols)} respondent information columns in mapping")

    questions_df
    return (value_survey_columns,)


@app.cell
def _(input_dir, pd):
    data_file = input_dir / "WVS_Wave_7_New_Zealand_Csv_v5.1.csv"
    text_file = input_dir / "WVS_Wave_7_New_Zealand_CsvText_v5.1.csv"
    wvs_df = pd.read_csv(data_file, sep=";", index_col=False)

    # CSVText has semicolons inside quoted text values (e.g.,
    # "Other missing; Multiple answers Mail (EVS) {other missing}")
    # so pd.read_csv misaligns columns. Use csv.reader instead.
    # Data rows have a trailing semicolon, creating an extra empty field;
    # trim to match header count.
    import csv as csv_mod

    with open(text_file, "r", encoding="utf-8-sig") as f:
        txt_reader = csv_mod.reader(f, delimiter=";")
        txt_headers = next(txt_reader)
        txt_rows = [row[: len(txt_headers)] for row in txt_reader]
    txt_headers_clean = [h.split(" ", 1)[0] if " " in h else h for h in txt_headers]
    wvs_text_df = pd.DataFrame(txt_rows, columns=txt_headers_clean)

    print(f"Loaded {wvs_df.shape[0]} rows, {wvs_df.shape[1]} columns from main CSV")
    return wvs_df, wvs_text_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Output integrity hash

    Compute a SHA-256 hash of the final output CSV so users can verify they
    produced the exact same dataset (raw WVS data cannot be redistributed due to
    licensing, but the output hash serves as a shared reference).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Clean data
    """)
    return


@app.cell
def _(wvs_df):
    # Some column values have a regional prefix (`554xxx`) that needs to be stripped.

    prefix_cols = ["N_REGION_ISO", "N_REGION_WVS", "Q223"]

    for col_to_fix in prefix_cols:
        wvs_df[col_to_fix] = wvs_df[col_to_fix].apply(
            lambda x: x - 554000 if isinstance(x, (int, float)) and x > 10 else x
        )

    print(f"Stripped 554 prefix from: {prefix_cols}")
    return


@app.cell
def _(pd, wvs_df):
    # Create canonical id and date columns
    wvs_df["id"] = (wvs_df["D_INTERVIEW"] - 554070000).astype(str)
    wvs_df["date"] = pd.to_datetime(
        wvs_df["J_INTDATE"], format="%Y%m%d", errors="coerce"
    )
    n_valid_dates = wvs_df["date"].notna().sum()
    print(f"Created id and date columns ({n_valid_dates} valid dates)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Decode numeric region/townsize/settlement codes into labels
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Add in useful respondent information columns

    Have good names as well as text values for the responses. This will make it easier to understand the data later down the pipeline.
    """)
    return


@app.cell
def _():
    using_text_cols = {
        "N_REGION_WVS": "region",
        "G_TOWNSIZE": "townsize",
        "H_SETTLEMENT": "settlement",
        "H_URBRURAL": "settlement_type",
        "Q260": "sex",
        "Q261": "birth_year",
        "Q262": "age",
        "Q263": "immigrant",
        "Q264": "mother_immigrant",
        "Q265": "father_immigrant",
        "Q266": "birth_country",
        "Q267": "mother_birth_country",
        "Q268": "father_birth_country",
        "Q269": "citizen",
        "Q270": "household_size",
        "Q271": "live_with_parents",
        "Q272": "home_language",
        "Q273": "marital_status",
        "Q274": "children",
        "Q275": "education_respondent",
        "Q276": "education_spouse",
        "Q277": "education_mother",
        "Q278": "education_father",
        "Q275A": "education_respondent_cs",
        "Q276A": "education_spouse_cs",
        "Q277A": "education_mother_cs",
        "Q278A": "education_father_cs",
        "Q279": "employment_respondent",
        "Q280": "employment_spouse",
        "Q281": "occupation_respondent",
        "Q282": "occupation_spouse",
        "Q283": "occupation_father",
        "Q284": "employment_sector",
        "Q285": "chief_wage_earner",
        "Q286": "savings",
        "Q287": "social_class",
        "Q288": "income_scale",
        "Q289": "religion",
    }

    print(
        f"Going to clean up {len(using_text_cols)} respondent information columns: \n{list(using_text_cols.values())}"
    )
    return (using_text_cols,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Handle missing values

    The WVS uses negative codes to indicate why a response is missing. The raw data
    contains four codes, which we consolidate into two categories.

    | Code | Meaning | Recoded to | Rationale |
    |------|---------|------------|-----------|
    | `-1` | **Don't know** | `-1` (kept) | Respondent engaged but couldn't choose |
    | `-2` | **Refused** | `-1` (merged) | Also an active non-response |
    | `-3` | **Not applicable** | `-5` (merged) | Structural — question didn't apply |
    | `-5` | **Missing / Not asked** | `-5` (kept) | Structural — question never shown |

    After consolidation:
    - `-1` = **Don't know** — respondent engaged but couldn't or wouldn't answer
    - `-5` = **Missing** — data was never collected (not asked or not applicable)
    """)
    return


@app.cell
def _(wvs_df):
    numeric_cols = wvs_df.select_dtypes("number").columns

    counts_before = {}
    for code in [-2, -3]:
        counts_before[code] = (wvs_df[numeric_cols] == code).sum().sum()

    wvs_df[numeric_cols] = wvs_df[numeric_cols].replace(-2.0, -1.0)
    wvs_df[numeric_cols] = wvs_df[numeric_cols].replace(-3.0, -5.0)

    counts_neg1 = (wvs_df[numeric_cols] == -1.0).sum().sum()
    counts_neg5 = (wvs_df[numeric_cols] == -5.0).sum().sum()

    print(f"Recoded -2 (Refused):      {counts_before[-2]:>6} cells → -1 (Don't know)")
    print(f"Recoded -3 (Not applicable): {counts_before[-3]:>6} cells → -5 (Missing)")
    print(
        f"Final: -1 (Don't know): {counts_neg1:>6} cells  |  -5 (Missing): {counts_neg5:>6} cells"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Subset to value survey + respondent info columns

    Keep metadata, all value survey columns, and the already-renamed
    respondent information columns (prefixed with `resp_info_`).
    """)
    return


@app.cell
def _(using_text_cols, value_survey_columns, wvs_df, wvs_text_df):
    metadata_columns = ["id", "date"]

    wvs_value_survey = wvs_df[metadata_columns + value_survey_columns].copy()

    for original_col, new_col in using_text_cols.items():
        wvs_value_survey[new_col] = wvs_text_df[original_col]

    print(
        f"Output: {wvs_value_survey.shape[1]} columns "
        f"({len(metadata_columns)} metadata + {len(value_survey_columns)} value survey + {len(using_text_cols)} resp_info)"
    )

    wvs_value_survey
    return (wvs_value_survey,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Save output
    """)
    return


@app.cell
def _(hashlib, output_dir, wvs_value_survey):
    import re as regex_mod

    out_path = output_dir / "wvs_value_survey.csv"
    wvs_value_survey.to_csv(out_path, index=False)
    print(f"Saved {wvs_value_survey.shape[0]} rows to {out_path}")

    h = hashlib.sha256()
    with open(out_path, "rb") as output_data_file:
        h.update(output_data_file.read())
    output_hash = h.hexdigest()

    readme_path = output_dir.parent / "README.md"
    readme_text = readme_path.read_text()
    readme_text = regex_mod.sub(
        r"<!-- HASH_START -->.*?<!-- HASH_END -->",
        f"<!-- HASH_START -->{output_hash}<!-- HASH_END -->",
        readme_text,
    )
    readme_path.write_text(readme_text)

    print(f"Output SHA-256: {output_hash}")
    print(f"Saved hash to {output_dir / 'output_hash.json'}")
    print(f"Updated {readme_path}")
    return


if __name__ == "__main__":
    app.run()
