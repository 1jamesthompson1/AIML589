import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This notebook wrangles the raw WVS Wave 7 New Zealand dataset into a clean format.

    **Input:** `input/WVS_Wave_7_New_Zealand_Csv_v5.1.csv`
    **Output:** `output/wvs_value_survey.csv` — metadata + value survey and resp_info columns.
    """)
    return


@app.cell
def _():
    import marimo as mo

    import pandas as pd
    import json
    from pathlib import Path

    input_dir = Path("input")
    output_dir = Path("output")
    return input_dir, json, mo, output_dir, pd


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
    for entry in question_mapping:
        sub_qs = entry.get("sub_questions") or [None] * len(entry["column_names"])
        for i, col in enumerate(entry["column_names"]):
            rows.append(
                {
                    "column_name": col,
                    "question": entry["question"],
                    "sub_question": sub_qs[i] if i < len(sub_qs) else None,
                    "question_type": entry["question_type"],
                    "question_format": entry["question_format"],
                    "survey_section": entry.get("survey_section"),
                    "id": entry["id"],
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
    return resp_info_cols, value_survey_columns


@app.cell
def _(input_dir, pd):
    data_file = input_dir / "WVS_Wave_7_New_Zealand_Csv_v5.1.csv"
    wvs_df = pd.read_csv(data_file, sep=";", index_col=False)
    print(f"Loaded {wvs_df.shape[0]} rows, {wvs_df.shape[1]} columns")
    return (wvs_df,)


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


@app.cell
def _(wvs_df):
    region_map = {
        1: "Northland",
        2: "Auckland",
        4: "Bay of Plenty",
        5: "Waikato",
        8: "Hawkes Bay",
        9: "Taranaki",
        11: "Manawatu-Rangitikei",
        13: "Wellington",
        15: "Nelson Bays",
        16: "Marlborough",
        17: "West Coast",
        18: "Canterbury",
        20: "Clutha-Central Otago",
        22: "Southland",
        23: "Gisborne",
        24: "Tasman",
    }
    wvs_df["region"] = wvs_df["N_REGION_WVS"].apply(
        lambda x: (
            region_map.get(int(x), -5)
            if isinstance(x, (int, float)) and x > 0
            else (-5 if x == -5 else x)
        )
    )

    towns_map = {
        1: "Under 2,000",
        3: "5,000-10,000",
        4: "10,000-20,000",
        5: "20,000-50,000",
        6: "50,000-100,000",
        7: "100,000-500,000",
        8: "500,000 and more",
    }
    wvs_df["townsize"] = wvs_df["G_TOWNSIZE"].apply(
        lambda x: (
            towns_map.get(int(x), "Missing")
            if isinstance(x, (int, float)) and x > 0
            else ("Missing" if x == -5 else x)
        )
    )

    settlement_map = {1: "Capital city", 2: "Regional center"}
    wvs_df["settlement"] = wvs_df["H_SETTLEMENT"].apply(
        lambda x: (
            settlement_map.get(int(x), "Missing")
            if isinstance(x, (int, float)) and x > 0
            else ("Missing" if x == -5 else x)
        )
    )

    urbrural_map = {1: "Urban", 2: "Rural"}
    wvs_df["settlement_tye"] = wvs_df["H_URBRURAL"].apply(
        lambda x: (
            urbrural_map.get(int(x), "Missing")
            if isinstance(x, (int, float)) and x > 0
            else ("Missing" if x == -5 else x)
        )
    )

    print("Decoded region, townsize, settlement, and urban/rural")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Handle missing values

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
    ## 8. Subset to value survey + respondent info columns

    Keep metadata, all value survey columns, and respondent information columns
    (renamed with `resp_info_` prefix to distinguish them).
    """)
    return


@app.cell
def _(resp_info_cols, value_survey_columns, wvs_df):
    metadata_columns = ["id", "date", "region", "townsize", "settlement_tye"]

    wvs_value_survey = wvs_df[metadata_columns + value_survey_columns].copy()

    for resp_info_col in resp_info_cols:
        wvs_value_survey[f"resp_info_{resp_info_col}"] = wvs_df[resp_info_col]

    print(
        f"Output: {wvs_value_survey.shape[1]} columns "
        f"({len(metadata_columns)} metadata + {len(value_survey_columns)} value survey + {len(resp_info_cols)} resp_info)"
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
def _(output_dir, wvs_value_survey):
    wvs_value_survey.to_csv(output_dir / "wvs_value_survey.csv", index=False)
    print(
        f"Saved {wvs_value_survey.shape[0]} rows to {output_dir / 'wvs_value_survey.csv'}"
    )
    return


if __name__ == "__main__":
    app.run()
