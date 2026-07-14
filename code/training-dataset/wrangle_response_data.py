import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This notebook is used to clean up the dataset.

    INputs is the WVS_Wave_7_New_Zealand_Csv_v5.1.csv file.

    The output is a clear question name and a clean response as both numeric or string. The output is saved in the `output` folder.
    """)
    return


@app.cell
def _():
    import marimo as mo

    import pandas as pd
    from pathlib import Path

    input_dir = Path("input")
    return input_dir, mo, pd


@app.cell
def _(input_dir, pd):
    text_names = pd.read_csv(
        input_dir / "WVS_Wave_7_New_Zealand_CsvText_v5.1.csv", sep=";", nrows=0
    ).columns.tolist()

    question_number_to_name = {
        i: name
        for i, name in [
            col.split(" ", maxsplit=1) for col in text_names if col.startswith("Q")
        ][1:]  # Remove "Q_Mode" from list
    }

    question_number_to_name
    return


@app.cell
def _(input_dir, pd):
    # Read in files

    data_file = input_dir / "WVS_Wave_7_New_Zealand_Csv_v5.1.csv"

    wvs_df = pd.read_csv(data_file, sep=";", index_col=False)
    wvs_df["id"] = wvs_df["D_INTERVIEW"].astype(str)
    wvs_df["date"] = pd.to_datetime(
        wvs_df["J_INTDATE"], format="%Y%m%d", errors="coerce"
    )

    columns_to_drop = list(wvs_df.columns[0:14]) + list(wvs_df.columns[20:26])

    print(f"Dropping {len(columns_to_drop)} columns: {columns_to_drop}")

    wvs_df.drop(columns_to_drop, axis=1, inplace=True)

    wvs_df.head()
    return


if __name__ == "__main__":
    app.run()
