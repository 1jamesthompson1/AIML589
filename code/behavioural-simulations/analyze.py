import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import pandas as pd

    # The profiles package lives next to this notebook; make it importable no
    # matter how the notebook is invoked (marimo, flat script, ...).
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from profiles import list_profiles, profile_spec, situations

    FIGS_DIR = Path(__file__).resolve().parent.parent / "figures"
    return FIGS_DIR, Path, list_profiles, pd, profile_spec, situations


@app.cell
def _(FIGS_DIR, Path, list_profiles, pd, profile_spec, situations):
    # The profiles x situations summary table. The profile (and situation)
    # form the DataFrame's index, so pandas' native sparsified index
    # rendering prints the profile label once per group -- no manual cell
    # blanking.

    def table_rows() -> list[tuple[str, str, str]]:
        """(work profile cell, situation name, interactive?) rows, one per
        profile x situation, profiles in registry order. The work profile
        cell is two lines - the job title on the first line and the
        organisation on the second (via \\shortstack, in the \\multirow
        profile label). Interactive situation names are wrapped in \\textbf
        so the table flags them."""
        rows = []
        for profile in list_profiles():
            spec = profile_spec(profile)
            cell = f"\\shortstack[l]{{{spec['name']} \\\\ {spec['organisation']}}}"
            for situation in situations(spec["id"]):
                interactive = situation.get("type") == "interactive"
                name = situation["name"]
                if interactive:
                    name = f"\\textbf{{{name}}}"
                rows.append((cell, name, "Yes" if interactive else "No"))
        return rows

    def render_table() -> pd.DataFrame:
        """The scenarios table: profile (and situation) as a MultiIndex --
        pandas' sparsified index output (\\multirow) then prints the profile
        label once per group of situations."""
        df = pd.DataFrame(
            table_rows(), columns=["Work profile", "Situation", "Interactive?"]
        )
        return df[["Work profile", "Situation"]].set_index(
            ["Work profile", "Situation"]
        )

    def write_table(df: pd.DataFrame, filename: str) -> Path:
        """Write the indexed scenarios table as a booktabs tabular to
        code/figures/ via pandas ``to_latex`` with the sparsified MultiIndex
        (\\multirow; the report loads the multirow package, see
        ``docs/common.tex``).
        """
        tex = df.style.to_latex(column_format="lll", hrules=True, sparse_index=True)
        lines = tex.splitlines(keepends=True)
        rows: list[str] = []
        for i, line in enumerate(lines):
            if (
                line.startswith("\\multirow")
                and i > 0
                and (not rows or rows[-1].strip() != "\\midrule")
            ):
                rows.append("\\midrule\n")
            rows.append(line)
        tex = "".join(rows).replace("\\bottomrule", "\\midrule\n\\bottomrule")
        out_path = FIGS_DIR / filename
        out_path.write_text(tex)
        return out_path

    scenarios_df = render_table()

    write_table(scenarios_df, "behavioural-sim-profiles.tex")

    scenarios_df
    return


if __name__ == "__main__":
    app.run()
