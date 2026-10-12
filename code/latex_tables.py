r"""Shared helpers for writing pandas DataFrames as booktabs LaTeX tables.

Tables are written to ``code/figures/`` and pulled into the report with
``\ctable{<name>}{<caption>}`` (see ``docs/common.tex``); they are wrapped in
``\resizebox{\textwidth}{!}{...}`` so they never overflow the page. Notebooks
in ``code/<stage>/`` import this module after putting ``code/`` on the import
path:

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from latex_tables import escape_latex, write_latex_table, write_tex

Note that pandas 3 ``Styler.to_latex`` does not escape its output, so header
labels and text cells are escaped explicitly with ``escape_header`` and
``escape_latex``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FIGURES_DIR = Path(__file__).resolve().parent / "figures"


def escape_header(label: str) -> str:
    """Escape LaTeX special chars in a column header."""

    return (
        label.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def escape_latex(text: object) -> str:
    """Escape LaTeX special chars in cell text (question/answer strings)."""

    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def highlight_best(df: pd.DataFrame, senses: dict | None = None):
    """Bold the best value per column: max for numeric columns by default,
    or per column via senses={label: "max"|"min"} (e.g. for MultiIndex
    columns keyed by metric)."""

    def highlight(series):
        if senses is None:
            if not pd.api.types.is_numeric_dtype(series):
                return ["" for _ in series]  # e.g. CI columns: no highlighting
            best = series.max()
        else:
            label = series.name[0] if isinstance(series.name, tuple) else series.name
            if label not in senses:
                return ["" for _ in series]
            best = series.max() if senses[label] == "max" else series.min()
        return ["font-weight: bold;" if v == best else "" for v in series]

    return df.style.apply(highlight, axis=0)


def resizebox(tex: str) -> str:
    r"""Wrap a tabular in \resizebox{\textwidth}{!}{...}. The trailing %
    stops TeX adding an end-of-file space, which would otherwise push the
    table a few points past the text width."""

    return "\\resizebox{\\textwidth}{!}{%\n" + tex + "}%"


def write_tex(tex: str, out_name: str, figs_dir: Path | None = None) -> Path:
    """Write raw LaTeX to a file in code/figures (or ``figs_dir``)."""

    out_path = (figs_dir or FIGURES_DIR) / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tex)
    return out_path


def write_latex_table(
    df: pd.DataFrame,
    fmt,
    n_cols: int,
    out_name: str,
    senses: dict | None = None,
    column_format: str | None = None,
    stub_label: str = "Model",
    figs_dir: Path | None = None,
) -> Path:
    """Style, format, and write a DataFrame to a resizebox-wrapped booktabs
    tabular in code/figures. Styler does not escape labels, so index values and
    column labels are escaped here (dilemma ids, headers with % and so on).
    The index name is suppressed (Styler would emit an extra header row), and
    ``stub_label`` is patched into the top-left corner of a multi-level column
    header (pass "" to leave the corner blank).

    ``senses`` keys match the *unescaped* labels; they are escaped with the
    labels so highlighting keeps working (e.g. ``{"TVD (%)": "min"}``)."""

    table = df.copy()
    if isinstance(table.index, pd.MultiIndex):
        table.index = pd.MultiIndex.from_tuples(
            [tuple(escape_header(str(value)) for value in row) for row in table.index],
            names=table.index.names,
        )
    else:
        table.index = pd.Index(
            [escape_header(str(value)) for value in table.index], name=table.index.name
        )
    if isinstance(table.columns, pd.MultiIndex):
        table.columns = pd.MultiIndex.from_tuples(
            [
                tuple(escape_header(str(value)) for value in column)
                for column in table.columns
            ],
            names=table.columns.names,
        )
    else:
        table.columns = pd.Index([escape_header(str(value)) for value in table.columns])
    if senses:
        senses = {escape_header(str(label)): sense for label, sense in senses.items()}

    tex = (
        highlight_best(table, senses=senses)
        .format(fmt, na_rep="--")
        .to_latex(
            convert_css=True,
            column_format=column_format or ("l" + "r" * n_cols),
            hrules=True,
            multicol_align="l",
        )
    )
    if stub_label:
        tex = tex.replace(" & \\multicolumn", f"{stub_label} & \\multicolumn", 1)
    return write_tex(resizebox(tex), out_name, figs_dir)


def write_text_table(
    df: pd.DataFrame,
    out_name: str,
    *,
    column_format: str,
    ragged_columns: tuple[str, ...] = (),
    resize: bool = True,
    figs_dir: Path | None = None,
) -> Path:
    """Write an index-hidden booktabs tabular for a table of prose text.

    Cells and headers are escaped with ``escape_latex``/``escape_header``.
    Columns named in ``ragged_columns`` are set ragged right inside their
    ``p{}`` cells so wrapped prose is not stretched; those columns must be
    declared as ``p{}`` in ``column_format``. Wrapped in a resizebox unless
    ``resize=False``."""

    escaped = df.map(escape_latex).rename(columns=escape_latex)
    for column in ragged_columns:
        # Braces + \par: \raggedright redefines \\, so it must be scoped to
        # the cell and the paragraph ended before the row terminator.
        escaped[column] = r"{\raggedright " + escaped[column] + r"\par}"
    tex = escaped.style.hide(axis="index").to_latex(
        column_format=column_format,
        hrules=True,
    )
    if resize:
        tex = resizebox(tex)
    return write_tex(tex, out_name, figs_dir)
