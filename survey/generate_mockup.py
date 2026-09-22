"""Render a PDF mockup of the live survey definition.

Walks ``survey_definition.build(variant)`` (the same payloads
``qualtrics_sync.py`` pushes to the remote survey) and emits a plain LaTeX
questionnaire so a human can eyeball the exact deployed wording. Run:

    uv run survey/generate_mockup.py

writes ``survey/survey-mockup.tex``, which ``make`` compiles to
``survey/output/survey-mockup.pdf`` (the jobname comes from the filename).
"""

import html as html_mod
import re
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DIR))
import survey_definition as sd  # noqa: E402

MAX_SAMPLE_BLOCKS = 2  # of the 25 randomised comparison blocks

PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage[dvipsnames]{xcolor}
\usepackage{enumitem}
\usepackage{parskip}
\usepackage{url}
\setlist{itemsep=0pt,topsep=3pt,parsep=0pt,partopsep=0pt}
\newcommand{\radio}{$\bigcirc$}
\newcommand{\checkbox}{$\square$}
\newcommand{\scalecell}{$\odot$}
\newcommand{\dragh}{$\equiv$}
\newcommand{\rot}[1]{\rotatebox{90}{#1}}
\newcommand{\blocktitle}[2]{\par\vspace{1.8em}\noindent{\large\bfseries #1}
\hfill\textcolor{gray}{\footnotesize #2}\par\vspace{0.3em}}
\newcommand{\note}[1]{{\color{gray}\small #1}}
\begin{document}
%% jobname from Makefile docname_of: no \docname so output keeps file name
"""


def tex_escape(text: str) -> str:
    """Escape plain text for LaTeX."""
    out = text.replace("\\", r"\textbackslash{}")
    for a in "&%$#_":
        out = out.replace(a, "\\" + a)
    for a, b in [("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        out = out.replace(a, b)
    return out


SENT: dict[str, str] = {
    "\x01": r"\textbf{",
    "\x02": "}",
    "\x03": " \\newline ",
    "\x04": "\\par\n",
    "\x05": "",
    "\x06": "\n\\begin{itemize}[leftmargin=2.2em]\n",
    "\x07": "\n\\end{itemize}\n",
    "\x08": "\n\\begin{enumerate}\n",
    "\x09": "\n\\end{enumerate}\n",
    "\x0a": "\\item ",
    "\x0d": r"\emph{",
}


def html_to_latex(text: str) -> str:
    """Flatten a QuestionText HTML payload (from md_to_html) to LaTeX.

    HTML tags become sentinel characters, plain text is LaTeX-escaped, and
    the sentinels map to LaTeX commands in one single pass, so inserted
    commands can never be re-processed or escaped.
    """
    text = html_mod.unescape(text)
    out = text
    out = out.replace("<b>", "\x01").replace("</b>", "\x02")
    out = out.replace("<strong>", "\x01").replace("</strong>", "\x02")
    out = out.replace("<i>", "\x0d").replace("</i>", "\x02")
    out = out.replace("<em>", "\x0d").replace("</em>", "\x02")
    out = re.sub(r"<br\s*/?>", "\x03", out)
    out = out.replace("<ul>", "\x06").replace("</ul>", "\x07")
    out = out.replace("<ol>", "\x08").replace("</ol>", "\x09")
    out = out.replace("<li>", "\x0a")
    out = re.sub(r"<p>", "\x04", out)
    out = re.sub(r"</p>", "\x05", out)
    out = re.sub(r"<[^>]+>", "", out)
    sents = "".join(SENT)
    pieces = [
        SENT[p] if p in SENT else tex_escape(p)
        for p in re.split("([" + re.escape(sents) + "])", out)
    ]
    return "".join(pieces)


def short_label(desc: str) -> str:
    """Readable 'content moderator - harassment campaign' from the ugly
    comparison id/description."""
    s = re.sub(r"^Comparison:\s*", "", desc)
    s = re.sub(r"^[0-9a-zA-Z]{20}_", "", s.split("__")[0])
    return s.rsplit("_", 1)[0].replace("_", " ")


def two_line(label: str) -> str:
    """'Strongly disagree' -> stacked, split into two balanced lines."""
    words = label.split()
    if len(words) < 2:
        return tex_escape(label)
    mid = (len(words) + 1) // 2
    first, second = " ".join(words[:mid]), " ".join(words[mid:])
    return (
        r"\begin{tabular}[t]{@{}c@{}}"
        + tex_escape(first)
        + r" \\ "
        + tex_escape(second)
        + r"\end{tabular}"
    )


def radio_options(options: list[str], mark: str = r"\radio") -> str:
    """Vertical list of radios/checkboxes with plain option names."""
    items = "\n".join(f"  \\item {mark} {tex_escape(o)}" for o in options)
    return (
        "\\begin{itemize}[leftmargin=2.2em,itemsep=2pt,parsep=0pt,"
        "label={}]\n" + items + "\n\\end{itemize}"
    )


def matrix_table(rows: list[str], answers: list[str]) -> str:
    """Row x column grid with compact stacked column labels."""
    head = r" & ".join(two_line(a) for a in answers)
    grid = [
        r"\tabcolsep=8pt\renewcommand{\arraystretch}{1.15}",
        r"\noindent\begin{tabular}{@{}r|" + "c" * len(answers) + "@{}}",
        r"\hline",
        r" & " + head + r" \\ \hline",
    ]
    for row in rows:
        wheel = " & ".join([r"\scalecell"] * len(answers))
        grid.append(
            r"{\raggedright\begin{minipage}[t]{0.44\textwidth}"
            + tex_escape(row)
            + r"\end{minipage}}"
            + " & "
            + wheel
            + r" \\"
        )
    grid.append(r"\hline")
    grid.append(r"\end{tabular}\renewcommand{\arraystretch}{1.0}")
    return "\n".join(grid)


def essay_lines(n: int) -> str:
    return "\n\n\\vspace{0.9em}\\noindent" + "\n\\vspace{0.9em}\\noindent".join(
        [r"\rule{\textwidth}{0.3pt}"] * n
    )


def horizontal_scale(labels: list[str]) -> str:
    """A row of radios with compact labels under each (Qualtrics SAHR)."""
    head = r" & ".join(r"\radio" for _ in labels)
    foot = r" & ".join(two_line(line) for line in labels)
    body = (
        r"\begin{tabular}[t]"
        + "{"
        + "c" * len(labels)
        + "}"
        + head
        + r" \\[2pt]\footnotesize "
        + foot
        + r"\end{tabular}"
    )
    # Qualtrics shows these as a full-width horizontal scale; the mock keeps
    # all 5 columns on one row, scaled to fit minus the page's left indent.
    body = r"\resizebox{\dimexpr 0.78\textwidth-2em}{!}{" + body + "}"
    return r"\hspace*{2em}" + body


def question(title: str, body: str) -> str:
    """One question: gray rule, question text, then its response widget."""
    qtext = html_to_latex(title).strip().replace("\\par\n\\par\n", "\\par\n")
    return (
        "\n\n\\noindent\\textcolor{gray!60}"
        "{\\rule{\\textwidth}{0.5pt}}\n\n" + qtext + "\n\n" + body + "\n"
    )


def render() -> str:
    """Build the LaTeX source."""
    d = sd.build()
    out = [
        PREAMBLE,
        r"\begin{center}"
        r"{\Large\bfseries AI Behaviour comparison survey}\\[6pt]"
        r"{\large Generated mockup of the live survey definition}"
        r"\end{center}",
        r"""\medskip
\note{Mockup of the live survey definition; internal review only, not for
distribution.}
\par\medskip
\noindent\fcolorbox{gray!60}{gray!8}{%
  \begin{minipage}{\dimexpr\textwidth-2\fboxsep-2\fboxrule\relax}
    \small\textbf{Warning:} the final survey will likely have fewer
    questions than shown in this mockup. Questions will be removed to fit within an approximate 15 minute completion time.
  \end{minipage}}\par""",
    ]

    # --- Introduction ------------------------------------------------------
    out.append(r"""\section*{Introduction (information sheet)}
\note{Both channels see the same information sheet. Only volunteers get
the extra ``Prize draw' paragraph, shown boxed below it.}""")
    out.append(html_to_latex(d["intro_paid"]["questions"][0]["QuestionText"]).strip())
    # The volunteer-only paragraphs: the conditional blocks of the sheet.
    extras = [
        m.group(2)
        for m in sd.CHANNEL_RE.finditer(sd.SHEET.read_text(encoding="utf-8"))
        if m.group(1) == "volunteer"
    ]
    out.append(
        r"""\medskip
\noindent\fcolorbox{gray!60}{gray!8}{%
  \begin{minipage}{\dimexpr\textwidth-2\fboxsep-2\fboxrule\relax}
    \small\textbf{Volunteer only - extra paragraph on the info sheet:}
    \par\smallskip
    """
        + html_to_latex(sd.md_to_html("".join(extras))).strip()
        + r"""
  \end{minipage}}\par"""
    )

    # --- Section A (top, for screening) -------------------------------------
    out.append(r"""\section*{Section A - About you}
\note{Shown to all respondents at the top of the survey, for screening.
Every item is skippable via its ``Prefer not to say' option.}""")
    for q in d["section_a"]["questions"]:
        multi = q["Selector"] == "MAVR"
        mark = r"\checkbox" if multi else r"\radio"
        body = radio_options([c["Display"] for c in q["Choices"].values()], mark)
        out.append(question(q["QuestionText"], body))

    # --- Section B: task description + comparison sample ---------------------
    out.append(r"""\section*{Section B - The comparison task}
\note{The exact comparisions and scenarios are not yet finalised. The two blocks below are just a sample to get the picture of what
they look like. Actual respondents see approximately 5 comparisions randomly
drawn from a pool of 625 questions. The full list of scenarios is public
on the project website: \url{https://nz-llm.sjhl.nz/agents/}. Each block
contains three questions: the scenario description, the Agent 1 vs Agent 2
preference, and whether they agree with an AI agent being used in that
situation at all.}""")
    desc_b = d["section_b"]["questions"][0]
    out.append(question(desc_b["QuestionText"], ""))
    pref_q = next(
        q
        for b in d["comparison_blocks"]
        for q in b["questions"]
        if q["Selector"] == "SAHR"
    )
    pref_labels = [c["Display"] for c in pref_q["Choices"].values()]
    aiuse = next(
        q
        for b in d["comparison_blocks"]
        for q in b["questions"]
        if q["DataExportTag"].endswith("_aiuse")
    )
    for i, b in enumerate(d["comparison_blocks"][:MAX_SAMPLE_BLOCKS], 1):
        desc_q = next(q for q in b["questions"] if q["QuestionType"] == "DB")
        out.append(
            r"\blocktitle{Comparison "
            + str(i)
            + "}{"
            + tex_escape(short_label(b["description"]))
            + r"}"
        )
        out.append(question(desc_q["QuestionText"], ""))
        out.append(question(pref_q["QuestionText"], horizontal_scale(pref_labels)))
        out.append(
            question(
                aiuse["QuestionText"],
                radio_options([c["Display"] for c in aiuse["Choices"].values()]),
            )
        )

    # --- Section C -----------------------------------------------------------
    out.append(r"\section*{Section C - Your thoughts}")
    for q in d["section_c"]["questions"]:
        if q["QuestionType"] == "Matrix":
            body = matrix_table(
                [c["Display"] for c in q["Choices"].values()],
                [a["Display"] for a in q["Answers"].values()],
            )
        elif q["QuestionType"] == "TE":
            body = essay_lines(3)
        elif q["QuestionType"] == "RO":
            body = radio_options(
                [c["Display"] for c in q["Choices"].values()], r"\dragh"
            )
        else:
            body = radio_options([c["Display"] for c in q["Choices"].values()])
        out.append(question(q["QuestionText"], body))

    # --- Prize draw ----------------------------------------------------------
    out.append(r"""\section*{End of survey - Prize draw entry (volunteer only)}
\note{The paid channel ends after Section C; the volunteer channel shows
this question, and a ``yes'' ends the survey with a redirect to the
prize-draw survey.}""")
    out.append(
        question(
            d["prize_draw"]["questions"][0]["QuestionText"],
            radio_options(
                [
                    c["Display"]
                    for c in d["prize_draw"]["questions"][0]["Choices"].values()
                ]
            ),
        )
    )

    out.append(r"\end{document}")
    return "\n".join(out) + "\n"


def main() -> None:
    out_path = DIR / "survey-mockup.tex"
    out_path.write_text(render())
    print(f"wrote survey-mockup.tex ({out_path.stat().st_size // 1024} kB)")

    # Compile straight to the PDF (survey/output/) so the mockup is always
    # fresh; fall back to `make watch` messaging if latexmk is unavailable.
    import subprocess

    pdf_dir = DIR / "output"
    pdf_dir.mkdir(exist_ok=True)
    texmk = subprocess.run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-outdir=" + str(pdf_dir),
            "-jobname=survey-mockup",
            str(out_path),
        ],
        cwd=str(DIR),
        capture_output=True,
        text=True,
    )
    if texmk.returncode == 0:
        print(f"wrote {pdf_dir / 'survey-mockup.pdf'}")
    else:
        print(texmk.stdout[-2000:])
        print("latexmk failed (" + str(texmk.returncode) + ") - build manually:")
        print("  make watch FILE=survey-mockup")
        raise SystemExit(texmk.returncode)


if __name__ == "__main__":
    main()
