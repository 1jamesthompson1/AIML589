"""Shared PDF-rendering helper for the profile environment builders.

The builders (``profiles/<id>/build_*.py``, uv inline-scripts) all render
fictional documents as PDFs - the recruitment CVs, the person-document
trays (``data/documents/``). This module holds the machinery they shared
(the latin-1 character folding required by FPDF's core fonts, the flow
defaults and the simple letterhead), so it is defined once. Builders pull
it in with:

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from docbuild import DocPDF, fold_latin1, new_doc_pdf
"""

from fpdf import FPDF

# Core PDF fonts are latin-1 only: fold common typographic characters.
# (applied with str.replace, so multi-character sequences are allowed)
_LATIN1_FOLD = [
    (" - ", "-"),
    ("\u2013", "-"),
    ("\u2018", "'"),
    ("\u2019", "'"),
    ("\u201c", '"'),
    ("\u201d", '"'),
    ("\u2026", "..."),
    ("\u00a0", " "),
]


def fold_latin1(text: str) -> str:
    """Fold non-latin-1 typographic characters to their core-font
    equivalents."""
    for old, new in _LATIN1_FOLD:
        text = text.replace(old, new)
    return text


class DocPDF(FPDF):
    """FPDF variant that folds non-latin-1 characters and defaults to the
    classic flow behaviour (continue below-left after each cell) unless a
    call passes ``new_x``/``new_y`` explicitly."""

    def _fold(self, args) -> tuple:
        if args and isinstance(args[2], str):
            args = args[:2] + (fold_latin1(args[2]),) + args[3:]
        return args

    @staticmethod
    def _flow(kwargs: dict) -> dict:
        kwargs.setdefault("new_x", "LMARGIN")
        kwargs.setdefault("new_y", "NEXT")
        return kwargs

    def cell(self, *args, **kwargs):  # noqa: D102
        return super().cell(*self._fold(args), **self._flow(kwargs))

    def multi_cell(self, *args, **kwargs):  # noqa: D102
        return super().multi_cell(*self._fold(args), **self._flow(kwargs))


def new_doc_pdf(margin: float = 14) -> DocPDF:
    """A fresh A4 document with the standard auto page break."""
    doc = DocPDF()
    doc.set_auto_page_break(auto=True, margin=margin)
    doc.add_page()
    return doc
