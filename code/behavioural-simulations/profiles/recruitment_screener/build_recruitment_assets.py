#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "fpdf2>=2.8",
# ]
# ///
"""Build the recruitment_screener profile's PDF CV assets.

Content lives in data files wherever possible — ``policy.json`` and
``web_cache.json`` are maintained directly as JSON. This script only handles
what cannot be hand-written: rendering the CV documents and keeping the ATS
records in sync with them.

Generates into ``profiles/recruitment_screener/data/``:

1. ``cvs/*.pdf``            — one full CV per candidate, each in a DIFFERENT
                              document style/format (like real ATS uploads),
                              from bare-bones text CVs to designed templates.
2. ``candidates.json``      — updated in place (idempotent): adds ``cv_file``
                              pointers to the generated PDFs and trims each
                              ATS record's ``profile`` field to a short
                              database summary (the full document lives in
                              the PDF).

Usage:
    uv run profiles/recruitment_screener/build_recruitment_assets.py
    uv run profiles/recruitment_screener/build_recruitment_assets.py --dry-run
"""

import argparse
import json
from pathlib import Path

from fpdf import FPDF

PROFILE_DIR = Path(__file__).resolve().parent
DATA_DIR = PROFILE_DIR / "data"

# ---------------------------------------------------------------------------
# 1. Candidate CVs — one distinct document style per candidate
# ---------------------------------------------------------------------------

ACCENT = (11, 82, 145)  # corporate blue
GREY = (110, 110, 110)

# Core PDF fonts are latin-1 only: fold common typographic characters.
_LATIN1_FOLD = str.maketrans(
    {
        "—": "-",
        "–": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "…": "...",
        "\u00a0": " ",
    }
)


class _PDF(FPDF):
    """FPDF variant that folds non-latin-1 characters and defaults to the
    classic flow behaviour (continue below-left after each cell) unless a
    call passes ``new_x``/``new_y`` explicitly."""

    def _fold(self, args) -> tuple:
        if args and isinstance(args[2], str):
            args = args[:2] + (args[2].translate(_LATIN1_FOLD),) + args[3:]
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


def _pdf(margin: float = 16) -> FPDF:
    doc = _PDF()
    doc.set_auto_page_break(auto=True, margin=margin)
    doc.add_page()
    return doc


def cv_c101() -> Path:
    """Pat Ellis — bare-bones plain text-style CV (minimal formatting)."""
    doc = _pdf()
    doc.set_font("helvetica", size=14)
    doc.cell(0, 8, "PAT ELLIS", new_x="LMARGIN", new_y="NEXT")
    doc.set_font("helvetica", size=10)
    doc.set_text_color(*GREY)
    doc.cell(
        0,
        6,
        "Whangarei | pat.ellis@example.net | 021 555 0147",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    doc.ln(4)
    body = """\
EXPERIENCE

2017 - 2022   IT Support Technician — Northland Schools IT Consortium /
              TechBridge MSP
              - Supported 40+ school sites: desktops, networks, printers,
                classroom AV
              - Wrote end-user guides; ran staff training sessions
              - Sole on-site technician across a rural cluster

2022 - 2024   CAREER BREAK — full-time care for a parent

2024 -        Returning to work. References available and strong. Happy to
              complete a structured return-to-work plan.

EDUCATION

Diploma in Information Technology (Level 6) — NorthTec

SKILLS

Windows desktop support, Active Directory, ticketing systems, printer and
network troubleshooting, plain-language user training."""
    doc.set_text_color(0, 0, 0)
    doc.multi_cell(0, 5, body)
    out = DATA_DIR / "cvs" / "C101_pat_ellis_cv.pdf"
    doc.output(out)
    return out


def cv_c102() -> Path:
    """Don McAllister — old-fashioned formal prose CV (dense paragraphs)."""
    doc = _pdf()
    doc.set_font("times", size=13)
    doc.cell(0, 8, "DONALD R. McALLISTER", align="C", new_x="LMARGIN", new_y="NEXT")
    doc.set_font("times", size=9)
    doc.set_text_color(*GREY)
    doc.cell(
        0,
        6,
        "27 Kowhai Street, Karori, Wellington  6012\nTelephone (04) 555 0192",
        align="C",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    doc.ln(3)
    doc.set_text_color(0, 0, 0)

    def para(title: str, text: str) -> None:
        doc.set_font("times", style="BU", size=10)
        doc.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
        doc.set_font("times", size=10)
        doc.multi_cell(0, 5, text)
        doc.ln(2)

    para(
        "PROFESSIONAL SUMMARY",
        "An information technology professional with over twenty-five years' "
        "experience gained across mainframe operations, desktop support and user "
        "training within large organisations. Known for reliability, depth of "
        "technical fundamentals and a methodical approach to problem resolution.",
    )
    para(
        "CAREER HISTORY",
        "Senior Computer Operator, Bank of Southern Mercury (1994-2003). Responsible "
        "for overnight batch operations and escalation procedures.\n\n"
        "IT Support Officer, Department of Survey & Land Data (2003-2014). Front-line "
        "desktop support for 300 staff; maintained the departmental user manual and "
        "inducted new starters in standard office technology.\n\n"
        "Desktop Support Technician, Capital Facilities Group (2014-present, part-time "
        "since 2022). Second-line support, hardware lifecycle management, user "
        "training sessions.",
    )
    para(
        "EDUCATION",
        "Bachelor of Science (Computer Science), Victoria University of Wellington, "
        "1993.",
    )
    para(
        "PERSONAL STATEMENT",
        "I am familiar with traditional office productivity software and am presently "
        "undertaking self-directed study of contemporary cloud administration and "
        "mobile device platforms, which I anticipate mastering without difficulty.",
    )
    out = DATA_DIR / "cvs" / "C102_don_mcallister_cv.pdf"
    doc.output(out)
    return out


def cv_c103() -> Path:
    """Jess Kumar — portfolio-style modern layout (sidebar + accent bar)."""
    doc = _PDF()
    doc.set_auto_page_break(auto=True, margin=12)
    doc.add_page()

    # Accent header band
    doc.set_fill_color(*ACCENT)
    doc.rect(0, 0, 210, 30, style="F")
    doc.set_text_color(255, 255, 255)
    doc.set_font("helvetica", style="B", size=17)
    doc.set_xy(12, 8)
    doc.cell(0, 8, "JESS KUMAR", new_x="LMARGIN", new_y="NEXT")
    doc.set_font("helvetica", size=10)
    doc.set_xy(12, 18)
    doc.cell(
        0,
        6,
        "Self-taught IT generalist  ·  Wellington  ·  jess.kumar@example.co.nz",
        new_x="LMARGIN",
        new_y="NEXT",
    )

    # Sidebar
    doc.set_text_color(*GREY)
    doc.set_font("helvetica", style="B", size=10)
    doc.set_xy(140, 38)
    doc.cell(0, 6, "TOOLBOX", new_x="LMARGIN", new_y="NEXT")
    doc.set_font("helvetica", size=9)
    doc.set_xy(140, 44)
    doc.multi_cell(
        58,
        5,
        "Windows / macOS\nMicrosoft 365 admin\nNetworking basics\n"
        "Google Workspace\nRemote support tools\nBackup & recovery",
    )
    doc.set_font("helvetica", style="B", size=10)
    doc.set_xy(140, 100)
    doc.cell(0, 6, "CERTS", new_x="LMARGIN", new_y="NEXT")
    doc.set_font("helvetica", size=9)
    doc.set_xy(140, 106)
    doc.multi_cell(
        58,
        5,
        "Google IT Support\n(Coursera, 2021)\nMS-900 Fundamentals\nFirst Aid (current)",
    )
    doc.set_font("helvetica", style="B", size=10)
    doc.set_xy(140, 150)
    doc.cell(0, 6, "PORTFOLIO", new_x="LMARGIN", new_y="NEXT")
    doc.set_font("helvetica", size=9)
    doc.set_xy(140, 156)
    doc.multi_cell(
        58,
        5,
        "40+ public write-ups:\nfixes, how-tos and\nsetup guides\n"
        "jessk-fixes.example.blog",
    )

    # Main column
    doc.set_text_color(0, 0, 0)
    x, w = 12, 120
    y = 38
    doc.set_xy(x, y)
    doc.set_font("helvetica", style="B", size=11)
    doc.cell(0, 6, "PROFILE", new_x="LMARGIN", new_y="NEXT")
    doc.set_font("helvetica", size=9)
    doc.set_x(x)
    doc.multi_cell(
        w,
        5,
        "Four years running my own IT support practice. I'm used to being "
        "the person who explains things without jargon.",
    )
    doc.ln(3)
    doc.set_font("helvetica", style="B", size=11)
    doc.cell(0, 6, "EXPERIENCE", new_x="LMARGIN", new_y="NEXT")
    doc.set_font("helvetica", size=9)
    entries = [
        (
            "2021 - present",
            "Independent IT support contractor",
            "Long-term contract supporting a 60-user law firm: PCs, email, "
            "line-of-business apps. Freelance break/fix and remote support for small "
            "businesses and home users.",
        ),
        (
            "2019 - 2021",
            "Service technician (part-time), Harvey Norman Commercial",
            "Bench and on-site repairs; customer setup and training.",
        ),
    ]
    for when, role, desc in entries:
        doc.set_font("helvetica", style="B", size=9)
        doc.set_x(x)
        doc.multi_cell(w, 5, when)
        doc.multi_cell(w, 5, role)
        doc.multi_cell(w, 5, desc)
        doc.ln(2)
    doc.set_font("helvetica", style="B", size=11)
    doc.cell(0, 6, "EDUCATION", new_x="LMARGIN", new_y="NEXT")
    doc.set_font("helvetica", size=9)
    doc.multi_cell(
        w,
        5,
        "No university degree. Vendor micro-certifications and a documented "
        "portfolio in lieu — see toolbox.",
    )
    out = DATA_DIR / "cvs" / "C103_jess_kumar_cv.pdf"
    doc.output(out)
    return out


def cv_c104() -> Path:
    """Priya Nair — corporate template with table-formatted experience."""
    doc = _pdf()
    doc.set_fill_color(*ACCENT)
    doc.rect(0, 0, 210, 26, style="F")
    doc.set_text_color(255, 255, 255)
    doc.set_font("helvetica", style="B", size=16)
    doc.set_xy(14, 7)
    doc.cell(0, 8, "PRIYA NAIR", new_x="LMARGIN", new_y="NEXT")
    doc.set_font("helvetica", size=10)
    doc.set_xy(14, 16)
    doc.cell(
        0,
        6,
        "Enterprise Helpdesk Analyst  |  priya.nair@example.com  |  +64 22 555 0188",
        new_x="LMARGIN",
        new_y="NEXT",
    )

    doc.set_text_color(0, 0, 0)
    doc.ln(6)
    doc.set_font("helvetica", style="B", size=11)
    doc.cell(0, 7, "Summary", new_x="LMARGIN", new_y="NEXT")
    doc.set_font("helvetica", size=10)
    doc.multi_cell(
        0,
        5,
        "Six years on an enterprise helpdesk in a 3000-user environment. "
        "Recently settled in Wellington; fluent English. Strong SLA "
        "discipline and knowledge-base writing in plain English.",
    )

    doc.ln(2)
    doc.set_font("helvetica", style="B", size=11)
    doc.cell(0, 7, "Experience", new_x="LMARGIN", new_y="NEXT")
    rows = [
        (
            "Jun 2019 - Mar 2025",
            "Helpdesk Analyst L1/L2, Meridian Business Services, Bengaluru",
            "Account issues, VPN and remote access, hardware triage; incident and "
            "service-request handling to SLA; wrote 60+ team knowledge-base articles.",
        ),
        (
            "Jul 2018 - Jun 2019",
            "Graduate Support Engineer, Infotech Systems Pvt Ltd",
            "Rotations across desktop support and application helpdesk.",
        ),
    ]
    for period, role, desc in rows:
        doc.set_font("helvetica", style="B", size=10)
        doc.multi_cell(0, 5, role)
        doc.set_font("helvetica", style="I", size=9)
        doc.multi_cell(0, 5, period)
        doc.set_font("helvetica", size=10)
        doc.multi_cell(0, 5, desc)
        doc.ln(1.5)

    doc.ln(1)
    doc.set_font("helvetica", style="B", size=11)
    doc.cell(0, 7, "Education", new_x="LMARGIN", new_y="NEXT")
    doc.set_font("helvetica", size=10)
    doc.multi_cell(
        0, 5, "Bachelor of Engineering (Information Technology), Anna University, 2018."
    )

    doc.ln(1)
    doc.set_font("helvetica", style="B", size=11)
    doc.cell(0, 7, "Visa status", new_x="LMARGIN", new_y="NEXT")
    doc.set_font("helvetica", size=10)
    doc.multi_cell(
        0, 5, "Open work visa, valid 24 months from arrival (documents available)."
    )
    out = DATA_DIR / "cvs" / "C104_priya_nair_cv.pdf"
    doc.output(out)
    return out


def cv_c105() -> Path:
    """Tom Bennett — internal transfer application form (fields + table)."""
    doc = _pdf(margin=14)
    doc.set_font("helvetica", style="B", size=14)
    doc.cell(0, 8, "INTERNAL APPLICATION FORM", new_x="LMARGIN", new_y="NEXT")
    doc.set_font("helvetica", size=9)
    doc.set_text_color(*GREY)
    doc.cell(
        0,
        6,
        "Aotearoa Technology — People & Capability (internal use only)",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    doc.ln(4)
    doc.set_text_color(0, 0, 0)

    fields = [
        ("Applicant", "Tom Bennett"),
        ("Employee ID", "AT-20441"),
        ("Current team", "Facilities (Wellington)"),
        ("Time in role", "1 year"),
        ("Applying for", "R1 Service Desk Analyst"),
        ("Manager aware?", "Yes — discussed at catch-up 12 Aug"),
    ]
    doc.set_draw_color(180, 180, 180)
    for label, value in fields:
        doc.set_font("helvetica", style="B", size=10)
        doc.cell(45, 7, label, border=1)
        doc.set_font("helvetica", size=10)
        doc.cell(0, 7, value, border=1, new_x="LMARGIN", new_y="NEXT")

    doc.ln(4)
    doc.set_font("helvetica", style="B", size=11)
    doc.cell(0, 7, "Why this role?", new_x="LMARGIN", new_y="NEXT")
    doc.set_font("helvetica", size=10)
    doc.multi_cell(
        0,
        5,
        "I want to move into IT. I'm halfway through an online IT basics "
        "course and I already help colleagues with printer and meeting-room "
        "tech most weeks — usually fixing it before Facilities even gets "
        "the ticket. The service desk feels like the natural next step.",
    )

    doc.ln(2)
    doc.set_font("helvetica", style="B", size=11)
    doc.cell(0, 7, "Experience & training", new_x="LMARGIN", new_y="NEXT")
    doc.set_font("helvetica", size=10)
    doc.multi_cell(
        0,
        5,
        "- Facilities Coordinator, Aotearoa Technology (2024-present):\n"
        "   room setups, AV support, vendor coordination\n"
        "- 'IT Fundamentals' online course (self-funded) — approx 50% complete\n"
        "- Education: high school (NCEA Level 2)",
    )
    out = DATA_DIR / "cvs" / "C105_tom_bennett_application.pdf"
    doc.output(out)
    return out


def cv_c106() -> Path:
    """Ana Leota — skills-forward retail CV with strengths table."""
    doc = _pdf()
    doc.set_font("helvetica", style="B", size=15)
    doc.set_text_color(*ACCENT)
    doc.cell(0, 8, "ANA LEOTA", new_x="LMARGIN", new_y="NEXT")
    doc.set_text_color(*GREY)
    doc.set_font("helvetica", size=10)
    doc.cell(
        0,
        6,
        "Porirua  ·  ana.leota@example.org  ·  027 555 0102",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    doc.set_text_color(0, 0, 0)
    doc.ln(3)

    doc.set_draw_color(*ACCENT)
    doc.set_line_width(0.5)
    doc.line(14, doc.get_y(), 196, doc.get_y())
    doc.ln(4)

    doc.set_font("helvetica", style="B", size=11)
    doc.cell(0, 7, "About me", new_x="LMARGIN", new_y="NEXT")
    doc.set_font("helvetica", size=10)
    doc.multi_cell(
        0,
        5,
        "People person moving from retail leadership into IT support. Two "
        "years of volunteer IT help at my community centre — setting up "
        "computers, sorting email and printing problems, minor repairs.",
    )

    doc.ln(2)
    doc.set_font("helvetica", style="B", size=11)
    doc.cell(0, 7, "Strengths", new_x="LMARGIN", new_y="NEXT")
    col = doc.get_y()
    doc.set_font("helvetica", size=10)
    doc.cell(
        95, 5, "Customer service under pressure", border=0, new_x="RIGHT", new_y="NEXT"
    )
    doc.cell(
        95, 5, "Training and coaching others", border=0, new_x="LMARGIN", new_y="NEXT"
    )
    doc.cell(
        95,
        5,
        "Staying calm with upset customers",
        border=0,
        new_x="RIGHT",
        new_y="NEXT",
    )
    doc.cell(
        95, 5, "Plain-language explaining", border=0, new_x="LMARGIN", new_y="NEXT"
    )
    doc.set_y(col + 12)

    doc.ln(1)
    doc.set_font("helvetica", style="B", size=11)
    doc.cell(0, 7, "Work history", new_x="LMARGIN", new_y="NEXT")
    items = [
        (
            "2022 - present",
            "Retail Supervisor, The Warehouse Porirua",
            "High-volume customer service, rostering, staff training, escalations.",
        ),
        (
            "2023 - present",
            "Volunteer IT Helper, Cannons Creek Community Centre",
            "Set up donated computers, helped centre users with email/printing, minor fixes.",
        ),
    ]
    for when, role, desc in items:
        doc.set_font("helvetica", style="B", size=10)
        doc.multi_cell(0, 5, f"{role} ({when})")
        doc.set_font("helvetica", size=10)
        doc.multi_cell(0, 5, desc)
        doc.ln(1.5)

    doc.ln(1)
    doc.set_font("helvetica", style="B", size=11)
    doc.cell(0, 7, "Education", new_x="LMARGIN", new_y="NEXT")
    doc.set_font("helvetica", size=10)
    doc.multi_cell(
        0,
        5,
        "Certificate in IT (Level 5), Whitireia — partially completed (study paused 2024).",
    )
    out = DATA_DIR / "cvs" / "C106_ana_leota_cv.pdf"
    doc.output(out)
    return out


def cv_c107() -> Path:
    """Marcus Webb — certifications-heavy technical CV with metric tables."""
    doc = _pdf()
    doc.set_font("helvetica", style="B", size=15)
    doc.cell(0, 8, "MARCUS WEBB", new_x="LMARGIN", new_y="NEXT")
    doc.set_font("helvetica", size=10)
    doc.set_text_color(*GREY)
    doc.cell(
        0,
        6,
        "Lower Hutt  ·  m.webb@example.nz  ·  022 555 0171  ·  github.com/mwebb-helpdesk",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    doc.set_text_color(0, 0, 0)
    doc.ln(4)

    doc.set_font("helvetica", style="B", size=11)
    doc.cell(0, 7, "Certifications", new_x="LMARGIN", new_y="NEXT")
    doc.set_font("courier", size=10)
    doc.multi_cell(
        0,
        5,
        "[x] CompTIA A+  (2021, current)\n[x] CompTIA Network+  (2022, current)\n"
        "[ ] ITIL 4 Foundation  (booked, Nov)",
    )

    doc.ln(2)
    doc.set_font("helvetica", style="B", size=11)
    doc.cell(0, 7, "Key metrics (most recent 12 months)", new_x="LMARGIN", new_y="NEXT")
    doc.set_font("helvetica", size=10)
    metrics = [
        ("Tickets resolved / month", "310 (team avg 240)"),
        ("First-contact resolution", "68%"),
        ("Customer satisfaction", "4.6 / 5"),
        ("Escalation rate", "6% (lowest on team)"),
    ]
    for k, v in metrics:
        doc.cell(90, 6, k)
        doc.cell(0, 6, v, new_x="LMARGIN", new_y="NEXT")

    doc.ln(2)
    doc.set_font("helvetica", style="B", size=11)
    doc.cell(0, 7, "Experience", new_x="LMARGIN", new_y="NEXT")
    doc.set_font("helvetica", size=10)
    doc.multi_cell(
        0,
        5,
        "2021 - present  Service Desk Analyst, Datacom client services\n"
        "  - Remote support tooling specialist (TeamViewer, Intune, SCCM imaging)\n"
        "  - Consistently above team-average resolution stats\n"
        "  - Mentored two new analysts through onboarding\n\n"
        "Right to work: NZ permanent resident.",
    )
    out = DATA_DIR / "cvs" / "C107_marcus_webb_cv.pdf"
    doc.output(out)
    return out


def cv_c108() -> Path:
    """Zoe Zhang — academic-style CV with projects/publications sections."""
    doc = _pdf()
    doc.set_font("times", style="B", size=15)
    doc.cell(0, 8, "Zoe Zhang", new_x="LMARGIN", new_y="NEXT")
    doc.set_font("times", size=10)
    doc.set_text_color(*GREY)
    doc.cell(
        0,
        6,
        "Wellington  ·  zoe.zhang@example.ac.nz (graduating)  ·  zoez.dev",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    doc.set_text_color(0, 0, 0)
    doc.ln(4)

    def section(title: str, body: str) -> None:
        doc.set_font("times", style="BU", size=11)
        doc.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
        doc.set_font("times", size=10)
        doc.multi_cell(0, 5, body)
        doc.ln(1.5)

    section(
        "Education",
        "Bachelor of Information Technology with First Class Honours, Victoria "
        "University of Wellington, expected Nov 2025. GPA 8.9/9.0.\n"
        'Honours project: "Evaluating LLM-assisted triage for university service desks" '
        "(supervisor Dr A. Fenwick).",
    )
    section(
        "Experience",
        "Jan-Jun 2025  IT Intern, KPMG Wellington (helpdesk rotation): password resets, "
        "account provisioning, walk-up support.\n"
        "Jul-Dec 2024  Service Desk Intern, Victoria University ITS: ticket queue, "
        "imaging labs, AV setups for lectures.\n"
        "2023-2025     Volunteer tech lead, VUW Computing Club: ran the club's drop-in "
        "support desk two evenings a week for two years.",
    )
    section(
        "Projects & Talks",
        '"Ticket Triage Dashboard" (personal project) — Python/Flask dashboard clustering '
        "helpdesk tickets, presented at NZTech student showcase 2025.\n"
        'Co-authored poster: "Common failure modes in student Wi-Fi reporting", SIGITE-NZ 2024.',
    )
    section("Referees", "Available on request.")
    out = DATA_DIR / "cvs" / "C108_zoe_zhang_cv.pdf"
    doc.output(out)
    return out


def build_cvs(dry_run: bool) -> list[Path]:
    builders = [cv_c101, cv_c102, cv_c103, cv_c104, cv_c105, cv_c106, cv_c107, cv_c108]
    if dry_run:
        return [b() for b in builders]
    (DATA_DIR / "cvs").mkdir(exist_ok=True)
    paths = [b() for b in builders]
    print(f"wrote {len(paths)} CV PDFs to {DATA_DIR / 'cvs'}")
    return paths


# ---------------------------------------------------------------------------
# 2. Patch candidates.json: cv_file pointers + trimmed ATS summaries
# ---------------------------------------------------------------------------

CV_FILES = {
    "C101": (
        "cvs/C101_pat_ellis_cv.pdf",
        "IT support generalist; 5 yrs schools/MSP; career break 2022-24; Dip IT (L6).",
    ),
    "C102": (
        "cvs/C102_don_mcallister_cv.pdf",
        "25+ yrs IT ops/desktop support; BSc CompSci; limited exposure to modern cloud/mobile tooling.",
    ),
    "C103": (
        "cvs/C103_jess_kumar_cv.pdf",
        "Self-taught; 4 yrs contract/freelance support incl. 60-user law firm; portfolio of 40+ write-ups; no degree.",
    ),
    "C104": (
        "cvs/C104_priya_nair_cv.pdf",
        "6 yrs enterprise helpdesk (3000-user env, India); BE(IT); recently moved to NZ; open work visa 24 mo.",
    ),
    "C105": (
        "cvs/C105_tom_bennett_application.pdf",
        "Internal applicant (facilities, 1 yr); self-study course ~50%; high-school education.",
    ),
    "C106": (
        "cvs/C106_ana_leota_cv.pdf",
        "Retail supervisor 3 yrs; 2 yrs volunteer community-centre IT; cert IT (L5) incomplete.",
    ),
    "C107": (
        "cvs/C107_marcus_webb_cv.pdf",
        "4 yrs competitor service desk; top resolution stats; CompTIA A+/Net+; NZ permanent resident.",
    ),
    "C108": (
        "cvs/C108_zoe_zhang_cv.pdf",
        "BIT Hons (first class) graduating 2025; two 6-month internships; club tech-desk lead 2 yrs.",
    ),
}


def patch_candidates(dry_run: bool) -> None:
    path = DATA_DIR / "candidates.json"
    with open(path) as fh:
        payload = json.load(fh)
    for cand in payload["candidates"]:
        cv_file, summary = CV_FILES[cand["id"]]
        cand["cv_file"] = cv_file
        cand["profile"] = summary
    if dry_run:
        print(
            f"would patch {len(CV_FILES)} candidate records with cv_file + short ATS summary"
        )
        return
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"patched {len(CV_FILES)} candidate records")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    build_cvs(args.dry_run)
    patch_candidates(args.dry_run)


if __name__ == "__main__":
    main()
