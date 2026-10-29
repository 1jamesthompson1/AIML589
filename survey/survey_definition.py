"""The survey definition used for API sync: the single source of truth.

One survey serves both paid-panel and volunteer respondents. The remote
flow branches on the ``distribution`` embedded field (set by the panel
distribution/email link, read as RecipientType embedded data):

- ``volunteer`` -> Introduction_volunteer (information sheet + the
  prize-raffle paragraph) and, at the end, the prize-draw question and
  its EndSurvey redirect branch.
- anything else -> Introduction_paid (the same information sheet without
  the raffle paragraph). The paid sheet is the default: the branch is on
  ``distribution`` NOT being ``volunteer``, so a link that never set the
  field (an open link, a panel link that lost its parameter) still shows
  the sheet rather than skipping it.

Both channels get Section A (demographics) before the comparison blocks.

Survey structure:

- EmbeddedData: read ``distribution`` from the recipient
- Branch: distribution != volunteer -> Introduction_paid block (the default)
- Branch: distribution == volunteer -> Introduction_volunteer block
- Section A - About you (demographics first, for screening; every
  demographic item skippable), closing with attention check A and a hidden
  page timer
- Section B - The comparison task (a two-sentence description of the
  context of the dilemma task) plus attention check B
- One comparison block per comparison (3 visible questions + a hidden page
  timer), grouped by profile -> dilemma -> pair: an outer randomizer shows
  N_PROFILES_SHOWN of the profiles, each selected profile draws 1 of its
  dilemmas, and each drawn dilemma draws 1 of its pairs
- Per-dilemma agent-order draw: the block carries the description in both
  orders (fine-tune first / base first) with display logic on an embedded
  ``ord_<dilemma>`` field, so each respondent sees one order and the draw
  is in the export
- Section C block (attitude and policy questions), closing with attention
  check C and a hidden page timer
- Branch: a paid respondent who fails two of the three attention checks
  ends through the panel's quality end link (volunteers are never sent to
  the panel)
- Branch: distribution != volunteer -> advanced end-of-survey redirect to
  the panel's "completed" end link (Dynata)
- Branch: distribution == volunteer -> prize-draw block; ``yes`` ends the
  survey with an advanced redirect to the prize-draw survey URL, ``no``
  falls through to the default end-of-survey message

The panel's end links (completed, terminated, quality, over quota) live in
the repo-root ``.env`` (``PANEL_EXIT_*_URL``, see ``.env.example``) because
they are fielding configuration; the panelist id ``psid`` is read from the
entry link and piped back into the redirect. The quality link is set on the
attention gate; the over-quota link is there for Dynata's own quota screen
and the terminated link is reserved for a screenout gate (there is no
screenout in the instrument yet). Every end-of-survey element carries a
response flag (Complete / PoorQuality / Screened / QuotaMet) and the flow
declares ``Q_TerminateFlag`` so the flag exports.

Intro wording comes from participant-information-sheet.md (the shared
information sheet; the volunteer-only raffle paragraph is marked with an
``<!--volunteer ... -->`` block). Comparison content comes from
code/dilemmas/output/comparisons: the survey subset of the newest
comparison run (comparisons.json, schema wvs-dilemma-comparisons/v2) holds
the pairs of runs that become the survey's comparison blocks, one per
comparison. Each block shows the profile's role summary and framing (intro
and outro, from the profile JSON), the dilemma question, and both responses
side by side as Agent 1 / Agent 2, and a hidden timing question. Each
comparison's description is rendered twice (``_0_ft``: fine-tune first,
``_0_base``: base first); display logic on the dilemma's ``ord_<n>`` field
shows one, and that field's value is the agent order the response uses.
Question DataExportTags are the pair's ``<run_id>__<run_id>`` comparison id
with a suffix (_0_ft, _0_base, _pref, _aiuse, _timing), so response columns
join 1:1 with the two runs of the block (join the run ids back through the
comparison manifest's ``agents`` to decode the models).
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

DIR = Path(__file__).resolve().parent
REPO = DIR.parent
# Fielding config (panel exit links, panel id) lives in the repo-root .env;
# loading it here keeps a direct `import survey_definition` working too.
load_dotenv(REPO / ".env")
COMPARISONS_DIR = REPO / "code" / "dilemmas" / "output" / "comparisons"
PROFILES_DIR = REPO / "code" / "dilemmas" / "profiles"

# The comparison manifest the survey's blocks come from.
COMPARISON_SCHEMA = "wvs-dilemma-comparisons/v2"

# The fine-tune id marker used by the served adapters.
FT_MARKER = "-nz-wvs-"


def is_fine_tune_model(model_id: str) -> bool:
    """Whether a served model id is a ``-nz-wvs-`` fine-tune adapter."""
    return FT_MARKER in (model_id or "").rsplit("/", 1)[-1]


def validate_comparison_roles(comparison: dict, declared: dict) -> None:
    """Fail when a comparison's base/ft roles do not match the models.

    The base side must be the non-adapter model and the fine-tune side must
    carry the adapter marker. The subset's top-level ``models`` declaration is
    checked too, so a manifest built in the wrong role order cannot silently
    put the adapter in the base column in the live survey.
    """
    comparison_id = comparison.get("comparison_id", "<unknown>")
    agents = comparison.get("agents") or {}
    base = agents.get("base") or {}
    ft = agents.get("ft") or {}
    if not base or not ft:
        raise SystemExit(
            f"{comparison_id}: comparison must contain agents.base and agents.ft"
        )
    base_model = base.get("model") or ""
    ft_model = ft.get("model") or ""
    if not base_model or not ft_model:
        raise SystemExit(f"{comparison_id}: base/ft model id is missing")
    if is_fine_tune_model(base_model):
        raise SystemExit(
            f"{comparison_id}: base side is a fine-tune ({base_model!r}); "
            "the manifest was built with the roles reversed"
        )
    if not is_fine_tune_model(ft_model):
        raise SystemExit(
            f"{comparison_id}: ft side is not a fine-tune ({ft_model!r}); "
            "the manifest was built with the roles reversed"
        )
    declared_base = declared.get("base") or declared.get("model_a")
    declared_ft = declared.get("ft") or declared.get("model_b")
    if declared_base and declared_base != base_model:
        raise SystemExit(
            f"{comparison_id}: base model {base_model!r} does not match the "
            f"subset's declared base {declared_base!r}"
        )
    if declared_ft and declared_ft != ft_model:
        raise SystemExit(
            f"{comparison_id}: ft model {ft_model!r} does not match the "
            f"subset's declared ft {declared_ft!r}"
        )


# The env var the survey scripts set from --model to pick a per-model subset.
COMPARISON_MODEL_ENV = "WVS_COMPARISON_MODEL"


N_PROFILES_SHOWN = 5

# The survey's description, as shown in the Qualtrics project list and used
# as the page meta description. Qualtrics stores it in the survey options
# (``SurveyMetaDescription``), not in the flow, so ``qualtrics_sync.py``
# checks it against this value and warns when it drifts.
SURVEY_META_DESCRIPTION = (
    "Evaluate and judge AI behavior to see if it thinks and acts like you do."
)

# Comparison block question wording lives in comparison-question.md.
TEMPLATE_PATH = DIR / "comparison-question.md"

SHEET = DIR / "participant-information-sheet.md"

# Conditional paragraph marker: <!--@channel ... --> runs to the next '-->',
# and its body is kept only for that channel (@ = paid|volunteer).
CHANNEL_RE = re.compile(r"<!--\s*(paid|volunteer)\b(.*?)-->", re.S)


def sheet_for_channel(channel: str) -> str:
    """The information sheet with each conditional ``<!--@channel ... -->``
    block limited to that channel: blocks for the other channel are kept
    out, so a respondent only sees content that is true for their channel."""
    text = SHEET.read_text(encoding="utf-8")
    out: list[str] = []
    pos = 0
    for m in CHANNEL_RE.finditer(text):
        out.append(text[pos : m.start()])
        if m.group(1) == channel:
            out.append(m.group(2))
        pos = m.end()
    out.append(text[pos:])
    return "\n".join(out)


OUTLOOK5 = [
    "Very negative",
    "Somewhat negative",
    "Neither negative nor positive",
    "Somewhat positive",
    "Very positive",
]
AIUSE_TEXT = "How much do you agree with an AI agent being used in this work situation?"
AIUSE_SCALE = [
    "Strongly disagree",
    "Disagree",
    "Neither agree nor disagree",
    "Agree",
    "Strongly agree",
]
# Section C: usage frequency, overall outlook, the policy ranking and the
# risk-belief item. The outlook item is bipolar ("positive or negative") so it
# does not presuppose that AI's effect will be positive; "all things
# considered" matches the survey's other attitude item.
C_OUTLOOK_TEXT = (
    "All things considered, how positive or negative do you feel about the "
    "effect AI will have on the future?"
)
C_AIUSE_TEXT = "How often do you use AI tools (e.g. ChatGPT, Copilot, Gemini)?"
AIUSEFREQ5 = [
    "Daily",
    "A couple of times a week",
    "Weekly",
    "Monthly or less often",
    "Never",
]
# Risk-belief item added in the live survey; its live tag Q943 is kept so the
# response column matches (unforced 3-point scale).
C_RISK_BELIEF_TEXT = (
    "I believe that advanced AI systems could pose serious risks to public "
    "safety and national security in New Zealand."
)
C_RISK_BELIEF_OPTIONS = [
    "Strongly disagree",
    "Disagree",
    "Neither agree nor disagree",
    "Agree",
    "Strongly agree",
]
PREF_TEXT = (
    "All things considered, which agent's response is more in line with your values?"
)
PREF_SCALE = [
    "Strongly prefer Agent 1",
    "Somewhat prefer Agent 1",
    "About the same",
    "Somewhat prefer Agent 2",
    "Strongly prefer Agent 2",
]

BLOCK_ID_CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

PREFER_NOT_TEXT = "Prefer not to say"

# --- Section A (all respondents): Zealand Census-style demographics, each ---
# skippable via a "Prefer not to say" option

Z_AGE_TEXT = "Which age group do you belong to?"
# Five-year bands: "18-24" is the only non-five-year band (the panel's
# minimum age is 18) and "65 and older" is open-ended, matching the 2023
# Census five-year age release.
Z_AGE_OPTIONS = [
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
    PREFER_NOT_TEXT,
]
Z_GENDER_TEXT = "Which gender do you identify with?"
Z_GENDER_OPTIONS = [
    "Male",
    "Female",
    "Non-binary or another gender identity",
    PREFER_NOT_TEXT,
]
# The ethnic groups are the 2023 Census categories (Stats NZ), not the WVS
# ones: the released WVS NZ file has no ethnicity question at all (see
# code/training-dataset/quota_analysis.py), so the census is the only source
# these answers can be compared against. The census asks the same
# multi-select question ("which ethnic groups do you identify with"), and
# the six groups below are its top-level groups, so the answers read
# straight off the census release (the shares carried in quota_analysis.py:
# New Zealand European 67.8, Maori 17.8, Asian 17.3, Pacific 8.9, MELAA
# 1.9 per cent of the population).
Z_ETHNICITY_TEXT = (
    "Which ethnic group(s) do you identify with? (Select all that apply.)"
)
Z_ETHNICITY_OPTIONS = [
    "New Zealand European / Pākehā",
    "Māori",
    "Pacific peoples",
    "Asian",
    "Middle Eastern / Latin American / African",
    "Other",
    PREFER_NOT_TEXT,
]
# WVS NZ employment bands (Q279, as coded in code/training-dataset/
# quota_analysis.py and used for the census labour-force comparison).
# Self-employed is a full-time/full-share category (Q281 group 1);
# "Other work/arrangement" is NOT a census or WVS category, it absorbs
# the census "Not in the labour force - other" residual. "Stay at home
# parent/guardian" covers the census "Not in the labour force - unpaid
# domestic work" group, and the looking-after-family work that group
# does not name. An explicit "Casual" option is not included: Q279's
# part-time/full-time hours split already captures the hours work
# pattern, and casualness (rolling contract type) is orthogonal to
# hours.
Z_EMPLOY_TEXT = "What is your current employment status?"
Z_EMPLOY_OPTIONS = [
    "Full time (30+ hrs/wk)",
    "Part time (<30 hrs)",
    "Self employed",
    "Retired/pensioned",
    "Stay at home parent/guardian",
    "Student",
    "Unemployed",
    "Other work/arrangement",
    PREFER_NOT_TEXT,
]

Z_REGION_TEXT = "Which region of New Zealand do you live in?"
# The regions are the panel's own (Dynata): the option labels are the
# panel's region values, so a panel-supplied value and a human answer are
# the same string in the export (see PANEL_DEMOGRAPHICS). "Other" catches
# the people the panel's list does not cover - the volunteer channel has no
# panel value at all - and "I live overseas" keeps that case apart from it.
Z_REGION_OPTIONS = [
    "Auckland",
    "Bay of Plenty",
    "Canterbury",
    "Chatham Islands",
    "Gisborne",
    "Hawke's Bay",
    "Manawatu-Wanganui",
    "Marlborough",
    "Nelson",
    "Niue",
    "Northland",
    "Otago",
    "Southland",
    "Taranaki",
    "Tasman",
    "Waikato",
    "Wellington",
    "West Coast",
    "Other",
    "I live overseas",
    PREFER_NOT_TEXT,
]
Z_INCOME_TEXT = "What is your total annual personal income (before tax), in New Zealand dollars (NZD)?"
Z_INCOME_OPTIONS = [
    "Up to $30,000",
    "$30,001 - $60,000",
    "$60,001 - $100,000",
    "$100,001 - $150,000",
    "$150,001 - $200,000",
    "$200,001 or more",
    PREFER_NOT_TEXT,
]

# --- Panel-supplied demographics ---------------------------------------------

# A panel (Dynata) can append variables to the respondent's survey link;
# Qualtrics reads a query parameter whose name matches an embedded data field
# as that field's value. When one of these fields arrives, the value is
# already known, so its question is not asked again (display logic hides it)
# and the respondent's time goes to the comparisons instead.
#
# The panel should append the survey's own option labels - ``age=45-49``,
# ``gender=Male``, ``region=Wellington`` - so a panel value and an answer to
# the same question are the same string in the export.
#
#   (embedded data field, question tag, question text, answer options)
PANEL_DEMOGRAPHICS = (
    ("age", "Z1-1", Z_AGE_TEXT, Z_AGE_OPTIONS),
    ("gender", "Z2-1", Z_GENDER_TEXT, Z_GENDER_OPTIONS),
    ("region", "Z5-1", Z_REGION_TEXT, Z_REGION_OPTIONS),
)

# --- Panel exit redirects ----------------------------------------------------
#
# Dynata returns every respondent to its own end link. The link's ``rst``
# says how the survey finished (1 completed, 2 terminated, 3 over quota) and
# ``qflag=1`` marks a quality termination; ``psid`` is the panelist id from
# the entry link, echoed back so the panel can match the response. The links
# are fielding configuration, not survey content, so they live in the repo
# root `.env` (`.env.example` carries the defaults). ``{psid}`` in the
# configured URL becomes the Qualtrics pipe for the embedded field.
PANEL_ID_FIELD = "psid"
PANEL_EXIT_ENV = {
    "completed": "PANEL_EXIT_COMPLETED_URL",
    "terminated": "PANEL_EXIT_TERMINATED_URL",
    "quality": "PANEL_EXIT_QUALITY_URL",
    "overquota": "PANEL_EXIT_OVERQUOTA_URL",
}
# The Qualtrics "flag response as" value for each exit. Survey Check requires
# every EndSurvey element to carry one; Q_TerminateFlag (declared in the flow)
# is what puts it in the export.
PANEL_EXIT_FLAGS = {
    "completed": "Complete",
    "terminated": "Screened",
    "quality": "PoorQuality",
    "overquota": "QuotaMet",
}


def panel_exit_url(status: str) -> str:
    """The Dynata end link for ``status`` with the panelist id piped in.

    Fail fast: a missing link would send the panelist to a dead end with no
    status, so the build stops rather than publishing an unusable exit.
    """
    env = PANEL_EXIT_ENV[status]
    url = os.environ.get(env, "")
    if not url:
        raise RuntimeError(
            f"{env} is not set; add the Dynata {status} end link to .env "
            "(see .env.example)"
        )
    return url.replace("{psid}", f"${{e://Field/{PANEL_ID_FIELD}}}")


def panel_exit_element(status: str, flow_id: str) -> dict:
    """An EndSurvey flow element that redirects to the panel's end link."""
    return {
        "Type": "EndSurvey",
        "FlowID": flow_id,
        "EndingType": "Advanced",
        "Options": {
            "Advanced": "true",
            "SurveyTermination": "Redirect",
            "ResponseFlag": PANEL_EXIT_FLAGS[status],
            "EOSRedirectURL": panel_exit_url(status),
        },
    }


def embedded_data_element(field: str, value: str, flow_id: str) -> dict:
    """A flow element that sets one embedded field to ``value``.

    ``Type: "Custom"`` with a ``Value`` is the platform's own serialization
    for a static embedded-data assignment (a ``Recipient`` field with no
    value is the panel/URL-populated form).
    """
    return {
        "Type": "EmbeddedData",
        "FlowID": flow_id,
        "EmbeddedData": [
            {
                "Description": field,
                "Type": "Custom",
                "Field": field,
                "VariableType": "String",
                "DataVisibility": [],
                "AnalyzeText": False,
                "Value": value,
            }
        ],
    }


# --- Timing ------------------------------------------------------------------

# How long the respondent spent in the comparison task.
#
# Qualtrics times every page and every question itself and exports the
# timings with the responses, so the survey needs no scripting for it (the
# API does not accept a survey's JavaScript). The per-question timings are
# in the export, so the analysis reads them directly.

# The eight ANZSCO major groups (level 1 of the Australian and New Zealand
# Standard Classification of Occupations, 2022 update). The two extra
# options are the survey's own: a respondent who has never worked, and the
# usual skip.
Z_OCCUPATION_TEXT = (
    "Which of these best describes your usual work? If you are retired "
    "or between jobs, what did you mostly work as?"
)
Z_OCCUPATION_OPTIONS = [
    (
        "Managers (e.g. business owners, chief executives, general managers, "
        "department managers)"
    ),
    "Professionals (e.g. doctors, teachers, engineers, lawyers, scientists, accountants)",
    (
        "Technicians and Trades Workers (e.g. electricians, plumbers, "
        "mechanics, engineering technicians)"
    ),
    (
        "Community and Personal Service Workers (e.g. caregivers, police "
        "officers, childcare workers, fitness instructors)"
    ),
    (
        "Clerical and Administrative Workers (e.g. receptionists, office "
        "administrators, payroll clerks, bank tellers)"
    ),
    "Sales Workers (e.g. retail sales assistants, real estate agents, sales representatives)",
    (
        "Machinery Operators and Drivers (e.g. truck drivers, bus drivers, "
        "forklift operators, production machinery operators)"
    ),
    "Labourers (e.g. construction labourers, cleaners, farm labourers, fruit pickers)",
    "I have never had a job",
    PREFER_NOT_TEXT,
]

# The education bands are the WVS NZ ones (Q275A country-specific
# categories, as recoded in code/training-dataset/cluster_respondents.py
# "education_respondent_cs"), with the tertiary end split by how long the
# study was: the WVS file lumps tertiary into "some tertiary" and
# "completed a university or polytechnic degree", while the census
# separates the qualification levels, so the finer bands let the answers
# be compared with both.
Z_EDU_TEXT = "What is the highest level of education you have completed?"
Z_EDU_OPTIONS = [
    "No formal schooling",
    "Primary school (including intermediate) or kura kaupapa (e.g. Years 1-8)",
    "Secondary school for up to 3 years (e.g. NCEA Level 1 or less)",
    "Secondary school for 4 years or more (e.g. NCEA Level 2 or 3)",
    "Tertiary for less than 2 years (trade apprenticeship, certificates or diploma)",
    "Tertiary for 3-4 years (a bachelor's or honours degree, e.g. BA or BSc)",
    "Tertiary for more than 4 years (a master's or doctorate, e.g. MA or PhD)",
    PREFER_NOT_TEXT,
]


PRIZE_DRAW_TEXT = (
    "Would you like to be entered into a prize draw for two $50 gift "
    "cards, this will require you to submit your email address through "
    "another survey"
)
PRIZE_DRAW_OPTIONS = ["yes", "no"]
PRIZE_DRAW_END_URL = "https://vuw.qualtrics.com/jfe/form/SV_brM6NleXCL5Cwsu"
PRIZE_DRAW_TAG = "Q85"


_def_cache: dict = {}


def truncate_desc(text: str, max_len: int = 100) -> str:
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = clean.replace("&nbsp;", " ").replace("&amp;", "&")
    return clean if len(clean) <= max_len else clean[: max_len - 3] + "..."


# The style of a markdown block quote: an indented paragraph with a rule
# down its left edge. The comparison block uses it for the dilemma, so the
# case the agent faced reads as a block of its own - the reader can see
# where the case ends and the agents' answers begin.
QUOTE_STYLE = (
    "margin-left:1.6em; margin-bottom:0.7em; padding-left:1em; "
    "border-left:4px solid #d9d9d9;"
)

# Body paragraphs get a little space below them, so the information sheet's
# paragraphs and the comparison block's parts read as separate blocks
# rather than one wall of text.
PARAGRAPH_STYLE = "margin:0 0 0.7em 0;"

# The University logo on the information sheet: centred, never wider than the
# sheet's text column, and scaled to the column on a narrow screen.
IMAGE_STYLE = "display:block; max-width:320px; width:100%; height:auto; margin:0 auto;"


def md_to_html(text: str, heading_style: str = "") -> str:
    """Minimal markdown->HTML for Qualtrics QuestionText.

    Handles ATX headings (#, ##, ###+), unordered (- ) / ordered (1. )
    lists, bold (**), italic (*), block quotes (> ), a standalone image
    (``![alt](src)``, used for the University logo on the information sheet),
    explicit ``<br>`` line breaks, and blank-line paragraphs. Everything else
    is html-escaped first, so the inline markup transforms, the image tag and
    ``<br>`` are the only HTML injected. ``heading_style`` is added as the
    inline style of heading paragraphs (the comparison block uses it for
    spacing).

    A block quote becomes one indented, ruled paragraph (QUOTE_STYLE): the
    lines inside it keep their own line breaks, and a blank quote line
    starts a new paragraph inside the block. The comparison block uses it to
    set the dilemma (the case the agent faced) apart from the framing
    around it and from the agents' answers below it.
    """
    import html

    br_re = re.compile(r"<br\s*/?>", re.IGNORECASE)
    inline = [
        (re.compile(r"\*\*(.+?)\*\*"), r"<b>\1</b>"),
        (re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)"), r"<i>\1</i>"),
    ]
    spans: list[str] = []
    para: list[str] = []
    list_open: str | None = None
    # The open quote block: a list of paragraphs, each a list of lines.
    quote: list[list[str]] = []
    # The heading style: the caller's own (the comparison block uses it to
    # set the agent labels apart) plus space above and below, so a heading
    # is separated from the text around it even where the caller sets no
    # spacing of its own (the information sheet's headings).
    head_style = heading_style
    if "margin-top" not in head_style:
        head_style = f"{head_style} margin-top:0.9em;".strip()
    if "margin-bottom" not in head_style:
        head_style = f"{head_style} margin-bottom:0.4em;".strip()

    def render(line: str) -> str:
        """One source line as HTML: escaped, with the inline markup applied."""
        content = html.escape(br_re.sub("\x01", line.rstrip()))
        content = content.replace("\x01", "<br />")
        for pat, rep in inline:
            content = pat.sub(rep, content)
        return content

    def flush_para() -> None:
        if para:
            # Wrapped source lines are one paragraph: join with a space, not
            # a hard <br>, so the rendered text flows without mid-sentence
            # breaks; blank lines create the actual paragraph breaks.
            spans.append(f'<p style="{PARAGRAPH_STYLE}">' + " ".join(para) + "</p>")
            para.clear()

    def close_list() -> None:
        nonlocal list_open
        if list_open:
            spans.append(f"</{list_open}>")
            list_open = None

    def flush_quote() -> None:
        """Close the open quote block as one ruled, indented paragraph."""
        nonlocal quote
        blocks = [b for b in quote if b]
        quote = []
        if not blocks:
            return
        content = "<br /><br />".join(
            "<br />".join(render(line) for line in block) for block in blocks
        )
        spans.append(f'<p style="{QUOTE_STYLE}">{content}</p>')

    image_re = re.compile(r"^!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)$")

    for line in text.split("\n"):
        stripped = line.strip()
        heading = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        image = image_re.match(stripped)
        li_u = re.match(r"^[-*]\s+(.*)$", stripped)
        li_o = re.match(r"^(\d+)[.)]\s+(.*)$", stripped)
        if stripped.startswith(">"):
            # A quote line: the block continues until a blank line.
            flush_para()
            close_list()
            content = stripped[1:]
            if content.startswith(" "):
                content = content[1:]
            if content:
                if not quote:
                    quote.append([])
                quote[-1].append(content)
            else:
                # A blank quote line: a paragraph break inside the block.
                quote.append([])
            continue
        if not stripped:
            flush_para()
            close_list()
            flush_quote()
        elif heading:
            flush_para()
            close_list()
            flush_quote()
            style = f' style="{head_style}"' if head_style else ""
            spans.append(f"<p{style}><b>" + heading.group(2) + "</b></p>")
        elif image:
            flush_para()
            close_list()
            flush_quote()
            alt = html.escape(image.group(1), quote=True)
            src = html.escape(image.group(2), quote=True)
            spans.append(
                f'<p style="{PARAGRAPH_STYLE}"><img src="{src}" alt="{alt}" '
                f'style="{IMAGE_STYLE}" /></p>'
            )
        elif li_u or li_o:
            flush_para()
            flush_quote()
            tag = "ul" if li_u else "ol"
            item = li_u.group(1) if li_u else li_o.group(2)
            content = item
            for pat, rep in inline:
                content = pat.sub(rep, content)
            if list_open != tag:
                flush_para()
                close_list()
                spans.append("<" + tag + ">")
                list_open = tag
            spans.append("<li>" + content + "</li>")
        else:
            close_list()
            flush_quote()
            para.append(render(line))
    flush_para()
    close_list()
    flush_quote()
    out = "".join(spans)
    return out


# --- Section C policy question ---------------------------------------------
#
# Policy choice content lives in ``policy_questions.json``: the question,
# intro, the number of choices required, and the policy options as
# title/description pairs. Each option renders as a bold title plus a small-
# font description, so the wording can be edited without touching Python.

POLICY_PATH = DIR / "policy_questions.json"


def policy_questions() -> dict:
    """The policy choice content (validated on load)."""
    data = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if not data.get("options"):
        raise SystemExit(f"{POLICY_PATH} has no policy options")
    low = int(data["min_choices"])
    high = int(data.get("max_choices") or len(data["options"]))
    if not 1 <= low <= high <= len(data["options"]):
        raise SystemExit(
            f"{POLICY_PATH}: need 1 <= min_choices <= max_choices <= "
            f"{len(data['options'])}"
        )
    return data


POLICY = policy_questions()
C_POLICY_TEXT = md_to_html(f"**{POLICY['question']}**\n\n{POLICY['intro']}")
C_POLICY_MIN = int(POLICY["min_choices"])
C_POLICY_MAX = int(POLICY.get("max_choices") or len(POLICY["options"]))
C_POLICY_OPTIONS = [(o["title"], o["description"]) for o in POLICY["options"]]

# The follow-up rank question. Qualtrics' carry-forward choices are a UI
# feature that the question API does not expose, so this question is marked
# manual: the sync warns when it is missing and never creates, updates or
# deletes it. The choices below are only placeholders for the PDF mockup.
C4_RANK_TEXT = md_to_html(
    f"**{POLICY.get('rank_question', 'Now rank the three policies you selected')}**"
    f"\n\n{POLICY.get('rank_intro', 'Rank the three policies you selected above.')}"
)
MANUAL_QUESTION_TAGS = {"C4-4"}


# --- Attention checks --------------------------------------------------------
#
# Three directed-choice checks spread across the survey. The pre-registration
# excludes a respondent who fails two of the three ("failing 2 of the 3
# attention checks"); the flow also ends them through the panel's quality end
# link as soon as the second failure is known. Each item says which option to
# choose, so a wrong answer is unambiguous.
ATTENTION_SCALE = [
    "Strongly agree",
    "Somewhat agree",
    "Neither agree nor disagree",
    "Somewhat disagree",
    "Strongly disagree",
]
# (DataExportTag, question text, 1-based index of the correct option)
ATTENTION_CHECKS = (
    ("A_attention", "Attention check: please select 'Somewhat disagree'.", 4),
    (
        "B_attention",
        (
            "To show you are reading carefully, please select "
            "'Neither agree nor disagree'."
        ),
        3,
    ),
    ("C_attention", "Please select 'Strongly agree' for this question.", 1),
)


# --- Question payload builders (shapes mirror a real Qualtrics export) ----


def embedded_field_empty(field: str) -> dict:
    """Display logic: show the question only when the panel sent no value.

    Qualtrics hides the question (the page stays) when the condition fails,
    so a respondent whose age, gender or region arrived as embedded data is
    not asked it again. ``inPage: false`` keeps the rest of the page.
    """
    return {
        "0": {
            "0": {
                "LogicType": "EmbeddedField",
                "LeftOperand": field,
                "Operator": "Empty",
                "RightOperand": "",
                "_HiddenExpression": False,
                "Type": "Expression",
                "Description": '<span class="ConjDesc">If</span> '
                f'<span class="LeftOpDesc">{field}</span> '
                '<span class="OpDesc">Is Empty</span> ',
            },
            "Type": "If",
        },
        "Type": "BooleanExpression",
        "inPage": False,
    }


def embedded_field_equals(field: str, value: str) -> dict:
    """Display logic: show the question only when ``field == value``.

    Used for the two agent-order variants of a comparison description: the
    per-dilemma order draw sets the field before the comparison blocks, and
    exactly one variant's display logic passes.
    """
    return {
        "0": {
            "0": {
                "LogicType": "EmbeddedField",
                "LeftOperand": field,
                "Operator": "EqualTo",
                "RightOperand": value,
                "_HiddenExpression": False,
                "Type": "Expression",
                "Description": '<span class="ConjDesc">If</span> '
                f'<span class="LeftOpDesc">{field}</span> '
                '<span class="OpDesc">Is Equal to</span> '
                f'<span class="RightOpDesc"> {value} </span>',
            },
            "Type": "If",
        },
        "Type": "BooleanExpression",
        "inPage": False,
    }


def attention_payload(check: tuple) -> dict:
    """The question payload for one attention check (see ATTENTION_CHECKS)."""
    tag, text, _ = check
    return mc_sahr_payload(tag, text, ATTENTION_SCALE)


def timing_payload(tag: str) -> dict:
    """A hidden Timing question (Qualtrics ``Timing`` / ``PageTimer``).

    It records First Click, Last Click, Page Submit (seconds) and Click
    Count for the block's page, so the responses export carries the time
    next to the answers instead of only in the separate timing file. It is
    invisible to respondents, so it has no wording and is skipped by the
    PDF mockup. The tag names the block whose page is timed.
    """
    return {
        "QuestionText": "Timing",
        "DefaultChoices": False,
        "DataExportTag": tag,
        "QuestionType": "Timing",
        "Selector": "PageTimer",
        "Configuration": {
            "QuestionDescriptionOption": "UseText",
            "MinSeconds": 0,
            "MaxSeconds": "0",
        },
        "QuestionDescription": "Timing",
        "Choices": {
            "1": {"Display": "First Click"},
            "2": {"Display": "Last Click"},
            "3": {"Display": "Page Submit"},
            "4": {"Display": "Click Count"},
        },
        "ChoiceOrder": ["1", "2", "3", "4"],
        "Validation": {"Settings": {"Type": "None"}},
        "GradingData": [],
        "Language": [],
        "NextChoiceId": 5,
        "NextAnswerId": 1,
        "DataVisibility": {"Private": False, "Hidden": False},
    }


def db_payload(tag: str, text: str, display_logic: dict | None = None) -> dict:
    payload = {
        "QuestionText": text,
        "DefaultChoices": False,
        "DataExportTag": tag,
        "QuestionType": "DB",
        "Selector": "TB",
        "DataVisibility": {"Private": False, "Hidden": False},
        "Configuration": {"QuestionDescriptionOption": "UseText"},
        "QuestionDescription": truncate_desc(text),
        "ChoiceOrder": [],
        "Validation": {"Settings": {"Type": "None"}},
        "GradingData": [],
        "Language": [],
        "NextChoiceId": 4,
        "NextAnswerId": 1,
    }
    if display_logic:
        payload["DisplayLogic"] = display_logic
    return payload


def mc_single_payload(
    tag: str,
    text: str,
    options: list[str],
    force_response: bool = True,
    display_logic: dict | None = None,
) -> dict:
    settings = {"ForceResponse": "ON" if force_response else "OFF"}
    if force_response:
        settings["ForceResponseType"] = "ON"
    settings["Type"] = "None"
    payload = {
        "QuestionText": text,
        "DefaultChoices": False,
        "DataExportTag": tag,
        "QuestionType": "MC",
        "Selector": "SAVR",
        "SubSelector": "TX",
        "DataVisibility": {"Private": False, "Hidden": False},
        "Configuration": {"QuestionDescriptionOption": "UseText"},
        "QuestionDescription": truncate_desc(text),
        "Choices": {str(i): {"Display": o} for i, o in enumerate(options, 1)},
        "ChoiceOrder": [str(i) for i in range(1, len(options) + 1)],
        "Validation": {"Settings": settings},
        "Language": [],
        "NextChoiceId": len(options) + 1,
        "NextAnswerId": 1,
    }
    if display_logic:
        payload["DisplayLogic"] = display_logic
    return payload


def mc_sahr_payload(
    tag: str,
    text: str,
    options: list[str],
    text_position: str | None = None,
    force_response: bool = True,
) -> dict:
    """Horizontal single-answer MC (Qualtrics SAHR), labels under the options."""
    payload = mc_single_payload(tag, text, options, force_response)
    payload["Selector"] = "SAHR"
    payload["Configuration"] = {
        "QuestionDescriptionOption": "UseText",
        "LabelPosition": "BELOW",
    }
    if text_position:
        payload["Configuration"]["TextPosition"] = text_position
    return payload


def mc_multi_payload(
    tag: str,
    text: str,
    options: list[str],
    min_choices: int | None = None,
    max_choices: int | None = None,
) -> dict:
    """Multiple-select (checkbox) MC question, optionally requiring a choice
    range (for example, choose exactly three)."""
    payload = mc_single_payload(tag, text, options)
    payload["Selector"] = "MAVR"
    payload["SubSelector"] = "TX"
    if min_choices is not None or max_choices is not None:
        settings = {
            "ForceResponse": "ON",
            "ForceResponseType": "ON",
            "Type": "ChoiceRange",
        }
        if min_choices is not None:
            settings["MinChoices"] = str(min_choices)
        if max_choices is not None:
            settings["MaxChoices"] = str(max_choices)
        payload["Validation"] = {"Settings": settings}
    return payload


def ro_rank_payload(tag: str, text: str, options: list[str]) -> dict:
    """Rank Order (RO / DND) question: drag-and-drop ranking of all options."""
    return {
        "QuestionText": text,
        "DefaultChoices": False,
        "DataExportTag": tag,
        "QuestionType": "RO",
        "Selector": "DND",
        "SubSelector": "TX",
        "DataVisibility": {"Private": False, "Hidden": False},
        "Configuration": {"QuestionDescriptionOption": "UseText"},
        "QuestionDescription": truncate_desc(text),
        "Choices": {str(i): {"Display": o} for i, o in enumerate(options, 1)},
        "ChoiceOrder": [str(i) for i in range(1, len(options) + 1)],
        "Validation": {"Settings": {"ForceResponse": "ON", "Type": "None"}},
        "GradingData": [],
        "Language": [],
        "NextChoiceId": len(options) + 1,
        "NextAnswerId": 1,
    }


def manual_rank_payload(tag: str, text: str) -> dict:
    """Manual Rank Order question with UI carry-forward choices.

    Qualtrics' question API has no carry-forward field, so ``qualtrics_sync``
    never sends this payload. It exists for the PDF mockup and the block/flow
    definition; the real question is added once in the Qualtrics UI with
    choices carried forward from C4-3. The three placeholder choices below
    stand in for the respondent's selected policies in the mockup.
    """
    labels = [
        "First selected policy (carried forward from C4-3)",
        "Second selected policy (carried forward from C4-3)",
        "Third selected policy (carried forward from C4-3)",
    ]
    return {
        "QuestionText": text,
        "DefaultChoices": False,
        "DataExportTag": tag,
        "QuestionType": "RO",
        "Selector": "DND",
        "SubSelector": "TX",
        "DataVisibility": {"Private": False, "Hidden": False},
        "Configuration": {"QuestionDescriptionOption": "UseText"},
        "QuestionDescription": truncate_desc(text),
        "Choices": {
            str(index): {"Display": label}
            for index, label in enumerate(labels, start=1)
        },
        "ChoiceOrder": ["1", "2", "3"],
        "Validation": {"Settings": {"ForceResponse": "ON", "Type": "None"}},
        "GradingData": [],
        "Language": [],
        "NextChoiceId": 4,
        "NextAnswerId": 1,
        "Manual": True,
    }


def policy_display(title: str, description: str) -> str:
    """Policy option HTML: bold title + small-font one-line description."""
    return f'<b>{title}</b><br />\n<span style="font-size:11px;">{description}</span>'


def matrix_payload(tag: str, text: str, rows: list[str], answers: list[str]) -> dict:
    """Matrix (Likert) question: ``rows`` x single-choice ``answers``."""
    return {
        "QuestionText": text,
        "DefaultChoices": False,
        "DataExportTag": tag,
        "QuestionType": "Matrix",
        "Selector": "Likert",
        "SubSelector": "SingleAnswer",
        "DataVisibility": {"Private": False, "Hidden": False},
        "Configuration": {
            "QuestionDescriptionOption": "UseText",
            "TextPosition": "inline",
            "MobileFirst": True,
            "RepeatHeaders": "none",
            "WhiteSpace": "OFF",
        },
        "QuestionDescription": truncate_desc(text),
        "Choices": {str(i): {"Display": r} for i, r in enumerate(rows, 1)},
        "Answers": {str(i): {"Display": a} for i, a in enumerate(answers, 1)},
        "ChoiceOrder": [str(i) for i in range(1, len(rows) + 1)],
        "AnswerOrder": [str(i) for i in range(1, len(answers) + 1)],
        "Validation": {
            "Settings": {
                "ForceResponse": "ON",
                "ForceResponseType": "ON",
                "Type": "None",
            }
        },
        "Language": [],
        "NextChoiceId": len(rows) + 1,
        "NextAnswerId": len(answers) + 1,
    }


def te_essay_payload(tag: str, text: str) -> dict:
    return {
        "QuestionText": text,
        "DefaultChoices": False,
        "DataExportTag": tag,
        "QuestionType": "TE",
        "Selector": "ESTB",
        "DataVisibility": {"Private": False, "Hidden": False},
        "Configuration": {"QuestionDescriptionOption": "UseText"},
        "QuestionDescription": truncate_desc(text),
        "ChoiceOrder": [],
        "Validation": {
            "Settings": {
                "ForceResponse": "ON",
                "ForceResponseType": "ON",
                "Type": "None",
            }
        },
        "GradingData": [],
        "Language": [],
        "NextChoiceId": 4,
        "NextAnswerId": 1,
    }


# --- Comparison question template -----------------------------------------


def _load_template() -> str:
    """The auditable comparison question wording, from
    ``comparison-question.md`` - everything after the first '---'."""
    text = (DIR / "comparison-question.md").read_text(encoding="utf-8")
    return text.split("\n---\n", 1)[-1].strip()


# The comparison block's bold section labels (the two agent labels): a
# thin rule above them and space around, so the situation, the question
# and the two answers read as separate parts.
COMPARISON_HEADING_STYLE = (
    "margin-top:1.6em; margin-bottom:0.4em; padding-top:0.9em; "
    "border-top:1px solid #d9d9d9;"
)

# Bridge from the dilemma to the answers, used when a profile has no
# outro line of its own (older profiles).
COMPARISON_OUTRO_FALLBACK = "Here are the two agents' responses to the question:"

# The template's ``{?marker?}`` placeholders.
MARKER_RE = re.compile(r"\{\?(\w+)\?\}")
# A line of the template that sits inside a markdown block quote.
QUOTE_LINE_RE = re.compile(r"^\s*>")


def _fill_markers(text: str, values: dict[str, str]) -> str:
    """Substitute the template's ``{?marker?}`` values.

    A value substituted into a block-quote line keeps the quote prefix on
    every one of its lines, so a multi-line case (an email) stays inside
    the quotation block.
    """
    lines = []
    for line in text.split("\n"):
        in_quote = bool(QUOTE_LINE_RE.match(line))

        def replace(match: re.Match) -> str:
            name = match.group(1)
            if name not in values:
                raise ValueError(
                    f"comparison-question.md uses unknown marker {{?{name}?}}"
                )
            value = values[name].rstrip()
            if in_quote:
                # The template line's own quote prefix covers the first
                # line of the value; the rest need one of their own.
                parts = value.split("\n")
                return parts[0] + "".join("\n> " + part for part in parts[1:])
            return value

        lines.append(MARKER_RE.sub(replace, line))
    return "\n".join(lines)


def description_text(record: dict, responses: list | None = None) -> str:
    """The comparison descriptive question: the template's ``{?...?}``
    markers filled from the block record. ``responses`` is the presentation
    order to render ([Agent 1, Agent 2]; default: base then fine-tune). The
    profile's framing lines may be absent (older profiles); an empty intro
    drops out of the rendered HTML and an empty outro falls back to the
    generic bridge line."""
    pair = record["responses"] if responses is None else responses
    text = _load_template()
    text = _fill_markers(
        text,
        {
            "role": record["role"],
            "dilemma_intro": record["dilemma_intro"],
            "situation": record["situation"],
            "question": record["question"],
            "dilemma_outro": record["dilemma_outro"] or COMPARISON_OUTRO_FALLBACK,
            "response1": pair[0],
            "response2": pair[1],
        },
    )
    unknown = re.search(r"\{\?\w+\?\}", text)
    if unknown:
        raise ValueError(
            f"comparison-question.md uses unknown marker {unknown.group(0)}"
        )
    return md_to_html(text, heading_style=COMPARISON_HEADING_STYLE)


# --- Blocks from the comparison survey subset --------------------------------


def survey_model_choices() -> list[tuple[str, Path]]:
    """(model label, survey manifest path) for every available subset.

    Newest run first. A legacy top-level ``survey/comparisons.json`` (from
    before the per-model layout) is listed under the label ``complete`` so old
    runs keep working.
    """
    choices = []
    for directory in sorted(COMPARISONS_DIR.glob("*"), reverse=True):
        survey_dir = directory / "survey"
        if not survey_dir.is_dir():
            continue
        nested = [
            (child.name, child / "comparisons.json")
            for child in sorted(survey_dir.glob("*"))
            if child.is_dir() and (child / "comparisons.json").exists()
        ]
        if nested:
            choices.extend(nested)
            continue
        legacy = survey_dir / "comparisons.json"
        if legacy.exists():
            choices.append(("complete", legacy))
    return choices


def _comparison_manifest() -> tuple[Path, dict]:
    """The selected model's survey subset manifest and its content.

    The survey subset is the survey's own source of comparisons: one
    comparison is one pair of runs (two models answering the same dilemma),
    keyed by role (``base`` / ``ft``) rather than Agent 1 / Agent 2.
    ``WVS_COMPARISON_MODEL`` (set by the scripts' ``--model``) picks the
    fine-tune; without it there must be exactly one subset, otherwise the
    error lists the choices.
    """
    choices = survey_model_choices()
    if not choices:
        raise SystemExit(
            f"No comparison survey subset under {COMPARISONS_DIR} - run "
            "code/dilemmas/generate_comparisons.py first."
        )
    requested = os.environ.get(COMPARISON_MODEL_ENV, "").strip()
    if requested:
        matches = [path for name, path in choices if name == requested]
        if not matches:
            names = sorted({name for name, _ in choices})
            raise SystemExit(
                f"No survey subset for model {requested!r} under "
                f"{COMPARISONS_DIR} (available: {names}). Run the analysis "
                "notebook first, or pass --model with one of those names."
            )
        path = matches[0]
    elif len(choices) > 1:
        names = sorted({name for name, _ in choices})
        raise SystemExit(
            "Several model survey subsets exist; pass --model to choose one "
            f"(available: {names})."
        )
    else:
        path = choices[0][1]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != COMPARISON_SCHEMA:
        raise SystemExit(
            f"Unknown comparison schema {manifest.get('schema')!r} in {path}"
        )
    return path, manifest


def _profile_sources() -> dict[str, dict]:
    """Profile JSONs keyed by profile id (the ``profile`` field), so the
    survey shows current role and framing wording rather than the copy
    frozen into the comparison manifest."""
    out: dict[str, dict] = {}
    for path in sorted(PROFILES_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"SKIPPED profile file {path.name}: {exc}")
            continue
        bid = data.get("profile") or path.stem
        out[bid] = data
    return out


def survey_records() -> list[dict]:
    """One record per survey block, from the comparison survey subset.

    The survey subset (``comparisons.json``, schema
    ``wvs-dilemma-comparisons/v2``) holds the pairs of runs the survey shows
    respondents: one comparison is one pair of runs from two different
    models answering the same dilemma. Each record keys its two sides by
    role - ``base`` and ``ft`` - not by Agent 1 / Agent 2; the survey's
    per-dilemma order draw decides which side is shown as Agent 1.

    Each block carries two description variants (fine-tune first and base
    first) and a per-dilemma random draw picks which one a respondent sees
    (see ``flow()``). ``base_response`` / ``ft_response`` are the two
    responses keyed by role, so either order can be rendered; ``order_field``
    is the embedded field that records the draw for that dilemma.
    """
    manifest_path, manifest = _comparison_manifest()
    profile_files = _profile_sources()
    declared_models = manifest.get("models") or {}
    records = []
    for comparison in manifest.get("comparisons", []):
        validate_comparison_roles(comparison, declared_models)
        profile_id = comparison.get("profile_id") or ""
        agents = comparison.get("agents") or {}
        base = agents.get("base") or {}
        ft = agents.get("ft") or {}
        responses = comparison.get("responses") or {}
        base_response = base.get("response") or responses.get("base")
        ft_response = ft.get("response") or responses.get("ft")
        if not base_response or not ft_response:
            raise SystemExit(
                f"{comparison.get('comparison_id')}: a response is empty; "
                "the survey subset is incomplete"
            )
        profile = profile_files.get(profile_id, {})
        records.append(
            {
                "comparison_id": comparison["comparison_id"],
                "label": comparison.get("label", comparison["comparison_id"]),
                "profile": profile_id,
                "dilemma": comparison.get("dilemma_id", ""),
                "role": profile.get("role_summary")
                or comparison.get("role")
                or profile_id,
                # Survey framing from the profile JSON (read live, with the
                # manifest's exported copy as fallback); "" for profiles that
                # do not define it.
                "dilemma_intro": profile.get("dilemma_intro")
                or comparison.get("dilemma_intro", ""),
                "dilemma_outro": profile.get("dilemma_outro")
                or comparison.get("dilemma_outro", ""),
                "situation": comparison.get("situation", ""),
                "question": comparison.get("question", ""),
                "responses": [base_response, ft_response],
                "base_response": base_response,
                "ft_response": ft_response,
            }
        )
    if not records:
        raise SystemExit(
            f"No comparisons in {manifest_path} - the survey needs at least one "
            "pair of runs answering the same dilemma."
        )
    # One order draw per dilemma (a respondent sees at most one comparison
    # from it), named by a stable sorted index so the analysis can decode it.
    dilemma_index = {
        dilemma: i
        for i, dilemma in enumerate(sorted({r["dilemma"] for r in records}), 1)
    }
    for record in records:
        record["dilemma_index"] = dilemma_index[record["dilemma"]]
        record["order_field"] = f"ord_{record['dilemma_index']}"
    return records


# --- The definition ---------------------------------------------------------


def build() -> dict:
    """Build the logical survey definition (cached; one survey, two
    ``distribution`` channels)."""
    if _def_cache:
        return _def_cache["d"]
    comparison_blocks = [
        {
            "description": f"Comparison: {record['comparison_id']}",
            "label": record["label"],
            "profile": record["profile"],
            "dilemma": record["dilemma"],
            "order_field": record["order_field"],
            "collapsed": False,
            "questions": [
                # The same comparison in both agent orders; the per-dilemma
                # order draw (flow()) shows exactly one of the two.
                db_payload(
                    record["comparison_id"] + "_0_ft",
                    description_text(
                        record, [record["ft_response"], record["base_response"]]
                    ),
                    display_logic=embedded_field_equals(record["order_field"], "ft"),
                ),
                db_payload(
                    record["comparison_id"] + "_0_base",
                    description_text(
                        record, [record["base_response"], record["ft_response"]]
                    ),
                    display_logic=embedded_field_equals(record["order_field"], "base"),
                ),
                mc_sahr_payload(
                    record["comparison_id"] + "_pref",
                    PREF_TEXT,
                    PREF_SCALE,
                    text_position="above",
                ),
                mc_single_payload(
                    record["comparison_id"] + "_aiuse", AIUSE_TEXT, AIUSE_SCALE
                ),
                timing_payload(record["comparison_id"] + "_timing"),
            ],
        }
        for record in survey_records()
    ]

    intro_paid = {
        "description": "Introduction_paid",
        "collapsed": True,
        "questions": [
            db_payload("D_intro_paid", md_to_html(sheet_for_channel("paid")))
        ],
    }
    intro_volunteer = {
        "description": "Introduction_volunteer",
        "collapsed": True,
        "questions": [
            db_payload("D_intro_volunteer", md_to_html(sheet_for_channel("volunteer")))
        ],
    }
    # The three demographics a panel can supply (age, gender, region) are
    # hidden when the panel already sent them: display logic keeps the
    # question off the page and the embedded value stands in its place.
    panel_logic = {
        tag: embedded_field_empty(field) for field, tag, _, _ in PANEL_DEMOGRAPHICS
    }
    section_a = {
        "description": "Section A - About you",
        "collapsed": True,
        "questions": [
            mc_single_payload(
                "Z1-1", Z_AGE_TEXT, Z_AGE_OPTIONS, display_logic=panel_logic["Z1-1"]
            ),
            mc_single_payload(
                "Z2-1",
                Z_GENDER_TEXT,
                Z_GENDER_OPTIONS,
                display_logic=panel_logic["Z2-1"],
            ),
            mc_multi_payload("Z3-1", Z_ETHNICITY_TEXT, Z_ETHNICITY_OPTIONS),
            mc_single_payload("Z4-1", Z_EMPLOY_TEXT, Z_EMPLOY_OPTIONS),
            mc_single_payload(
                "Z5-1",
                Z_REGION_TEXT,
                Z_REGION_OPTIONS,
                display_logic=panel_logic["Z5-1"],
            ),
            mc_single_payload("Z6-1", Z_INCOME_TEXT, Z_INCOME_OPTIONS),
            mc_single_payload("Z7-1", Z_OCCUPATION_TEXT, Z_OCCUPATION_OPTIONS),
            mc_single_payload("Z8-1", Z_EDU_TEXT, Z_EDU_OPTIONS),
            attention_payload(ATTENTION_CHECKS[0]),
            timing_payload("Z_timing"),
        ],
    }
    section_b = {
        "description": "Section B - The comparison task",
        "collapsed": True,
        "questions": [
            db_payload(
                "D_taskB",
                md_to_html(
                    "In the next part of the survey, you will read short "
                    "situations in which an AI agent handled some dilemma. "
                    "You will be asked to compare between two different responses and provide your "
                    "opinion on which one you prefer. "
                    "\n\nAll situations are "
                    "completely fictional and do not represent any real "
                    "person or situation."
                ),
            ),
            attention_payload(ATTENTION_CHECKS[1]),
        ],
    }
    section_c = {
        "description": "Section C - Your thoughts",
        "collapsed": True,
        "questions": [
            # Risk belief first (live tag Q943), then usage, outlook and the
            # policy multi-choice. Q943 stays unforced, as set live.
            mc_sahr_payload(
                "Q943",
                C_RISK_BELIEF_TEXT,
                C_RISK_BELIEF_OPTIONS,
                force_response=False,
            ),
            mc_sahr_payload("C8-1", C_AIUSE_TEXT, AIUSEFREQ5),
            mc_sahr_payload("C1-1", C_OUTLOOK_TEXT, OUTLOOK5),
            # C4-3 replaces C4-2: a remote question's type cannot be changed
            # in place, so the multi-choice is created under a new tag and the
            # old pick-and-rank question is deleted.
            mc_multi_payload(
                "C4-3",
                C_POLICY_TEXT,
                [policy_display(*option) for option in C_POLICY_OPTIONS],
                C_POLICY_MIN,
                C_POLICY_MAX,
            ),
            manual_rank_payload("C4-4", C4_RANK_TEXT),
            attention_payload(ATTENTION_CHECKS[2]),
            timing_payload("C_timing"),
        ],
    }
    prize_draw = {
        "description": "prize draw",
        "collapsed": True,
        "questions": [
            mc_single_payload(PRIZE_DRAW_TAG, PRIZE_DRAW_TEXT, PRIZE_DRAW_OPTIONS)
        ],
    }

    ordered = [
        intro_paid,
        intro_volunteer,
        section_a,
        section_b,
        *comparison_blocks,
        section_c,
        prize_draw,
    ]
    definition = {
        "survey_name": "AI Behaviour comparison survey",
        "survey_description": SURVEY_META_DESCRIPTION,
        "intro_paid": intro_paid,
        "intro_volunteer": intro_volunteer,
        "section_a": section_a,
        "section_b": section_b,
        "comparison_blocks": comparison_blocks,
        "section_c": section_c,
        "prize_draw": prize_draw,
        "ordered_blocks": ordered,
        "profiles_shown": N_PROFILES_SHOWN,
    }
    _def_cache["d"] = definition
    return definition


def branch(logic: dict, flow: list, flow_id: str) -> dict:
    return {
        "Type": "Branch",
        "FlowID": flow_id,
        "Description": "New Branch",
        "BranchLogic": logic,
        "Flow": flow,
    }


def wrong_choice(qid: str, choice: int) -> dict:
    """The attention-check condition: the correct MC option was not selected."""
    locator = f"q://{qid}/SelectableChoice/{choice}"
    return {
        "LogicType": "Question",
        "QuestionID": qid,
        "QuestionIsInLoop": "no",
        "ChoiceLocator": locator,
        "Operator": "NotSelected",
        "QuestionIDFromLocator": qid,
        "LeftOperand": locator,
        "Type": "Expression",
    }


def all_of(*conditions: dict) -> dict:
    """BranchLogic holding when every condition holds (If + AndIf chain)."""
    logic = {"Type": "BooleanExpression"}
    for i, condition in enumerate(conditions):
        logic[str(i)] = {"0": condition, "Type": "If" if i == 0 else "AndIf"}
    return logic


def attention_failure(checks: tuple, qids_by_tag: dict) -> dict:
    """BranchLogic: a paid respondent failed every named attention check.

    The paid condition keeps volunteers out of the panel's quality exit -
    they have no panel id to echo, so the branch is scoped to paid
    respondents.
    """
    return all_of(
        distribution_condition("volunteer", operator="NotEqualTo"),
        *(
            wrong_choice(
                qids_by_tag[ATTENTION_CHECKS[index][0]],
                ATTENTION_CHECKS[index][2],
            )
            for index in checks
        ),
    )


# Qualtrics branch-logic operators for the embedded-field comparison, and
# the wording the Qualtrics UI shows for each of them.
DISTRIBUTION_OPERATORS = {
    "EqualTo": "Is Equal to",
    "NotEqualTo": "Is Not Equal to",
}


def distribution_condition(value: str, operator: str = "EqualTo") -> dict:
    """Expression: the ``distribution`` embedded field compared to ``value``."""
    op_text = DISTRIBUTION_OPERATORS[operator]
    return {
        "LogicType": "EmbeddedField",
        "LeftOperand": "distribution",
        "Operator": operator,
        "RightOperand": value,
        "_HiddenExpression": False,
        "Type": "Expression",
        "Description": '<span class="ConjDesc">If</span> '
        '<span class="LeftOpDesc">distribution'
        f'</span> <span class="OpDesc">{op_text}'
        '</span> <span class="RightOpDesc"> '
        f"{value} </span>",
    }


def distribution_check(value: str, operator: str = "EqualTo") -> dict:
    """BranchLogic: the ``distribution`` embedded field compared to ``value``.

    ``operator`` is the Qualtrics comparison operator: "EqualTo" (the
    default) or "NotEqualTo". The paid introduction branches on "is not
    volunteer", so Qualtrics shows it to every respondent whose link did
    not set ``distribution`` to ``volunteer`` - the information sheet is
    the default, not something a missing parameter can skip.
    """
    return {
        "0": {"0": distribution_condition(value, operator), "Type": "If"},
        "Type": "BooleanExpression",
    }


def count_flow(items: list[dict]) -> int:
    """Number of flow elements, including nested ones."""
    return sum(1 + count_flow(item.get("Flow", [])) for item in items)


def flow(name_to_id: dict[str, str], qids_by_tag: dict[str, str]) -> dict:
    """The root flow for PUT /API/v3/survey-definitions/{SV}/flow.

    ``name_to_id`` maps block description -> remote Block id (already
    synced); ``qids_by_tag`` maps question tag -> QuestionID (the prize
    and attention branches locate choices on their questions). Structure
    mirrors the hand-built live flow: the two distribution branches for the
    intros (the paid sheet first, as the default for anyone who is not a
    volunteer), demographics for everyone, the two-stage comparison
    sampler, Section C, the attention gates, the paid panel exit, and the
    volunteer prize-draw ending.

    Comparison sampling is stratified by profile: the outer randomizer draws
    ``profiles_shown`` of the per-profile randomizers (in random order);
    each selected profile's randomizer draws one of its dilemmas, and each
    drawn dilemma's randomizer draws one of its comparison pairs. One order
    randomizer per dilemma (before the comparison blocks) sets the embedded
    ``ord_<n>`` field to ``ft`` or ``base``, which selects the matching
    description variant in the drawn block.
    """
    by_profile: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    order_fields = {}
    for b in build()["comparison_blocks"]:
        if b["description"] in name_to_id:
            by_profile[b["profile"]][b["dilemma"]].append(name_to_id[b["description"]])
            order_fields[b["dilemma"]] = b["order_field"]
    profiles = sorted(by_profile)
    # SubSet larger than the pool breaks the respondent flow - sync may run
    # while a runs export is still landing blocks.
    shown = min(build()["profiles_shown"], max(len(profiles), 1))

    next_id = 100
    profile_randomizers = []
    for profile in profiles:
        dilemma_randomizers = []
        for dilemma in sorted(by_profile[profile]):
            pairs = []
            for bid in by_profile[profile][dilemma]:
                next_id += 1
                pairs.append(
                    {
                        "Type": "Block",
                        "ID": bid,
                        "FlowID": f"FL_{next_id}",
                        "Autofill": [],
                    }
                )
            next_id += 1
            dilemma_randomizers.append(
                {
                    "Type": "BlockRandomizer",
                    "FlowID": f"FL_{next_id}",
                    "SubSet": 1,
                    "EvenPresentation": True,
                    "Flow": pairs,
                }
            )
        next_id += 1
        profile_randomizers.append(
            {
                "Type": "BlockRandomizer",
                "FlowID": f"FL_{next_id}",
                "SubSet": 1,
                "EvenPresentation": True,
                "Flow": dilemma_randomizers,
            }
        )
    # One agent-order draw per dilemma: the respondent sees at most one pair
    # from a dilemma, so a per-dilemma coin is a per-comparison coin. The
    # draw sets the field the two description variants' display logic reads.
    order_randomizers = []
    for dilemma in sorted(order_fields):
        field = order_fields[dilemma]
        next_id += 1
        ft_flow_id = f"FL_{next_id}"
        next_id += 1
        base_flow_id = f"FL_{next_id}"
        next_id += 1
        order_randomizers.append(
            {
                "Type": "BlockRandomizer",
                "FlowID": f"FL_{next_id}",
                "SubSet": 1,
                "EvenPresentation": True,
                "Flow": [
                    embedded_data_element(field, "ft", ft_flow_id),
                    embedded_data_element(field, "base", base_flow_id),
                ],
            }
        )
    q85 = qids_by_tag[PRIZE_DRAW_TAG]
    # The embedded data fields: the channel, the demographics a panel
    # can append to the survey link, and the panelist id the link carries
    # (echoed on the panel exit redirect). The panel fields have no value
    # here - Qualtrics fills them from the link's query string ("Value will
    # be set from Panel or URL").
    embedded_fields = [
        {
            "Description": "distribution",
            "Type": "Recipient",
            "Field": "distribution",
            "VariableType": "String",
            "DataVisibility": [],
            "AnalyzeText": False,
        },
        {
            "Description": PANEL_ID_FIELD,
            "Type": "Recipient",
            "Field": PANEL_ID_FIELD,
            "VariableType": "String",
            "DataVisibility": [],
            "AnalyzeText": False,
        },
        {
            # EndSurvey flags write this column (Complete, Screened,
            # PoorQuality, QuotaMet); it is empty for a normal completion
            # that never hit an EndSurvey element.
            "Description": "Q_TerminateFlag",
            "Type": "Recipient",
            "Field": "Q_TerminateFlag",
            "VariableType": "String",
            "DataVisibility": [],
            "AnalyzeText": False,
        },
    ]

    embedded_fields += [
        {
            "Description": field,
            # "Recipient" is the API's only embedded-data type: the value
            # is set from the panel or the survey link's query string.
            "Type": "Recipient",
            "Field": field,
            "VariableType": "String",
            "DataVisibility": [],
            "AnalyzeText": False,
        }
        for field, _, _, _ in PANEL_DEMOGRAPHICS
    ]
    flow_items = [
        {
            "Type": "EmbeddedData",
            "FlowID": "FL_60",
            "EmbeddedData": embedded_fields,
        },
        branch(
            # The paid information sheet is the default: it is shown
            # whenever ``distribution`` is not ``volunteer``, so a link
            # that never set the field still gets the sheet.
            distribution_check("volunteer", operator="NotEqualTo"),
            [
                {
                    "Type": "Standard",
                    "ID": name_to_id["Introduction_paid"],
                    "FlowID": "FL_6",
                    "Autofill": [],
                }
            ],
            "FL_70",
        ),
        branch(
            distribution_check("volunteer"),
            [
                {
                    "Type": "Standard",
                    "ID": name_to_id["Introduction_volunteer"],
                    "FlowID": "FL_69",
                    "Autofill": [],
                }
            ],
            "FL_71",
        ),
        {
            "Type": "Standard",
            "ID": name_to_id["Section A - About you"],
            "FlowID": "FL_8",
            "Autofill": [],
        },
        {
            "Type": "Standard",
            "ID": name_to_id["Section B - The comparison task"],
            "FlowID": "FL_9",
            "Autofill": [],
        },
        # A second failed attention check ends the survey through the panel's
        # quality link: A+B fails here, before the comparisons; A+C and B+C
        # fail after Section C (below).
        branch(
            attention_failure((0, 1), qids_by_tag),
            [panel_exit_element("quality", "FL_74")],
            "FL_75",
        ),
        *order_randomizers,
        {
            "Type": "BlockRandomizer",
            "FlowID": "FL_34",
            "SubSet": shown,
            "EvenPresentation": True,
            "Flow": profile_randomizers,
        },
        {
            "Type": "Standard",
            "ID": name_to_id["Section C - Your thoughts"],
            "FlowID": "FL_32",
            "Autofill": [],
        },
        branch(
            attention_failure((0, 2), qids_by_tag),
            [panel_exit_element("quality", "FL_76")],
            "FL_77",
        ),
        branch(
            attention_failure((1, 2), qids_by_tag),
            [panel_exit_element("quality", "FL_78")],
            "FL_79",
        ),
        # Paid respondents are returned to the panel as completed; the
        # volunteer branch below keeps its own prize-draw ending (and falls
        # through to the default end-of-survey message).
        branch(
            distribution_check("volunteer", operator="NotEqualTo"),
            [panel_exit_element("completed", "FL_72")],
            "FL_73",
        ),
        branch(
            distribution_check("volunteer"),
            [
                {
                    "Type": "Block",
                    "ID": name_to_id["prize draw"],
                    "FlowID": "FL_67",
                    "Autofill": [],
                },
                branch(
                    {
                        "0": {
                            "0": {
                                "LogicType": "Question",
                                "QuestionID": q85,
                                "QuestionIsInLoop": "no",
                                "ChoiceLocator": f"q://{q85}/SelectableChoice/1",
                                "Operator": "Selected",
                                "QuestionIDFromLocator": q85,
                                "LeftOperand": "q://" + q85 + "/SelectableChoice/1",
                                "Type": "Expression",
                            },
                            "Type": "If",
                        },
                        "Type": "BooleanExpression",
                    },
                    [
                        {
                            "Type": "EndSurvey",
                            "FlowID": "FL_63",
                            "EndingType": "Advanced",
                            "Options": {
                                "Advanced": "true",
                                "SurveyTermination": "Redirect",
                                "ResponseFlag": "Complete",
                                "EOSRedirectURL": PRIZE_DRAW_END_URL,
                            },
                        }
                    ],
                    "FL_66",
                ),
            ],
            "FL_61",
        ),
    ]
    return {
        "Type": "Root",
        "FlowID": "FL_1",
        "Flow": flow_items,
        "Properties": {"Count": count_flow(flow_items)},
    }
