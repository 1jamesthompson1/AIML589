"""The survey definition used for API sync: the single source of truth.

One survey serves both paid-panel and volunteer respondents. The remote
flow branches on the ``distribution`` embedded field (set by the panel
distribution/email link, read as RecipientType embedded data):

- ``paid``      -> Introduction_paid   (information sheet, no raffle paragraph)
- ``volunteer`` -> Introduction_volunteer (same sheet + prize-raffle
  paragraph) and, at the end, the prize-draw question and its EndSurvey
  redirect branch.

Both channels get Section A (demographics) before the comparison blocks.

Survey structure:

- EmbeddedData: read ``distribution`` from the recipient
- Branch: distribution == paid    -> Introduction_paid block
- Branch: distribution == volunteer -> Introduction_volunteer block
- Section A - About you (demographics first, for screening; every item
  skippable)
- Section B - The comparison task (a two-sentence description of the
  context of the scenario task)
- One comparison block per scenario pair (3 questions each), wired into a
  BlockRandomizer showing N_PAIRS_SHOWN of them per respondent
- Section C block (attitude and policy questions)
- Branch: distribution == volunteer -> prize-draw block; ``yes`` ends the
  survey with an advanced redirect to the prize-draw/website URL

Intro wording comes from participant-information-sheet.md (the shared
information sheet; the volunteer-only raffle paragraph is marked with an
``<!--volunteer ... -->`` block). Comparison content comes from
code/behavioural-simulations/output/comparisons: one <pair>.json per
scenario (schema wvs-comparison/v1). Question DataExportTags are the
pair's comparison_id with a suffix (_0, _pref, _aiuse) so response columns
join 1:1 with comparison records (which carry the agents per pair).
"""

import json
import re
from pathlib import Path

DIR = Path(__file__).resolve().parent
REPO = DIR.parent
COMPARISONS_DIR = REPO / "code" / "behavioural-simulations" / "output" / "comparisons"

N_PAIRS_SHOWN = 5

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


KNOWLEDGE5 = ["Nothing", "A little", "A moderate amount", "A lot", "A great deal"]
AGREE5 = [
    "Strongly disagree",
    "Disagree",
    "Neither agree nor disagree",
    "Agree",
    "Strongly agree",
]
SUPPORT5 = [
    "Strongly oppose",
    "Somewhat oppose",
    "Neither support nor oppose",
    "Somewhat support",
    "Strongly support",
]
AIUSE_TEXT = (
    "How much do you agree with an AI agent being used in this work situation at all?"
)
AIUSE_SCALE = [
    "Strongly disagree",
    "Disagree",
    "Neither agree nor disagree",
    "Agree",
    "Strongly agree",
]
C_KNOWLEDGE_TEXT = "How much would you say you know about Artificial Intelligence (AI)?"
C_SUPPORT_TEXT = (
    "All things considered, how much do you support further AI development?"
)
C_AUTOMATION_TEXT = "How much do you support automation in the workplace by AI?"
C_AIUSE_TEXT = "How often do you use AI tools (e.g. ChatGPT, Copilot, Gemini)?"
AIUSEFREQ5 = [
    "Daily",
    "A couple of times a week",
    "Weekly",
    "Monthly or less often",
    "Never",
]
C_JOBLOSS_TEXT = "How concerned are you about AI causing job losses in New Zealand?"
CONCERN5 = [
    "Not at all concerned",
    "Slightly concerned",
    "Moderately concerned",
    "Very concerned",
    "Extremely concerned",
]
C_GOV_TEXT = "Do you think the New Zealand government should do each of the following?"
C_GOV_ROWS = [
    "Increase state capacity to research, develop and deploy AI",
    "Act as a regulator of AI",
    "Have an AI safety institute",
    "Use AI internally to be more efficient",
]
C_TRUST_TEXT = "I trust the organisations developing AI to develop it safely."
C_RISK_TEXT = (
    "Rank the following from 1 (greatest risk to humanity) to 5 (least "
    "risk to humanity). Drag and drop the items to rank them."
)
C_RISK_OPTIONS = [
    "Climate change (sudden and severe changes to Earth's climate)",
    "War / weapons of mass destruction (nuclear war)",
    "Disease / biological threats (natural and synthetic severe pandemics)",
    "Artificial Intelligence (AI)",
    "Natural catastrophes (asteroid/comet impacts and supervolcanoes)",
]
# Rank Order question (Qualtrics RO / DND selector, i.e. drag-and-drop),
# the same type as the remote scratch question Q93. Each respondent ranks
# all options from greatest to least risk.

C_COMMENT_TEXT = (
    "What causes your optimism or pessimism about AI? "
    "Please explain in a few sentences."
)
PREF_TEXT = "All things considered which agent acted more in line with your values?"

BLOCK_ID_CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

PREFER_NOT_TEXT = "Prefer not to say"

# --- Section A (all respondents): Zealand Census-style demographics, each ---
# skippable via a "Prefer not to say" option

Z_AGE_TEXT = "Which age group do you belong to?"
Z_AGE_OPTIONS = [
    "18-24",
    "25-34",
    "35-44",
    "45-54",
    "55-64",
    "65-74",
    "75 or older",
    PREFER_NOT_TEXT,
]
Z_GENDER_TEXT = "Which gender do you identify with?"
Z_GENDER_OPTIONS = [
    "Man",
    "Woman",
    "Non-binary or another gender identity",
    PREFER_NOT_TEXT,
]
Z_ETHNICITY_TEXT = "Which ethnic group(s) do you belong to? (Select all that apply.)"
Z_ETHNICITY_OPTIONS = [
    "New Zealand European / Pākehā",
    "Māori",
    "Samoan",
    "Cook Island Māori",
    "Tongan",
    "Niuean",
    "Other Pacific peoples",
    "Chinese",
    "Indian",
    "Other Asian",
    "Middle Eastern / Latin American / African",
    "Other ethnicity",
    PREFER_NOT_TEXT,
]
Z_REGION_TEXT = "Which region of New Zealand do you live in?"
Z_REGION_OPTIONS = [
    "Northland",
    "Auckland",
    "Waikato",
    "Bay of Plenty",
    "Gisborne",
    "Hawke's Bay",
    "Taranaki",
    "Manawatū-Whanganui",
    "Wellington",
    "Tasman / Nelson / Marlborough / West Coast",
    "Canterbury",
    "Otago",
    "Southland",
    "I live overseas",
    PREFER_NOT_TEXT,
]
Z_INCOME_TEXT = "What is your total annual personal income (before tax)?"
Z_INCOME_OPTIONS = [
    "Up to $30,000",
    "$30,001 - $60,000",
    "$60,001 - $100,000",
    "$100,001 - $150,000",
    "$150,001 or more",
    PREFER_NOT_TEXT,
]

Z_OCCUPATION_TEXT = (
    "Which of these best describes your usual work? If you are retired "
    "or between jobs, what did you mostly work as?"
)
Z_OCCUPATION_OPTIONS = [
    "Professional or technical work (e.g. doctor, teacher, engineer, "
    "nurse, accountant)",
    "High-level administrative work (e.g. banker, senior manager, "
    "senior public servant)",
    "Clerical work (e.g. secretary, clerk, office manager)",
    "Sales work (e.g. shop owner or assistant, sales agent)",
    "Service work (e.g. police officer, waiter, barber, caretaker)",
    "Skilled worker (e.g. electrician, motor mechanic, printer, foreman)",
    "Semi-skilled worker (e.g. bricklayer, bus driver, carpenter, assembler)",
    "Unskilled work (e.g. labourer, porter, cleaner)",
    "I have never had a job",
    PREFER_NOT_TEXT,
]

# WVS NZ education bands (Q275A country-specific categories, as recoded in
# code/training-dataset/cluster_respondents.py "education_respondent_cs").
Z_EDU_TEXT = "What is the highest level of education you have completed?"
Z_EDU_OPTIONS = [
    "No formal schooling",
    "Primary school (including intermediate) or kura kaupapa",
    "Secondary school for up to 3 years",
    "Secondary school for 4 years or more",
    "Some tertiary education (university, polytechnic, wananga, trades or "
    "other training), up to a completed bachelor's degree",
    "Completed a postgraduate degree (e.g. honours, master's or doctorate)",
    PREFER_NOT_TEXT,
]


PRIZE_DRAW_TEXT = (
    "Would you like to be entered into a prize draw for two $50 gift "
    "cards, this will require you to submit your email address through "
    "another survey"
)
PRIZE_DRAW_OPTIONS = ["yes", "no"]
PRIZE_DRAW_END_URL = "https://nz-llm.sjhl.nz"
PRIZE_DRAW_TAG = "Q85"


# --- Census-matched quotas ---------------------------------------------------
#
# Independent hard quotas (EndCurrentSurvey when full) on the Section A
# age (Z1-1) and gender (Z2-1) questions, filled in proportion to the
# 2023 Census shares documented in
# code/training-dataset/quota_analysis.py (the CENSUS dict there).
# Census 5-year bands are regrouped onto the survey's options: "18-24"
# is the notebook's 16-24 approximation (6.2 + 0.4 * 6.4); "65+" splits
# into "65-74" (65-69: 5.1 + 70-74: 4.3) and "75 or older" (75-79: 3.3
# + 80-84: 2.2 + 85+: 1.2 + 0.7). "Non-binary or another gender
# identity" and "Prefer not to say" have no census share and are left
# unquota'd (they absorb rounding slack).
QUOTA_N = 400

CENSUS_AGE_SHARES = {
    "18-24": 6.2 + 0.4 * 6.4,  # 8.76
    "25-34": 6.7 + 7.5,  # 14.2
    "35-44": 6.9 + 6.3,  # 13.2
    "45-54": 6.1 + 6.5,  # 12.6
    "55-64": 6.1 + 5.9,  # 12.0
    "65-74": 5.1 + 4.3,
    "75 or older": 3.3 + 2.2 + 1.2 + 0.7,
}
CENSUS_GENDER_SHARES = {"Man": 49.3, "Woman": 50.3}

# EndSurveyOptions for a hard quota; mirrors the remote default set.
QUOTA_END_OPTIONS = {
    "EndingType": "Default",
    "ResponseFlag": "QuotaMet",
    "SurveyTermination": "DefaultMessage",
    "EmailThankYou": "false",
    "ResponseSummary": "No",
    "ConfirmResponseSummary": "",
    "CountQuotas": "Yes",
    "Screenout": "No",
    "AnonymizeResponse": "No",
    "IgnoreResponse": "No",
}


def quota_definitions() -> list[dict]:
    """Census-matched quota specs: label, question tag and its options,
    per-option census shares."""
    return [
        ("Age", "Z1-1", Z_AGE_TEXT, Z_AGE_OPTIONS, CENSUS_AGE_SHARES),
        ("Gender", "Z2-1", Z_GENDER_TEXT, Z_GENDER_OPTIONS, CENSUS_GENDER_SHARES),
    ]


def _quota_pct_by_option(shares: dict[str, float]) -> dict[str, int]:
    """Percent of QUOTA_N per quota'd option, proportional to the shares
    (largest-remainder rounding so the weights sum to 100). These feed
    the Cross-logic quota's Conjuction weights; Qualtrics derives each
    cell's occurrence count from its weight (pct * QUOTA_N / 100, so
    the 1%-grain cost is QUOTA_N / 100 respondents per point)."""
    denom = sum(shares.values())
    raw = {o: 100 * s / denom for o, s in shares.items()}
    pcts = {o: int(v) for o, v in raw.items()}
    remainder = 100 - sum(pcts.values())
    for o, _ in sorted(raw.items(), key=lambda kv: -(kv[1] % 1))[:remainder]:
        pcts[o] += 1
    return pcts


def quota_payloads(qids_by_tag: dict) -> list[dict]:
    """One hard Qualtrics Cross-logic quota per quota'd Section A
    question (age, gender), referencing the synced question id for its
    tag. Cross quotas keep the quota panel to just two entries: each
    quota is a single ``CrossLogicDef`` with a cell per answer option
    ("option selected AND distribution = paid", census-matched
    ``Occurrences``), like the manually built "Example quota".

    Each cell requires the ``distribution`` embedded field to be
    ``paid``, so volunteer respondents (open link) answer the
    demographic questions but never increment a quota nor get screened
    out by one.
    """
    payloads = []
    for label, tag, text, options, shares in quota_definitions():
        qid = qids_by_tag[tag]
        assert set(shares) <= set(options), f"quota {label}: shares missing"
        # cells exist only for quota'd options, in survey order; the
        # unquota'd ones (non-binary etc.) absorb rounding slack
        pcts = _quota_pct_by_option(shares)
        cells = []
        for i, option in enumerate(options, start=1):
            if option not in shares:
                continue
            n = pcts[option] * QUOTA_N // 100
            cond = {
                "LogicType": "Question",
                "QuestionID": qid,
                "QuestionIsInLoop": "no",
                "ChoiceLocator": f"q://{qid}/SelectableChoice/{i}",
                "Operator": "Selected",
                "QuestionIDFromLocator": qid,
                "LeftOperand": f"q://{qid}/SelectableChoice/{i}",
                "Type": "Expression",
                "Description": '<span class="ConjDesc">And</span> '
                '<span class="QuestionDesc">'
                f"{truncate_desc(text, 60)}</span> "
                f'<span class="LeftOpDesc">{truncate_desc(option, 30)}'
                '</span> <span class="OpDesc">Is Selected</span> ',
                "Conjuction": f"{pcts[option]}%",
            }
            paid = {
                "LogicType": "EmbeddedField",
                "LeftOperand": "distribution",
                "Operator": "EqualTo",
                "RightOperand": "paid",
                "Type": "Expression",
                "Description": '<span class="ConjDesc">And</span> '
                '<span class="LeftOpDesc">distribution</span> '
                '<span class="OpDesc">Is Equal to</span> '
                '<span class="RightOpDesc"> paid </span>',
                "Conjuction": "100%",
            }
            desc = cond["Description"] + " X " + paid["Description"]
            cells.append(
                {
                    "Occurrences": n,
                    "Logic": {
                        "0": {"0": cond, "Type": "If"},
                        "Type": "BooleanExpression",
                        "1": {"0": paid, "Type": "AndIf"},
                    },
                    "Description": desc,
                    "ID": f"{i - 1}X0",
                    "Count": 0,
                }
            )
        # the quota-level ``Logic`` mirrors what the GET returns for a
        # Cross quota: all the cells' weighted question conditions in
        # one If group, then the distribution condition (paid only)
        qconds = {str(j): cell["Logic"]["0"]["0"] for j, cell in enumerate(cells)}
        payloads.append(
            {
                "Name": f"{label} quota"[:40],
                "LogicType": "Cross",
                "Logic": [
                    {"0": dict(qconds, Type="If"), "Type": "BooleanExpression"},
                    {"0": {"0": paid, "Type": "If"}, "Type": "BooleanExpression"},
                ],
                "Occurrences": QUOTA_N,
                "QuotaAction": "EndCurrentSurvey",
                "OverQuotaAction": None,
                "QuotaRealm": "Survey",
                "ActionInfo": {
                    "0": {
                        "0": {
                            "ActionType": "EndCurrentSurvey",
                            "LogicType": "QuotaAction",
                            "Type": "Expression",
                        },
                        "Type": "If",
                    },
                    "Type": "BooleanExpression",
                },
                "EndSurveyOptions": dict(QUOTA_END_OPTIONS),
                "CrossLogicDef": cells,
            }
        )
    return payloads


_def_cache: dict = {}


def truncate_desc(text: str, max_len: int = 100) -> str:
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = clean.replace("&nbsp;", " ").replace("&amp;", "&")
    return clean if len(clean) <= max_len else clean[: max_len - 3] + "..."


def md_to_html(text: str) -> str:
    """Minimal markdown->HTML for Qualtrics QuestionText.

    Handles ATX headings (#, ##, ###+), unordered (- ) / ordered (1. )
    lists, bold (**), italic (*), and blank-line paragraphs. Everything is
    html-escaped first, so the inline markup transforms are the only HTML
    injected.
    """
    import html

    inline = [
        (re.compile(r"\*\*(.+?)\*\*"), r"<b>\1</b>"),
        (re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)"), r"<i>\1</i>"),
    ]
    spans: list[str] = []
    para: list[str] = []
    list_open: str | None = None

    def flush_para() -> None:
        if para:
            # Wrapped source lines are one paragraph: join with a space, not
            # a hard <br>, so the rendered text flows without mid-sentence
            # breaks; blank lines create the actual paragraph breaks.
            spans.append("<p>" + " ".join(para))
            para.clear()

    def close_list() -> None:
        nonlocal list_open
        if list_open:
            spans.append(f"</{list_open}>")
            list_open = None

    for line in text.split("\n"):
        stripped = line.strip()
        heading = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        li_u = re.match(r"^[-*]\s+(.*)$", stripped)
        li_o = re.match(r"^(\d+)[.)]\s+(.*)$", stripped)
        if not stripped:
            flush_para()
            close_list()
        elif heading:
            flush_para()
            close_list()
            spans.append("<p><b>" + heading.group(2) + "</b></p>")
        elif li_u or li_o:
            flush_para()
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
            content = html.escape(line.rstrip())
            for pat, rep in inline:
                content = pat.sub(rep, content)
            para.append(content)
    flush_para()
    close_list()
    out = "".join(spans)
    return out


# --- Question payload builders (shapes mirror a real Qualtrics export) ----


def db_payload(tag: str, text: str) -> dict:
    return {
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


def mc_single_payload(tag: str, text: str, options: list[str]) -> dict:
    return {
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
        "Validation": {
            "Settings": {
                "ForceResponse": "ON",
                "ForceResponseType": "ON",
                "Type": "None",
            }
        },
        "Language": [],
        "NextChoiceId": len(options) + 1,
        "NextAnswerId": 1,
    }


def mc_sahr_payload(tag: str) -> dict:
    """Preference: horizontal MC, 5-point Agent 1 ... Agent 2 scale."""
    payload = mc_single_payload(
        tag,
        PREF_TEXT,
        [
            "Strongly Agent 1",
            "Somewhat Agent 1",
            "About the same",
            "Somewhat Agent 2",
            "Strongly Agent 2",
        ],
    )
    payload["Selector"] = "SAHR"
    payload["Configuration"] = {
        "QuestionDescriptionOption": "UseText",
        "TextPosition": "above",
        "LabelPosition": "BELOW",
    }
    return payload


def mc_multi_payload(tag: str, text: str, options: list[str]) -> dict:
    """Multiple-select (checkbox) MC question."""
    payload = mc_single_payload(tag, text, options)
    payload["Selector"] = "MAVR"
    payload["SubSelector"] = "TX"
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


def description_text(record: dict) -> str:
    """The comparison descriptive question: the template's ``{?...?}``
    markers filled from the pair's record (audits ignored for now)."""
    s = record["scenario"]
    text = _load_template()
    for marker, value in (
        ("profile", s["profile_summary"]),
        ("situation", s["situation_summary"]),
        ("summary", record["summary"]["text"]),
    ):
        text = text.replace("{?%s?}" % marker, value)
    return md_to_html(text)


# --- The definition ---------------------------------------------------------


def build() -> dict:
    """Build the logical survey definition (cached; one survey, two
    ``distribution`` channels)."""
    if _def_cache:
        return _def_cache["d"]
    comparison_blocks = []
    records = sorted(COMPARISONS_DIR.glob("*/*/*.json"))
    if not records:
        raise SystemExit(
            f"No comparison <pair>.json files under {COMPARISONS_DIR} - run "
            "code/behavioural-simulations/build_comparisons.py first."
        )
    for json_path in records:
        record = json.loads(json_path.read_text(encoding="utf-8"))
        if record.get("schema") != "wvs-comparison/v1":
            print(f"SKIPPED {json_path.name}: unknown schema {record.get('schema')!r}")
            continue
        cid = record["comparison_id"]
        comparison_blocks.append(
            {
                "description": f"Comparison: {cid}",
                "collapsed": False,
                "questions": [
                    db_payload(cid + "_0", description_text(record)),
                    mc_sahr_payload(cid + "_pref"),
                    mc_single_payload(cid + "_aiuse", AIUSE_TEXT, AIUSE_SCALE),
                ],
            }
        )

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
    section_a = {
        "description": "Section A - About you",
        "collapsed": True,
        "questions": [
            mc_single_payload("Z1-1", Z_AGE_TEXT, Z_AGE_OPTIONS),
            mc_single_payload("Z2-1", Z_GENDER_TEXT, Z_GENDER_OPTIONS),
            mc_multi_payload("Z3-1", Z_ETHNICITY_TEXT, Z_ETHNICITY_OPTIONS),
            mc_single_payload("Z5-1", Z_REGION_TEXT, Z_REGION_OPTIONS),
            mc_single_payload("Z6-1", Z_INCOME_TEXT, Z_INCOME_OPTIONS),
            mc_single_payload("Z7-1", Z_OCCUPATION_TEXT, Z_OCCUPATION_OPTIONS),
            mc_single_payload("Z8-1", Z_EDU_TEXT, Z_EDU_OPTIONS),
        ],
    }
    section_b = {
        "description": "Section B - The comparison task",
        "collapsed": True,
        "questions": [
            db_payload(
                "D_taskB",
                md_to_html(
                    "In the next part of the survey, you will read short work "
                    "scenarios in which an AI agent acted, each shown through two "
                    "different versions of the agent.\n\nAll simulations are "
                    "completely fictional and do not represent any real person, "
                    "company or situation."
                ),
            )
        ],
    }
    section_c = {
        "description": "Section C - Your thoughts",
        "collapsed": True,
        "questions": [
            mc_single_payload("C1-1", C_KNOWLEDGE_TEXT, KNOWLEDGE5),
            mc_single_payload("C8-1", C_AIUSE_TEXT, AIUSEFREQ5),
            mc_single_payload("C2-1", C_SUPPORT_TEXT, SUPPORT5),
            mc_single_payload("C3-1", C_AUTOMATION_TEXT, SUPPORT5),
            mc_single_payload("C9-1", C_JOBLOSS_TEXT, CONCERN5),
            matrix_payload("C4-1", C_GOV_TEXT, C_GOV_ROWS, AGREE5),
            mc_single_payload("C5-1", C_TRUST_TEXT, AIUSE_SCALE),
            ro_rank_payload("C6-1", C_RISK_TEXT, C_RISK_OPTIONS),
            te_essay_payload("C7-1", C_COMMENT_TEXT),
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
        "intro_paid": intro_paid,
        "intro_volunteer": intro_volunteer,
        "section_a": section_a,
        "section_b": section_b,
        "comparison_blocks": comparison_blocks,
        "section_c": section_c,
        "prize_draw": prize_draw,
        "ordered_blocks": ordered,
        "randomizer_subset": N_PAIRS_SHOWN,
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


def distribution_check(value: str) -> dict:
    """BranchLogic: distribution embedded field equals ``value``."""
    return {
        "0": {
            "0": {
                "LogicType": "EmbeddedField",
                "LeftOperand": "distribution",
                "Operator": "EqualTo",
                "RightOperand": value,
                "_HiddenExpression": False,
                "Type": "Expression",
                "Description": '<span class="ConjDesc">If</span> '
                '<span class="LeftOpDesc">distribution'
                '</span> <span class="OpDesc">Is Equal to'
                '</span> <span class="RightOpDesc"> '
                f"{value} </span>",
            },
            "Type": "If",
        },
        "Type": "BooleanExpression",
    }


def flow(name_to_id: dict[str, str], qids_by_tag: dict[str, str]) -> dict:
    """The root flow for PUT /API/v3/survey-definitions/{SV}/flow.

    ``name_to_id`` maps block description -> remote Block id (already
    synced); ``qids_by_tag`` maps question tag -> QuestionID (the prize
    branch locates choices on Q85). Structure mirrors the hand-built live
    flow: distribution branches for the two intros, then demographics for
    everyone, the comparison randomizer, Section C, and the volunteer
    prize-draw ending.
    """
    comps = [
        name_to_id[b["description"]]
        for b in build()["comparison_blocks"]
        if b["description"] in name_to_id
    ]
    # SubSet larger than the pool breaks the respondent flow - sync may run
    # while a build_comparisons batch is still landing pairs.
    subset = min(build()["randomizer_subset"], max(len(comps), 1))
    q85 = qids_by_tag[PRIZE_DRAW_TAG]
    flow_items = [
        {
            "Type": "EmbeddedData",
            "FlowID": "FL_60",
            "EmbeddedData": [
                {
                    "Description": "distribution",
                    "Type": "Recipient",
                    "Field": "distribution",
                    "VariableType": "String",
                    "DataVisibility": [],
                    "AnalyzeText": False,
                }
            ],
        },
        branch(
            distribution_check("paid"),
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
        {
            "Type": "BlockRandomizer",
            "FlowID": "FL_34",
            "SubSet": subset,
            "EvenPresentation": True,
            "Flow": [
                {"Type": "Block", "ID": bid, "FlowID": f"FL_{35 + i}", "Autofill": []}
                for i, bid in enumerate(comps)
            ],
        },
        {
            "Type": "Standard",
            "ID": name_to_id["Section C - Your thoughts"],
            "FlowID": "FL_32",
            "Autofill": [],
        },
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
        "Properties": {"Count": len(flow_items) + len(comps)},
    }
