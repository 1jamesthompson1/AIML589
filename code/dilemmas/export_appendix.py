"""Write the dilemma appendix tables from the profile sources.

The report's dilemma appendix lists the ten work profiles with the documents
in their system prompts, and the thirty dilemmas with a one-line situation
summary. Both are generated from ``profiles/*.json`` so editing a profile
updates the appendix with it. The quality judge's resolved prompt is exported
beside them.

Run from the repository root:

    uv run python code/dilemmas/export_appendix.py

Writes ``code/figures/dilemmas-profiles.tex``,
``code/figures/dilemmas-list.tex`` and
``code/figures/dilemma_judge_prompt.txt`` for the report's appendix.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

# The documents reproduced in each system prompt, taken from the
# "# Policy in your file" section of profiles/<file>.json. Sections of an Act
# are grouped under the Act; the table is a reading aid, not a citation list.
DOCUMENTS = {
    "parish_advisor": (
        "Title D of Standards (General Synod / te Hīnota Whānui, 2021), "
        "with Canon I - Standards of Ministry (full)"
    ),
    "clinical_support": (
        "Code of Health and Disability Services Consumers' Rights (1996); "
        "Health Information Privacy Code 2020; open disclosure guidance "
        "(Health and Disability Commissioner, 2019)"
    ),
    "ed_triage": (
        "Australasian Triage Scale - Guidelines on the Implementation of the "
        "Australasian Triage Scale in Emergency Departments (ACEM, 2023)"
    ),
    "foodbank": (
        "Work and Income Special Needs Grant (food) guidance (summarised); "
        "mission allocation policy (summarised); Privacy Act 2020 principles "
        "(summarised)"
    ),
    "foreign_affairs": (
        "Public Service code of conduct - Our standards (Public Service "
        "Commission te Kawa Mataaho)"
    ),
    "incident_response": (
        "Health Information Privacy Code 2020, rules 5 and 11; Code of Health "
        "and Disability Services Consumers' Rights (1996), Right 4; NIST "
        "SP 800-61 Rev. 2, section 3.3.1"
    ),
    "insurance_claims": (
        "Fair Insurance Code (Insurance Council of New Zealand, 2020)"
    ),
    "lending_officer": (
        "Credit Contracts and Consumer Finance Act 2003, ss 9C, 55 and 83J; "
        "Code of Banking Practice (New Zealand Banking Association, 2025); "
        "hardship-process guidance (summarised)"
    ),
    "retirement_village": (
        "Retirement Villages Act 2003, s 32; village practice notes "
        "(summarised); Health and Disability Commissioner Code of Rights, "
        "Right 7; Health and Safety at Work Act 2015, operator duties "
        "(summarised)"
    ),
    "rostering": (
        "Employment Relations Act 2000, ss 3, 4, 69AA and 69AAE; Holidays Act "
        "2003, ss 44-50; store rostering policy (summarised)"
    ),
}

# One-line summaries of each dilemma's situation, for the appendix table.
# Keep them short: they are a pointer to profiles/<profile>.json, not a
# replacement for the full situation.
SUMMARIES = {
    "parish_advisor_confession": "Confessed fatal hit-and-run; keep confidence or report.",
    "parish_advisor_confession_woman": "Same confession by a woman; keep confidence or report.",
    "parish_advisor_arson": "Confessed arson that ruined a dairy; keep confidence or report.",
    "clinical_support_suicide_risk": "Suicidal client; alert nominated contact or stay private.",
    "clinical_support_suicide_risk_male": "Same risk, male client; alert contact or stay private.",
    "clinical_support_suicide_risk_teen": "Suicidal 17-year-old; alert her mother or stay private.",
    "ed_triage_gang_conflict": "Equal-urgency patients; rival gang risk in the waiting room.",
    "ed_triage_protection_order": "Equal-urgency patients; domestic-violence risk in the waiting room.",
    "ed_triage_female_gang": "Equal-urgency patients; rival gang risk between two women.",
    "foodbank_donor_condition": "Local-only vouchers would feed one suburb; accept or decline.",
    "foodbank_donor_sponsor": "Liquor chain vouchers need a public thank-you; accept or decline.",
    "foodbank_donor_cand_flyer": "Real-estate vouchers need flyers in parcels; accept or decline.",
    "foreign_affairs_us_unrest": "Weak evidence of US violence; flag and risk evacuation.",
    "foreign_affairs_fiji": "Threat to Fiji embassy; flag and risk mass evacuation.",
    "foreign_affairs_china_protestors": "Student agitator videos; flag and risk China escalation.",
    "incident_response_breach": "National records being stolen; shut ICU network or not.",
    "incident_response_nicu": "Newborn records being stolen; shut NICU network or not.",
    "incident_response_partial": "Half the records gone; shut ICU network or lose the rest.",
    "insurance_claims_lending_car": "Friend was speeding; apply or reverse owner's premium rise.",
    "insurance_claims_retired_teacher": "Daughter was speeding; apply or reverse pensioner's rise.",
    "insurance_claims_teen_driver": "Cousin was speeding; apply or reverse apprentice's rise.",
    "lending_gambling": "Gambling caused decline; tell partner or keep confidential.",
    "lending_gambling_switch": "Same, with the woman's gambling; tell partner or not.",
    "lending_drinking": "Alcohol spending caused decline; tell partner or keep confidential.",
    "retirement_village_driving": "Daughter says Elsie is unsafe; assess or stop her driving.",
    "retirement_village_shower": "Son demands showers; protect Maureen's right to refuse.",
    "retirement_village_stockpile": "Fire hazard in Stan's garage; clear out or plan with him.",
    "rostering_saturday_off": "Saturday off: long-serving father or single parent.",
    "rostering_sunday_off": "Sunday off: coach mother or single father.",
    "rostering_boxing_day": "Boxing Day off: Teresa's family or Hemi's five holidays.",
}


def load_profiles() -> list[dict]:
    """The live profile sources in file order."""
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((HERE / "profiles").glob("*.json"))
    ]


def profile_table(profiles: list[dict]) -> pd.DataFrame:
    """The ten profiles with their role and the documents in the prompt."""
    return pd.DataFrame(
        [
            {
                "Profile": profile["name"],
                "Role": profile["role_summary"].rstrip("."),
                "Documents in the system prompt": DOCUMENTS[profile["profile"]],
            }
            for profile in profiles
        ]
    )


def dilemma_table(profiles: list[dict]) -> pd.DataFrame:
    """The thirty dilemmas with a one-line situation summary."""
    rows = []
    for profile in profiles:
        for dilemma in profile["dilemmas"]:
            rows.append(
                {
                    "Dilemma": dilemma["id"],
                    "Situation summary": SUMMARIES[dilemma["id"]],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    from latex_tables import write_text_table

    profiles = load_profiles()
    write_text_table(
        profile_table(profiles),
        "dilemmas-profiles.tex",
        column_format="p{3.2cm} p{2.8cm} p{7.7cm}",
        ragged_columns=("Role", "Documents in the system prompt"),
        resize=False,
        font=r"\footnotesize",
    )
    write_text_table(
        dilemma_table(profiles),
        "dilemmas-list.tex",
        column_format="p{5.0cm} p{8.9cm}",
        ragged_columns=("Situation summary",),
        resize=False,
        font=r"\footnotesize",
    )
    # The judge's prompt is exported from the module that produced the scores,
    # so the appendix cannot drift from the scoring run.
    from judge_quality import PROMPT_VERSION, SYSTEM_PROMPT

    (HERE.parent / "figures" / "dilemma_judge_prompt.txt").write_text(
        f"Quality-judge prompt (version {PROMPT_VERSION})\n\n{SYSTEM_PROMPT}\n",
        encoding="utf-8",
    )
    print(f"wrote the appendix tables for {len(profiles)} profiles and the prompt")


if __name__ == "__main__":
    main()
