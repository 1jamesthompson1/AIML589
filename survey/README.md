# Surveys

The public consultation: one Qualtrics survey that shows 500 respondents five
pairwise comparisons between the base model and a fine-tune. The survey is
built from a comparison run's **survey subset**. Methodology:
`docs/report/chapters/survey.tex`.

## Workflow

```bash
# 1. Build the survey subset (writes .../survey/<model>/comparisons.json)
uv run marimo edit code/dilemmas/analyze.py

# 2. Sync the chosen subset to Qualtrics
uv run survey/qualtrics_sync.py --model modal_response-overall --workers 8

# 3. Preview the exact respondent-facing survey
uv run survey/generate_mockup.py --model modal_response-overall

# 4. Download responses (writes survey/output/responses.csv)
uv run survey/download_results.py

# 5. Power analyses
uv run survey/power_analysis.py   # full design: tables + coverage figure
uv run survey/h2_power.py         # H2 only: LMM and Bradley-Terry

# 6. Interactive analysis: H2, policy preferences, dilemmas
uv run marimo edit survey/analyze.py
```

`analyze.py` is a Marimo notebook. It reads the downloaded export, applies the
pre-registered exclusions, and produces the H2 LMM/Bradley-Terry table, the
census-weighted policy top-three shares and tiers, sample-vs-census demographic graph, and the dilemma support/time summary.

The sync deletes and recreates questions. Add `--clear` when Qualtrics
rejects new questions because the survey is too full (`--clear` empties the
survey first). Without `--model`, a single subset is used; several subsets is
an error that lists the choices. Syncing/downloading needs
`QUALTRICS_API_KEY` and `QUALTRICS_SURVEY_ID` in `.env`. Each script's
docstring, and `--help` where it has flags, covers its outputs. The
definition fails fast when the subset is missing or empty.

## Survey design

- One survey, two channels: the panel distribution sets `distribution=paid`;
  the open link sets `volunteer`. Paid is the default, so a link without the
  field still shows the paid information sheet (no raffle paragraph).
- Section A: demographics. Section B: five comparison blocks. Section C:
  AI attitudes and policy, including C4-3 (choose three policies) and C4-4
  (rank the three selected policies). C4-3 choices are shown in a random
  order per respondent (Qualtrics choice randomization, stored as the
  canonical `Randomization` object); choice IDs, and so the `C4-3_1` ...
  `C4-3_9` export columns, keep the `policy_questions.json` order.
- C4-4 is **manual**: Qualtrics exposes carry-forward choices in the UI but
  not in the question API. Add it once in Qualtrics as a Rank Order question,
  tag `C4-4`, with Carry Forward -> Selected Choices -> `C4-3`, force
  response ON. `qualtrics_sync.py` checks it and warns with the exact text
  when it is missing or its force-response setting is off; it never creates
  or updates the question. `--clear` deletes it, so re-add it after a clear
  rebuild.
- Each block shows a dilemma and two responses, blind and in random order.
  Sampling is outer 5-of-8 profiles -> 1 dilemma per selected profile -> 1
  pair per dilemma, so 500 respondents x 5 = 2,500 ratings.
- `ord_<n>` in the export records the order: `ft` means the fine-tune was
  Agent 1; `base` the reverse. Block tags are
  `<base_run_id>__<ft_run_id>` plus `_0_ft`/`_0_base`, `_pref`, `_aiuse` or
  `_timing`; join them to the runs manifest.
- The 5-of-8 draw weights profiles equally, not dilemmas. A dilemma in a
  profile with `n_p` dilemmas gets `(5/8)/n_p` of respondents: about 312,
  156 and 104 ratings per dilemma for the 1-, 2- and 3-dilemma profiles.
  Weight by `n_p` for the pair-average (+3% SE). Do not cap profiles
  mid-fielding to equalise counts: caps confound profile with arrival time
  and give late respondents fewer than five comparisons.
- Hidden timers cover the intro/information sheet, Section B instructions,
  each comparison page, Section A, Section C, and the volunteer prize draw.
- Paid respondents whose `Q_TotalDuration` is under 240 seconds (4 minutes)
  are sent to the panel quality exit (`Q_TerminateFlag = PoorQuality`). This
  is a live screenout and a deviation from the pre-registered relative speed
  rule.
- Three attention checks (`A_attention`, `B_attention`, `C_attention`).
  Paid respondents who fail two go to the panel's quality exit; volunteers
  are only recorded. Correct options live in `ATTENTION_CHECKS`.
- Ethnicity uses the 2023 Census top-level groups, not the WVS categories;
  the released WVS NZ file has no ethnicity item.
- Each block has a hidden timing question; Sections A and C have page timers.
  The standard responses export carries First/Last Click, Page Submit and
  Click Count, so no separate timing script is needed.

## Sources

`survey_definition.py` is the single source of truth. Wording comes from
`participant-information-sheet.md`, `comparison-question.md` and
`policy_questions.json`; blocks come from
`code/dilemmas/output/comparisons/<run-id>/survey/<model>/comparisons.json`
(built by the Survey design section of `code/dilemmas/analyze.py`); profile
wording is read live from `code/dilemmas/profiles/<profile>.json`. The public
website mirrors the text and scales in `website/src/lib/survey.ts`; update it
when they change.

## Panel configuration

Dynata appends `age`, `gender` and `region` on the entry link; the matching
questions are hidden when a value arrives. The values must use the survey's
own labels (`age=45-49`, `gender=Male`, `region=Wellington`); the fields are
declared in `PANEL_DEMOGRAPHICS` in `survey_definition.py`. Panel region ids:

```text
21167544 West Coast          21167545 Waikato          21167546 Auckland
21167547 Canterbury          21167548 Northland        21167549 Southland
21167550 Tasman              21167551 Chatham Islands  21167552 Nelson
21167553 Manawatu-Wanganui   21167554 Wellington       21167555 Gisborne
21167556 Otago               21167557 Taranaki         21167558 Bay of Plenty
21167559 Marlborough         21167560 Hawke's Bay       25064601 Niue
```

Auckland's panel id is `21167546` (confirmed in the current export). The age/gender quota cells
match either the question answer or the appended value, so panel-supplied
respondents still count towards the census-matched quota; preview the quota
in Qualtrics before fielding. The exit links live in `.env`; `psid` is the
panelist id from the entry link, echoed on the exit redirect. See
`.env.example`. Paid respondents use the completed/quality links; volunteers
keep the prize-draw ending and never use a panel link.

## Notes

- `download_results.py` writes `survey/output/responses.csv` and
  `responses-question-ids.csv` (gitignored; may contain confidential data).
  Qualtrics can truncate long DataExportTags in the first CSV header; recover
  the full tag from the `ImportId` QID in the third header row and join it to
  `question_id` in the map.
- `generate_mockup.py` writes `survey-mockup.tex` and compiles
  `survey/output/survey-mockup.pdf`. `--all` includes retired profiles.
- `power_analysis.py` writes `code/figures/survey-power-estimands.tex`,
  `survey-power-coverage.tex` and `survey-power-coverage.png`.
- `h2_power.py` writes `code/figures/survey-h2-power.tex` and
  `survey-h2-power.png`.
- `analyze.py` is a Marimo notebook with shared helper cells at the top. It
  writes `code/figures/survey-exclusions.tex`,
  `code/figures/survey-h2-results.tex`,
  `code/figures/survey-policy-shares.png`,
  `code/figures/survey-demographics-vs-census.png`,
  `code/figures/survey-dilemma-preference.png`,
  `code/figures/survey-dilemma-ai-support.png`,
  `code/figures/survey-dilemma-mean-time.png`,
  `code/figures/survey-timing-allocation.tex`, and
  `code/figures/survey-response-time-histogram.png`.
- The notebook excludes development tests (before the fielding cutoff, test or
  preview), incompletes, attention-check failures, responses under 4 minutes
  or over 2 hours, and duplicate panelists. `Q_TerminateFlag = PoorQuality`
  and straightlining are flags for manual review, not automatic exclusions.
- `reference.qsf` is a gitignored reference export, not read by the scripts.

## Files

| File | Purpose |
| --- | --- |
| `survey_definition.py` | survey payloads and flow (single source of truth) |
| `qualtrics_sync.py` | create or update the live Qualtrics survey |
| `download_results.py` | download responses via the Qualtrics API |
| `generate_mockup.py` | render the PDF mockup |
| `power_analysis.py` | full design power analysis |
| `h2_power.py` | H2-only power analysis |
| `analyze.py` | Marimo notebook: wrangling, timing, H2, policy, dilemmas |
| `comparison-question.md` | respondent-facing comparison template |
| `participant-information-sheet.md` | information sheet / title page |
| `policy_questions.json` | Section C policy question and options |
| `recruitment-flyer.tex`, `recruitment-post.txt` | recruitment material |
| `hec-template-for-anonymous-surveys.docx`, `vuw-logo.pdf` | ethics/branding assets |
