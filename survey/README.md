# Surveys

Survey design assets for the public consultation and the script that
generates the live comparison survey. Survey methodology:
`docs/report/chapters/survey.tex`.

## The comparison survey

One Qualtrics survey serves both recruited channels. The panel email
distribution sets a `distribution` embedded field (`paid`); the open link
sets `volunteer`. The survey flow (mirrored by `survey_definition.build()`):

1. EmbeddedData reads `distribution`
2. Branch on `volunteer` -> the volunteer introduction block (the
   information sheet plus the prize-raffle paragraph); everyone else ->
   the paid block, i.e. the same sheet without the raffle paragraph. The
   paid sheet is the default (the branch is on `distribution` *not* being
   `volunteer`), so a link that never set the field still shows it
3. Section A - About you: demographics, every item skippable ("Prefer not
   to say"), top of the survey for screening
4. Section B - The comparison task: a two-sentence description of the
   task context
5. Two-stage comparison sampling: an outer BlockRandomizer draws 5 of the
   10 per-profile randomizers (random order), and each selected profile's
   inner BlockRandomizer draws 1 of its dilemma blocks (one block per
   comparison in the survey subset - the questions chosen out of the
   comparison pool, one per selected profile; Agent 1/Agent 2 order is a
   stable pseudo-random assignment per run pair, so neither model is
   systematically first). Each respondent therefore sees 5 comparisons, one
   per selected profile.
6. Section C - Your thoughts: attitude and policy questions
7. Branch on `volunteer` -> prize-draw question; a `yes` answer ends the
   survey with a redirect

- `survey_definition.py` - the definition (single source of truth).
  Intro wording comes from `participant-information-sheet.md` including the
  `<!--volunteer ... -->` conditional paragraph (parsed by
  `sheet_for_channel`); comparison wording from `comparison-question.md`, and
  the comparison blocks themselves from a comparison run's **survey
  subset** (`code/dilemmas/output/comparisons/<run-id>/survey/<model>/comparisons.json`,
  schema `wvs-dilemma-comparisons/v1`): one block per comparison, each
  comparison being a pair of runs from two different models answering the
  same dilemma. The survey subset is written by the analysis notebook's
  **Survey design** section (`code/dilemmas/analyze.py`), which pairs each
  fine-tune's runs with the base model's by sample index, removes dilemmas
  where every pair was judged the same option, drops pairs whose answers are
  close in embedding space and judged the same option, draws up to 10
  questions per dilemma, and writes one `survey/<model>/` subset per
  fine-tune. Choose the subset to build with `--model
  <model>` (the directory name, e.g. `modal_response-overall`) when running
  `qualtrics_sync.py` or `generate_mockup.py`; without it a single subset is
  used and several subsets are an error that lists the choices. Each block
  reads the profile's one-sentence role summary and framing live from the
  profile JSON (dilemma intro opening the block, outro bridging the case and
  the answers, so wording edits need no re-export), then the dilemma, and
  both runs' responses under Agent 1 / Agent 2 headings. The definition fails
  fast when the survey subset is missing or holds no comparison (run
  `code/dilemmas/analyze.py` first).
  The public website mirrors the respondent-facing question text and answer
  scales in `website/src/lib/survey.ts`; update that file when they change.

- The dilemma (the situation and the question) is shown as a quotation
  block: in `comparison-question.md` the `>` lines become one indented
  paragraph with a rule down its left edge (`md_to_html` in
  `survey_definition.py`, style `QUOTE_STYLE`). The case the agent faced
  therefore reads as a block of its own - the respondent reads it as their
  own situation before switching to judging what the agents did with it.
  A value substituted into a quote line keeps the quote prefix on every one
  of its lines, so a multi-line case (an email) stays inside the block.
  `generate_mockup.py` renders the same block in the PDF mockup (the
  `framed` package's `leftbar`).

- `qualtrics_sync.py` - syncs questions from definition to live qualtrics survey. Adds in quotas where Non-binary / "Prefer not to say" are unquota'd
  and absorb rounding slack.

  ```bash
  uv run survey/qualtrics_sync.py --model modal_response-overall
  ```

  Agent 1/Agent 2 labels per block: the DataExportTag is
  `<run_id>__<run_id>` (+ `_0`, `_pref`, `_aiuse`), so the two run ids join
  the response row back to the runs manifest to decode which model played
  which agent.

## Panel-supplied demographics (Dynata appends)

A panel can append variables to the respondent's survey link; Qualtrics
reads a query parameter whose name matches an embedded data field as that
field's value. The survey declares three such fields - `age`, `gender` and
`region` - and the matching Section A questions (Z1-1, Z2-1, Z5-1) carry
**display logic** that hides them when the panel already supplied the
value: the respondent is not asked what the panel already knows, and the

```bash
# the panel's link carries the demographics
https://vuw.qualtrics.com/j/XXXXXXXX?age=45-54&gender=Male&region=Wellington
```

- The panel should append the survey's **own option labels** (`age=45-54`,
  `gender=Male`, `region=Wellington`), so a panel value and an answer to the
  same question are the same string in the export. Dynata's "appends"
  (Samplify) are configured per project; the field names must match
  `PANEL_DEMOGRAPHICS` in `survey_definition.py`.
- The age and gender **quota** cells match either source - the question
  answer or the embedded value - so a panel-supplied respondent still
  counts towards the census-matched quota. (Preview the quota in
  Qualtrics before the campaign: the cell condition is
  "option selected OR age = <option>", which Qualtrics expresses as an
  `OrIf` inside the cell.)
- The panel's values land in the export as their own columns, next to the
  question answers, so the analysis can use the exact age rather than a
  band.

### The region question (Z5-1) uses the panel's regions

The region options are the panel's own region names, so a panel-supplied
value and a human answer are the same string. The panel's identifiers
(the values Dynata holds for each region) are:

| Panel id | Region (the survey's option label) | | Panel id | Region |
| --- | --- | --- | --- | --- |
| 21167546 | Auckland (unconfirmed) | | 21167551 | Chatham Islands |
| 21167558 | Bay of Plenty | | 21167552 | Nelson |
| 21167547 | Canterbury | | 25064601 | Niue |
| 21167555 | Gisborne | | 21167548 | Northland |
| 21167560 | Hawke's Bay | | 21167556 | Otago |
| 21167553 | Manawatu-Wanganui | | 21167549 | Southland |
| 21167559 | Marlborough | | 21167557 | Taranaki |
| 21167550 | Tasman | | 21167545 | Waikato |
| 21167554 | Wellington | | 21167544 | West Coast |

Auckland's identifier is the one the panel's list does not name; the
identifiers run 21167544-21167560, so 21167546 is the gap - **check it
before the campaign**. The survey also has "Other" (for the people the
panel's list does not cover - the volunteer channel has no panel value at
all), "I live overseas" and "Prefer not to say".

### The ethnic groups question (Z3-1) uses the census categories

The options are the 2023 Census ethnic groups (Stats NZ), not the WVS
ones: the released WVS NZ file has no ethnicity question at all (see
`code/training-dataset/quota_analysis.py`), so the census is the only
source these answers can be compared against. The census asks the same
multi-select question, and the survey's options are its top-level groups
(New Zealand European / Pakeha, Maori, Pacific peoples, Asian, Middle
Eastern / Latin American / African, Other), so the answers read straight
off the census release.

## Time spent on the comparisons

Qualtrics times every page and every question and exports the timings with
the responses, so the survey needs no scripting for this (the API does not
accept a survey's JavaScript). `timing.py` turns the exported timings into
the two numbers the analysis wants: the time spent on **each comparison**
the respondent saw, and the time spent in the **comparison task** (Section
B) as a whole.

```bash
uv run survey/timing.py \
    --responses artifacts/dilemmas/survey-timings/responses.csv \
    --timing artifacts/dilemmas/survey-timings/timing.csv
```

The inputs are the standard responses export (with "Timing Data" included)
and its timing file; the script joins them on the response id and writes
`comparison_timings.csv` (one row per response and comparison),
`response_timings.csv` (the time in the comparison task per response) and
`timing_summary.md` (the summary for the report). The time on a
comparison is the time on its description question - the block's three
questions share one page - and the comparison task's time is that plus the
time on its introduction page.

## Survey documents

Two parts to the survey: the main part is ranking which agent's answer the
respondent prefers in a given dilemma (the comparison blocks in
Section B), and a second part about how they view the world and AI
(Section C; the WVS-derived cluster-assignment questions).

Section C's policy question is a multiple-choice (Qualtrics MAVR, tag
`C4-3`): nine proposed policy responses, each a bold title plus a one-line
description, of which respondents choose exactly three. It replaced the
`C4-2` pick-and-rank question (itself a replacement for the `C4-1`
government-capability matrix); a new tag is needed for each type change
because Qualtrics cannot change a remote question's type in place. Question,
intro, the required choice range and the option wording live in
`policy_questions.json` (edit that, not Python).

- `participant-information-sheet.md` - the information sheet respondents
  see before starting. The `.md` is the source the generator converts into
  the survey's title page (edit the `.md`; the sheet's contact list is the
  student researcher and the supervisor, and it carries the Hōkai
  application number).
- `generate_mockup.py` - renders the live definition to
  `output/survey-mockup.pdf` (LaTeX via the gitignored `survey-mockup.tex`).
  Section B shows one comparison block per current profile - its first
  dilemma in file order - for review; `--all` includes the retired profiles'
  first dilemmas too, and `--model <model>` picks the per-model survey subset
  (required when several exist).
- `recruitment-flyer.tex` / `recruitment-post.txt` - recruitment material.
- `hec-template-for-anonymous-surveys.docx` / `vuw-logo.pdf` - ethics /
  branding assets.
