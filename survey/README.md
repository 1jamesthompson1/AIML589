# Surveys

Survey design assets for the public consultation and the script that
generates the live comparison survey. Survey methodology:
`docs/report/chapters/survey.tex`.

## The comparison survey

One Qualtrics survey serves both recruited channels. The panel email
distribution sets a `distribution` embedded field (`paid`); the open link
sets `volunteer`. The survey flow (mirrored by `survey_definition.build()`):

1. EmbeddedData reads `distribution`
2. Branch on `paid`/`volunteer` -> the matching introduction block
   (information sheet; the volunteer one adds the prize-raffle paragraph)
3. Section A - About you: demographics, every item skippable ("Prefer not
   to say"), top of the survey for screening
4. Section B - The comparison task: a two-sentence description of the
   task context
5. Two-stage comparison sampling: an outer BlockRandomizer draws 5 of the
   10 per-profile randomizers (random order), and each selected profile's
   inner BlockRandomizer draws 1 of its dilemma blocks (one block per
   repetition where the fine-tuned adapter and the base model both have an
   exported run; Agent 1/Agent 2 order is a stable pseudo-random assignment
   per run pair, so neither model is systematically first). Each respondent
   therefore sees 5 comparisons, one per selected profile.
6. Section C - Your thoughts: attitude and policy questions
7. Branch on `volunteer` -> prize-draw question; a `yes` answer ends the
   survey with a redirect

- `survey_definition.py` - the definition (single source of truth).
  Intro wording comes from `participant-information-sheet.md` including the
  `<!--volunteer ... -->` conditional paragraph (parsed by
  `sheet_for_channel`); comparison wording from `comparison-question.md`, and
  the comparison blocks themselves from the exported runs
  (`code/dilemmas/output/runs/index.json`). Each block reads the profile's
  one-sentence role summary and framing live from the profile JSON (dilemma
  intro opening the block, outro bridging the situation and question, so
  wording edits need no re-export), then the dilemma situation and question,
  and both runs' responses under Agent 1 / Agent 2 headings. The definition
  fails fast when the manifest is missing or has no version-matched pairs (run
  `code/dilemmas/export_results.py` first).
  The public website mirrors the respondent-facing question text and answer
  scales in `website/src/lib/survey.ts`; update that file when they change.

- `qualtrics_sync.py` - syncs questions from definition to live qualtrics survey. Adds in quotas where Non-binary / "Prefer not to say" are unquota'd
  and absorb rounding slack.

  ```bash
  uv run survey/qualtrics_sync.py
  ```

  Agent 1/Agent 2 labels per block: the DataExportTag is
  `<run_id>__<run_id>` (+ `_0`, `_pref`, `_aiuse`), so the two run ids join
  the response row back to the runs manifest to decode which model played
  which agent.

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

- `participant-information-sheet.md` / `.tex` - the information sheet
  respondents see before starting. The `.md` is the source the generator
  converts into the survey's title page (edit the `.md`, the `.tex` is the
  printable/official copy).
- `generate_mockup.py` - renders the live definition to
  `output/survey-mockup.pdf` (LaTeX via the gitignored `survey-mockup.tex`).
  Section B shows one comparison block per current profile - its first
  dilemma in file order - for review; `--all` includes the retired profiles'
  first dilemmas too.
- `recruitment-flyer.tex` / `recruitment-post.txt` - recruitment material.
- `hec-template-for-anonymous-surveys.docx` / `vuw-logo.pdf` - ethics /
  branding assets.
