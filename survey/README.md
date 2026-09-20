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
5. BlockRandomizer: 5 of the 625 comparison blocks (25 scenarios with 5 samples each and 2 models generating samples) per respondent
6. Section C - Your thoughts: attitude and policy questions
7. Branch on `volunteer` -> prize-draw question; a `yes` answer ends the
   survey with a redirect

- `survey_definition.py` - the definition (single source of truth).
  Intro wording comes from `participant-information-sheet.md` including the
  `<!--volunteer ... -->` conditional paragraph (parsed by
  `sheet_for_channel`); comparison wording from `comparison-question.md`.

- `qualtrics_sync.py` - syncs questions from definition to live qualtrics survey. Adds in quotas where Non-binary / "Prefer not to say" are unquota'd
  and absorb rounding slack.

  ```bash
  uv run survey/qualtrics_sync.py
  ```

  Agent 1/Agent 2 labels per comparison: not stored anywhere key-based - join
  response data on the question tag's comparison_id with the pair record in
  `code/behavioural-simulations/output/comparisons` to decode which exported
  model played which agent.

## Survey documents

Two parts to the survey: the main part is ranking which agent's actions the
respondent prefers in a given scenario (the comparison blocks in
Section B), and a second part about how they view the world and AI
(Section C; the WVS-derived cluster-assignment questions).

- `participant-information-sheet.md` / `.tex` - the information sheet
  respondents see before starting. The `.md` is the source the generator
  converts into the survey's title page (edit the `.md`, the `.tex` is the
  printable/official copy).
- `recruitment-flyer.tex` / `recruitment-post.txt` - recruitment material.
- `hec-template-for-anonymous-surveys.docx` / `vuw-logo.pdf` - ethics /
  branding assets.
