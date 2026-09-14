# Surveys

Survey design assets for the public consultation and the script that
generates the live comparison survey. Survey methodology:
`docs/report/chapters/survey.tex`.

## The comparison survey

- `generate_survey_qsf.py` - the single standalone script (no reference
  files) that produces the full survey QSF. It builds all 25 comparison
  blocks (one per profile x situation: a descriptive question with the
  profile + situation summaries and both agents' self-reviews with their
  audit fact-checks, then the Agent 1/2 preference, process-quality,
  reasonableness and AI-use questions), the introduction and Section C, and wires the
  flow's BlockRandomizer to show 5 of the 25 blocks per respondent.

  Qualtrics identity (SurveyID/owner/response-set/quota-group ids) is read
  from the repo root `.env` (`QUALTRICS_*` entries, documented in
  `.env.example`); the importer requires those ids to reference real brand
  objects, and duplicates re-import fine. All
  `survey/*.qsf` copies are gitignored (they embed the real ids).

  ```bash
  uv run survey/generate_survey_qsf.py
  ```

  Outputs:
  - `ai-behaviour-survey.qsf` - import via Qualtrics (Surveys -> Create
    survey -> Import survey). This import has been verified to work.
  - `comparison-agent-map.json` - which exported model (glm-5.3-flash vs
    v4-flash) played "Agent 1"/"Agent 2" per comparison (shuffled per
    build); the analysis code needs this to decode response data.

  Machine-readability: every question carries a structured export tag
  visible in the response data: comparison questions are
  `B_<profile>_<situation>_{0,pref,process,reasonable,aiuse}`
  (`_0` = the scenario description, `_pref` the Agent 1/Agent 2
  preference, `_process`/`_reasonable` the two Matrix Likerts, `_aiuse`
  the 5-point agree-with-AI-agent-in-this-work-situation question). Combined
  with `comparison-agent-map.json` (which model played Agent 1 / Agent 2
  per comparison, written every build) the response data needs no
  other lookup. (Editor notes (`NT` elements) are not shipped: Qualtrics'
  QSF import does not accept them.)

  Question text comes from the latest export session
  (`code/behavioural-simulations/output/runs`); re-run `export_results.py`
  in that directory to refresh, then re-run this script. Note a rebuild
  re-shuffles the Agent 1/Agent 2 labels.

## Survey documents

Two parts to the survey: the main part is ranking which agent's actions the
respondent prefers in a given scenario (the comparison blocks above), and a
second part about how they view the world and AI (Section C; the WVS-derived
cluster-assignment questions described in `survey-mockup.tex`).

- `survey-mockup.tex` - PDF mockup of the planned questionnaire format
  (`make watch FILE=survey-mockup`; output in `output/`).
- `participant-information-sheet.md` / `.tex` - the information sheet
  respondents see before starting. The `.md` is the source the generator
  converts into the survey's title page (edit the `.md`, the `.tex` is the
  printable/official copy).
- `recruitment-flyer.tex` / `recruitment-post.txt` - recruitment material.
- `hec-template-for-anonymous-surveys.docx` / `vuw-logo.pdf` - ethics /
  branding assets.
