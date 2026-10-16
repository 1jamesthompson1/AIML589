# The comparison question template

Content below the `---` is the respondent-facing descriptive question for
each comparison block. `{?...?}` markers are substituted verbatim from the
block record built by `survey_definition.survey_records()`:

- `{?role?}` — the profile's one-sentence role summary
  (`code/dilemmas/profiles/<profile>.json`)
- `{?dilemma_intro?}` / `{?dilemma_outro?}` — the profile's framing lines
  from the same profile file: the intro opens the block, the outro bridges
  the situation and the question
- `{?situation?}` / `{?question?}` — the dilemma
  (`code/dilemmas/output/runs/index.json`)
- `{?response1?}` / `{?response2?}` — the two runs' exported response texts,
  shown as Agent 1 and Agent 2

Rendered to HTML by `survey_definition.description_text()` — edit the text
below to change the question wording. Respondents are never told which
models produced the responses.

---

Two different AI assistants were built to {?role?}

{?dilemma_intro?}

{?situation?}

{?dilemma_outro?}

{?question?}

Here are the two agents' responses:

**Agent 1:**

{?response1?}

**Agent 2:**

{?response2?}
