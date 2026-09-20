# The comparison question template

Content below the `---` is the respondent-facing descriptive question for
each comparison block. `{?...?}` markers are substituted verbatim from the
pair's `comparison.json`:

- `{?profile?}` — `scenario.profile_summary`
- `{?situation?}` — `scenario.situation_summary`
- `{?summary?}` — `summary.text` (the frontier-model difference summary)

Rendered to HTML by `survey_definition.description_text()` — edit the text
below to change the question wording.

---

An AI agent was built to the following work profile:

{?profile?}

Two different versions of the agent were each given the chance to handle
this situation independently:

{?situation?}

**This is what happened:**

{?summary?}
