# The comparison question template

Content below the `---` is the respondent-facing descriptive question for
each comparison block. `{?...?}` markers are substituted verbatim from the
block record built by `survey_definition.survey_records()`:

- `{?role?}` — the profile's one-sentence role summary
  (`code/dilemmas/profiles/<profile>.json`)
- `{?dilemma_intro?}` — the profile's framing line: it opens the block and
  introduces the case
- `{?dilemma_outro?}` — the profile's framing line: it bridges the case and
  the agents' answers (a profile without one falls back to a generic bridge
  line, `COMPARISON_OUTRO_FALLBACK`)
- `{?situation?}` / `{?question?}` — the dilemma
  (`code/dilemmas/output/comparisons/<run-id>/survey/<model>/comparisons.json`)
- `{?response1?}` / `{?response2?}` — the two runs' exported response texts,
  shown as Agent 1 and Agent 2

The dilemma (the case the agent faced, and the question put to it) sits in
a markdown block quote: the `>` lines become one indented paragraph with a
rule down its left edge. That sets the case apart from the framing around
it and from the answers below it, so the respondent reads the case as their
own before switching to judging what the agents did with it. A value
substituted into a quote line keeps the quote prefix on every one of its
lines, so a multi-line case (an email) stays inside the block.

The block's styling (the spacing between its parts, the rule above each
agent label) is set in `survey_definition.py`: `PARAGRAPH_STYLE` spaces the
body paragraphs, `QUOTE_STYLE` rules the case off, and
`COMPARISON_HEADING_STYLE` rules off the "Agent 1" and "Agent 2" labels, so
the two answers read as two labelled blocks.

Rendered to HTML by `survey_definition.description_text()` — edit the text
below to change the question wording. Respondents are never told which
models produced the responses.

---

An AI agent was built to *{?role?}*

{?dilemma_intro?}

> {?situation?}
>
> **{?question?}**

{?dilemma_outro?}

## Agent 1
{?response1?}

## Agent 2
{?response2?}
