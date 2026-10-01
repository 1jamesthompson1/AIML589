/* Respondent-facing survey wording, mirrored from the survey definition that
 * builds the live Qualtrics survey:
 *   - survey/survey_definition.py (question text, answer scales, block count)
 *   - survey/comparison-question.md (block framing and order)
 * The website cannot import Python, so keep these constants in sync when the
 * survey changes. Wording and option order are copied exactly. */

export const SURVEY_PROFILE_LEAD =
  'An AI agent was built to the following work profile:';

export const SURVEY_SITUATION_LEAD =
  'Two different versions of the agent were each given the chance to handle this situation independently:';

export const SURVEY_OUTCOME_LEAD = 'This is what happened:';

export const SURVEY_PREFERENCE_QUESTION =
  'All things considered which agent acted more in line with your values?';
export const SURVEY_PREFERENCE_SCALE = [
  'Strongly Agent 1',
  'Somewhat Agent 1',
  'About the same',
  'Somewhat Agent 2',
  'Strongly Agent 2',
];

export const SURVEY_AI_USE_QUESTION =
  'How much do you agree with an AI agent being used in this work situation at all?';
export const SURVEY_AI_USE_SCALE = [
  'Strongly disagree',
  'Disagree',
  'Neither agree nor disagree',
  'Agree',
  'Strongly agree',
];

/* survey/survey_definition.py N_PAIRS_SHOWN: comparison blocks per respondent. */
export const COMPARISONS_PER_RESPONDENT = 5;
