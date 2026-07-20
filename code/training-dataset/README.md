# Building the Value Alignment Training Dataset

This phase is about building a training dataset. This is done by

- Wrangling the WVS data to be in a format that can be used to generate the training dataset
- Creating clusters of respondents
- Creating templates for the question
  - Creating a mapping from the question response to the actual question asked (i.e in the right format).
  - Create system prompt templates for the different question styles
- Generating dataset by merging templates with the clusters responses in one of three ways
  - For each template question response simply use the most common response from the cluster
  - For each template question response sample from the cluster responses
  - For each template question response the answer is the response distribution of the cluster.

## Survey question formats

Each question in the WVS has been given a `question_format` that describes the structure of the question. There are generally two sorts of question: single questions and matrix/battery questions. For matrix/battery questions, the training dataset generation pipeline uses the per-question template from `prompt_templates.json` to generate a self-contained question for each sub-item. The following are the question formats used in the WVS:

### `single_select`

A single question where the respondent picks exactly one option from a list.

**Question example (Questionnaire):**
```
Taking all things together, would you say you are:

Please select one of the following options:
- Very happy
- Quite happy
- Not very happy
- Not at all happy
- Don't know
```

**Example response:** `Quite happy`

---

### `matrix_single_select`

A group of related items, each rated on the same single-select scale. The respondent picks one option per item.

**Question example:**
```
Please indicate, for each of the following, how important it is in your life.

For each item below, select one option:
- Family
- Friends
- Leisure time
- Politics
- Work
- Religion

Options:
- Very important
- Rather important
- Not very important
- Not at all important
- Don't know
```

**Example response:**
```
- Family: Very important
- Friends: Rather important
- Leisure time: Rather important
- Politics: Not very important
- Work: Very important
- Religion: Not at all important
```

---

### `matrix_multi_select`

A group of items where the respondent selects all that apply from a shared list of options.

**Question example:**
```
Here is a list of qualities that children can be encouraged to learn at home.
Which, if any, do you consider to be especially important?

Select all that apply from the following list:
- Good manners
- Independence
- Hard work
- Feeling of responsibility
- Imagination
- Tolerance and respect for other people
- Thrift, saving money and things
- Determination, perseverance
- Religious faith
- Unselfishness
- Obedience

Options:
- Important
- Not mentioned
```

**Example response:**
```
- Good manners: Important
- Independence: Important
- Hard work: Important
- Feeling of responsibility: Important
- Imagination: Not mentioned
- Tolerance and respect for other people: Important
- Thrift, saving money and things: Not mentioned
- Determination, perseverance: Important
- Religious faith: Not mentioned
- Unselfishness: Not mentioned
- Obedience: Not mentioned
```

---

### `matrix_likert`

A group of statements where the respondent indicates their level of agreement on a Likert scale.

**Question example:**
```
People talk about the changing roles of men and women today. For each of the
following statements, please indicate how much you agree or disagree.

For each statement below, indicate your level of agreement:
- One of my main goals in life has been to make my parents proud
- A preschool child suffers if their mother works
- Men make better political leaders than women do
- A university education is more important for a boy than a girl
- Men make better business executives than women do
- Being a housewife is just as fulfilling as working for pay

Options:
- Agree strongly
- Agree
- Disagree
- Strongly disagree
- Don't know
```

**Example response:**
```
- One of my main goals in life has been to make my parents proud: Agree
- A preschool child suffers if their mother works: Disagree
- Men make better political leaders than women do: Strongly disagree
- A university education is more important for a boy than a girl: Strongly disagree
- Men make better business executives than women do: Strongly disagree
- Being a housewife is just as fulfilling as working for pay: Disagree
```

---

### `matrix_binary`

A group of items each requiring a simple Yes/No (or binary) response.

**Question example:**
```
Which of the following things have you done for reasons of security?

For each item below, answer Yes or No:
- Didn't carry much money
- Preferred not to go out at night
- Carried a knife, gun or other weapon
```

**Example response:**
```
- Didn't carry much money: Yes
- Preferred not to go out at night: Yes
- Carried a knife, gun or other weapon: No
```

---

### `rating_scale`

A single question answered on a numeric scale, typically 1-10.

**Question example:**
```
All things considered, how satisfied are you with your life as a whole these
days? Please use this scale where 1 means you are "completely dissatisfied"
and 10 means you are "completely satisfied".

Please rate on a scale from 1 to 10.
```

**Example response:** `7`

---

### `ranking`

A question asking the respondent to rank a set of options in order of preference, typically with first and second choices.

**Question example:**
```
People sometimes talk about what the aims of this country should be for the
next ten years. Listed below are some of the goals which different people would
give top priority. Would you please say which one of these you, yourself,
consider the most important? And which would be the next most important?

Please select your most important and next most important:
- A high level of economic growth
- Strong defence forces
- People have more say at work and in their communities
- Making cities and countryside more beautiful
```

**Example response:**
```
1. A high level of economic growth
2. People have more say at work and in their communities
```

---

## Output

### `question_mapping.json`

A mapping of WVS questions into a structured format used by the dataset generation pipeline. Each entry represents either a single survey question or a group of questions that share the same response scale.

Generated using the help of AI, see [conversation](https://opncd.ai/share/0NdBka2x).

Example entry:

```json
{
  "id": 1,
  "question": "Please indicate, for each of the following, how important it is in your life.",
  "sub_questions": [
    "Family",
    "Friends",
    "Leisure time",
    "Politics",
    "Work",
    "Religion"
  ],
  "column_names": ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"],
  "word_response_types": [
    "Very important",
    "Rather important",
    "Not very important",
    "Not at all important",
    "Don't know"
  ],
  "numeric_response_types": [
    "1.0",
    "2.0",
    "3.0",
    "4.0",
    "-1.0"
  ],
  "survey_section": "The community and you",
  "question_type": "value survey",
  "question_format": "matrix_single_select"
}
```

**Fields:**

| Field | Description |
|-------|-------------|
| `id` | Sequential integer identifier (1–98) used as the canonical reference throughout the pipeline. |
| `question` | The full question text as presented to respondents. |
| `sub_questions` | For matrix/battery questions, the list of items rated on the same scale. For single questions, this is `[""]`. |
| `column_names` | The WVS column name(s) in the CSV data. Multiple names indicate a grouped entry. |
| `word_response_types` | The human-readable response options, in order. The last entry is `"Don't know"` if the option was offered in the questionnaire. |
| `numeric_response_types` | The numeric codes corresponding to each word response type, in the same order. `-1.0` is the WVS missing-value code for "Don't know". |
| `survey_section` | The WVS thematic section (e.g. "The community and you", "People, society and politics", "Respondent information"). |
| `question_type` | Broad category: `"value survey"` for attitudinal/value questions, `"respondent information"` for demographics. |
| `question_format` | Structural question type: `single_select`, `multi_select`, `matrix_single_select`, `matrix_multi_select`, `matrix_likert`, `matrix_binary`, `ranking`, `rating_scale`, `open_text`, `demographic`, `binary`, `demographic_composite`. |

`word_response_types` and `numeric_response_types` are parallel arrays — index `i` in one corresponds to index `i` in the other. Questions where "Don't know" was not offered in the questionnaire (or no respondent chose it) simply omit that entry.

### `prompt_templates.json`

Contains system prompts and per-question user prompt templates used by the dataset generation pipeline.

```json
{
  "system_prompts": {
    "ai_research_assistant": "You are an AI research assistant...",
    "survey_respondent": "You are a participant in the World Values Survey..."
  },
  "question_templates": {
    "1": "How important is {{sub_question}} in your life?",
    "2": "Do you consider {{sub_question}} to be an especially important quality for children to learn at home?",
    "6": null
  }
}
```

**`system_prompts`** — A dict of system prompt variants. Each prompt sets context for the AI model to respond as a survey participant. One is chosen per training example.

**`question_templates`** — A dict keyed by the `id` field (as a string) from each entry in `question_mapping.json`. Each value is either:
- `null` — the original question text from `question_mapping.json` is used verbatim (used for single‑select, rating scale, ranking, binary, and demographic questions).
- The question text template with the `{{sub_question}}` placeholder.

Single questions use the original question text verbatim, while matrix/battery questions use a template with a `{{sub_question}}` placeholder that is replaced with each sub-question in turn.

**Example prompt (with template, id=1, sub_question="Family"):**
```
Please indicate, for each of the following, how important it is in your life.

Family:
```

### `wvs_value_survey.csv`

The wrangled output dataset. Due to WVS licensing, raw input files cannot be redistributed,
but this output file can be independently verified.

| File | SHA-256 |
|------|---------|
| `wvs_value_survey.csv` | `<!-- HASH_START -->3b75f38e90f40d5ad023d7cc2150174dd871305209253111ac43861d7530a3e2<!-- HASH_END -->` |

### Generated Training Datasets

The `build_dataset.py` notebook produces three datasets in Parquet format:

| File | Description |
|------|-------------|
| `training_single_modal.parquet` | Single expected response (modal from cluster). Use for standard SFT. |
| `training_single_sample.parquet` | Single expected response (sampled from cluster distribution). Use for SFT with distribution matching in expectation. |
| `training_distributional.parquet` | Full probability distribution over response categories. Use for soft/distributional loss functions. |

Each dataset has the same row structure: every sub-question in every matrix/battery question is expanded into its own row, and demographic questions (`question_type: "respondent information"`) are excluded.

**Common columns:**

| Column | Description |
|--------|-------------|
| `system_prompt` | One of the system prompt variants from `prompt_templates.json`. |
| `user_prompt` | For template-based questions: the question text, a blank line, the pipe-separated response options, another blank line, then the sub-question followed by a colon. For verbatim questions: the original question text only. |
| `question_id` | The `id` from `question_mapping.json` this example originates from. |
| `sub_question` | The specific sub-question text (e.g. "Family"). `None` for single questions. |
| `column_name` | The WVS column name (e.g. "Q1"). |
| `question_format` | The structural type (e.g. `matrix_single_select`). |

**Single response datasets** additionally contain:

| Column | Description |
|--------|-------------|
| `expected_text` | The word-format expected response (e.g. "Very important"). |
| `expected_numeric` | The numeric code of the expected response (e.g. "1.0"). |

**Distributional dataset** additionally contains:

| Column | Description |
|--------|-------------|
| `expected_distribution` | An array of probabilities (summing to 1) over the response categories, aligned with `word_response_types` in `question_mapping.json`. |
| `categories` | The response category labels (copy of `word_response_types`). |

The cluster data is currently a placeholder (random Dirichlet-sampled distributions). When real WVS cluster data is available, replace the stub in the `Generate Placeholder Cluster Responses` cell to load actual response distributions per cluster.

## Input data

The input data you need to have is the World Values Survey Wave 7 data for New Zealand. This is a CSV file that can be downloaded from the [World Values Survey website](https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp). The file should be named `WVS_Wave_7_New_Zealand_Csv_v5.1.csv` and placed in the `code/training-dataset/input` directory.

### Data integrity verification

Due to WVS licensing restrictions, the raw input files cannot be included in this repository. Use the SHA-256 hashes below to verify your downloaded data is identical to the version this pipeline was built with.

Run the wrangle script; it will compute and print the hashes of your input files. Compare against the reference hashes below:

| File | SHA-256 |
|------|---------|
| `WVS_Wave_7_New_Zealand_Csv_v5.1.csv` | `5ff709a700adf966f809a388111632a6eec5d6f816558cf08d2f0dec013282f0` |
| `WVS_Wave_7_New_Zealand_CsvText_v5.1.csv` | `a99f40fbd6c509aa403b0c747193e2bf4efbeba3564cb9685aef76f0d364c6e2` |

To verify manually:

```bash
sha256sum input/WVS_Wave_7_New_Zealand_Csv_v5.1.csv
sha256sum input/WVS_Wave_7_New_Zealand_CsvText_v5.1.csv