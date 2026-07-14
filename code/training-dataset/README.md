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

## Output

### `output/question_mapping.json`

A mapping of WVS questions into a structured format used by the dataset generation pipeline. Each entry represents either a single survey question or a group of questions that share the same response scale.

Generated using the help of AI, see [conversation](https://opncd.ai/share/0NdBka2x).

Example entry:

```json
{
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
| `question` | The full question text as presented to respondents. |
| `sub_questions` | For matrix/battery questions, the list of items rated on the same scale. For single questions, this is `[""]`. |
| `column_names` | The WVS column name(s) in the CSV data. Multiple names indicate a grouped entry. |
| `word_response_types` | The human-readable response options, in order. The last entry is `"Don't know"` if the option was offered in the questionnaire. |
| `numeric_response_types` | The numeric codes corresponding to each word response type, in the same order. `-1.0` is the WVS missing-value code for "Don't know". |
| `survey_section` | The WVS thematic section (e.g. "The community and you", "People, society and politics", "Respondent information"). |
| `question_type` | Broad category: `"value survey"` for attitudinal/value questions, `"respondent information"` for demographics. |
| `question_format` | Structural question type: `single_select`, `multi_select`, `matrix_single_select`, `matrix_multi_select`, `matrix_likert`, `matrix_binary`, `ranking`, `rating_scale`, `open_text`, `demographic`, `binary`, `demographic_composite`. |

`word_response_types` and `numeric_response_types` are parallel arrays — index `i` in one corresponds to index `i` in the other. Questions where "Don't know" was not offered in the questionnaire (or no respondent chose it) simply omit that entry.

## Input data

The input data you need to have is the World Values Survey Wave 7 data for New Zealand. This is a CSV file that can be downloaded from the [World Values Survey website](https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp). The file should be named `WVS_Wave_7_New_Zealand_Csv_v5.1.csv` and placed in the `code/training-dataset/input` directory.