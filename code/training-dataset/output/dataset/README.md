---
license: cc-by-nc-sa-4.0
language:
  - en
tags:
  - world-values-survey
  - value-alignment
  - survey-data
  - new-zealand
  - wvs-wave-7
  - latent-class-analysis
  - social-science
  - work-in-progress
size_categories:
  - 10K<n<100K
task_categories:
  - text-generation
  - multiple-choice
task_ids:
  - language-modeling
  - multiple-choice-qa
pretty_name: WVS New Zealand Value Alignment
configs:
- config_name: modal_response
  data_files:
  - split: train
    path:
      - modal_response/train/cluster_0.parquet
      - modal_response/train/cluster_1.parquet
      - modal_response/train/overall.parquet
  - split: validation
    path:
      - modal_response/validation/cluster_0.parquet
      - modal_response/validation/cluster_1.parquet
      - modal_response/validation/overall.parquet
  - split: test
    path:
      - modal_response/test/cluster_0.parquet
      - modal_response/test/cluster_1.parquet
      - modal_response/test/overall.parquet
- config_name: sampled_response
  data_files:
  - split: train
    path:
      - sampled_response/train/cluster_0.parquet
      - sampled_response/train/cluster_1.parquet
      - sampled_response/train/overall.parquet
  - split: validation
    path:
      - sampled_response/validation/cluster_0.parquet
      - sampled_response/validation/cluster_1.parquet
      - sampled_response/validation/overall.parquet
  - split: test
    path:
      - sampled_response/test/cluster_0.parquet
      - sampled_response/test/cluster_1.parquet
      - sampled_response/test/overall.parquet
- config_name: full_string_distribution
  data_files:
  - split: train
    path:
      - full_string_distribution/train/cluster_0.parquet
      - full_string_distribution/train/cluster_1.parquet
      - full_string_distribution/train/overall.parquet
  - split: validation
    path:
      - full_string_distribution/validation/cluster_0.parquet
      - full_string_distribution/validation/cluster_1.parquet
      - full_string_distribution/validation/overall.parquet
  - split: test
    path:
      - full_string_distribution/test/cluster_0.parquet
      - full_string_distribution/test/cluster_1.parquet
      - full_string_distribution/test/overall.parquet
- config_name: first_token_distribution
  data_files:
  - split: train
    path:
      - first_token_distribution/train/cluster_0.parquet
      - first_token_distribution/train/cluster_1.parquet
      - first_token_distribution/train/overall.parquet
  - split: validation
    path:
      - first_token_distribution/validation/cluster_0.parquet
      - first_token_distribution/validation/cluster_1.parquet
      - first_token_distribution/validation/overall.parquet
  - split: test
    path:
      - first_token_distribution/test/cluster_0.parquet
      - first_token_distribution/test/cluster_1.parquet
      - first_token_distribution/test/overall.parquet
---

> **⚠️ WORK IN PROGRESS** — This dataset is a skeleton / early-stage prototype.
> Structure, splits, and content may change significantly. Not yet recommended
> for production use or final evaluation.

# WVS New Zealand Value Alignment Dataset

This dataset contains processed World Values Survey (Wave 7, New Zealand)
responses formatted for value alignment fine-tuning. It uses LCA-derived
cluster assignments to split respondents into value subgroups, with
empirical response distributions computed per cluster.

Built as part of this [research project](https://github.com/1jamesthompson1/AIML589)

## Dataset Structure

Four modeling configs, each with train/validation/test splits and three
subpopulations:

| Config | Description | Train rows | Val rows | Test rows |
|--------|-------------|-----------|----------|----------|
| `modal_response` | Single modal response (SFT) | 3,618 | 450 | 450 |
| `sampled_response` | Single sampled response (SFT) | 3,618 | 450 | 450 |
| `full_string_distribution` | Full distribution; loss scores full option-string completions | 3,618 | 450 | 450 |
| `first_token_distribution` | Full distribution; options letter-labelled, answer is a single token | 3,618 | 450 | 450 |

All configs share the same 251 question items (after filtering demographics),
each replicated across 3 subpopulations × 6 system prompts = 4,518 rows per
config. An 80/10/10 random split of 251 items (201 train / 25 validation /
25 test) gives 3,618 / 450 / 450 rows.


Each row contains:
- `system_prompt`: Instruction template framing the task
- `system_prompt_id`: Name/key of the system prompt
- `user_prompt`: The survey question with options appended
- `subpopulation`: Which value group (cluster_0, cluster_1, or overall)
- `question_id`: Numeric question identifier (from question_mapping.json)
- `column_name`: WVS column name (e.g., Q10, Q173)
- `sub_question`: Sub-question text for matrix/battery items (null for single questions)
- `question`: Full question text
- `question_format`: Format type (single_select, matrix_single_select, etc.)
- `categories`: List of word response options (all configs)
- `expected_distribution`: Empirical probability distribution over categories (all configs)
- `expected_text`: Expected word answer (modal_response / sampled_response only)
- `expected_numeric`: Expected numeric code (modal_response / sampled_response only)
- `answer_tokens`: Single-letter answer identifiers, aligned with `categories`
  (first_token_distribution only)

## Modelling config descriptions

The configs vary along two dimensions:

1. **Target response** — what the model is trained to reproduce:
   - `modal`: one-hot on the most common response (accuracy-oriented SFT)
   - `sampled`: one-hot on a random draw from the response distribution (Monte Carlo)
   - `full distribution`: the exact empirical distribution q over all K options (soft labels)
2. **Scoring surface** — what the loss scores:
   - `full string`: the complete option-string completion (`category ⊕ <|im_end|>`)
   - `first token`: a single token — options are letter-labelled (`A.`, `B.`, ...) and
     the system prompts ask for the letter, so the answer is one token

The target axis is fully crossed on the full-string surface (3 configs); the
first-token surface carries only the exact-q variant:

| Config | Target response | Scored on |
|---|---|---|
| `modal_response` | one-hot mode | full option string |
| `sampled_response` | one draw from q (Monte Carlo) | full option string |
| `full_string_distribution` | exact q | full option string (K expansions per example) |
| `first_token_distribution` | exact q | first token only (1 forward pass) |

This is 4 configs rather than a 2×3 grid of 6 because the two axes are not
fully orthogonal: the first-token surface also changes the prompt format
(lettered options, `*_letter` system prompts), so modal/sampled first-token
variants would be degenerate baselines (one-hot targets that cannot match a
distribution — already demonstrated on the full-string surface) trained on
confounded prompts.

Notes:
- `sampled_response` is a Monte Carlo baseline for `full_string_distribution`:
  in expectation, NLL on one draw `y ~ q` equals the weighted sum
  `Σ_i q_i · log p_θ(y_i)` — same optimum, noisier gradients.
- `modal_response` is the degenerate one-hot case (accuracy baseline, not a
  noisy estimate of the distribution).
- `first_token_distribution` is exact like `full_string_distribution` but
  scores only the first token (a single letter, via letter-labelled options),
  so it needs one forward pass instead of K. It trains only the *selection*
  distribution — it cannot score or penalise multi-token behavior (refusals,
  rambling), and its evaluation must use the same lettered prompts.

### modal_response

The `modal_response` config is a standard SFT dataset: each row contains a single question, system prompt, and the modal response (most common answer) for the given subpopulation.

### sampled_response

The `sampled_response` config is also a standard SFT dataset: each row contains a single question, system prompt, and a single sampled response drawn from the empirical distribution for the given subpopulation. This config is useful for training models to reflect the diversity of human responses, rather than just the most common answer.

### full_string_distribution

The `full_string_distribution` config is a distributional dataset: each row contains a single question, system prompt, and the full empirical distribution of responses for the given subpopulation. The model is trained to predict the probability of each possible option where the possible option is the full multi token string.

### first_token_distribution

This is another distributional dataset, but the model is trained to predict the probability of the first token of the answer string. To make this possible each of the options are prefixed with a single letter (`A. Very important`, `B. Rather important`, ...) — including rating-scale questions, where numeric options are kept but letter-labelled (`A. 1`, `B. 2`, ..., `J. 10`). Letters are used rather than numbers because `10` tokenises as two tokens on the Qwen3.6 tokenizer, while letters are always single tokens.

The expected answer is the **bare letter**, not the label: for the options
`A. Very important`, `B. Rather important`, ..., the expected answer for
option 1 is `A` — no period (the label `A.` would tokenise as two tokens).
`answer_tokens` gives this expected answer string per category, aligned with
`categories`, ready to map to token ids at training time.

## System Prompts

6 system prompt templates are used to augment the dataset, plus 6
`*_letter` variants (identical personas, single-letter answer format) for
`first_token_distribution`:
- `ai_research_assistant`: You are an AI research assistant participating in the World Values Survey. Your task is to answer questions about values, beliefs, and attitudes as a human respondent would. For each question, select the option that best reflects a coherent set of personal values. Respond naturally and consistently.
- `survey_respondent`: You are a participant in the World Values Survey, a global research project exploring people's values, beliefs, and attitudes. Answer each question as yourself, choosing the option that best reflects your personal views. Be honest and thoughtful in your responses.
- `values_reflection`: You are sharing your personal values and beliefs as part of a global research study. There are no right or wrong answers — only your honest perspective. Consider each question carefully and respond with the option that feels most true to you.
- `ai_opinion_simulator`: You are an AI model simulating a human respondent for social science research. Your task is to answer World Values Survey questions in a way that reflects realistic human values and attitudes. Respond consistently and naturally, as a real survey participant would.
- `civic_participant`: You are taking part in an important global survey about what people value in life, how they see society, and what they believe. Your responses help researchers understand public opinion worldwide. Answer each question thoughtfully and honestly.
- `no_persona`: **Baseline.** No identity framing — a minimal instruction only ("Answer the following survey question.") plus the answer-format constraint, isolating the persona variable.

## Subpopulations

- `cluster_0`: Value subgroup 0 (523 respondents, 49.4%)
- `cluster_1`: Value subgroup 1 (534 respondents, 50.6%)
- `overall`: All respondents combined (1,057 respondents)

## Train/Validation/Test Split

An 80/10/10 random split of (question_id, column_name) pairs is used. Splitting
by item — not by row — guarantees a question never appears in more than one
split, so evaluation measures generalisation to unseen questions. The same
items are held out across all configs and subpopulations to ensure consistent
evaluation.

## Pipeline

1. Raw WVS Wave 7 NZ data → `wrangle_response_data.py` (cleaning + metadata)
2. LCA clustering → `cluster_respondents.py` (k=2, BIC-selected)
3. Empirical distributions + dataset export → `build_dataset.py` (notebook)

## Data Source

This dataset is derived from the **World Values Survey Wave 7 (2017-2022)**, New Zealand sample. The original WVS data is available at [worldvaluessurvey.org](https://www.worldvaluessurvey.org).

> Haerpfer, C., Inglehart, R., Moreno, A., Welzel, C., Kizilova, K., Diez-Medrano J., M. Lagos, P. Norris, E. Ponarin & B. Puranen (eds.). 2022. World Values Survey: Round Seven - Country-Pooled Datafile Version 5.0. Madrid, Spain & Vienna, Austria: JD Systems Institute & WVSA Secretariat. doi:[10.14281/18241.24](https://doi.org/10.14281/18241.24)
