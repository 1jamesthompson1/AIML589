import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Building the Value Alignment Training Dataset

    This notebook takes the `question_mapping.json`, placeholder cluster responses and
    `prompt_templates.json` to produce two training dataset variants:

    1. **Single response dataset** — each example has one expected answer (text + numeric).
       Used for supervised fine-tuning (SFT) with a cross-entropy loss, either using the
       modal response or by sampling from the response distribution.

    2. **Distributional response dataset** — each example has an expected probability
       distribution over response categories. Used for distribution-matching fine-tuning.

    Each sub-question in a matrix/battery is expanded into its own training example.
    Demographic questions are excluded as they are not relevant for value alignment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Imports
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import json
    import random
    import numpy as np
    from pathlib import Path

    return Path, json, mo, np, pd, random


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Load Input Data
    """)
    return


@app.cell
def _(Path, json):
    output_dir = Path("output")

    with open(output_dir / "question_mapping.json") as f:
        question_mapping = json.load(f)

    with open(output_dir / "prompt_templates.json") as f:
        prompt_templates = json.load(f)

    system_prompts = list(prompt_templates["system_prompts"].values())
    question_templates = prompt_templates["question_templates"]

    f"Loaded {len(question_mapping)} question entries and {len(system_prompts)} system prompts"
    return output_dir, question_mapping, question_templates, system_prompts


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Filter Out Demographic Questions

    Demographic questions (e.g., sex, age, income, employment) are excluded from the
    training dataset as they describe the respondent rather than their values.
    """)
    return


@app.cell
def _(question_mapping):
    value_entries = [
        e
        for e in question_mapping
        if e.get("question_type") != "respondent information"
    ]
    removed = len(question_mapping) - len(value_entries)
    f"Kept {len(value_entries)} value entries, removed {removed} demographic entries"
    return (value_entries,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Expand Entries into Individual Items

    Each matrix/battery entry is expanded so every sub-question becomes its own
    training item. Single questions and ranking entries produce one item per column.
    """)
    return


@app.cell
def _(value_entries):
    def expand_items(entries):
        items = []
        for entry in entries:
            eid = entry["id"]
            qfmt = entry["question_format"]
            question = entry["question"]
            sq_list = entry.get("sub_questions")
            # Handle missing/empty sub_questions
            if not sq_list or sq_list == [""] or all(s == "" for s in sq_list):
                sq_list = None
            col_names = entry["column_names"]
            word_opts = entry["word_response_types"]
            num_opts = entry["numeric_response_types"]

            if sq_list and len(sq_list) == len(col_names):
                # Matrix question: one item per sub_question
                for sq, col in zip(sq_list, col_names):
                    items.append(
                        {
                            "id": eid,
                            "question": question,
                            "sub_question": sq,
                            "column_name": col,
                            "word_options": word_opts,
                            "numeric_options": num_opts,
                            "question_format": qfmt,
                        }
                    )
            else:
                # Single / ranking: one item per column
                for col in col_names:
                    items.append(
                        {
                            "id": eid,
                            "question": question,
                            "sub_question": None,
                            "column_name": col,
                            "word_options": word_opts,
                            "numeric_options": num_opts,
                            "question_format": qfmt,
                        }
                    )
        return items

    items = expand_items(value_entries)
    f"Expanded {len(value_entries)} entries into {len(items)} individual items"
    return (items,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Build User Prompts from Templates

    For each item, the user prompt is constructed from the question template.
    - If the template is `null`, the original question text is used verbatim.
    - If the template contains `{{sub_question}}`, it is substituted.
    - The response options are appended.
    """)
    return


@app.cell
def _(items, question_templates):
    def build_user_prompt(item):
        eid = str(item["id"])
        tmpl_data = question_templates.get(eid)

        if tmpl_data is not None and item["sub_question"] is not None:
            prompt = tmpl_data + "\n\n" + f"{item['sub_question']}:"
        else:
            prompt = item["question"]

        return prompt

    sample = [(i["id"], i["sub_question"], build_user_prompt(i)) for i in items[:4]]
    f"Example prompts: {sample}"
    return (build_user_prompt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate Placeholder Cluster Responses

    Until real WVS cluster data is available, we generate placeholder response
    distributions. For each item, we create a cluster entry with:
    - `mode`: the most common word response (modal response)
    - `distribution`: probability over each word response category
    - `sample`: a draw from the distribution
    """)
    return


@app.cell
def _(items, np, random):
    random.seed(42)
    np.random.seed(42)

    def generate_placeholder_clusters(items):
        clusters = {}
        for item in items:
            key = (item["id"], item["column_name"])
            n_opts = len(item["word_options"])
            # Random Dirichlet distribution
            probs = np.random.dirichlet(np.ones(n_opts) * 0.5)
            mode_idx = int(np.argmax(probs))
            clusters[key] = {
                "distribution": probs.tolist(),
                "mode": mode_idx,
                "word_options": item["word_options"],
                "numeric_options": item["numeric_options"],
            }
        return clusters

    placeholder_clusters = generate_placeholder_clusters(items)
    f"Generated placeholder data for {len(placeholder_clusters)} items"
    return (placeholder_clusters,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Build Single Response Dataset

    Each example has a single expected response (text + numeric).
    This is used for standard SFT. Two sampling strategies are available:
    - `mode`: the modal (most common) response from the cluster
    - `sample`: a random draw from the cluster distribution
    """)
    return


@app.cell
def _(
    build_user_prompt,
    items,
    pd,
    placeholder_clusters,
    random,
    system_prompts,
):
    random.seed(42)

    def build_single_dataset(items, clusters, strategy="mode"):
        dataset = []
        for item in items:
            key = (item["id"], item["column_name"])
            cluster = clusters[key]

            if strategy == "mode":
                idx = cluster["mode"]
            else:
                probs = cluster["distribution"]
                idx = random.choices(range(len(probs)), weights=probs, k=1)[0]

            word_answer = cluster["word_options"][idx]
            num_answer = cluster["numeric_options"][idx]

            user_prompt = build_user_prompt(item)
            system = system_prompts[item["id"] % len(system_prompts)]

            dataset.append(
                {
                    "system_prompt": system,
                    "user_prompt": user_prompt,
                    "expected_text": word_answer,
                    "expected_numeric": num_answer,
                    "question_id": item["id"],
                    "sub_question": item["sub_question"],
                    "column_name": item["column_name"],
                    "question_format": item["question_format"],
                }
            )
        return dataset

    ds_mode = build_single_dataset(items, placeholder_clusters, "mode")
    ds_sample = build_single_dataset(items, placeholder_clusters, "sample")

    df_sample = pd.DataFrame(ds_sample)

    df_mode = pd.DataFrame(ds_mode)

    df_mode
    return df_mode, df_sample


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Build Distributional Response Dataset

    Each example has an expected probability distribution over the response
    categories. This is used for distribution-matching fine-tuning approaches
    (e.g., soft cross-entropy loss).
    """)
    return


@app.cell
def _(build_user_prompt, items, pd, placeholder_clusters, system_prompts):
    def build_distributional_dataset(items, clusters):
        dataset = []
        for item in items:
            key = (item["id"], item["column_name"])
            cluster = clusters[key]
            user_prompt = build_user_prompt(item)
            system = system_prompts[item["id"] % len(system_prompts)]

            dataset.append(
                {
                    "system_prompt": system,
                    "user_prompt": user_prompt,
                    "expected_distribution": cluster["distribution"],
                    "categories": cluster["word_options"],
                    "question_id": item["id"],
                    "sub_question": item["sub_question"],
                    "column_name": item["column_name"],
                    "question_format": item["question_format"],
                }
            )
        return dataset

    ds_distribution = build_distributional_dataset(items, placeholder_clusters)
    f"Distributional dataset: {len(ds_distribution)} rows"

    df_dist = pd.DataFrame(ds_distribution)

    df_dist
    return (df_dist,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Export Datasets
    """)
    return


@app.cell
def _(df_dist, df_mode, df_sample, output_dir):
    df_mode.to_parquet(output_dir / "training_single_modal.parquet")
    df_sample.to_parquet(output_dir / "training_single_sample.parquet")
    df_dist.to_parquet(output_dir / "training_distributional.parquet")

    for name in [
        "training_single_modal",
        "training_single_sample",
        "training_distributional",
    ]:
        size = (output_dir / f"{name}.parquet").stat().st_size
        print(f"  {name}.parquet: {size / 1024:.1f} KB")
    return


if __name__ == "__main__":
    app.run()
