import marimo

__generated_with = "0.23.14"
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


@app.cell
def _():
    import json
    import marimo as mo
    import pandas as pd
    import random
    import numpy as np
    from pathlib import Path

    return Path, json, mo, np, pd, random


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Train/Test Split

    A naive 5% random sample of items (question_id × column_name) is held out
    for testing. The split keys are computed once and reused across all three
    modeling approaches and all subpopulations, ensuring consistent partitions.

    The test set is used to add an `expected_distribution` column marking each
    row as train or test.
    """)
    return


@app.cell
def _(items, random):
    random.seed(42)

    all_item_keys = sorted({(item["id"], item["column_name"]) for item in items})
    n_test = max(1, len(all_item_keys) * 5 // 100)
    test_keys = set(random.sample(all_item_keys, n_test))
    train_keys = set(all_item_keys) - test_keys

    print(
        f"Train items: {len(train_keys)}, Test items: {len(test_keys)} ({n_test / len(all_item_keys) * 100:.1f}%)"
    )
    return test_keys, train_keys


@app.cell
def _(items, test_keys, train_keys):
    def split_rows(rows):
        train_rows = []
        test_rows = []
        for r in rows:
            key = (r["question_id"], r["column_name"])
            if key in train_keys:
                train_rows.append(r)
            else:
                test_rows.append(r)
        return train_rows, test_rows

    # Verify consistency across all items
    all_train = 0
    all_test = 0
    for _item in items:
        _key = (_item["id"], _item["column_name"])
        if _key in train_keys:
            all_train += 1
        elif _key in test_keys:
            all_test += 1
    print(f"Split covers {all_train + all_test}/{len(items)} item expansions")
    return (split_rows,)


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

    system_prompts = prompt_templates["system_prompts"]
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
    ## Compute Empirical Response Distributions

    Cluster assignments from LCA are used to split respondents into groups.
    Per-cluster and overall response distributions are computed **empirically**
    from the processed WVS value survey data — the actual proportion of each
    response category for each question among respondents in each group.

    **Why empirical instead of model-implied?** The LCA model estimates
    conditional item probabilities per cluster (its best guess of the response
    profile). But since we have hard cluster assignments for every respondent,
    we can simply compute the empirical proportions directly from their actual
    answers. This gives us the true observed distribution for each cluster,
    which is more faithful to the data.

    The empirical distributions are also saved to `empirical_cluster_distributions.json`
    for downstream use (replacing the old LCA model-implied distributions from
    `cluster_response_distributions.json`). The JSON structure is identical so
    any downstream code that reads the old file can seamlessly switch to the new one.
    """)
    return


@app.cell
def _(output_dir):
    """Load processed WVS survey data and merge with LCA cluster assignments."""
    import pandas as _pd

    # Load the already-processed value survey data (numeric response codes,
    # with -1 for "Don't know" and -5 for missing, handled by wrangle_response_data.py)
    wvs_value_survey = _pd.read_csv(output_dir / "wvs_value_survey.csv")
    wvs_value_survey["respondent_id"] = wvs_value_survey["id"].astype(int)

    # Load cluster assignments from the LCA model (id -> cluster label)
    cluster_assignments = _pd.read_csv(output_dir / "cluster_assignments.csv")
    cluster_assignments["respondent_id"] = cluster_assignments["id"].astype(int)

    # Merge so each row has both the survey responses and its cluster label
    respondents_with_clusters = wvs_value_survey.merge(
        cluster_assignments[["respondent_id", "cluster"]],
        on="respondent_id",
        how="inner",
    )

    unique_cluster_ids = sorted(respondents_with_clusters["cluster"].unique())
    total_respondents = len(respondents_with_clusters)

    print(
        f"Merged {total_respondents} respondents into "
        f"{len(unique_cluster_ids)} clusters: {unique_cluster_ids}"
    )
    return respondents_with_clusters, total_respondents, unique_cluster_ids


@app.cell
def _(
    items,
    json,
    np,
    output_dir,
    respondents_with_clusters,
    total_respondents,
    unique_cluster_ids,
):
    """
    For a given subset of respondents, compute the empirical proportion of
    each valid response category for a single survey question column.

    Only responses matching the known numeric codes (defined in the question
    mapping) are counted — missing codes (-1, -5) are excluded since the
    distribution should only reflect valid responses.
    """

    def _category_proportions(respondent_subset, column_name, known_numeric_codes):
        valid_responses = respondent_subset[column_name].dropna()
        valid_responses = valid_responses[valid_responses.isin(known_numeric_codes)]
        counts = [
            float((valid_responses == code).sum()) for code in known_numeric_codes
        ]
        total = sum(counts)
        return [count / total for count in counts] if total else None

    def _build_distribution_lookup(respondent_subset, items):
        """
        Build a lookup dict keyed by (question_id, column_name) containing
        the empirical distribution, mode index, and response options.
        """
        distribution_lookup = {}
        for item in items:
            column_name = item["column_name"]
            lookup_key = (item["id"], column_name)
            known_codes = [float(n) for n in item["numeric_options"]]

            proportions = _category_proportions(
                respondent_subset, column_name, known_codes
            )
            if proportions is None:
                num_categories = len(known_codes)
                proportions = [1.0 / num_categories] * num_categories

            distribution_lookup[lookup_key] = {
                "distribution": proportions,
                "mode": int(np.argmax(proportions)),
                "word_options": item["word_options"],
                "numeric_options": item["numeric_options"],
            }
        return distribution_lookup

    # Build per-cluster and overall distribution lookups
    cluster_distribution_lookups = {}
    per_cluster_json_output = {}

    for cluster_id in unique_cluster_ids:
        cluster_subset = respondents_with_clusters[
            respondents_with_clusters["cluster"] == cluster_id
        ]
        cluster_lookup = _build_distribution_lookup(cluster_subset, items)
        lookup_name = f"cluster_{cluster_id}"
        cluster_distribution_lookups[lookup_name] = cluster_lookup

        # Build JSON-serializable version for the saved file
        cluster_items_json = {}
        for item in items:
            column_name = item["column_name"]
            lookup_key = (item["id"], column_name)
            entry = cluster_lookup[lookup_key]
            cluster_items_json[column_name] = {
                "numeric_codes": [float(n) for n in entry["numeric_options"]],
                "word_labels": entry["word_options"],
                "distribution": entry["distribution"],
            }

        per_cluster_json_output[str(cluster_id)] = {
            "size": len(cluster_subset),
            "weight": len(cluster_subset) / total_respondents,
            "items": cluster_items_json,
        }

    # Overall population distribution (all respondents, no cluster split)
    cluster_distribution_lookups["overall"] = _build_distribution_lookup(
        respondents_with_clusters, items
    )

    # Save to JSON with the same structure as the original LCA model file
    empirical_distributions_file = output_dir / "empirical_cluster_distributions.json"
    with open(empirical_distributions_file, "w") as fh:
        json.dump(
            {
                "method": "empirical (computed from WVS data using LCA cluster assignments)",
                "n_clusters": len(unique_cluster_ids),
                "total_respondents": total_respondents,
                "clusters": per_cluster_json_output,
            },
            fh,
            indent=2,
        )

    print(
        f"Built {len(cluster_distribution_lookups)} distribution lookups "
        f"(clusters {unique_cluster_ids} + overall)"
    )
    print(f"Saved empirical distributions to {empirical_distributions_file.name}")
    return (cluster_distribution_lookups,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Build Single Response Datasets

    Each example has a single expected response (text + numeric).
    This is used for standard SFT. Two sampling strategies are available:
    - `mode`: the modal (most common) response from the cluster
    - `sample`: a random draw from the cluster distribution

    Each item is replicated across all system prompt templates,
    producing multiple training examples per question with different
    framing instructions. This augments the dataset and improves
    robustness to prompt variation at inference time.
    """)
    return


@app.cell
def _(
    build_user_prompt,
    cluster_distribution_lookups,
    items,
    random,
    system_prompts,
):
    random.seed(42)

    def build_single_dataset(items, lookup, strategy="mode"):
        dataset = []
        for item in items:
            key = (item["id"], item["column_name"])
            cluster = lookup[key]

            if strategy == "mode":
                idx = cluster["mode"]
            else:
                probs = cluster["distribution"]
                idx = random.choices(range(len(probs)), weights=probs, k=1)[0]

            word_answer = cluster["word_options"][idx]
            num_answer = cluster["numeric_options"][idx]
            user_prompt = build_user_prompt(item)

            for sp_idx, (sp_name, sp_text) in enumerate(system_prompts.items()):
                dataset.append(
                    {
                        "system_prompt": sp_text,
                        "system_prompt_id": sp_name,
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

    mode_sets = {}
    sample_sets = {}
    for _vs_name, _lookup in cluster_distribution_lookups.items():
        mode_sets[_vs_name] = build_single_dataset(items, _lookup, "mode")
        sample_sets[_vs_name] = build_single_dataset(items, _lookup, "sample")

    _example_count = len(next(iter(mode_sets.values())))
    f"Built single-response datasets for {len(mode_sets)} value sets ({_example_count} rows each, {len(system_prompts)} system prompts × {len(items)} items)"
    return mode_sets, sample_sets


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Build Distributional Response Datasets

    Each example has an expected probability distribution over the response
    categories. This is used for distribution-matching fine-tuning approaches
    (e.g., soft cross-entropy loss). Datasets are generated for each value set
    (cluster_0, cluster_1, overall).

    Each item is replicated across all system prompt templates.
    """)
    return


@app.cell
def _(build_user_prompt, cluster_distribution_lookups, items, system_prompts):
    def build_distributional_dataset(items, lookup):
        dataset = []
        for item in items:
            key = (item["id"], item["column_name"])
            cluster = lookup[key]
            user_prompt = build_user_prompt(item)

            for sp_idx, (sp_name, sp_text) in enumerate(system_prompts.items()):
                dataset.append(
                    {
                        "system_prompt": sp_text,
                        "system_prompt_id": sp_name,
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

    dist_sets = {}
    for _vs_name, _lookup in cluster_distribution_lookups.items():
        dist_sets[_vs_name] = build_distributional_dataset(items, _lookup)

    _example_count = len(next(iter(dist_sets.values())))
    f"Distributional datasets: {sum(len(v) for v in dist_sets.values())} total rows across {len(dist_sets)} value sets ({_example_count} rows each, {len(system_prompts)} system prompts × {len(items)} items)"
    return (dist_sets,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Export Datasets

    ### Hugging Face dataset structure
    The same data is organized under `output/dataset/` as a Hugging Face
    `datasets` repository. Each modeling approach (distributional,
    single_sample, single_modal) is a **config** with `train` and `test`
    splits. Each split references the per-subpopulation files.

    **Repository structure:**
    ```
    output/dataset/
      README.md
      distributional/
        train/
          cluster_0.parquet
          cluster_1.parquet
          overall.parquet
        test/
          cluster_0.parquet
          cluster_1.parquet
          overall.parquet
      single_sample/
        train/  (same layout)
        test/
      single_modal/
        train/
        test/
    ```
    """)
    return


@app.cell
def _(dist_sets, mode_sets, output_dir, pd, sample_sets, split_rows):
    _dataset_dir = output_dir / "dataset"

    _variants = {
        "single_modal": mode_sets,
        "single_sample": sample_sets,
        "distributional": dist_sets,
    }
    _subpops = ["cluster_0", "cluster_1", "overall"]

    for _config_name, _sets in _variants.items():
        for _subpop in _subpops:
            _all_rows = _sets[_subpop]
            for _r in _all_rows:
                _r["subpopulation"] = _subpop
            _train_rows, _test_rows = split_rows(_all_rows)

            _train_df = pd.DataFrame(_train_rows)
            _train_path = _dataset_dir / _config_name / "train" / f"{_subpop}.parquet"
            _train_path.parent.mkdir(parents=True, exist_ok=True)
            _train_df.to_parquet(_train_path)

            _test_df = pd.DataFrame(_test_rows)
            _test_path = _dataset_dir / _config_name / "test" / f"{_subpop}.parquet"
            _test_path.parent.mkdir(parents=True, exist_ok=True)
            _test_df.to_parquet(_test_path)

        # Log sizes
        for _split in ["train", "test"]:
            _total = 0
            for _subpop in _subpops:
                _path = _dataset_dir / _config_name / _split / f"{_subpop}.parquet"
                _size = _path.stat().st_size
                _n = len(pd.read_parquet(_path))
                _total += _n
                print(
                    f"  {_config_name}/{_split}/{_subpop}.parquet: {_size / 1024:.1f} KB ({_n} rows)"
                )
            print(f"  -> {_config_name}/{_split} total: {_total} rows")

    f"Exported {len(_variants)} configs to {_dataset_dir}"
    return


if __name__ == "__main__":
    app.run()
