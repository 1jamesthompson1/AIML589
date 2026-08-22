import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Building the Value Alignment Training Dataset

    This notebook takes the `question_mapping.json`, placeholder cluster responses and
    `prompt_templates.json` to produce four training dataset variants:

    1. **modal_response** — each example has one expected answer (text + numeric):
       the modal (most common) response from the cluster. Used for SFT.
    2. **sampled_response** — each example has one expected answer: a random draw
       from the cluster response distribution. Used for SFT.
    3. **full_string_distribution** — each example has an expected probability
       distribution over response categories; the training loss scores full
       option-string completions (weighted NLL). Used for distribution matching.
    4. **first_token_distribution** — like (3) but options are labelled with
       single letters (A., B., ...) and the prompts ask for the single letter,
       so the expected completion is one token. Used for first-token losses
       (Cao et al., NAACL 2025).

    Each sub-question in a matrix/battery is expanded into its own training example.
    Demographic questions are excluded as they are not relevant for value alignment.
    """)
    return


@app.cell
def _():
    import hashlib
    import json
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import random
    import numpy as np
    import os
    import pickle
    import re
    import shutil
    import textwrap
    from itertools import combinations
    from pathlib import Path
    from scipy.stats import chi2_contingency
    from matplotlib.patches import Patch
    from dotenv import load_dotenv

    load_dotenv()

    # Which LCA model to use: the number of clusters from cluster_respondents.py.
    # Must be 2 or 3 (or any k with a completed run), i.e. the assignments are
    # read from output/cluster_analysis/k{SELECTED_K}_analysis/cluster_assignments.csv.
    SELECTED_K = 2

    # Which subpopulations to build datasets for. Options are the LCA cluster
    # names ("cluster_0", "cluster_1", ...) plus "overall". The available
    # cluster names depend on the loaded cluster_assignments.csv (e.g.
    # cluster_0 .. cluster_1 for k=2, cluster_0 .. cluster_2 for k=3).
    SELECTED_SUBPOPS = ["cluster_0", "cluster_1", "overall"]
    return (
        Patch,
        Path,
        SELECTED_K,
        SELECTED_SUBPOPS,
        chi2_contingency,
        combinations,
        hashlib,
        json,
        mo,
        np,
        os,
        pd,
        pickle,
        plt,
        random,
        re,
        shutil,
        textwrap,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Train/Validation Split

    The validation set is deliberately **out of distribution** and serves two
    sanity checks:

    1. **Held-out questions** — 5 questions, one per redundancy cluster and
       battery, whose responses are most predictable from the *other*
       questions (max cross-battery Cramer's V). The model has the
       value-relevant information to answer them but never sees the exact
       question text, so reproducing their empirical distributions shows it
       learned values rather than memorised question→answer pairs.
    2. **Held-out system prompt** — the prompt least similar to the other
       five (mean token-Jaccard), never used in training, so any behaviour
       difference under it measures genuine prompt sensitivity.

    There is no test split: everything not held out goes to training.
    Splits are keyed by item (question_id × column_name) and reused across
    all four modeling approaches and all subpopulations, ensuring consistent
    partitions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Select held out prompts

    Select the held-out system prompt: the least similar to the other
    five by mean token-Jaccard. It is never used in training, so any
    behaviour difference under it measures genuine prompt sensitivity.
    """)
    return


@app.cell
def _(pd, re, system_prompts):
    def _tokens(s):
        return set(re.findall(r"[a-z]+", s.lower()))

    prompt_names = list(system_prompts.keys())
    prompt_sim = pd.DataFrame(index=prompt_names, columns=prompt_names, dtype=float)
    for a in prompt_names:
        for b in prompt_names:
            ta, tb = _tokens(system_prompts[a]), _tokens(system_prompts[b])
            prompt_sim.loc[a, b] = len(ta & tb) / max(len(ta | tb), 1)
    held_out_prompt = prompt_sim.mean(axis=1).idxmin()

    print(f"Held-out system prompt: {held_out_prompt}")
    print("Pairwise token-Jaccard similarities (rows/cols = prompts):")
    print(prompt_sim.round(3).to_string())
    return (held_out_prompt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Held out survey questions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Compute the redundancy structure between all question pairs.

    A question is a good held-out pick when its responses are most
    predictable from the OTHER questions (max cross-battery Cramer's V) —
    the model has the value-relevant information to answer it but never
    sees its exact text. All computation lives inside one function so
    the global namespace stays clean; the results feed the held-out
    selection cell below.

    This is the slowest step in the notebook, so the result is cached to
    `output/redundancy.pkl`, keyed by a hash of the input files
    (question mapping, survey, cluster assignments). Reruns load the
    cache; it invalidates automatically when any input changes (delete
    the file to force a recompute).
    """)
    return


@app.cell
def _(
    SELECTED_K,
    chi2_contingency,
    combinations,
    hashlib,
    items,
    np,
    output_dir,
    pd,
    pickle,
    respondents_with_clusters,
):
    def compute_redundancy(items, respondents_with_clusters):
        """Return the cross-battery redundancy structure as a dict:

        - scores: {(c1, c2): Cramer's V} for every cross-battery pair
        - redundancy: per-question DataFrame (max_cross_v, top_partner,
          battery_id, sub_question, question), sorted by max_cross_v desc
        - parent_map: union-find roots linking questions strongly
          associated cross-battery (V > 0.5) into redundancy components
        - top_partners: per-question top-3 most similar cross-battery
          questions (kept in train), for diagnostics
        - col2battery / col2sub / col2q: column lookup helpers
        """
        value_cols = [it["column_name"] for it in items]
        survey = respondents_with_clusters[value_cols].replace(-5.0, np.nan)

        def cramers_v(c1, c2):
            pair = survey[[c1, c2]].dropna()
            if len(pair) < 50:
                return float("nan")
            ct = pd.crosstab(pair[c1], pair[c2])
            if min(ct.shape) < 2:
                return float("nan")
            chi2, *_ = chi2_contingency(ct)
            n = ct.values.sum()
            return float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))

        col2battery = {it["column_name"]: it["id"] for it in items}
        col2sub = {it["column_name"]: it["sub_question"] for it in items}
        col2q = {it["column_name"]: it["question"] for it in items}

        # Pairwise Cramer's V. Only cross-battery pairs are scored:
        # same-battery questions are trivially related.
        all_pairs = list(combinations(value_cols, 2))
        scores = {}
        for i, (c1, c2) in enumerate(all_pairs):
            if i % 5000 == 0:
                print(f"  [redundancy] {i}/{len(all_pairs)} pairs...")
            if col2battery[c1] == col2battery[c2]:
                continue
            scores[(c1, c2)] = cramers_v(c1, c2)

        # Per-question redundancy: max V against a question from another
        # battery.
        rows = []
        for c in value_cols:
            rel = [(o, v) for (a, o), v in scores.items() if a == c] + [
                (a, v) for (a, o), v in scores.items() if o == c
            ]
            cross = [(o, v) for o, v in rel if col2battery[c] != col2battery[o]]
            if not cross:
                continue
            rows.append(
                {
                    "column": c,
                    "sub_question": col2sub.get(c),
                    "battery_id": col2battery[c],
                    "question": col2q.get(c),
                    "max_cross_v": round(max(v for _, v in cross), 3),
                    "top_partner": max(cross, key=lambda x: x[1])[0],
                }
            )
        redundancy = pd.DataFrame(rows).sort_values("max_cross_v", ascending=False)

        # Redundancy components via union-find: questions linked when
        # strongly associated cross-battery (V > 0.5) measure the same
        # value dimension.
        def parent(x, p):
            while p[x] != x:
                p[x] = p[p[x]]
                x = p[x]
            return x

        parent_map = {c: c for c in value_cols}
        for (c1, c2), v in scores.items():
            if v > 0.5 and col2battery[c1] != col2battery[c2]:
                r1, r2 = parent(c1, parent_map), parent(c2, parent_map)
                if r1 != r2:
                    parent_map[r1] = r2

        # Top cross-battery partners per question (for diagnostics: which
        # *training* questions are most similar to each held-out one).
        partner_rows = []
        for c in value_cols:
            rel = [(o, v) for (a, o), v in scores.items() if a == c] + [
                (a, v) for (a, o), v in scores.items() if o == c
            ]
            cross = sorted(rel, key=lambda x: -x[1])[:3]
            for o, v in cross:
                partner_rows.append(
                    {
                        "column": c,
                        "similar_to": o,
                        "similar_sub": col2sub.get(o),
                        "similar_question": col2q.get(o),
                        "similar_battery": col2battery[o],
                        "v": round(v, 3),
                    }
                )
        top_partners = pd.DataFrame(partner_rows)

        return {
            "scores": scores,
            "redundancy": redundancy,
            "parent_map": parent_map,
            "top_partners": top_partners,
            "col2battery": col2battery,
            "col2sub": col2sub,
            "col2q": col2q,
        }

    # Disk cache: the pairwise Cramer's V over all cross-battery pairs is
    # the slowest step in this notebook, and it only depends on the
    # question mapping plus the survey/cluster files. The result is saved
    # to output/redundancy.pkl, keyed by a hash of those inputs, so
    # reruns load it instantly. The cache invalidates automatically when
    # any input file changes; delete the file to force a recompute.
    _cache_file = output_dir / "redundancy.pkl"

    def redundancy_cache_key():
        """Hash of every input the redundancy structure depends on."""
        h = hashlib.sha256()
        for p in (
            output_dir / "question_mapping.json",
            output_dir / "wvs_value_survey.csv",
            output_dir
            / "cluster_analysis"
            / f"k{SELECTED_K}_analysis"
            / "cluster_assignments.csv",
        ):
            if p.exists():
                h.update(p.name.encode())
                h.update(p.read_bytes())
        return h.hexdigest()[:16]

    def load_or_compute_redundancy(items, respondents_with_clusters, key):
        """Load the cached redundancy if the key matches, else compute and
        cache it."""
        if _cache_file.exists():
            try:
                with open(_cache_file, "rb") as fh:
                    cached = pickle.load(fh)
                if cached["key"] == key:
                    print(f"Loaded redundancy from cache: {_cache_file} (key {key})")
                    return cached["redundancy_data"]
                print(f"Inputs changed (key {key}) — recomputing redundancy")
            except Exception as exc:
                print(f"Cache unreadable ({exc}) — recomputing redundancy")
        redundancy_data = compute_redundancy(items, respondents_with_clusters)
        with open(_cache_file, "wb") as fh:
            pickle.dump({"key": key, "redundancy_data": redundancy_data}, fh)
        print(f"Computed redundancy, cached to {_cache_file} (key {key})")
        return redundancy_data

    _key = redundancy_cache_key()
    redundancy_data = load_or_compute_redundancy(items, respondents_with_clusters, _key)
    return (redundancy_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Pick how many to hold out

    Candidates are ranked greedy by max_cross_v, one per battery AND per
    redundancy component (cross-battery V > 0.5). The first N_HELD_OUT
    become the validation set. The prints, ranking table and bar chart
    below show the marginal redundancy of each successive pick, so you
    can judge whether N_HELD_OUT questions is the right amount.
    """)
    return


@app.cell
def _(Path, pd, redundancy_data, textwrap):

    # How many questions to hold out for validation (one per battery AND
    # per redundancy component).
    N_PRINTOUT = 15
    N_HELD_OUT = 6

    redundancy = redundancy_data["redundancy"]
    parent_map = redundancy_data["parent_map"]
    top_partners = redundancy_data["top_partners"]
    col2sub = redundancy_data["col2sub"]

    def parent(x, p):
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    # Rank every candidate: one per battery AND per redundancy component.
    candidates, seen_batts, seen_comps = [], set(), set()
    for r in redundancy.itertuples():
        comp_root = parent(r.column, parent_map)
        if r.battery_id in seen_batts or comp_root in seen_comps:
            continue
        candidates.append(r)
        seen_batts.add(r.battery_id)
        seen_comps.add(comp_root)

    held_out_questions = pd.DataFrame(candidates[:N_HELD_OUT])
    held_cols = set(held_out_questions["column"])

    # Marginal redundancy: the V of each successive pick, so you can see
    # where holding out an extra question stops buying much.
    print(
        "\nMarginal redundancy of each successive pick (one per battery + component):"
    )
    for i, r in enumerate(candidates[:15], 1):
        print(
            f"  pick {i:2d}: {r.column:<6} V={r.max_cross_v:.3f}  "
            f"{col2sub.get(r.column)}"
        )

    def print_held_out_similarity(questions, top_partners, out_path):
        """Show, for each held-out question, the training questions most
        similar to it (cross-battery Cramer's V, top 3, kept in train).
        Writes the block to out_path (plain text, ASCII-only and
        line-wrapped so it embeds cleanly in LaTeX verbatim, ready for
        the report appendix)."""

        def _wrap_line(line, width=75):
            """Wrap via ``textwrap``, keeping the leading indentation on
            continuation lines (long question texts would otherwise
            overflow the LaTeX verbatim box)."""
            indent = line[: len(line) - len(line.lstrip())]
            return textwrap.fill(
                line.strip(),
                width=width,
                initial_indent=indent,
                subsequent_indent=indent,
            )

        lines = []
        for r in questions.itertuples():
            lines.append("\n" + "=" * 70)
            if r in held_out_questions.itertuples():
                lines.append(f" Actual HELD OUT: {r.column} - {r.sub_question}")
            else:
                lines.append(f"Suggested HELD OUT: {r.column} - {r.sub_question}")
            lines.append(f"  Q: {r.question}")
            lines.append(f"  (max cross-battery V={r.max_cross_v})")
            lines.append("-" * 70)
            lines.append("  Most similar TRAINING questions:")
            sims = top_partners[
                (top_partners["column"] == r.column)
                & (~top_partners["similar_to"].isin(held_cols))
            ]
            for s in sims.itertuples():
                lines.append(f"\n    ~ {s.similar_to} - {s.similar_sub}  (V={s.v})")
                lines.append(f"      Q: {s.similar_question}")
        text = "\n".join(_wrap_line(line) for line in lines)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(text)

    print_held_out_similarity(
        pd.DataFrame(candidates[:N_PRINTOUT]),
        top_partners,
        "../figures/held_out_similarity.txt",
    )

    # Redundancy components with >= 2 questions: these are the only
    # places where holding out several questions would be redundant.
    comp_sizes = {}
    for c in redundancy["column"]:
        root = parent(c, parent_map)
        comp_sizes[root] = comp_sizes.get(root, 0) + 1
    multi_comps = sorted((r, n) for r, n in comp_sizes.items() if n >= 2)
    print("\nRedundancy components (cross-battery V > 0.5) with >= 2 questions:")
    for root, n in multi_comps:
        members = [c for c in redundancy["column"] if parent(c, parent_map) == root]
        print(f"  {root}: {n} questions -> {', '.join(members)}")
    return (
        N_HELD_OUT,
        candidates,
        col2sub,
        held_cols,
        held_out_questions,
        redundancy,
    )


@app.cell
def _(N_HELD_OUT, candidates, col2sub, mo, pd):
    # Ranking table so you can see what holding out 6, 7, ... would add.
    cand_df = pd.DataFrame(
        [
            {
                "pick_rank": i + 1,
                "column": r.column,
                "battery_id": r.battery_id,
                "sub_question": col2sub.get(r.column),
                "max_cross_v": r.max_cross_v,
                "held_out": i < N_HELD_OUT,
            }
            for i, r in enumerate(candidates[:25])
        ]
    )
    mo.ui.table(
        cand_df, label="Candidates by pick order — first N_HELD_OUT are held out"
    )
    return


@app.cell
def _(
    Patch,
    held_cols,
    held_out_questions,
    np,
    plt,
    redundancy,
    redundancy_data,
):
    # Bar chart of max cross-battery V, coloured by redundancy group.
    _parent_map = redundancy_data["parent_map"]

    def _root(c):
        while _parent_map[c] != c:
            c = _parent_map[c]
        return c

    _group_of, _root_members = {}, {}
    for _c in redundancy["column"]:
        _r = _root(_c)
        _group_of[_c] = _r
        _root_members.setdefault(_r, []).append(_c)

    # Order groups by their best member's rank so colours follow the
    # chart order (left to right).
    _multi_roots = [r for r, m in _root_members.items() if len(m) >= 2]
    _first_rank = {
        r: redundancy.index[redundancy["column"].isin(_root_members[r])].min()
        for r in _multi_roots
    }
    _multi_roots.sort(key=lambda r: _first_rank[r])

    _cmap = plt.cm.tab20
    _group_colors = {
        r: _cmap(i / max(len(_multi_roots), 1)) for i, r in enumerate(_multi_roots)
    }
    _grey = "#c8c8c8"

    _bar_colors = [_group_colors.get(_group_of[c], _grey) for c in redundancy["column"]]
    _edge_colors = ["black" if c in held_cols else "none" for c in redundancy["column"]]
    _linewidths = [1.3 if c in held_cols else 0.0 for c in redundancy["column"]]

    # Held-out groups, in pick order (one per battery + redundancy group).
    _held_group_roots, _seen = [], set()
    for _c in held_out_questions["column"]:
        _r = _group_of[_c]
        if _r not in _seen:
            _seen.add(_r)
            _held_group_roots.append(_r)
    _held_out_in_group = {_r: _c for _c in held_cols for _r in [_group_of[_c]]}

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(
        np.arange(len(redundancy)),
        redundancy["max_cross_v"],
        color=_bar_colors,
        edgecolor=_edge_colors,
        linewidth=_linewidths,
        width=0.9,
    )
    ax.axhline(0.5, color="black", ls="--", lw=0.8)
    ax.text(len(redundancy) - 1, 0.52, "V = 0.5 threshold", ha="right", fontsize=9)
    for _i, _r in enumerate(redundancy.head(15).itertuples()):
        ax.annotate(
            _r.column,
            (_i, _r.max_cross_v),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set_xlabel("questions ranked by max cross-battery V")
    ax.set_ylabel("max cross-battery Cramer's V")
    ax.set_title(
        "Redundancy of all questions (colour = redundancy group, black edge = held out)"
    )

    _handles = []
    for _i, _r in enumerate(_held_group_roots):
        _members = sorted(_root_members[_r])
        _handles.append(
            Patch(
                facecolor=_group_colors.get(_r, _grey),
                edgecolor="black",
                label=(
                    f"held-out group {_i + 1} — {', '.join(_members)} "
                    f"(held out: {_held_out_in_group[_r]})"
                ),
            )
        )
    _handles.append(
        Patch(facecolor=_grey, label="other questions (no redundancy group)")
    )
    ax.legend(
        handles=_handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=7,
        frameon=False,
    )

    fig.savefig("../figures/redundancy_bar_chart.pdf", bbox_inches="tight")

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Build the train/validation split and the row assigner.

    Validation = held-out questions (any prompt) + every item under the
    held-out system prompt. Train = everything else. There is no test
    split; splits are keyed by item (question_id × column_name) and reused
    across all four modeling approaches and all subpopulations.
    """)
    return


@app.cell
def _(first_token_system_prompts, held_out_prompt, held_out_questions, items):
    # First-token prompts are named "{name}_letter", so the matching
    # first-token variant of the held-out prompt must be held out too.
    held_out_prompt_ids = {held_out_prompt} | {
        name
        for name in first_token_system_prompts
        if name.removesuffix("_letter") == held_out_prompt
    }

    def make_splits(items, held_out_questions):
        """Item keys per split."""
        all_item_keys = sorted({(item["id"], item["column_name"]) for item in items})
        held_out_keys = set(
            (int(row.battery_id), row.column) for row in held_out_questions.itertuples()
        )
        train_keys = set(all_item_keys) - held_out_keys
        return held_out_keys, train_keys

    held_out_keys, train_keys = make_splits(items, held_out_questions)
    val_keys = held_out_keys

    def split_rows(rows):
        train_rows, val_rows = [], []
        for r in rows:
            key = (r["question_id"], r["column_name"])
            if key in val_keys or r.get("system_prompt_id") in held_out_prompt_ids:
                val_rows.append(r)
            else:
                train_rows.append(r)
        return train_rows, val_rows

    print(f"Train items: {len(train_keys)}, Validation items: {len(val_keys)}")
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
    first_token_system_prompts = prompt_templates["system_prompts_first_token"]
    question_templates = prompt_templates["question_templates"]

    f"Loaded {len(question_mapping)} question entries, {len(system_prompts)} system prompts, {len(first_token_system_prompts)} first-token system prompts"
    return (
        first_token_system_prompts,
        output_dir,
        question_mapping,
        question_templates,
        system_prompts,
    )


@app.cell
def _(system_prompts, textwrap):
    # output system prompts in nice format for LaTeX report, wrapped at
    # 75 chars so they embed cleanly in LaTeX verbatim blocks
    with open("../figures/system_prompts.txt", "w") as sys_f:
        for name, text in system_prompts.items():
            wrapped = "\n\n".join(
                textwrap.fill(p, width=75) for p in text.split("\n\n")
            )
            sys_f.write(f"--- {name} ---\n{wrapped}\n\n")
    return


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
    def option_identifier(idx):
        """Single-letter identifier for option index 0..25 (A..Z).

        Letters are used for every question format — not numbers — because
        single-token-ness must be guaranteed: verified on the Qwen3.6-27B
        tokenizer, digits 1-9 are single tokens but "10" (the top rating-scale
        option) tokenises as two tokens. Letters A-K are always single tokens.
        """
        return chr(ord("A") + idx)

    def build_user_prompt(item):
        eid = str(item["id"])
        tmpl_data = question_templates.get(eid)

        options_text = "\n\nOptions:\n" + "\n".join(item["word_options"])

        if tmpl_data is not None and item["sub_question"] is not None:
            prompt = tmpl_data + options_text + "\n\n" + f"{item['sub_question']}:"
        else:
            prompt = item["question"] + options_text

        return prompt

    def build_user_prompt_lettered(item):
        """Same prompt, but options prefixed with single-letter identifiers
        (A., B., C., ...) so the expected answer can be a single token.

        For rating-scale questions the numeric options are kept intact and
        still get letter prefixes (e.g. "A. 1", "B. 2", ..., "J. 10") so the
        answer token is always a single letter.
        """
        eid = str(item["id"])
        tmpl_data = question_templates.get(eid)

        options_text = "\n\nOptions:\n" + "\n".join(
            f"{option_identifier(i)}. {opt}"
            for i, opt in enumerate(item["word_options"])
        )

        if tmpl_data is not None and item["sub_question"] is not None:
            prompt = tmpl_data + options_text + "\n\n" + f"{item['sub_question']}:"
        else:
            prompt = item["question"] + options_text

        return prompt

    sample = [(i["id"], i["sub_question"], build_user_prompt(i)) for i in items[:4]]
    f"Example prompts: {sample}"
    return build_user_prompt, build_user_prompt_lettered, option_identifier


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
def _(SELECTED_K, output_dir, pd):
    """Load processed WVS survey data and merge with LCA cluster assignments."""

    # Load the already-processed value survey data (numeric response codes,
    # with -1 for "Don't know" and -5 for missing, handled by wrangle_response_data.py)
    wvs_value_survey = pd.read_csv(output_dir / "wvs_value_survey.csv")
    wvs_value_survey["respondent_id"] = wvs_value_survey["id"].astype(int)

    # Load cluster assignments from the selected LCA run
    # (output/cluster_analysis/k{SELECTED_K}_analysis/cluster_assignments.csv)
    assignments_path = (
        output_dir
        / "cluster_analysis"
        / f"k{SELECTED_K}_analysis"
        / "cluster_assignments.csv"
    )
    if not assignments_path.exists():
        raise FileNotFoundError(
            f"{assignments_path} not found — run cluster_respondents.py with "
            f"--n_clusters={SELECTED_K} first, or change SELECTED_K at the top "
            "of this notebook."
        )
    cluster_assignments = pd.read_csv(assignments_path)
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
            word_options = cluster["word_options"]
            true_dist = cluster["distribution"]

            for sp_idx, (sp_name, sp_text) in enumerate(system_prompts.items()):
                dataset.append(
                    {
                        "system_prompt": sp_text,
                        "system_prompt_id": sp_name,
                        "user_prompt": user_prompt,
                        "expected_text": word_answer,
                        "expected_numeric": num_answer,
                        "categories": word_options,
                        "expected_distribution": true_dist,
                        "question_id": item["id"],
                        "question": item["question"],
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
                        "question": item["question"],
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
    ## Build First-Token Distributional Response Datasets

    Same empirical distribution as `full_string_distribution`, but optimised
    for first-token training (Cao et al., NAACL 2025): options are labelled
    with single letters (A., B., ...) and the system prompts instruct the
    model to answer with ONLY the letter, so the expected completion is a
    single token. Each row stores `answer_tokens` (the letter for each
    category, aligned with `categories`) for the training loop to map to
    token ids.
    """)
    return


@app.cell
def _(
    build_user_prompt_lettered,
    cluster_distribution_lookups,
    first_token_system_prompts,
    items,
    option_identifier,
):
    def build_first_token_dataset(items, lookup):
        dataset = []
        for item in items:
            key = (item["id"], item["column_name"])
            cluster = lookup[key]
            user_prompt = build_user_prompt_lettered(item)
            answer_tokens = [
                option_identifier(i) for i in range(len(cluster["word_options"]))
            ]

            for sp_idx, (sp_name, sp_text) in enumerate(
                first_token_system_prompts.items()
            ):
                dataset.append(
                    {
                        "system_prompt": sp_text,
                        "system_prompt_id": sp_name,
                        "user_prompt": user_prompt,
                        "expected_distribution": cluster["distribution"],
                        "categories": cluster["word_options"],
                        "answer_tokens": answer_tokens,
                        "question_id": item["id"],
                        "question": item["question"],
                        "sub_question": item["sub_question"],
                        "column_name": item["column_name"],
                        "question_format": item["question_format"],
                    }
                )
        return dataset

    ft_dist_sets = {}
    for _vs_name, _lookup in cluster_distribution_lookups.items():
        ft_dist_sets[_vs_name] = build_first_token_dataset(items, _lookup)

    _example_count = len(next(iter(ft_dist_sets.values())))
    f"First-token datasets: {sum(len(v) for v in ft_dist_sets.values())} total rows across {len(ft_dist_sets)} value sets ({_example_count} rows each, {len(first_token_system_prompts)} system prompts × {len(items)} items)"
    return (ft_dist_sets,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Export Datasets

    ### Hugging Face dataset structure
    The same data is organized under `output/dataset/` as a Hugging Face
    `datasets` repository. Each modeling approach (modal_response,
    sampled_response, full_string_distribution, first_token_distribution) is a
    **config** with `train` and `validation` splits. Each split references the
    per-subpopulation files.

    **Repository structure:**
    ```
    output/dataset/
      README.md
      modal_response/
        {train,validation}/
          cluster_0.parquet
          cluster_1.parquet
          overall.parquet
      sampled_response/   (same layout)
      full_string_distribution/   (same layout)
      first_token_distribution/   (same layout, adds answer_tokens column)
    ```
    """)
    return


@app.cell
def _(
    SELECTED_SUBPOPS,
    dist_sets,
    ft_dist_sets,
    mode_sets,
    output_dir,
    pd,
    sample_sets,
    shutil,
    split_rows,
):
    _dataset_dir = output_dir / "dataset"

    _variants = {
        "modal_response": mode_sets,
        "sampled_response": sample_sets,
        "full_string_distribution": dist_sets,
        "first_token_distribution": ft_dist_sets,
    }
    _subpops = SELECTED_SUBPOPS
    _splits = ["train", "validation"]

    _missing = [s for s in _subpops if s not in mode_sets]
    if _missing:
        raise ValueError(
            f"Selected subpopulations {_missing} were not built; "
            f"available: {sorted(mode_sets)}"
        )

    for _config_name, _sets in _variants.items():
        _config_dir = _dataset_dir / _config_name

        # Remove stale split directories (e.g. the old test split) so the
        # exported dataset only ever contains the current splits.
        for _existing in _config_dir.iterdir():
            if _existing.is_dir() and _existing.name not in _splits:
                shutil.rmtree(_existing)
                print(
                    f"  removed stale split dir: {_existing.relative_to(_dataset_dir)}"
                )

        for _subpop in _subpops:
            _all_rows = _sets[_subpop]
            for _r in _all_rows:
                _r["subpopulation"] = _subpop
            _train_rows, _val_rows = split_rows(_all_rows)

            for _split, _rows in zip(_splits, [_train_rows, _val_rows]):
                _df = pd.DataFrame(_rows)
                _path = _dataset_dir / _config_name / _split / f"{_subpop}.parquet"
                _path.parent.mkdir(parents=True, exist_ok=True)
                _df.to_parquet(_path)

        # Log sizes
        for _split in _splits:
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Publish to the Hugging Face Hub

    The exported datasets under `output/dataset/` are published to the Hub
    so the fine-tuning pipeline can load them directly:
    [`1jamesthompson1/wvs-nz-value-alignment`](https://huggingface.co/datasets/1jamesthompson1/wvs-nz-value-alignment).
    Set `HF_TOKEN` (with write permission) in the environment first.

    Uploading **overwrites** the public dataset and cannot be undone, so
    the cells below require explicit confirmation: tick the
    acknowledgement checkbox, type the repo id, then press the upload
    button. Nothing touches the network until all of that is done.
    """)
    return


@app.cell
def _(mo, output_dir, pd):
    # Publish to the Hub overwrites the public dataset, so require explicit
    # confirmation: an acknowledgement checkbox AND the repo id typed in
    # full, with the actual checks done in the upload cell below (marimo
    # forbids reading UI values in the cell that creates them).
    WVS_REPO_ID = "1jamesthompson1/wvs-nz-value-alignment"
    dataset_dir = output_dir / "dataset"

    def _summarize_dataset(dataset_dir):
        """Per-file summary of the exported dataset: rows and size per
        config / split / subpopulation."""
        rows = []
        for p in sorted(dataset_dir.rglob("*.parquet")):
            rows.append(
                {
                    "config": p.parent.parent.name,
                    "split": p.parent.name,
                    "subpopulation": p.stem,
                    "rows": len(pd.read_parquet(p)),
                    "size_kb": round(p.stat().st_size / 1024, 1),
                }
            )
        return pd.DataFrame(rows)

    summary = _summarize_dataset(dataset_dir)
    _total_rows = int(summary["rows"].sum())
    _total_size_kb = float(summary["size_kb"].sum())

    upload_ack = mo.ui.checkbox(
        label=(
            f"I understand this OVERWRITES the public dataset `{WVS_REPO_ID}` "
            "and cannot be undone."
        )
    )
    upload_typed = mo.ui.text(placeholder=f'Type "{WVS_REPO_ID}" exactly to confirm')
    # run_button (not button): pressing it runs the cells that reference it,
    # which works even where UI interactions don't auto-rerun dependents
    # (e.g. the VS Code extension).
    upload_go = mo.ui.run_button(
        label="Upload to the Hugging Face Hub",
        kind="danger",
    )

    mo.vstack(
        [
            mo.md(
                f"**{len(summary)} parquet files, {_total_rows:,} rows, "
                f"{_total_size_kb:,.0f} KB** will be uploaded to "
                f"`{WVS_REPO_ID}`."
            ),
            summary,
            upload_ack,
            upload_typed,
            upload_go,
        ]
    )
    return WVS_REPO_ID, dataset_dir, upload_ack, upload_go, upload_typed


@app.cell
def _(WVS_REPO_ID, dataset_dir, mo, os, upload_ack, upload_go, upload_typed):
    mo.stop(
        not upload_go.value,
        mo.md(
            "Nothing uploaded yet — tick the checkbox, type the repo id, "
            "then press the **Upload to the Hugging Face Hub** button above."
        ),
    )
    mo.stop(
        not upload_ack.value,
        mo.md("Tick the acknowledgement checkbox before uploading."),
    )
    mo.stop(
        upload_typed.value != WVS_REPO_ID,
        mo.md(f"Type `{WVS_REPO_ID}` exactly in the text box to confirm."),
    )

    token = os.environ.get("HF_TOKEN")
    mo.stop(
        not token,
        mo.md(
            "**`HF_TOKEN` is not set** — export it first, e.g. "
            "`export HF_TOKEN=hf_...`, then run this cell again."
        ),
    )

    # huggingface_hub is imported here (like in fine-tuning/finetune.py) so
    # the marimo global namespace stays clean.
    def upload_to_hub(repo_id, folder, token):
        from huggingface_hub import HfApi, create_repo

        api = HfApi(token=token)
        create_repo(repo_id, token=token, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=folder,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Regenerated by build_dataset.py",
        )
        return repo_id

    uploaded = upload_to_hub(WVS_REPO_ID, dataset_dir, token)
    f"Done — https://huggingface.co/datasets/{uploaded}"
    return


if __name__ == "__main__":
    app.run()
