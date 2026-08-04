#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "numpy>=2.0.0",
#   "pandas>=2.0.0",
#   "matplotlib>=3.8.0",
#   "datasets>=5.0.0",
#   "openai>=1.80.0",
#   "python-dotenv>=1.1.0",
# ]
# ///
"""
Evaluate a served model on the WVS-NZ test set.

Connects to a running vLLM server (OpenAI-compatible API), runs the
test split of the dataset, and reports:

  1. Per-question results (model answer, expected answer, distributions, etc.)
  2. Summary metrics (cross-entropy, KL-divergence, accuracy, etc.)
  3. Optional per-question distribution comparison plots

Usage:
    uv run evaluate.py \
        --port 8087 \
        --model Qwen/Qwen3.6-27B-FP8 \
        --dataset modal_response

    # For quick checks without plots:
    uv run evaluate.py \
        --port 8000 \
        --model Qwen/Qwen3.6-27B \
        --dataset modal_response \
        --no-plots \
        --num-test-examples 20
"""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _model_short_name(model: str) -> str:
    return model.split("/", 1)[-1]


def _model_safe_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_").replace(" ", "_")


def _model_sha(
    model_id: str, hf_token: str | None = None, server_created: int | None = None
) -> str | None:
    import os
    from datetime import datetime, timezone

    lookup = model_id
    if "/" not in lookup:
        hf_org = os.environ.get("HF_ORG")
        if not hf_org:
            return None
        lookup = f"{hf_org}/{lookup}"
    try:
        from huggingface_hub import model_info

        info = model_info(lookup, token=hf_token)
        if server_created and info.lastModified:
            if isinstance(info.lastModified, str):
                hub_updated = datetime.fromisoformat(
                    info.lastModified.replace("Z", "+00:00")
                )
            else:
                hub_updated = info.lastModified
            server_dt = datetime.fromtimestamp(int(server_created), tz=timezone.utc)
            diff = abs((hub_updated - server_dt).total_seconds())
            if diff < 300:
                print()
                print("!" * 60)
                print("WARNING: Model may have changed since server started!")
                print(
                    f"  Server loaded:     {server_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}"
                )
                print(
                    f"  Hub last modified: {hub_updated.strftime('%Y-%m-%d %H:%M:%S UTC')}"
                )
                print(f"  Gap:               {diff:.0f}s")
                print("  If you just uploaded a new version, restart the server.")
                print("!" * 60)
                print()
        model_sha = info.sha
        if model_sha:
            print(f"[eval] model commit: {model_sha}")
        else:
            print(f"[eval] model commit: Not found for '{lookup}'")
        return model_sha
    except Exception as e:
        print(f"  [warn] could not fetch SHA for model '{lookup}': {e}")
        return None


def _load_dataset_and_sha(
    dataset_id: str,
    config_name: str,
    hf_token: str | None = None,
    num_test_examples: int | None = None,
):
    from datasets import load_dataset
    from huggingface_hub import dataset_info

    dataset_sha = None
    try:
        repo_info = dataset_info(dataset_id, token=hf_token)
        dataset_sha = repo_info.sha
    except Exception:
        print("[data] could not fetch dataset SHA from Hub")

    ds = load_dataset(dataset_id, config_name, split="test", token=hf_token or None)
    if num_test_examples:
        ds = ds.select(range(min(num_test_examples, len(ds))))

    subpops = (
        set(ds["subpopulation"]) if "subpopulation" in ds.features else {"unknown"}
    )
    print(
        f"[data] loaded {len(ds)} test examples"
        f" (config={config_name}, subpops={', '.join(sorted(subpops))})"
    )
    if dataset_sha:
        print(f"[data] dataset commit: {dataset_sha}")

    return ds, dataset_sha


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Evaluate a served model on the WVS-NZ test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--api-url",
        default=None,
        help="vLLM server URL (default: http://localhost:<port>)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8000,
        help="vLLM server port (used when --api-url is not set)",
    )
    p.add_argument("--api-key", default="EMPTY", help="API key for the vLLM server")
    p.add_argument(
        "--model",
        default="Qwen/Qwen3.6-27B",
        help="Model name as registered on the server",
    )
    p.add_argument(
        "--dataset",
        default="modal_response",
        choices=["modal_response", "sampled_response"],
        help="Dataset config to evaluate on. "
        "The distributional configs (full_string_distribution, "
        "first_token_distribution) are not evaluable via served logprobs yet — "
        "modal_response gives both accuracy and KL/CE.",
    )
    p.add_argument(
        "--hf-token",
        default=None,
        help="HF token for loading dataset (or HF_TOKEN env var)",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save evaluation results. "
        "Default: output/evals/<model>/<dataset>-<timestamp>/",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max total tokens for each query (thinking + answer). "
        "Default: 1000 (no reasoning) / 4096 (reasoning).",
    )
    p.add_argument(
        "--top-logprobs", type=int, default=20, help="Number of top logprobs to request"
    )
    p.add_argument(
        "--num-test-examples",
        type=int,
        default=None,
        help="Limit to first N test examples (for quick checks)",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        default=False,
        help="Skip generating per-question distribution plots",
    )
    p.add_argument(
        "--reasoning",
        action="store_true",
        default=False,
        help="Enable reasoning mode (chain-of-thought)",
    )

    return p.parse_args(argv)


def compute_option_probs(
    categories: list[str],
    logprobs_data,
    answer_tokens: list | None = None,
) -> dict[str, float]:
    """Compute full-sequence probability for each category option.

    Uses the per-position logprobs from the model response to trace the
    most likely token path matching each option string. Falls back to
    first-token-only probabilities when the full path can't be resolved.
    """
    if not logprobs_data or not logprobs_data.content:
        return {}

    toks = logprobs_data.content

    # Find answer start (after </think>)
    ans_start = 0
    for _i, _t in enumerate(toks):
        if _t.token == "</think>":
            ans_start = _i + 1
            break
    ans_toks = toks[ans_start:]

    if not ans_toks:
        return {}

    # Build per-position lookup: token -> logprob
    pos_lps = []
    for _t in ans_toks:
        pos_lps.append({_tp.token: _tp.logprob for _tp in (_t.top_logprobs or [])})

    # Token texts
    token_texts = [_t.token for _t in ans_toks]

    result = {}
    for _cat in categories:
        _opt = _cat.lstrip()
        _running = ""
        _chain = []

        for _j, _tok_text in enumerate(token_texts):
            _new_run = _running + _tok_text
            _new_stripped = _new_run.lstrip()

            if _opt.startswith(_new_stripped):
                _chain.append((_j, _tok_text, True))
                _running = _new_run
                if _new_stripped == _opt:
                    _next = _j + 1
                    if _next < len(pos_lps) and "<|im_end|>" in pos_lps[_next]:
                        _chain.append((_next, "<|im_end|>", False))
                    elif "<|im_end|>" in pos_lps[_j]:
                        _chain.append((_j, "<|im_end|>", False))
                    break
            else:
                if _j < len(pos_lps):
                    for _alt_token in pos_lps[_j]:
                        _alt_run = _running + _alt_token
                        if _opt.startswith(_alt_run.lstrip()):
                            _chain.append((_j, _alt_token, False))
                            _running = _alt_run
                            if _alt_run.lstrip() == _opt:
                                _next = _j + 1
                                if (
                                    _next < len(pos_lps)
                                    and "<|im_end|>" in pos_lps[_next]
                                ):
                                    _chain.append((_next, "<|im_end|>", False))
                            break
                break

        if not _chain:
            for _tp in ans_toks[0].top_logprobs:
                if _opt.startswith(_tp.token.lstrip()):
                    _chain.append((0, _tp.token, False))
                    break

        _total_lp = 0.0
        for _pos, _tok, _ in _chain:
            if _pos < len(pos_lps) and _tok in pos_lps[_pos]:
                _total_lp += pos_lps[_pos][_tok]

        result[_cat] = math.exp(_total_lp) if _chain else 0.0

    return result


def cross_entropy(
    true_label: str | list[float],
    top_logprobs: dict[str, float],
    categories: list[str] | None = None,
) -> float:
    if categories and isinstance(true_label, list):
        logprobs_for_cats = []
        for cat in categories:
            lp = top_logprobs.get(cat)
            if lp is None:
                fuzzy = [t for t in top_logprobs if cat in t or t in cat]
                lp = top_logprobs.get(fuzzy[0]) if fuzzy else None
            if lp is not None:
                logprobs_for_cats.append(lp)
            else:
                logprobs_for_cats.append(-100.0)
        probs = [math.exp(lp) for lp in logprobs_for_cats]
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        return sum(
            -true_p * math.log(max(p, 1e-10)) for true_p, p in zip(true_label, probs)
        )

    # For single-response: cross-entropy of the true label
    lp = top_logprobs.get(true_label)
    if lp is None:
        fuzzy = [t for t in top_logprobs if true_label in t or t in true_label]
        lp = top_logprobs.get(fuzzy[0]) if fuzzy else None
    if lp is not None:
        return -lp
    return float("nan")


def kl_divergence(p: list[float], q: list[float]) -> float:
    return sum(
        pi * math.log(max(pi / max(qi, 1e-10), 1e-10)) for pi, qi in zip(p, q) if pi > 0
    )


def plot_distribution_comparison(
    per_question_df: pd.DataFrame, output_dir: Path, max_plots: int = 100
):
    import matplotlib.pyplot as plt
    import textwrap

    figs_dir = output_dir / "figures"
    figs_dir.mkdir(exist_ok=True)
    plot_count = 0

    for col, group in per_question_df.groupby("column_name"):
        if plot_count >= max_plots:
            break

        row = group.iloc[0]
        cats = row.get("categories")
        true_dist = row.get("true_distribution")
        if not cats or not true_dist:
            continue

        sq = row.get("sub_question", "")
        qtext = row.get("question", "")
        if not sq or (isinstance(sq, float) and math.isnan(sq)):
            sq = ""

        title_parts = []
        if qtext:
            title_parts.append(textwrap.fill(qtext, width=70))
        if sq:
            title_parts.append(sq)
        title = "\n".join(title_parts)

        def _make_plot(model_dist, label_suffix, filename_suffix):
            """Helper to create a single comparison plot."""
            x = np.arange(len(cats))
            width = 0.35

            fig, ax = plt.subplots(figsize=(max(9, len(cats) * 0.8), 5.5))
            bars1 = ax.bar(
                x - width / 2,
                true_dist,
                width,
                label="Expected",
                color="#DD8452",
                alpha=0.85,
            )
            bars2 = ax.bar(
                x + width / 2,
                model_dist,
                width,
                label="Model",
                color="#4C72B0",
                alpha=0.85,
            )

            for b1, b2 in zip(bars1, bars2):
                for b, d in [(b1, true_dist), (b2, model_dist)]:
                    h = b.get_height()
                    if h > 0.01:
                        ax.annotate(
                            f"{h:.0%}",
                            xy=(b.get_x() + b.get_width() / 2, h),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            fontsize=7,
                        )

            ax.set_xticks(x)
            ax.set_xticklabels(cats, fontsize=8, rotation=20, ha="right")
            ax.set_ylabel("Probability")
            ax.set_title(title, fontsize=10, linespacing=1.3, pad=12)
            ax.legend(fontsize=8, loc="upper right")

            _p = [max(p, 1e-10) for p in true_dist]
            _q = [max(p, 1e-10) for p in model_dist]
            kl = sum(pi * math.log(pi / qi) for pi, qi in zip(_p, _q) if pi > 0)
            ax.text(
                0.98,
                0.95,
                f"KL={kl:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7),
            )

            footer = f"Column: {col}  {label_suffix}"
            fig.text(
                0.5,
                0.01,
                footer,
                ha="center",
                va="bottom",
                fontsize=7,
                color="gray",
                style="italic",
            )

            ax.set_ylim(0, max(max(true_dist), max(model_dist)) * 1.25)
            fig.subplots_adjust(top=0.88, bottom=0.08)
            plt.tight_layout(rect=[0, 0.04, 1, 0.92])

            safe_name = f"{col}{filename_suffix}".replace(" ", "_").replace("/", "_")[
                :80
            ]
            figpath = figs_dir / f"{safe_name}.png"
            fig.savefig(figpath, dpi=150, bbox_inches="tight")
            plt.close(fig)

        # --- Per-system-prompt plots ---
        for _, r in group.iterrows():
            md = r.get("model_distribution")
            if md and isinstance(md, (list, tuple)):
                sp = r.get("system_prompt_id", "unknown")
                _make_plot(md, f"System Prompt: {sp}", f"_{sp}")

        # --- Averaged plot ---
        model_dists = group["model_distribution"].tolist()
        model_dists = [d for d in model_dists if isinstance(d, (list, tuple))]
        if model_dists:
            avg_dist = [sum(vals) / len(vals) for vals in zip(*model_dists)]
            _make_plot(avg_dist, "Averaged across prompts", "_avg")

        plot_count += 1 + len(group)

    print(f"[plots] saved plots to {figs_dir}")


def _load_question_options():
    import json
    from pathlib import Path

    qm_path = (
        Path(__file__).resolve().parent.parent
        / "training-dataset"
        / "output"
        / "question_mapping.json"
    )
    lookup = {}
    if qm_path.exists():
        with open(qm_path) as f:
            for entry in json.load(f):
                for cn in entry["column_names"]:
                    lookup[cn] = entry["word_response_types"]
    return lookup


def _run_evaluation(
    client, ds, model, max_tokens, top_logprobs, reasoning, output_dir, no_plots
):
    import json
    import time

    start = time.time()
    has_expected_text = "expected_text" in ds.features
    has_categories = "categories" in ds.features

    options_lookup = _load_question_options()

    results = []
    total_ce = 0.0
    total_kl = 0.0
    correct_count = 0
    nan_count = 0

    print("[eval] running test set...")
    for i, example in enumerate(ds):
        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(ds)}]...")

        messages = [
            {"role": "system", "content": example["system_prompt"]},
            {"role": "user", "content": example["user_prompt"]},
        ]

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                logprobs=True,
                top_logprobs=top_logprobs,
                temperature=0.7,
                top_p=0.8,
                presence_penalty=1.5,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": reasoning},
                },
            )
        except Exception as e:
            print(f"  [error] query failed at example {i}: {e}")
            nan_count += 1
            continue

        choice = resp.choices[0]
        msg = choice.message
        logprobs_data = choice.logprobs

        answer_text = ""
        reasoning_text = ""
        answer_start_found = False
        if logprobs_data and logprobs_data.content:
            toks = logprobs_data.content
            ans_start = 0
            for k, t in enumerate(toks):
                if t.token == "</think>":
                    ans_start = k + 1
                    answer_start_found = True
                    break
            if ans_start > 0 or not msg.content:
                if ans_start > 0:
                    reasoning_text = "".join(
                        t.token for t in toks[: ans_start - 1]
                    ).strip()
                answer_text = "".join(t.token for t in toks[ans_start:]).strip()
                if answer_text.endswith("<|im_end|>"):
                    answer_text = answer_text[: -len("<|im_end|>")].strip()
        if not answer_text and msg.content:
            content = msg.content
            think_end = content.find("</think>")
            if think_end >= 0:
                reasoning_text = content[:think_end].strip()
                answer_text = content[think_end + len("</think>") :].strip()
                answer_start_found = True
            if not answer_text:
                answer_text = content.strip()
        if not answer_text:
            print(f"  [error] empty response at example {i}")
            nan_count += 1
            continue
        if reasoning and not answer_start_found:
            print(
                "  [warn] answer start not found (</think> missing) — response may be truncated"
            )

        option_probs = {}
        categories = list(example.get("categories", [])) if has_categories else []
        if not categories:
            col = example.get("column_name", "")
            categories = options_lookup.get(col, [])
        if categories:
            if reasoning and not answer_start_found:
                pass
            else:
                option_probs = compute_option_probs(categories, logprobs_data)

        row = {
            "question_id": example["question_id"],
            "question": example.get("question", ""),
            "sub_question": example.get("sub_question", ""),
            "column_name": example.get("column_name", ""),
            "question_format": example.get("question_format", ""),
            "system_prompt_id": example.get("system_prompt_id", ""),
            "subpopulation": example.get("subpopulation", ""),
            "model_answer": answer_text,
            "model_reasoning": reasoning_text if reasoning else "",
        }

        if categories:
            row["categories"] = categories
            cat_probs = {c: option_probs.get(c, 0.0) for c in categories}
            total_prob = sum(cat_probs.values())
            if total_prob > 0:
                for k in cat_probs:
                    cat_probs[k] /= total_prob
            model_dist = [cat_probs.get(c, 0.0) for c in categories]
            row["model_distribution"] = model_dist

            if "expected_distribution" in example:
                true_dist = list(example["expected_distribution"])
                row["true_distribution"] = true_dist
            else:
                true_dist = None

            if true_dist:
                p_clamped = [max(p, 1e-10) for p in true_dist]
                q_clamped = [max(p, 1e-10) for p in model_dist]
                kl = kl_divergence(p_clamped, q_clamped)
                row["kl_divergence"] = kl
                total_kl += kl
                ce_dist = sum(
                    -tp * math.log(max(mp, 1e-10))
                    for tp, mp in zip(true_dist, model_dist)
                )
                row["cross_entropy"] = ce_dist
                total_ce += ce_dist

        if has_expected_text:
            expected_text = example["expected_text"]
            row["expected_text"] = expected_text
            if expected_text.strip().lower() == answer_text.strip().lower():
                correct_count += 1

        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "per_question_results.csv", index=False)
    print(f"[save] per-question results -> {output_dir / 'per_question_results.csv'}")

    if not no_plots and any("categories" in r for r in results):
        plot_distribution_comparison(df, output_dir)

    print()
    print("=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"  Examples evaluated:  {len(results)}")
    print(f"  Nan / failures:      {nan_count}")

    if len(results) > nan_count:
        avg_ce = total_ce / max(len(results) - nan_count, 1)
        print(f"  Avg cross-entropy:   {avg_ce:.4f}")
    if has_expected_text and len(results) > nan_count:
        accuracy = correct_count / max(len(results) - nan_count, 1) * 100
        print(f"  Accuracy (exact):    {accuracy:.1f}%")
        print(f"  Correct:             {correct_count}/{len(results) - nan_count}")
    if total_kl > 0 and len(results) > 0:
        avg_kl = total_kl / max(len(results), 1)
        print(f"  Avg KL-divergence:   {avg_kl:.4f}")

    elapsed = time.time() - start
    config_path = output_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        config["elapsed_seconds"] = round(elapsed)
        config_path.write_text(json.dumps(config, indent=2))
    print()
    print(f"[done] evaluation complete ({elapsed:.0f}s)")
    print(f"       Results saved to {output_dir.resolve()}")


def main():
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv())
    args = parse_args()

    if args.hf_token is None:
        args.hf_token = os.environ.get("HF_TOKEN")

    from openai import OpenAI

    if args.api_url is None:
        args.api_url = f"http://localhost:{args.port}"

    print("=" * 60)
    print("Evaluation configuration")
    print("=" * 60)
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print("=" * 60)

    import json
    from datetime import datetime
    import urllib.request

    # Query server for available models (fail fast if model doesn't exist)
    models_url = f"{args.api_url}/v1/models"
    try:
        with urllib.request.urlopen(models_url, timeout=5) as resp:
            models_data = json.loads(resp.read().decode())
        available = [m["id"] for m in models_data.get("data", [])]
    except Exception as e:
        print(f"[eval] could not fetch model list from {models_url}: {e}")
        available = []

    if available and args.model not in available:
        print(f"\n[error] Model '{args.model}' not found on server.")
        print(f"Available models ({len(available)}):")
        for m in available:
            print(f"  - {m}")
        print("\nSet --model to one of the above.")
        sys.exit(1)
    elif available:
        print(
            f"[eval] model '{args.model}' found on server ({len(available)} total models)"
        )

    run_name = f"{args.dataset}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if args.output_dir is None:
        model_name = _model_short_name(args.model)
        output_dir = Path("output") / "evals" / model_name / run_name
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds, dataset_sha = _load_dataset_and_sha(
        "1jamesthompson1/wvs-nz-value-alignment",
        args.dataset,
        hf_token=args.hf_token,
        num_test_examples=args.num_test_examples,
    )

    server_created = None
    if available:
        for m in models_data.get("data", []):
            if m["id"] == args.model:
                server_created = m["created"]
                break

    if args.max_tokens is None:
        args.max_tokens = 4096 if args.reasoning else 1000

    model_sha = _model_sha(
        args.model, hf_token=args.hf_token, server_created=server_created
    )

    config = {
        "target": args.model,
        "dataset": args.dataset,
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "output_dir": str(output_dir.resolve()),
        "api_url": args.api_url,
        "api_key": args.api_key,
        "max_tokens": args.max_tokens,
        "top_logprobs": args.top_logprobs,
        "num_test_examples": args.num_test_examples,
        "no_plots": args.no_plots,
        "reasoning": args.reasoning,
        "dataset_sha": dataset_sha,
        "model_sha": model_sha,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[save] config -> {output_dir / 'config.json'}")

    client = OpenAI(
        base_url=f"{args.api_url}/v1",
        api_key=args.api_key,
    )

    _run_evaluation(
        client=client,
        ds=ds,
        model=args.model,
        max_tokens=args.max_tokens,
        top_logprobs=args.top_logprobs,
        reasoning=args.reasoning,
        output_dir=output_dir,
        no_plots=args.no_plots,
    )


if __name__ == "__main__":
    main()
