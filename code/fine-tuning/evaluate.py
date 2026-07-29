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
        --dataset distributional \
        --subpopulation overall

    # For quick checks without plots:
    uv run code/fine-tuning/evaluate_model_cli.py \
        --port 8000 \
        --model Qwen/Qwen3.6-27B \
        --dataset distributional \
        --subpopulation overall \
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


def _model_safe_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_").replace(" ", "_")


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
        default="distributional",
        choices=["single_modal", "single_sample", "distributional"],
        help="Dataset config to evaluate on",
    )
    p.add_argument(
        "--subpopulation",
        default="overall",
        choices=["cluster_0", "cluster_1", "overall"],
        help="Subpopulation to evaluate",
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
        "Default: output/eval/<model>_<dataset>_<subpopulation>/",
    )
    p.add_argument(
        "--max-tokens", type=int, default=1000, help="Max tokens for each query"
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


def main():
    from dotenv import load_dotenv

    load_dotenv()
    args = parse_args()

    if args.hf_token is None:
        args.hf_token = os.environ.get("HF_TOKEN")

    from datasets import load_dataset
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

    if args.output_dir is None:
        safe_name = _model_safe_name(args.model)
        output_dir = (
            Path("output") / "eval" / f"{safe_name}_{args.dataset}_{args.subpopulation}"
        )
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "target": args.model,
        "dataset": args.dataset,
        "population": args.subpopulation,
        "timestamp": datetime.now().isoformat(),
        "output_dir": str(output_dir.resolve()),
        "api_url": args.api_url,
        "api_key": args.api_key,
        "max_tokens": args.max_tokens,
        "top_logprobs": args.top_logprobs,
        "num_test_examples": args.num_test_examples,
        "no_plots": args.no_plots,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[save] config -> {output_dir / 'config.json'}")

    print("[data] loading test split...")
    ds = load_dataset(
        "1jamesthompson1/wvs-nz-value-alignment",
        args.dataset,
        split="test",
        token=args.hf_token or None,
    )
    ds = ds.filter(lambda x: x["subpopulation"] == args.subpopulation)
    if args.num_test_examples:
        ds = ds.select(range(min(args.num_test_examples, len(ds))))
    print(
        f"[data] loaded {len(ds)} test examples"
        f" (config={args.dataset}, subpop={args.subpopulation})"
    )

    has_expected_text = "expected_text" in ds.features
    has_categories = "categories" in ds.features

    print("[eval] connecting to server...")
    client = OpenAI(
        base_url=f"{args.api_url}/v1",
        api_key=args.api_key,
    )

    # Load response options from question mapping (used for all dataset configs)
    _qm_path = (
        Path(__file__).resolve().parent.parent
        / "training-dataset"
        / "output"
        / "question_mapping.json"
    )
    _options_lookup = {}
    if _qm_path.exists():
        with open(_qm_path) as _f:
            for _entry in json.load(_f):
                for _cn in _entry["column_names"]:
                    _options_lookup[_cn] = _entry["word_response_types"]

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
                model=args.model,
                messages=messages,
                max_tokens=args.max_tokens,
                logprobs=True,
                top_logprobs=args.top_logprobs,
                temperature=0.7,
                top_p=0.8,
                presence_penalty=1.5,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
        except Exception as e:
            print(f"  [error] query failed at example {i}: {e}")
            nan_count += 1
            continue

        choice = resp.choices[0]
        msg = choice.message
        logprobs_data = choice.logprobs

        # Extract answer text, handling thinking mode
        answer_text = ""
        if logprobs_data and logprobs_data.content:
            toks = logprobs_data.content
            ans_start = 0
            for k, t in enumerate(toks):
                if t.token == "</think>":
                    ans_start = k + 1
                    break
            if ans_start > 0 or not msg.content:
                answer_text = "".join(t.token for t in toks[ans_start:]).strip()
                # Strip trailing EOS token
                if answer_text.endswith("<|im_end|>"):
                    answer_text = answer_text[: -len("<|im_end|>")].strip()
        if not answer_text:
            answer_text = (msg.content or "").strip()
        if not answer_text:
            print(f"  [error] empty response at example {i}")
            nan_count += 1
            continue

        # Compute full option probabilities
        option_probs = {}
        categories = list(example.get("categories", [])) if has_categories else []
        if not categories:
            _col = example.get("column_name", "")
            categories = _options_lookup.get(_col, [])
        if categories:
            option_probs = compute_option_probs(categories, logprobs_data)

        row = {
            "question_id": example["question_id"],
            "question": example.get("question", ""),
            "sub_question": example.get("sub_question", ""),
            "column_name": example.get("column_name", ""),
            "question_format": example.get("question_format", ""),
            "system_prompt_id": example.get("system_prompt_id", ""),
            "model_answer": answer_text,
        }

        # Always compute distributional metrics when options are available
        if categories:
            row["categories"] = categories

            cat_probs = {c: option_probs.get(c, 0.0) for c in categories}
            _total_prob = sum(cat_probs.values())
            if _total_prob > 0:
                for k in cat_probs:
                    cat_probs[k] /= _total_prob
            model_dist = [cat_probs.get(c, 0.0) for c in categories]
            row["model_distribution"] = model_dist

            # Load true distribution from dataset (all configs have it now)
            if "expected_distribution" in example:
                true_dist = list(example["expected_distribution"])
                row["true_distribution"] = true_dist
            else:
                true_dist = None

            if true_dist:
                _p_clamped = [max(p, 1e-10) for p in true_dist]
                _q_clamped = [max(p, 1e-10) for p in model_dist]
                kl = kl_divergence(_p_clamped, _q_clamped)
                row["kl_divergence"] = kl
                total_kl += kl

                ce_dist = sum(
                    -tp * math.log(max(mp, 1e-10))
                    for tp, mp in zip(true_dist, model_dist)
                )
                row["cross_entropy"] = ce_dist
                total_ce += ce_dist

        # Accuracy for single-response configs
        if has_expected_text:
            expected_text = example["expected_text"]
            row["expected_text"] = expected_text
            if expected_text.strip().lower() == answer_text.strip().lower():
                correct_count += 1

        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "per_question_results.csv", index=False)
    print(f"[save] per-question results -> {output_dir / 'per_question_results.csv'}")

    if not args.no_plots and any("categories" in r for r in results):
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

    print()
    print("[done] evaluation complete")
    print(f"       Results saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
