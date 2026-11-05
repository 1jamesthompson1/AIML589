#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "inspect-ai==0.3.287",
#   "inspect-evals==0.27.0",
#   "openai",  # inspect's OpenAI-compatible client for the vLLM endpoint
#   # ifeval's verifier is git-only. Declared explicitly (rather than via the
#   # [ifeval] extra) because uv does not apply [tool.uv.sources] to extras.
#   "instruction-following-eval",
#   "langdetect",
# ]
#
# [tool.uv.sources]
# instruction-following-eval = { git = "https://github.com/josejg/instruction_following_eval", rev = "0c495b2" }
# ///
"""
Capability sanity check for a served model (base or LoRA adapter) against vLLM.

Purpose: show that fine-tuning on the WVS value dataset did not remove general
capability. It answers two questions, not "what is the model's leaderboard
score":

  1. did the adapter keep *knowledge and reasoning* -> mmlu_pro
  2. is the effect visible in the *answer distribution*, even where the
     argmax is unchanged -> mmlu_pro_logprob (1 output token per item; the
     A-J distribution is scored directly, so this is the cheap, high-n view
     of the same items)
  3. did it keep *instruction following*            -> ifeval   (the likely
     breakage: SFT on near-one-hot answer codes makes a model emit "3"
     regardless of the question)

`mmlu_pro` and `mmlu_pro_logprob` build the identical dataset and use the same
seed/limit, so they select the same items. That means the two rows can be read
side by side: a capability drop that shows up in logprob accuracy but not in
generative accuracy is a *reasoning* effect, while one that shows up in the
generative row only is a *format/CoT* effect.

``gpqa_diamond`` is available with ``--tasks gpqa_diamond`` but is not in the
default set: this model reasons for ~4-6k tokens per item even when asked for
a direct answer, so it costs more than everything else combined and n=100 is
still a coarse tripwire.

Tasks are fixed, deterministic subsets (``sample_shuffle=<seed>`` + ``limit``)
so the base model and every adapter see the SAME items. That pairing is what
makes the deltas trustworthy: compare paired differences, not raw accuracies.

Outputs, per model, under ``output/capability/<model>/<run>/``:
  config.json, summary.json, per-sample rows in ``summary_samples.csv``,
  plus the raw inspect logs in ``inspect-logs/``.

With ``--models a b c`` the whole grid runs back to back and a paired
comparison is written to ``output/capability/comparison-<ts>/``
(``comparison.md`` + ``comparison.csv``), with base-vs-adapter deltas,
bootstrap CIs and exact McNemar tests.

Usage (against the vLLM server from serve.py / run.sh serve):
    # the standard sanity check: 5 models, ~Tier A
    uv run capability_eval.py --port 8087 \
        --models Qwen/Qwen3.8-27B \
                 Qwen3.8-27B-nz-wvs-modal_response-overall \
                 Qwen3.8-27B-nz-wvs-sampled_response-overall \
                 Qwen3.8-27B-nz-wvs-full_string_distribution-overall \
                 Qwen3.8-27B-nz-wvs-first_token_distribution-overall

    # one model, a smaller subset while iterating
    uv run capability_eval.py --port 8087 --model Qwen/Qwen3.8-27B \
        --tasks mmlu_pro --max-samples 50

    # enable reasoning (10x cost; off by default to match the fine-tuning evals)
    uv run capability_eval.py --port 8087 --model Qwen/Qwen3.8-27B --thinking

Notes:
  * ``--concurrency`` is inspect's ``max_samples`` (samples in flight). The
    vLLM server runs ``--max-num-seqs 256``; measured aggregate decode on an
    RTX PRO 6000 is ~420 tok/s at 16 in flight vs ~1,870 at 128, so the
    default of 128 is worth ~4x wall clock. ``max_connections`` is set to the
    same value because inspect otherwise starts at 20 connections and ramps.
  * SWE-bench is available with ``--include-swe`` but is NOT part of the
    sanity check: docker sandboxes, agent loops, 24k context, and a handful
    of samples give no statistical power.
  * gpqa_diamond defaults to 4 epochs in inspect_evals; epochs is forced to 1
    here (4x cheaper).
  * ``invalid_format`` is reported next to accuracy on purpose: a collapsed
    adapter that answers "3" to every question is a format failure, not
    knowledge loss, and the two must not be conflated.
"""

import argparse
import csv
import json
import math
import random
import re
from datetime import datetime
from pathlib import Path

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver
from inspect_evals.gpqa import gpqa_diamond
from inspect_evals.ifeval import ifeval
from inspect_evals.mmlu_pro import mmlu_pro
from inspect_evals.mmlu_pro.mmlu_pro import (
    DATASET_PATH,
    MMLU_PRO_DATASET_REVISION,
    record_to_sample,
)
from inspect_evals.utils.huggingface import hf_dataset

# Default fixed-subset sizes (override with --max-samples).
# Tier A: ~900-1,700 samples per model, all tasks, full determinism.
TASK_DEFAULTS = {
    "mmlu_pro": 1000,
    "mmlu_pro_logprob": 1000,  # same subset as mmlu_pro, so the two pair up
    "gpqa_diamond": 100,  # full set is 198; opt-in only
    "ifeval": 541,  # the whole benchmark
}

# Per-task output caps. These bound runaway generation (this model reasons
# even when the prompt asks for a direct answer); they are not targets.
# mmlu_pro/gpqa_diamond CoT averages ~1.4k / ~4-6k tokens, and a truncated
# sample scores incorrect, so the cap has to be high enough that truncation
# stays a small, reportable tail rather than a systematic bias.
TASK_MAX_TOKENS = {
    "mmlu_pro": 6144,
    "mmlu_pro_logprob": 1,  # one token: the answer letter
    "gpqa_diamond": 8192,
    "ifeval": 2048,
    "swe_bench": 4096,
}

# Tasks scored from the first-token logprobs rather than by parsing generated
# text. They cost ~1 output token per item, so they trade generation cost for
# prefill and give the full answer distribution instead of a single letter.
TASK_LOGPROBS = {"mmlu_pro_logprob": True}

# Part of the sanity check.
TASKS = ["mmlu_pro_logprob", "mmlu_pro", "ifeval"]
# Available but opt-in: expensive (reasoning runs long) and low statistical
# power at small n. Use --tasks gpqa_diamond if a hard-reasoning axis is wanted.
OPTIONAL_TASKS = ["swe_bench"]

SWE_DATASETS = {
    "verified_mini": None,  # swe_bench_verified_mini() (public, 50)
    "mini": {"dataset": "princeton-nlp/SWE-bench_Mini", "revision": None},
    "verified": None,  # swe_bench() default (gated, pinned revision)
}

# Scorer values from inspect_ai: C = correct, I = incorrect, N = no answer.
CORRECT, INCORRECT, NOANSWER = "C", "I", "N"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Direct-answer prompt for the logprob-scored variant. No chain of thought:
# the measurement is the next-token distribution over A-J, so the model has to
# emit the letter as its first token. (Verified against the served Qwen3.8:
# with thinking off it answers "B" at p=0.997 rather than starting to reason.)
MMLU_PRO_LETTER_PROMPT = """Answer the following multiple choice question. Reply with only the letter of the correct option, nothing else.

Question:
{question}
Options:
{options}
Answer:"""


@solver
def letter_prompt() -> Solver:
    """Render the question + A..J options and hand off to a 1-token generate."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        options = "\n".join(
            f"{chr(65 + i)}) {choice}" for i, choice in enumerate(state.choices or [])
        )
        state.messages = [
            ChatMessageUser(
                content=MMLU_PRO_LETTER_PROMPT.format(
                    question=state.input_text, options=options
                )
            )
        ]
        return state

    return solve


def letter_probs(output, letters):
    """Normalised P(letter) from the first generated token's top-k logprobs.

    Letters can appear under several token spellings ("A", " A", "a"), so
    their probability mass is summed before normalising over valid letters.
    Letters outside the returned top-k contribute nothing, which slightly
    understates the tail but never changes the argmax in practice.
    """
    if not output.choices:
        return {}
    logprobs = output.choices[0].logprobs
    if not logprobs or not logprobs.content:
        return {}
    first = logprobs.content[0]
    items = [(t.token, t.logprob) for t in (first.top_logprobs or [])]
    if first.token:
        items.append((first.token, first.logprob))

    mass = dict.fromkeys(letters, 0.0)
    for token, logprob in items:
        key = token.strip().upper()
        if key in mass:
            mass[key] += math.exp(logprob)
    total = sum(mass.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in mass.items()}


@scorer(metrics=[accuracy(), stderr()])
def letter_choice() -> Scorer:
    """Score by argmax over the letter distribution, keeping the full vector.

    The per-item probability vector is stored in Score.metadata so the
    comparison can ask whether fine-tuning shifted the *distribution* even
    where it did not change the argmax.
    """

    async def score(state: TaskState, target: Target) -> Score:
        letters = [chr(65 + i) for i in range(len(state.choices or []))]
        probs = letter_probs(state.output, letters)
        if not probs:
            return Score(value=NOANSWER, answer="", explanation="no letter logprobs")
        predicted = max(probs, key=probs.get)
        gold = (target.text or "").strip().upper()
        return Score(
            value=CORRECT if predicted == gold else INCORRECT,
            answer=predicted,
            explanation=f"P({gold})={probs.get(gold, 0.0):.3f}",
            metadata={"probs": probs, "p_gold": probs.get(gold, 0.0)},
        )

    return score


# How to read each task's results.
#   metrics     - (accuracy, stderr) keys in log.results.scores[0].metrics
#   bool_key    - for dict-valued scorers, the per-sample field that is a bool
#                 (ifeval scores are dicts, not C/I/N)
TASK_SPEC = {
    # lenient=True adds a second correctness test that accepts any of the
    # output conventions these adapters were fine-tuned on, not just the
    # benchmark's `ANSWER: $LETTER` template. See lenient_letter().
    "mmlu_pro": {"metrics": ("accuracy", "stderr"), "bool_key": None, "lenient": True},
    "mmlu_pro_logprob": {
        "metrics": ("accuracy", "stderr"),
        "bool_key": None,
        "lenient": False,
    },
    "gpqa_diamond": {
        "metrics": ("accuracy", "stderr"),
        "bool_key": None,
        "lenient": True,
    },
    # prompt-level strict accuracy is the headline IFEval number; its per
    # sample truth value is the prompt_level_strict flag.
    "ifeval": {
        "metrics": ("prompt_strict_acc", "prompt_strict_stderr"),
        "bool_key": "prompt_level_strict",
        "lenient": False,
    },
    "swe_bench": {
        "metrics": ("accuracy", "stderr"),
        "bool_key": None,
        "lenient": False,
    },
}


def build_task(name, swe_dataset="verified_mini"):
    """Instantiate one eval task.

    mmlu_pro: task-level shuffle is disabled so eval(sample_shuffle=seed,
    max_samples=N) selects the SAME deterministic subset for every model.
    gpqa_diamond: plain inspect_evals task (no sandbox); its 4-epoch default
    is overridden by the epoch=1 passed to eval().
    ifeval: short prompts, objective verifier, ~1 min per model.
    """
    if name == "mmlu_pro":
        # shuffle=False is load-bearing: hf_dataset(shuffle=True) draws a
        # RANDOM seed, so two processes see different dataset orders and
        # sample_shuffle + limit then select DIFFERENT items. That silently
        # breaks base-vs-adapter pairing. Order is left fixed here and all
        # randomness comes from eval(sample_shuffle=args.seed) instead.
        return mmlu_pro(shuffle=False, fewshot=0)
    if name == "mmlu_pro_logprob":
        # Same dataset/build as mmlu_pro, so sample_shuffle + limit select the
        # identical items; only the prompt and the scoring channel differ.
        dataset = hf_dataset(
            path=DATASET_PATH,
            split="test",
            sample_fields=record_to_sample,
            shuffle=False,  # see build_task("mmlu_pro"): must match exactly
            revision=MMLU_PRO_DATASET_REVISION,
        )
        return Task(
            dataset=dataset,
            solver=[letter_prompt(), generate()],
            scorer=letter_choice(),
            name="mmlu_pro_logprob",
            version="letter-logprob-v1",
        )
    if name == "gpqa_diamond":
        # cot=False on purpose: with CoT this model averages ~3.3k output
        # tokens per item and ~60% hit a 4k cap, which costs more than
        # mmlu_pro and turns truncation into noise. Without CoT gpqa_diamond
        # is a coarse "is reasoning still there at all" tripwire, which is
        # all the sanity check needs; mmlu_pro carries the finer signal.
        return gpqa_diamond(cot=False)
    if name == "ifeval":
        return ifeval()
    if name == "swe_bench":
        from inspect_evals.swe_bench import swe_bench, swe_bench_verified_mini

        if swe_dataset == "verified_mini":
            return swe_bench_verified_mini()
        return swe_bench(**SWE_DATASETS[swe_dataset])
    raise SystemExit(f"unknown task: {name}")


def short_label(model):
    """`Qwen3.8-27B-nz-wvs-modal_response-overall` -> `modal_response`.

    Used only for table headings and comparison pairs; the full model name is
    kept everywhere on disk and in config.json.
    """
    name = model.split("/", 1)[-1]
    if "-nz-wvs-" in name:
        name = name.split("-nz-wvs-", 1)[1]
    for suffix in ("-overall",):
        name = name.removesuffix(suffix)
    return name or model


def lenient_letter(completion, choices):
    """Extract a choice letter from any convention these adapters were trained on.

    The adapters were SFT'd with system prompts like "Respond with ONLY the
    exact text of your chosen option. No extra words, no explanations, no
    formatting." MMLU-Pro instead demands a trailing `ANSWER: $LETTER`. Scored
    strictly, a perfectly correct bare `D` counts as a failure, which measures
    instruction conflict rather than capability.

    Accepted, in order of preference:
      ANSWER: D  /  ANSWER: (D)     - the benchmark's own template
      D) text    /  D. text         - an echo of the rendered option label
      D  /  (D)  /  D.              - the trained "answer only" convention
      the exact option text         - also trained ("exact text of your chosen option")

    Returns None when no unique choice can be recovered.
    """
    text = (completion or "").strip()
    if not text or not choices:
        return None
    letters = LETTERS[: len(choices)]

    match = re.search(rf"ANSWER:\s*\(?([{letters}])\b", text)
    if match:
        return match.group(1)
    match = re.match(rf"^\(?([{letters}])[\)\.\:]", text)
    if match:
        return match.group(1)
    match = re.fullmatch(rf"\(?([{letters}])\)?[\.\)]?", text)
    if match:
        return match.group(1)

    lowered = text.lower()
    hits = [
        letters[i]
        for i, choice in enumerate(choices)
        if choice is not None and str(choice).strip().lower() == lowered
    ]
    return hits[0] if len(hits) == 1 else None


def sample_scores(log, bool_key=None, ignore_truncation=False, lenient=False):
    """Map sample id -> per-sample correctness and failure flags.

    bool_key handles dict-valued scorers (ifeval); MCQ scorers use C/I/N.
    ignore_truncation suppresses the flag for 1-token logprob tasks, where
    stop_reason is always max_tokens by construction.
    """
    out = {}
    for s in log.samples or []:
        score = None
        if s.scores:
            score = next(iter(s.scores.values()))
        if score is None:
            continue
        value = score.value
        if bool_key and isinstance(value, dict):
            correct = bool(value.get(bool_key))
        else:
            correct = value == CORRECT
        meta = score.metadata if isinstance(score.metadata, dict) else {}
        row = {
            "value": value,
            "reason": score.reason,
            "correct": correct,
            "truncated": (
                not ignore_truncation
                and getattr(s.output, "stop_reason", None) == "max_tokens"
            ),
            "probs": meta.get("probs"),
            "p_gold": meta.get("p_gold"),
        }
        if lenient:
            # target is a Target on a live run but a plain str when read back
            # from a log, so accept either.
            gold = str(getattr(s.target, "text", s.target) or "").strip()
            predicted = lenient_letter(s.output.completion, s.choices)
            row["lenient_correct"] = bool(gold) and predicted == gold
            row["lenient_parsed"] = predicted is not None
        out[str(s.id)] = row
    return out


def summarise(log, task_name=None):
    """Extract a small summary dict from an inspect EvalLog.

    Counts the two failure modes that a near-one-hot adapter produces, so a
    collapse reads as `invalid_format=88%` rather than being mistaken for
    knowledge loss:
      no_response      - empty completion
      invalid_format   - non-empty completion with no usable answer
      truncated        - hit the max_tokens cap
    """
    task = (task_name or (log.eval.task if log.eval else "unknown")).split("/")[-1]
    spec = TASK_SPEC.get(task, TASK_SPEC["mmlu_pro"])
    if not log.results or not log.results.scores:
        return {
            "task": task,
            "error": log.error.message if log.error else "no results",
        }

    score = log.results.scores[0]
    metrics = score.metrics or {}

    def metric(key):
        m = metrics.get(key)
        if isinstance(m, dict):
            return m.get("value")
        return getattr(m, "value", m)

    rows = sample_scores(
        log, spec["bool_key"], TASK_LOGPROBS.get(task, False), spec["lenient"]
    )
    n = len(rows) or log.results.completed_samples or score.scored_samples
    no_response = sum(r["reason"] == "no_response" for r in rows.values())
    invalid_format = sum(
        r["reason"] == "invalid_response_format" for r in rows.values()
    )
    truncated = sum(r["truncated"] for r in rows.values())
    correct = sum(r["correct"] for r in rows.values())

    acc = metric(spec["metrics"][0])
    if acc is None and n:
        acc = correct / n
    stderr = metric(spec["metrics"][1])

    summary = {
        "task": task,
        "n": n,
        "accuracy": round(acc, 4) if isinstance(acc, (int, float)) else acc,
        "stderr": round(stderr, 4) if isinstance(stderr, (int, float)) else stderr,
        "no_response": no_response,
        "invalid_format": invalid_format,
        "truncated": truncated,
    }
    if spec["lenient"] and rows:
        summary["lenient_accuracy"] = round(
            sum(r["lenient_correct"] for r in rows.values()) / len(rows), 4
        )
        summary["lenient_unparsed"] = sum(
            not r["lenient_parsed"] for r in rows.values()
        )
    return summary


def paired_delta(base_rows, other_rows, key="correct", n_boot=2000, seed=0):
    """Paired accuracy difference (other - base) with a bootstrap 95% CI.

    Pairing removes item difficulty, which is the dominant variance term for
    benchmarks like MMLU-Pro: unpaired stderr at n=1000 is ~1.5pp, the paired
    difference is much tighter. `key` selects the correctness field, so the
    same test runs on both the strict and lenient scorings.
    """
    ids = sorted(set(base_rows) & set(other_rows))
    ids = [i for i in ids if key in base_rows[i] and key in other_rows[i]]
    if not ids:
        return None
    diffs = [int(other_rows[i][key]) - int(base_rows[i][key]) for i in ids]
    delta = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        boots.append(sum(rng.choice(diffs) for _ in diffs) / len(diffs))
    boots.sort()
    lo = boots[int(0.025 * (n_boot - 1))]
    hi = boots[int(0.975 * (n_boot - 1))]
    # Discordant pairs for McNemar.
    b = sum(1 for i in ids if base_rows[i][key] and not other_rows[i][key])
    c = sum(1 for i in ids if not base_rows[i][key] and other_rows[i][key])
    return {
        "n_paired": len(ids),
        "delta_pp": round(100 * delta, 2),
        "ci_lo_pp": round(100 * lo, 2),
        "ci_hi_pp": round(100 * hi, 2),
        "b": b,
        "c": c,
        "mcnemar_p": round(mcnemar_exact(b, c), 4),
    }


def mcnemar_exact(b, c):
    """Two-sided exact McNemar test (binomial on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    if n > 2000:  # normal approximation; exact sum over huge ints is pointless
        z = (b - c) / math.sqrt(n)
        return math.erfc(abs(z) / math.sqrt(2))
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def distribution_shift(base_rows, other_rows):
    """How far the answer *distribution* moved, independent of the argmax.

    Mean TVD and KL(base || other) over the normalised A-J letter
    distributions, plus top-1 agreement. Accuracy can be unchanged while TVD
    is large, which is the case this test exists to detect.
    """
    ids = sorted(set(base_rows) & set(other_rows))
    ids = [i for i in ids if base_rows[i].get("probs") and other_rows[i].get("probs")]
    if not ids:
        return None

    tvd = kl = agreement = 0.0
    for i in ids:
        p = base_rows[i]["probs"]
        q = other_rows[i]["probs"]
        for letter in set(p) | set(q):
            pv = p.get(letter, 0.0)
            qv = q.get(letter, 0.0)
            tvd += 0.5 * abs(pv - qv)
            kl += pv * math.log((pv + 1e-12) / (qv + 1e-12))
        agreement += max(p, key=p.get) == max(q, key=q.get)
    n = len(ids)
    return {
        "n": n,
        "tvd": round(tvd / n, 4),
        "kl": round(kl / n, 4),
        "top1_agreement": round(agreement / n, 4),
    }


def mean_p_gold(rows):
    """Mean probability the model put on the correct letter (calibration proxy)."""
    vals = [r["p_gold"] for r in rows.values() if r.get("p_gold") is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def write_comparison(results, out_dir):
    """Write comparison.md/.csv from {task: {model: summary}} plus sample rows.

    results: {task: {"summaries": {model: dict}, "rows": {model: {id: row}}}}
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "comparison.csv"
    md_path = out_dir / "comparison.md"

    models = list(next(iter(results.values()))["summaries"])
    base = models[0]

    fields = [
        "task",
        "model",
        "label",
        "n",
        "accuracy",
        "stderr",
        "invalid_format_pct",
        "no_response",
        "truncated",
        "delta_pp_vs_base",
        "ci_lo_pp",
        "ci_hi_pp",
        "mcnemar_p",
        "b",
        "c",
        "lenient_accuracy",
        "lenient_unparsed",
        "delta_pp_lenient",
        "ci_lo_pp_lenient",
        "ci_hi_pp_lenient",
        "mcnemar_p_lenient",
        "mean_p_gold",
        "mean_tvd_vs_base",
        "mean_kl_vs_base",
        "top1_agreement",
    ]
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for task, block in results.items():
            base_rows = block["rows"].get(base, {})
            for model in models:
                s = block["summaries"][model]
                row = {
                    "task": task,
                    "model": model,
                    "label": short_label(model),
                    "n": s.get("n"),
                    "accuracy": s.get("accuracy"),
                    "stderr": s.get("stderr"),
                    "invalid_format_pct": round(
                        100 * s.get("invalid_format", 0) / max(1, s.get("n") or 1), 2
                    ),
                    "no_response": s.get("no_response"),
                    "truncated": s.get("truncated"),
                    "lenient_accuracy": s.get("lenient_accuracy"),
                    "lenient_unparsed": s.get("lenient_unparsed"),
                    "mean_p_gold": mean_p_gold(block["rows"].get(model, {})),
                }
                if model != base:
                    d = paired_delta(base_rows, block["rows"].get(model, {}))
                    if d:
                        row.update(
                            delta_pp_vs_base=d["delta_pp"],
                            ci_lo_pp=d["ci_lo_pp"],
                            ci_hi_pp=d["ci_hi_pp"],
                            mcnemar_p=d["mcnemar_p"],
                            b=d["b"],
                            c=d["c"],
                        )
                    dl = paired_delta(
                        base_rows, block["rows"].get(model, {}), key="lenient_correct"
                    )
                    if dl:
                        row.update(
                            delta_pp_lenient=dl["delta_pp"],
                            ci_lo_pp_lenient=dl["ci_lo_pp"],
                            ci_hi_pp_lenient=dl["ci_hi_pp"],
                            mcnemar_p_lenient=dl["mcnemar_p"],
                        )
                    shift = distribution_shift(base_rows, block["rows"].get(model, {}))
                    if shift:
                        row.update(
                            mean_tvd_vs_base=shift["tvd"],
                            mean_kl_vs_base=shift["kl"],
                            top1_agreement=shift["top1_agreement"],
                        )
                w.writerow(row)

    lines = [
        "# Capability sanity check",
        "",
        f"Base model: `{base}`. Deltas are **paired** on identical items.",
        "",
        (
            "`invalid_format` counts responses the benchmark's scorer could not "
            "map to a choice. For these adapters that is almost always a *format* "
            "mismatch rather than a wrong answer: they were fine-tuned to emit "
            "only the option text or the bare letter. Use `lenient` for "
            "capability and `strict` for template compliance."
        ),
        "",
    ]

    # Headline matrix: one column per task, using the lenient value where one
    # exists (it is the capability number).
    def headline(summary):
        value = summary.get("lenient_accuracy", summary.get("accuracy"))
        return f"{100 * value:.1f}%" if isinstance(value, (int, float)) else "n/a"

    lines.append("## Headline")
    lines.append("")
    lines.append("| model | " + " | ".join(results) + " |")
    lines.append("|---" * (len(results) + 1) + "|")
    for model in models:
        marker = " (base)" if model == base and short_label(model) != "base" else ""
        cells = [headline(results[t]["summaries"][model]) for t in results]
        name = (
            f"**{short_label(model)}**{marker}" if model == base else short_label(model)
        )
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "`mmlu_pro_logprob` is the direct-answer (no chain-of-thought) channel; "
        "`mmlu_pro` is scored leniently here, so its column mixes CoT with "
        "format tolerance. Read the two MMLU columns against each other: base "
        "gains ~20pp from chain-of-thought, the adapters do not reason at all."
    )
    lines.append("")
    for task, block in results.items():
        lines.append(f"## {task}")
        lines.append("")
        has_lenient = any("lenient_accuracy" in s for s in block["summaries"].values())
        if has_lenient:
            lines.append(
                "`strict` is the benchmark's own template (`ANSWER: $LETTER`); "
                "`lenient` also accepts a bare letter, `X) text`, or the exact "
                "option text — the conventions these adapters were fine-tuned "
                "to emit. **Use lenient as the capability number**; strict "
                "measures template compliance."
            )
            lines.append("")
            lines.append(
                "Reference point: base's lenient score here is its "
                "**chain-of-thought** score. Its score in the same "
                "direct-answer mode as the adapters is the "
                "`mmlu_pro_logprob` column above. Read against *that*, the "
                "adapters are close to base; read against base's CoT score "
                "they are ~18-22pp down, because they answer immediately "
                "instead of reasoning."
            )
            lines.append("")
            lines.append(
                "| model | n | strict | lenient | Δ lenient vs base | 95% CI | McNemar p | invalid fmt |"
            )
            lines.append("|---|---|---|---|---|---|---|---|")
        else:
            lines.append(
                "| model | n | accuracy | Δ vs base | 95% CI | McNemar p | invalid fmt |"
            )
            lines.append("|---|---|---|---|---|---|---|")
        base_rows = block["rows"].get(base, {})
        for model in models:
            s = block["summaries"][model]
            n = s.get("n") or 0
            acc = s.get("accuracy")
            acc_s = f"{100 * acc:.1f}%" if isinstance(acc, (int, float)) else "n/a"
            inv = 100 * s.get("invalid_format", 0) / max(1, n)
            marker = " (base)" if model == base and short_label(model) != "base" else ""
            len_acc = s.get("lenient_accuracy")
            len_s = (
                f"{100 * len_acc:.1f}%" if isinstance(len_acc, (int, float)) else "n/a"
            )

            if model == base:
                if has_lenient:
                    lines.append(
                        f"| **{short_label(model)}**{marker} | {n} | {acc_s} | {len_s} | — | — | — | {inv:.1f}% |"
                    )
                else:
                    lines.append(
                        f"| **{short_label(model)}**{marker} | {n} | {acc_s} | — | — | — | {inv:.1f}% |"
                    )
                continue

            if has_lenient:
                dl = paired_delta(
                    base_rows, block["rows"].get(model, {}), key="lenient_correct"
                )
                if dl:
                    lines.append(
                        f"| {short_label(model)} | {n} | {acc_s} | {len_s} | "
                        f"{dl['delta_pp']:+.1f}pp | "
                        f"[{dl['ci_lo_pp']:+.1f}, {dl['ci_hi_pp']:+.1f}] | "
                        f"{dl['mcnemar_p']:.4f} | {inv:.1f}% |"
                    )
                else:
                    lines.append(
                        f"| {short_label(model)} | {n} | {acc_s} | {len_s} | n/a | n/a | n/a | {inv:.1f}% |"
                    )
                continue

            d = paired_delta(base_rows, block["rows"].get(model, {}))
            if d:
                lines.append(
                    f"| {short_label(model)} | {n} | {acc_s} | "
                    f"{d['delta_pp']:+.1f}pp | "
                    f"[{d['ci_lo_pp']:+.1f}, {d['ci_hi_pp']:+.1f}] | "
                    f"{d['mcnemar_p']:.4f} | {inv:.1f}% |"
                )
            else:
                lines.append(
                    f"| {short_label(model)} | {n} | {acc_s} | n/a | n/a | n/a | {inv:.1f}% |"
                )
        lines.append("")

        has_probs = any(
            any(r.get("probs") for r in block["rows"].get(m, {}).values())
            for m in models
        )
        if has_probs:
            lines.append(
                "**Answer-distribution shift vs base** — same items, so an "
                "unchanged accuracy with a large TVD means the adapter moved "
                "the distribution without changing the argmax."
            )
            lines.append("")
            lines.append(
                "| model | mean P(gold) | TVD vs base | KL(base‖model) | top-1 agree |"
            )
            lines.append("|---|---|---|---|---|")
            for model in models:
                pg = mean_p_gold(block["rows"].get(model, {}))
                pg_s = f"{pg:.3f}" if pg is not None else "n/a"
                if model == base:
                    lines.append(f"| **{short_label(model)}** | {pg_s} | — | — | — |")
                    continue
                shift = distribution_shift(base_rows, block["rows"].get(model, {}))
                if shift:
                    lines.append(
                        f"| {short_label(model)} | {pg_s} | {shift['tvd']:.3f} | "
                        f"{shift['kl']:.3f} | {100 * shift['top1_agreement']:.1f}% |"
                    )
                else:
                    lines.append(f"| {short_label(model)} | {pg_s} | n/a | n/a | n/a |")
            lines.append("")

    md_path.write_text("\n".join(lines))
    return md_path, csv_path


def write_run_artefacts(out, output_dir):
    """Write summary.json + summary_samples.csv for one model's run dir.

    Shared by the live path and --compare-only so a re-analysis refreshes the
    per-run summaries instead of leaving them stale.
    """
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({t: v[0] for t, v in out.items()}, indent=2))
    rows_path = output_dir / "summary_samples.csv"
    with rows_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "task",
                "sample_id",
                "correct",
                "lenient_correct",
                "value",
                "reason",
                "truncated",
            ]
        )
        for task, (_, rows, _) in out.items():
            for sid, r in rows.items():
                w.writerow(
                    [
                        task,
                        sid,
                        int(r["correct"]),
                        "" if "lenient_correct" not in r else int(r["lenient_correct"]),
                        r["value"],
                        r["reason"],
                        int(r["truncated"]),
                    ]
                )
    print(f"[save] {output_dir}")


def latest_run_dir(model_name, tasks=None):
    """Newest run dir for a model, preferring one that covers ``tasks``.

    A newer partial run (a smoke test, or a rerun that died early) must not
    shadow the completed grid run, so runs whose summary.json does not cover
    every requested task are skipped when a complete one exists.
    """
    model_dir = Path("output") / "capability" / model_name.split("/", 1)[-1]
    if not model_dir.is_dir():
        raise SystemExit(f"no previous runs under {model_dir}")
    runs = sorted(
        (d for d in model_dir.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        raise SystemExit(f"no previous runs under {model_dir}")
    if tasks:
        for run in runs:
            summary_path = run / "summary.json"
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text())
            if all(
                (summary.get(task) or {}).get("accuracy") is not None for task in tasks
            ):
                return run
    return runs[0]


def find_task_log(run_dir, task):
    """Most recent inspect log for a task inside a run dir (best match by name)."""
    logs_dir = run_dir / "inspect-logs"
    if not logs_dir.is_dir():
        print(f"  [warn] no inspect-logs dir in {run_dir}")
        return None
    tag = task.replace("_", "-")
    # Anchor the tag so mmlu-pro does not also match mmlu-pro-logprob logs.
    pattern = re.compile(rf"(^|_){re.escape(tag)}(_|$)")
    matches = [
        p
        for p in logs_dir.iterdir()
        if p.suffix in (".json", ".eval") and pattern.search(p.name)
    ]
    if not matches:
        print(f"  [warn] no log for task {task} in {logs_dir}")
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def load_run(model_name, run_dir, tasks):
    """Rebuild {task: (summary, rows, log)} from a completed run directory.

    Used by --compare-only so the comparison table can be regenerated from
    the artefacts without touching the GPU. The per-run summary.json is
    rewritten too, so downstream readers (analyze.py) see the same numbers —
    including fields added after the run itself (lenient scoring).
    """
    out = {}
    for task in tasks:
        log_path = find_task_log(run_dir, task)
        if log_path is None:
            continue
        log = read_eval_log(log_path)
        out[task] = (
            summarise(log, task),
            sample_scores(
                log,
                TASK_SPEC.get(task, {}).get("bool_key"),
                TASK_LOGPROBS.get(task, False),
                TASK_SPEC.get(task, {}).get("lenient", False),
            ),
            log,
        )
    if out:
        write_run_artefacts(out, run_dir)
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Capability sanity check (mmlu_pro, gpqa_diamond, ifeval) "
        "against a served model. Compare the base model with its adapters using "
        "the same deterministic item subsets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--api-url", default=None, help="vLLM server URL")
    p.add_argument("--port", type=int, default=8087, help="vLLM server port")
    p.add_argument("--api-key", default="EMPTY", help="API key for the vLLM server")
    p.add_argument("--model", default=None, help="Single model name on the server")
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Run this whole grid back to back and write a paired comparison. "
        "The first entry is treated as the base model.",
    )
    p.add_argument("--tasks", nargs="+", default=TASKS, help="Tasks to run")
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per task (default: per-task default, "
        f"{', '.join(f'{k}={v}' for k, v in TASK_DEFAULTS.items())})",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=128,
        help="Samples in flight (inspect's max_samples). vLLM serves "
        "--max-num-seqs 256; 128 measured ~4x faster than 16.",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override the per-task output cap "
        f"({', '.join(f'{k}={v}' for k, v in TASK_MAX_TOKENS.items())}).",
    )
    p.add_argument(
        "--thinking",
        action="store_true",
        help="Enable chain-of-thought (<think>...</think>). Off by default, "
        "matching the fine-tuning evals; leaving it on costs ~10x.",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Seed for the deterministic subset"
    )
    p.add_argument(
        "--include-swe",
        action="store_true",
        help="Also run SWE-bench (docker sandbox, agentic, very slow; not part "
        "of the sanity check).",
    )
    p.add_argument(
        "--swe-dataset",
        default="verified_mini",
        choices=list(SWE_DATASETS),
        help="SWE-bench dataset (only with --include-swe)",
    )
    p.add_argument(
        "--display",
        default="plain",
        choices=["full", "conversation", "rich", "plain", "log", "none"],
        help="inspect_ai display",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="inspect_ai console log level",
    )
    p.add_argument(
        "--log-realtime",
        action="store_true",
        help="Stream eval logs to the console in realtime",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Results dir (default: output/capability/<model>/<run>)",
    )
    p.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        metavar="RUN_DIR",
        help="Resume a single-model run (not usable with --models).",
    )
    p.add_argument(
        "--compare-only",
        action="store_true",
        help="Run nothing: rebuild the paired comparison from each model's "
        "most recent run directory (needs >= 2 --models).",
    )
    args = p.parse_args(argv)

    if args.models and args.model:
        raise SystemExit("pass either --model or --models, not both")
    if not args.models and not args.model:
        raise SystemExit("pass --model or --models")
    if args.resume and args.models and len(args.models) > 1:
        raise SystemExit("--resume only works for a single model")
    if args.output_dir and args.models and len(args.models) > 1:
        raise SystemExit("--output-dir only works for a single model")
    if args.include_swe:
        args.tasks = list(args.tasks) + [
            t for t in OPTIONAL_TASKS if t not in args.tasks
        ]
    return args


def evaluate_model(model_name, args):
    """Run the task list for one model. Returns {task: (summary, rows, log)}."""
    if args.api_url is None:
        args.api_url = f"http://localhost:{args.port}"
    model = f"openai/{model_name}"

    # responses_api=False: inspect_ai treats any unrecognized model name as an
    # OpenAI "latest" model, which flips on the Responses API (incl. token
    # counting via /responses/input_tokens). vLLM doesn't implement that
    # endpoint, so force the chat-completions path for self-hosted servers.
    model_args = {"api_key": args.api_key, "responses_api": False}
    base_generate_args = {
        "temperature": 0.0,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": bool(args.thinking)}
        },
        "log_samples": True,
        # Explicit max_connections disables inspect's adaptive "slow start"
        # (which begins at 20 in flight and ramps up, wasting minutes).
        "max_connections": args.concurrency,
    }

    resume_dir = None
    if args.resume:
        resume_dir = (
            Path(args.resume) if args.resume != "latest" else latest_run_dir(model_name)
        )
        print(f"[resume] continuing run: {resume_dir}")
        output_dir = resume_dir
    elif args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        run_name = f"capability-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = (
            Path("output") / "capability" / model_name.split("/", 1)[-1] / run_name
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "target": model_name,
        "label": short_label(model_name),
        "tasks": list(args.tasks),
        "max_samples": args.max_samples,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "max_tokens_per_task": {
            t: (args.max_tokens or TASK_MAX_TOKENS.get(t, 2048)) for t in args.tasks
        },
        "thinking": bool(args.thinking),
        "temperature": 0.0,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
        "api_url": args.api_url,
        "note": "Fixed deterministic subsets; base and every adapter see the "
        "same items, so compare paired deltas rather than raw accuracies.",
    }
    config_path = output_dir / "config.json"
    if resume_dir is not None and config_path.exists():
        prev = json.loads(config_path.read_text())
        prev["resume"] = {"at": datetime.now().isoformat(), "tasks": list(args.tasks)}
        config = prev
    config_path.write_text(json.dumps(config, indent=2))

    out = {}
    for task in args.tasks:
        print(f"\n=== {model_name} :: {task} ===")
        limit = args.max_samples or TASK_DEFAULTS.get(task, 100)
        task_max_tokens = args.max_tokens or TASK_MAX_TOKENS.get(task, 2048)
        generate_args = dict(base_generate_args, max_tokens=task_max_tokens)
        if TASK_LOGPROBS.get(task):
            # top_logprobs=20 covers A-J plus the model's likely filler
            # tokens; vLLM caps it at 20.
            generate_args["logprobs"] = True
            generate_args["top_logprobs"] = 20
        print(f"  limit={limit}, max_tokens={task_max_tokens}")
        log = None
        try:
            if resume_dir is not None:
                log_path = find_task_log(resume_dir, task)
                if log_path is not None:
                    prev = read_eval_log(log_path)
                    if prev.status == "success":
                        print(f"  {task}: already successful, nothing to retry")
                        log = prev
            if log is None:
                logs = inspect_eval(
                    tasks=[build_task(task, args.swe_dataset)],
                    model=model,
                    model_base_url=f"{args.api_url}/v1",
                    model_args=model_args,
                    limit=limit,
                    max_samples=args.concurrency,
                    sample_shuffle=args.seed,
                    epochs=1,  # overrides gpqa_diamond's 4-epoch default
                    retry_on_error=3,
                    max_retries=2,
                    display=args.display,
                    log_level=args.log_level,
                    log_realtime=args.log_realtime,
                    log_dir=str(output_dir / "inspect-logs"),
                    log_format="json",
                    **generate_args,
                )
                log = logs[0] if logs else None
                if log is None:
                    raise RuntimeError("no log returned")
        except Exception as e:  # noqa: BLE001 - one bad task must not kill the grid
            print(f"[error] {task} failed: {e}")
            out[task] = (
                {"task": task, "error": str(e)},
                {},
                None,
            )
            continue

        summary = summarise(log, task)
        rows = sample_scores(
            log,
            TASK_SPEC.get(task, {}).get("bool_key"),
            TASK_LOGPROBS.get(task, False),
            TASK_SPEC.get(task, {}).get("lenient", False),
        )
        out[task] = (summary, rows, log)
        if "error" not in summary:
            acc = summary["accuracy"]
            acc_s = f"{acc * 100:.1f}%" if acc is not None else "n/a"
            len_s = (
                f", lenient={100 * summary['lenient_accuracy']:.1f}%"
                if "lenient_accuracy" in summary
                else ""
            )
            print(
                f"  {task}: n={summary['n']}, accuracy={acc_s}{len_s}, "
                f"invalid_format={summary['invalid_format']}, "
                f"truncated={summary['truncated']}"
            )
        else:
            print(f"  {task}: ERROR {summary['error']}")

    # Per-model artefacts.
    write_run_artefacts(out, output_dir)
    return out


def main():
    args = parse_args()
    models = args.models or [args.model]

    print("=" * 60)
    print("Capability sanity check")
    print("=" * 60)
    for k in ("tasks", "max_samples", "concurrency", "max_tokens", "thinking", "seed"):
        print(f"  {k}: {getattr(args, k)}")
    print(f"  models: {len(models)}")
    for i, m in enumerate(models):
        role = "base" if i == 0 and len(models) > 1 else ""
        print(f"    - {m} {role}")
    print("=" * 60)

    results = {}
    if args.compare_only:
        if len(models) < 2:
            raise SystemExit("--compare-only needs at least two --models")
        for model_name in models:
            run_dir = latest_run_dir(model_name, args.tasks)
            print(f"[load] {short_label(model_name):<28} <- {run_dir}")
            out = load_run(model_name, run_dir, args.tasks)
            for task, (summary, rows, _) in out.items():
                expected = args.max_samples or TASK_DEFAULTS.get(task)
                if (
                    isinstance(expected, int)
                    and isinstance(summary.get("n"), int)
                    and summary["n"] < expected
                ):
                    print(
                        f"  [warn] {task}: loaded n={summary['n']} < expected "
                        f"{expected} — latest run was a smaller subset?"
                    )
                block = results.setdefault(task, {"summaries": {}, "rows": {}})
                block["summaries"][model_name] = summary
                block["rows"][model_name] = rows
    else:
        for model_name in models:
            out = evaluate_model(model_name, args)
            for task, (summary, rows, _) in out.items():
                block = results.setdefault(task, {"summaries": {}, "rows": {}})
                block["summaries"][model_name] = summary
                block["rows"][model_name] = rows

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for task, block in results.items():
        print(f"\n{task}")
        for model, s in block["summaries"].items():
            if "error" in s:
                print(f"  {short_label(model):<28} ERROR: {s['error']}")
                continue
            acc = s["accuracy"]
            acc_s = f"{100 * acc:.1f}%" if acc is not None else "n/a"
            len_s = (
                f"lenient={100 * s['lenient_accuracy']:.1f}%"
                if "lenient_accuracy" in s
                else ""
            )
            print(
                f"  {short_label(model):<28} n={s['n']:<5} acc={acc_s:<7} "
                f"{len_s:<18} invalid_fmt={s['invalid_format']}"
            )

    if len(models) > 1:
        if not results:
            print("[warn] no task produced results; skipping comparison")
            return
        out_dir = (
            Path("output")
            / "capability"
            / ("comparison-" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        )
        md, csv_path = write_comparison(results, out_dir)
        print(f"\n[paired] {md}")
        print(f"[paired] {csv_path}")


if __name__ == "__main__":
    main()
