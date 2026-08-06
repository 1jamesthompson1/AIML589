#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "torch>=2.6.0",
#   "transformers>=4.50.0",
#   "trl>=0.16.0",
#   "datasets>=5.0.0",
#   "accelerate>=1.6.0",
#   "python-dotenv>=1.1.0",
#   "peft>=0.15.0",
#   "bitsandbytes>=0.46.0",
#   "huggingface-hub>=0.30.0",
#   "jinja2>=3.1.0",
#   # torchvision + pillow: newer transformers auto-loads image processors
#   # for multimodal models (e.g. Qwen3.5-VL) during SFTTrainer init
#   "torchvision>=0.20.0",
#   "pillow>=11.0.0",
# ]
# ///
"""Fine-tune an LLM on the WVS-NZ value alignment dataset using LoRA.

Uploads the adapter to its own HF Hub repo at
``{HF_ORG}/{model_slug}-nz-wvs-{dataset}-{subpopulation}``
for use with the LoRA-aware vLLM server (see ``serve.py``).

Examples:
    uv run finetune.py --model Qwen/Qwen3.6-27B \\
        --dataset full_string_distribution --subpopulation overall \\
        --upload-to-hf --hf-collection "1jamesthompson1/wvs-nz-lora-adapters"

    # SCP to a GPU machine and run:
    scp finetune.py user@host:~
    ssh user@host ./finetune.py --model Qwen/Qwen3.6-27B \\
        --dataset full_string_distribution --subpopulation overall \\
        --hf-token "$HF_TOKEN" --upload-to-hf

    uv run code/fine-tuning/finetune.py --help
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Use expandable CUDA segments by default to reduce memory fragmentation
# (large tensors like logits/logsumexp allocate in contiguous blocks).
# setdefault so an explicitly-set environment variable still wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Silence HF Hub's "HTTP Request: HEAD ..." INFO logs (emitted while probing
# processor/preprocessor configs, e.g. for vision-config models). This must
# be set before huggingface_hub is imported — it is read at import time.
os.environ.setdefault("HF_HUB_VERBOSITY", "error")


def _apply_gpu_env():
    """Set ``CUDA_VISIBLE_DEVICES`` before transformers is imported.

    The transformers import chain initialises CUDA, and ``CUDA_VISIBLE_DEVICES``
    is only honoured at CUDA init time. So ``--gpu`` is pre-parsed from
    argv here, before the eager import below. Keep the default (``"0"``) in
    sync with the argparse default in parse_args.
    """
    val = "0"
    argv = sys.argv[1:]
    for a in argv:
        if a == "--gpu":
            i = argv.index(a)
            if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                val = argv[i + 1]
        elif a.startswith("--gpu="):
            val = a.split("=", 1)[1]
    if val != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = val


_apply_gpu_env()

# Eager import (unlike the rest of the heavy imports, which are lazy): the
# DistributionalCollator/Trainer classes are defined at module level and must
# be picklable, because DataLoader worker processes serialize them with pickle
# when dataloader_num_workers > 0.
from transformers import Trainer, TrainingArguments  # noqa: E402

log = logging.getLogger(__name__)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Fine-tune an LLM on the WVS-NZ value alignment dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--model",
        default="Qwen/Qwen3.6-27B",
        help="Base model name on Hugging Face Hub (use non-quantized model for fine-tuning)",
    )
    p.add_argument(
        "--gpu",
        default="0",
        help="Comma-separated CUDA device(s) to use, e.g. '0' for a single "
        "GPU or '0,1' to spread across two. 'auto' leaves device_map='auto' "
        "to decide. Defaults to a single GPU: device_map='auto' otherwise "
        "splits the model across every visible GPU, which gives no speedup "
        "for LoRA training and blocks the other GPU(s).",
    )
    p.add_argument(
        "--dataset",
        default="modal_response",
        choices=[
            "modal_response",
            "sampled_response",
            "full_string_distribution",
            "first_token_distribution",
        ],
        help="Dataset config",
    )
    p.add_argument(
        "--subpopulation",
        default="overall",
        choices=["cluster_0", "cluster_1", "overall"],
        help="Subpopulation to train on",
    )
    p.add_argument(
        "--output-dir",
        default="./output",
        help="Local output directory for checkpoints & logs",
    )

    p.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (higher = more capacity, more memory)",
    )
    p.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (scaling, usually 2× rank)",
    )
    p.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    p.add_argument(
        "--dora",
        action="store_true",
        default=False,
        help="Use DoRA (weight-decomposed LoRA) for better capacity at same rank",
    )

    p.add_argument(
        "--quantization",
        default=None,
        choices=[None, "4bit"],
        help="Quantization method",
    )

    p.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate")
    p.add_argument("--num-epochs", type=int, default=3, help="Number of epochs")
    p.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Per-device eval batch size (default: same as --batch-size). "
        "Note: with distributional training each example expands into K "
        "completions, so sequences per step = batch_size * K.",
    )
    p.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Maximum sequence length for tokenization",
    )
    p.add_argument(
        "--warmup-ratio", type=float, default=0.1, help="Warmup ratio for scheduler"
    )
    p.add_argument("--logging-steps", type=int, default=10, help="Log every N steps")
    p.add_argument(
        "--save-steps", type=int, default=200, help="Save checkpoint every N steps"
    )
    p.add_argument("--eval-steps", type=int, default=10, help="Evaluate every N steps")

    p.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16"],
        help="Compute dtype for training (bf16 recommended for modern GPUs)",
    )

    p.add_argument(
        "--hf-token", default=None, help="Hugging Face token (or HF_TOKEN env var)"
    )
    p.add_argument(
        "--hf-repo",
        default=None,
        help="HF repo to upload to (default: "
        "{HF_ORG}/{model_slug}-nz-wvs-{dataset}-{subpopulation}). "
        "Set HF_ORG env var for the org/username prefix.",
    )
    p.add_argument(
        "--hf-collection",
        default=None,
        help="HF collection slug to add the uploaded repo to "
        "(e.g. '1jamesthompson1/wvs-nz-lora-adapters', "
        "or HF_COLLECTION env var)",
    )
    p.add_argument(
        "--upload-to-hf",
        action="store_true",
        default=False,
        help="Upload the fine-tuned model to Hugging Face Hub",
    )

    return p.parse_args(argv)


def load_dataset(args, split="train"):
    from datasets import load_dataset

    ds = load_dataset(
        "1jamesthompson1/wvs-nz-value-alignment",
        args.dataset,
        split=split,
        token=args.hf_token,
    )
    ds = ds.filter(lambda x: x["subpopulation"] == args.subpopulation)
    log.info(
        "[data] loaded %d %s examples (config=%s, subpop=%s)",
        len(ds),
        split,
        args.dataset,
        args.subpopulation,
    )
    return ds


def format_chat(system_prompt: str, user_prompt: str, response: str | None = None):
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if response is not None:
        msgs.append({"role": "assistant", "content": response})
    return msgs


def load_tokenizer(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        token=args.hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(args):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    model_kwargs = dict(
        device_map="auto",
        trust_remote_code=True,
        token=args.hf_token,
    )
    if args.quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16" if args.dtype == "bf16" else "float16",
            bnb_4bit_use_double_quant=True,
        )

    return AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)


def make_peft_config(args):
    from peft import LoraConfig

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_dora=args.dora,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def save_model(output_dir, model, tokenizer):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    log.info("    saved to %s", output_dir)


def _get_gpu_info() -> str:
    import torch

    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


def setup_logging(output_dir: Path):
    log_path = output_dir / "training.log"
    handler = logging.FileHandler(str(log_path))
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(handler)
    log.info("logging to %s", log_path)
    return log_path


# ----------------------------------------------------
# Supervised fine-tuning (SFT) with one completion per example
# ----------------------------------------------------


def run_sft(args, ds, model_path):
    """Run supervised fine-tuning (SFT) on a dataset with one completion per example.

    Depending on dataset config this is either the modal response (most common
    answer) or a sampled response from the empirical distribution. The
    SFTTrainer masks the prompt tokens in the loss, so only the completion is
    scored. The validation split is tokenized identically and evaluated every
    ``--eval-steps`` steps (completion NLL), matching the distributional paths.

    Args:
        args: Parsed command-line arguments.
        ds: Training dataset.
        model_path: Output directory for the fine-tuned model.

    Returns:
        Tuple of ``(model_path, log_history, gpu_name, train_time_s)``.
    """

    import time
    from functools import partial

    from trl import SFTConfig, SFTTrainer

    log.info("[sft] loading tokenizer...")
    tokenizer = load_tokenizer(args)

    log.info("[sft] loading model...")
    model = load_base_model(args)

    peft_config = make_peft_config(args)

    log.info("[sft] tokenizing %d examples...", len(ds))
    tok_ds = ds.map(
        partial(
            tokenize_sft,
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
        ),
        batched=False,
        remove_columns=ds.column_names,
    )

    log.info("[sft] tokenizing validation split for eval...")
    tok_eval_ds = load_dataset(args, split="validation").map(
        partial(
            tokenize_sft,
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
        ),
        batched=False,
        remove_columns=ds.column_names,
    )
    log.info("[sft] eval split: %d examples", len(tok_eval_ds))

    training_args = SFTConfig(
        output_dir=str(model_path),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size or args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_total_limit=2,
        fp16=(args.dtype == "fp16"),
        bf16=(args.dtype == "bf16"),
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        max_length=args.max_seq_length,
        loss_type="nll",
        # We pre-tokenize with completion-only labels ourselves (see
        # tokenize_sft) because TRL >= 1.9 refuses assistant-only loss for
        # vision-language models and crashes on dict-returning
        # formatting_funcs.
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tok_ds,
        eval_dataset=tok_eval_ds,
        peft_config=peft_config,
        data_collator=SftCollator(tokenizer, args.max_seq_length),
    )

    log.info("[sft] starting training...")
    gpu_name = _get_gpu_info()
    t0 = time.time()
    trainer.train()
    train_time_s = time.time() - t0

    log.info("[sft] saving...")
    trainer.save_model(str(model_path))
    tokenizer.save_pretrained(str(model_path))
    log.info(f"    saved to {model_path}")
    log.info(f"    gpu: {gpu_name}  time: {train_time_s:.0f}s")

    # Capture training logs
    log_history = getattr(trainer.state, "log_history", [])

    save_training_log(model_path, log_history)
    return model_path, log_history, gpu_name, train_time_s


# ----------------------------------------------------
# Shared tokenization helpers + single response SFT
# ----------------------------------------------------


def _prompt_ids(example, tokenizer, max_length):
    """Tokenize prompt (system + user) via the chat template.

    Uses the chat template with a generation prompt (ends in
    ``<|im_start|>assistant\\n``), truncated from the left to fit
    ``max_length`` so the answer position stays at the end of the sequence.

    Args:
        example: Dataset row with ``system_prompt`` and ``user_prompt`` keys.
        tokenizer: Tokeniser instance.
        max_length: Maximum sequence length.

    Returns:
        List of token ids for the prompt.
    """
    prompt_text = tokenizer.apply_chat_template(
        format_chat(example["system_prompt"], example["user_prompt"]),
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tokenizer(prompt_text)["input_ids"]
    return ids[-max_length:] if len(ids) > max_length else ids


def _completion_ids(text, tokenizer):
    """Tokenise one option/completion string plus the end-of-turn token.

    Args:
        text: Completion text (without EOT).
        tokenizer: Tokeniser instance.

    Returns:
        List of completion token ids including ``<|im_end|>``.
    """
    return tokenizer(text + "<|im_end|>", add_special_tokens=False)["input_ids"]


def _with_completion(prompt_ids, comp_ids, max_length):
    """Concatenate prompt + completion with prompt-left truncation.

    Truncates the prompt from the left (never the completion) when the
    combined sequence exceeds ``max_length``. Labels mask prompt tokens to
    ``-100`` so only the completion is scored.

    Args:
        prompt_ids: List of token ids for the prompt.
        comp_ids: List of token ids for the completion.
        max_length: Maximum total sequence length.

    Returns:
        Tuple of ``(sequence_ids, label_ids)``.
    """
    if len(prompt_ids) + len(comp_ids) > max_length:
        keep = max_length - len(comp_ids)
        prompt_ids = prompt_ids[-keep:] if keep > 0 else []
    seq = (prompt_ids + comp_ids)[:max_length]
    lab = ([-100] * len(prompt_ids) + comp_ids)[:max_length]
    return seq, lab


def tokenize_sft(example, tokenizer, max_length):
    """Tokenize one SFT example: shared prompt + single completion.

    Prompt tokens are masked to ``-100`` so only the completion is scored —
    the same completion-only objective as the distributional path.

    Used instead of TRL's own dataset preparation: TRL >= 1.9 refuses
    assistant-only loss for models detected as vision-language (Qwen3.5-9B
    and Qwen3.6-27B both carry ``vision_config``), and its formatting_func
    path wraps output under ``"text"`` and crashes. We pre-tokenize with
    ``skip_prepare_dataset=True`` instead.

    Args:
        example: Dataset row with ``expected_text`` key.
        tokenizer: Tokeniser instance.
        max_length: Maximum sequence length.

    Returns:
        Dict with ``input_ids`` and ``labels``.
    """
    prompt_ids = _prompt_ids(example, tokenizer, max_length)
    comp_ids = _completion_ids(example.get("expected_text", ""), tokenizer)
    seq, lab = _with_completion(prompt_ids, comp_ids, max_length)
    return {"input_ids": seq, "labels": lab}


def _pad_sequence(seq, max_len, pad_value):
    """Truncate or pad a sequence to ``max_len``.

    Args:
        seq: List of token ids.
        max_len: Target length.
        pad_value: Value used for padding.

    Returns:
        Tuple of ``(padded_seq, attention_mask)``.
    """
    seq = seq[:max_len]
    n_pad = max_len - len(seq)
    return seq + [pad_value] * n_pad, [1] * len(seq) + [0] * n_pad


class SftCollator:
    """Pad tokenized SFT examples to a batch (labels padded with ``-100``).

    Args:
        tokenizer: Tokeniser instance (used for pad token id).
        max_length: Maximum sequence length.
    """

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        import torch

        max_len = min(self.max_length, max(len(ex["input_ids"]) for ex in examples))
        pad_id = self.tokenizer.pad_token_id

        input_ids, attention_mask, labels = [], [], []
        for ex in examples:
            seq, mask = _pad_sequence(ex["input_ids"], max_len, pad_id)
            lab, _ = _pad_sequence(ex["labels"], max_len, -100)
            input_ids.append(seq)
            attention_mask.append(mask)
            labels.append(lab)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ----------------------------------------------------
# Full string distributional training (K completions per example, soft CE loss)
# ----------------------------------------------------


def tokenize_distributional(example, tokenizer, max_length):
    """Expand one distributional example into its K option-string completions.

    Each example has a shared prompt (system + user + options block) and an
    empirical response distribution q over K categories. The prompt is
    rendered once, then one sequence per category (``y_i = category_i + "<|im_end|>"``)
    is built with prompt tokens masked to ``-100`` in ``labels``, so training
    only scores the completion tokens.

    Args:
        example: Dataset row with ``expected_distribution`` and ``categories`` keys.
        tokenizer: Tokeniser instance.
        max_length: Maximum sequence length.

    Returns:
        Dict with ``input_ids`` (list of K lists), ``labels`` (list of K
        lists), and ``q`` (list of K floats).
    """
    prompt_ids = _prompt_ids(example, tokenizer, max_length)

    input_ids, labels, q = [], [], []
    for q_i, category in zip(example["expected_distribution"], example["categories"]):
        comp_ids = _completion_ids(category, tokenizer)
        # Never mutate `prompt_ids` — all K sequences share the same prefix;
        # _with_completion truncates a copy from the left when needed.
        seq, lab = _with_completion(prompt_ids, comp_ids, max_length)

        input_ids.append(seq)
        labels.append(lab)
        q.append(q_i)

    return {"input_ids": input_ids, "labels": labels, "q": q}


def distributional_loss(q, logits, labels, row_ids):
    """Compute renormalised soft cross-entropy over K option-string completions.

    .. math::

        L = -\\sum_i q_i \\log \\tilde{p}_i,
        \\qquad \\tilde{p}_i = \\text{softmax over the K completions of } \\log p(y_i | x)

    Soft-label training over full option strings (LDL-over-sequences
    transplanted to an LLM). Per-completion log-probability is the joint over
    the option string (sum over completion tokens, not length-normalised),
    matching the eval's full-sequence scoring. Loss is mean over examples in
    the batch.

    Args:
        q: Soft labels, one per completion. Shape ``[S]``.
        logits: Model logits. Shape ``[S, T, V]``.
        labels: Token labels, ``-100`` on prompt/padding tokens. Shape ``[S, T]``.
        row_ids: Example index each completion belongs to, groups the K
            completions of one prompt for the renormalisation. Shape ``[S]``.

    Returns:
        Scalar loss tensor, mean over examples in the batch.
    """
    import torch
    import torch.nn.functional as F

    # Shift: logits[t] predicts token t+1, so drop the last logit and the
    # first label to keep them aligned.
    shifted_logits = logits[:, :-1]
    targets = labels[:, 1:].clamp_min(0)  # -100 -> 0 (safe gather index)
    valid = labels[:, 1:] != -100

    # log p = gathered logit - logsumexp(logits): avoids materialising the
    # full [S, T, V] log_softmax output, which is the largest allocation in a
    # step (vocab ~151k -> several GiB per batch). The logsumexp reduction
    # accumulates in fp32 internally, so precision is comparable.
    token_log_probs = torch.gather(
        shifted_logits, dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1) - torch.logsumexp(shifted_logits, dim=-1)
    token_log_probs = token_log_probs.masked_fill(~valid, 0.0)

    # Joint log-probability per completion (sum over tokens, not length-
    # normalised), matching the eval's full-sequence scoring.
    completion_log_probs = token_log_probs.sum(dim=-1)  # [S]

    # Renormalised soft CE, mirroring the eval estimator which renormalises
    # the model's K option probabilities before computing CE:
    #   p̃_i = softmax over the K completions of one example
    #   L    = -sum_i q_i * log p̃_i
    # mean over examples in the batch.
    unique_row_ids = row_ids.unique()
    loss = torch.tensor(0.0, device=logits.device)
    for row_id in unique_row_ids:
        m = row_ids == row_id
        p_tilde = F.softmax(completion_log_probs[m], dim=0)
        loss = loss - (q[m] * p_tilde.log()).sum()
    return loss / unique_row_ids.numel()


class DistributionalCollator:
    """Flatten B examples x K completions into one padded batch.

    Each dataset row holds K sequences (shared prompt + one option
    completion). The collator flattens them to ``S = B*K`` rows and returns
    ``q`` (soft labels) and ``row_ids`` (which example each sequence belongs
    to) aligned with the flattened batch.

    Args:
        tokenizer: Tokeniser instance (used for pad token id).
        max_length: Maximum sequence length.
    """

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        import torch

        rows = []
        for row_id, ex in enumerate(examples):
            for i in range(len(ex["q"])):
                rows.append((row_id, ex["q"][i], ex["input_ids"][i], ex["labels"][i]))
        max_len = min(self.max_length, max(len(r[2]) for r in rows))
        pad_id = self.tokenizer.pad_token_id

        input_ids, attention_mask, labels, q, row_ids = [], [], [], [], []
        for row_id, qi, seq, lab in rows:
            seq, mask = _pad_sequence(seq, max_len, pad_id)
            lab, _ = _pad_sequence(lab, max_len, -100)
            input_ids.append(seq)
            attention_mask.append(mask)
            labels.append(lab)
            q.append(qi)
            row_ids.append(row_id)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "q": torch.tensor(q, dtype=torch.float),
            "row_ids": torch.tensor(row_ids, dtype=torch.long),
        }


class CustomLossTrainer(Trainer):
    """Trainer that pops extra fields and delegates loss to an external function.

    Stock Trainer losses can only score one response per example with no soft
    labels. This trainer pops the specified keys from the collated batch before
    the forward pass and passes them (plus the logits) to ``loss_fn``.

    Args:
        loss_fn: Callable that receives the popped keys as kwargs plus
            ``logits`` and returns a scalar loss tensor.
        pop_keys: List of input keys to pop before the forward pass.
    """

    def __init__(self, *, loss_fn, pop_keys, **kwargs):
        super().__init__(**kwargs)
        self._loss_fn = loss_fn
        self._pop_keys = pop_keys

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        loss_kwargs = {k: inputs.pop(k) for k in self._pop_keys}
        outputs = model(**inputs)
        loss = self._loss_fn(**loss_kwargs, logits=outputs.logits)
        return (loss, outputs) if return_outputs else loss


# ----------------------------------------------------
# First-token distributional training (K single-token answers per example, soft CE loss)
# ----------------------------------------------------


def tokenize_first_token(example, tokenizer, max_length):
    """Tokenize one first-token example: prompt only, plus K answer tokens and q.

    Unlike :func:`tokenize_distributional`, no expansion happens: the expected
    answer is a single token (a letter from ``answer_tokens``), so one forward
    pass over the prompt gives the logits at the answer position for all K
    options at once. The prompt is truncated from the left if needed so the
    answer position (the final token) is always valid.

    Args:
        example: Dataset row with ``answer_tokens`` and ``expected_distribution``.
        tokenizer: Tokeniser instance.
        max_length: Maximum sequence length.

    Returns:
        Dict with ``input_ids``, ``answer_token_ids``, and ``q``.

    Raises:
        ValueError: If any answer token encodes to more than one token id.
    """
    input_ids = _prompt_ids(example, tokenizer, max_length)

    answer_token_ids = []
    for tok in example["answer_tokens"]:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"answer token {tok!r} encodes to {len(ids)} tokens ({ids}) — "
                "the first-token approach requires single-token answers"
            )
        answer_token_ids.append(ids[0])

    return {
        "input_ids": input_ids,
        "answer_token_ids": answer_token_ids,
        "q": example["expected_distribution"],
    }


def first_token_loss(q, logits, answer_token_ids, option_mask, attention_mask):
    """Compute soft cross-entropy over K answer-token logits (Cao et al. 2025).

    The prompt is scored once: the logits at the last real token position are
    the distribution over the answer token. K single-letter answer tokens are
    gathered, renormalised over the K options, and compared to the empirical
    distribution q using soft cross-entropy:

    .. math::

        L = -\\sum_i q_i \\log \\tilde{p}_i,
        \\qquad \\tilde{p} = \\text{softmax over the K answer logits}

    Args:
        q: Soft labels. Shape ``[B, K]`` (0-padded rows with fewer options).
        logits: Model logits. Shape ``[B, T, V]``.
        answer_token_ids: Token ids of the answer letters (0-padded).
            Shape ``[B, K]``.
        option_mask: 1 for real options, 0 for padding slots. Padding slots
            are excluded from the softmax denominator (but ``q=0`` *real*
            options, e.g. "Don't know", are kept, matching the eval).
            Shape ``[B, K]``.
        attention_mask: Used to find the answer position per row.
            Shape ``[B, T]``.

    Returns:
        Scalar loss, mean over examples in the batch.
    """
    import torch

    last_pos = attention_mask.sum(dim=-1) - 1  # [B] answer position
    ans_logits = logits[
        torch.arange(logits.shape[0], device=logits.device), last_pos
    ]  # [B, V]
    gathered = ans_logits.gather(-1, answer_token_ids)  # [B, K]
    gathered = gathered.masked_fill(~option_mask.bool(), -1e9)
    log_probs = gathered - torch.logsumexp(gathered, dim=-1, keepdim=True)
    return -(q * log_probs).sum(dim=-1).mean()


class FirstTokenCollator:
    """Collate first-token examples: one prompt per example plus K answer
    token ids and soft labels q, zero-padded to the max K in the batch.
    ``option_mask`` marks the real options (padding slots are excluded from
    the softmax denominator in the loss).

    Args:
        tokenizer: Tokeniser instance (used for pad token id).
        max_length: Maximum sequence length.
    """

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        import torch

        max_len = min(self.max_length, max(len(ex["input_ids"]) for ex in examples))
        max_k = max(len(ex["q"]) for ex in examples)
        pad_id = self.tokenizer.pad_token_id

        input_ids, attention_mask = [], []
        answer_token_ids, q, option_mask = [], [], []
        for ex in examples:
            seq, mask = _pad_sequence(ex["input_ids"], max_len, pad_id)
            k = len(ex["q"])
            input_ids.append(seq)
            attention_mask.append(mask)
            answer_token_ids.append(ex["answer_token_ids"] + [0] * (max_k - k))
            q.append(ex["q"] + [0.0] * (max_k - k))
            option_mask.append([1] * k + [0] * (max_k - k))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "answer_token_ids": torch.tensor(answer_token_ids, dtype=torch.long),
            "q": torch.tensor(q, dtype=torch.float),
            "option_mask": torch.tensor(option_mask, dtype=torch.long),
        }


def run_distributional(args, ds, model_path, variant):
    """Run distributional fine-tuning — one shared loop for both variants.

    The two distributional configs share every step of the loop (tokenize,
    collate, custom-loss ``compute_loss``, eval on validation, save). They
    differ only in four pieces, selected by ``variant``:

    ============  =======================  ======================
    variant       tokenize_fn              collator
    ============  =======================  ======================
    full_string   tokenize_distributional  DistributionalCollator
    first_token   tokenize_first_token     FirstTokenCollator
    ============  =======================  ======================

    ``full_string`` expands each example into K option-string completions and
    scores them with the weighted soft-CE loss; ``first_token`` scores the K single-token letter answers at the answer position in
    one forward pass.

    Why a custom Trainer: stock SFTTrainer/Trainer losses can only score one
    response per example with no soft labels. The K options of an example must
    reach the loss together, which is why the collator keeps ``q`` (and
    ``row_ids`` / ``answer_token_ids`` / ``option_mask``) alongside the batch
    so the loss can re-associate them.

    Args:
        args: Parsed command-line arguments.
        ds: Training dataset.
        model_path: Output directory for the fine-tuned model.
        variant: One of ``"full_string"`` or ``"first_token"``.

    Returns:
        Tuple of ``(model_path, log_history, gpu_name, train_time_s)``.

    Raises:
        ValueError: If ``variant`` is not one of the known configs.
    """
    import time
    from functools import partial

    from peft import get_peft_model

    variant_configs = {
        "full_string": (
            tokenize_distributional,
            DistributionalCollator,
            distributional_loss,
            ["q", "labels", "row_ids"],
            "dist",
        ),
        "first_token": (
            tokenize_first_token,
            FirstTokenCollator,
            first_token_loss,
            ["q", "answer_token_ids", "option_mask", "attention_mask"],
            "ft",
        ),
    }
    try:
        tokenize_fn, collator_cls, loss_fn, pop_keys, tag = variant_configs[variant]
    except KeyError:
        raise ValueError(f"Unknown distributional variant: {variant}")

    log.info("[%s] loading tokenizer...", tag)
    tokenizer = load_tokenizer(args)

    log.info("[%s] tokenizing %d examples...", tag, len(ds))
    tok_ds = ds.map(
        partial(
            tokenize_fn,
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
        ),
        batched=False,
        remove_columns=ds.column_names,
    )
    n_opt = sum(len(ex["q"]) for ex in tok_ds)
    log.info("[%s] %d options/completions across %d examples", tag, n_opt, len(tok_ds))

    log.info("[%s] tokenizing validation split for eval...", tag)
    tok_eval_ds = load_dataset(args, split="validation").map(
        partial(
            tokenize_fn,
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
        ),
        batched=False,
        remove_columns=ds.column_names,
    )
    log.info("[%s] eval split: %d examples", tag, len(tok_eval_ds))

    log.info("[%s] loading model...", tag)
    model = get_peft_model(load_base_model(args), make_peft_config(args))

    training_args = TrainingArguments(
        output_dir=str(model_path),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size or args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_total_limit=2,
        fp16=(args.dtype == "fp16"),
        bf16=(args.dtype == "bf16"),
        report_to=[],
        # collator output carries q/row_ids/answer_token_ids/option_mask which
        # are not model args; without this Trainer would strip them before
        # they reach compute_loss
        remove_unused_columns=False,
        dataloader_num_workers=2,
        gradient_checkpointing=True,
    )

    trainer = CustomLossTrainer(
        loss_fn=loss_fn,
        pop_keys=pop_keys,
        model=model,
        args=training_args,
        train_dataset=tok_ds,
        eval_dataset=tok_eval_ds,
        data_collator=collator_cls(tokenizer, args.max_seq_length),
    )

    log.info("[%s] starting training...", tag)
    gpu_name = _get_gpu_info()
    t0 = time.time()
    trainer.train()
    train_time_s = time.time() - t0

    log.info("[%s] saving...", tag)
    trainer.save_model(str(model_path))
    tokenizer.save_pretrained(str(model_path))

    log_history = getattr(trainer.state, "log_history", [])
    save_training_log(model_path, log_history)
    return model_path, log_history, gpu_name, train_time_s


def upload_to_hf(args, model_path: Path):
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=args.hf_token)
    create_repo(args.hf_repo, token=args.hf_token, exist_ok=True)

    log.info(
        "[upload] uploading %s to https://huggingface.co/%s...",
        model_path,
        args.hf_repo,
    )

    api.upload_folder(
        folder_path=model_path,
        repo_id=args.hf_repo,
        repo_type="model",
        token=args.hf_token,
        ignore_patterns=["checkpoint-*"],
    )

    # Add to collection if requested
    if args.hf_collection:
        parts = args.hf_collection.split("/", 1)
        title = parts[1] if len(parts) > 1 else parts[0]
        collection = api.create_collection(
            title=title,
            description="Dataset and LoRA adpaters for a NZ value alignment project that uses World Value Survey data.",
            exists_ok=True,
        )
        # Use the returned collection slug (includes namespace)
        collection_slug = collection.slug
        api.add_collection_item(
            collection_slug=collection_slug,
            item_id=args.hf_repo,
            item_type="model",
            exists_ok=True,
        )
        log.info("[upload] added to collection %s", collection_slug)

    log.info("[upload] done — https://huggingface.co/%s", args.hf_repo)
    return args.hf_repo


def save_run_metadata(output_dir: Path, args, gpu_name: str | None = None):
    import json
    from datetime import datetime

    dataset_sha = None
    model_sha = None
    try:
        from huggingface_hub import dataset_info, model_info

        dataset_sha = dataset_info(
            "1jamesthompson1/wvs-nz-value-alignment", token=os.environ.get("HF_TOKEN")
        ).sha
        model_sha = model_info(args.model, token=os.environ.get("HF_TOKEN")).sha
    except Exception:
        pass

    metadata = {
        "model": args.model,
        "model_sha": model_sha,
        "dataset": args.dataset,
        "dataset_sha": dataset_sha,
        "subpopulation": args.subpopulation,
        "gpu": gpu_name,
        "library_name": "peft",
        "pipeline_tag": "text-generation",
        "timestamp": datetime.now().isoformat(),
        "hyperparameters": {
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "dora": args.dora,
            "quantization": args.quantization,
            "lr": args.lr,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_seq_length": args.max_seq_length,
            "warmup_ratio": args.warmup_ratio,
            "dtype": args.dtype,
        },
    }

    meta_path = output_dir / "finetune_config.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    log.info("    run metadata -> %s", meta_path)


def save_training_log(output_dir: Path, log_history: list[dict]):
    import json

    log_path = output_dir / "training_log.json"
    log_path.write_text(json.dumps(log_history, indent=2))
    log.info("    training log -> %s", log_path)


def _script_dependencies() -> list[str]:
    """Parse package names from this script's PEP 723 dependency block.

    Returns:
        List of package name strings found in the ``dependencies`` list.
    """
    import re

    try:
        src = Path(__file__).read_text()
        match = re.search(r"dependencies\s*=\s*\[(.*?)\]", src, re.DOTALL)
        if not match:
            return []
        names = []
        for line in match.group(1).splitlines():
            line = line.strip()
            if not line:
                continue
            line = line.lstrip("#").strip()
            # Skip nested comment lines (## or # text without quotes)
            if not line or line.startswith("#") or not line.startswith('"'):
                continue
            line = line.strip(",").strip("\"'")
            name = re.split(r"[>=<~!]", line)[0].strip()
            if name:
                names.append(name)
        return names
    except Exception:
        return []


def dump_environment(output_dir: Path, key_names: list[str] | None = None):
    import importlib.metadata
    import json

    if key_names is None:
        key_names = _script_dependencies()

    all_packages = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name", "")
        if name:
            all_packages[name] = dist.version

    env_path = output_dir / "environment.json"
    env_path.write_text(json.dumps(all_packages, indent=2, sort_keys=True))
    log.info("    environment -> %s", env_path)
    return all_packages, key_names


def generate_readme(
    output_dir: Path,
    args,
    log_history: list[dict],
    gpu_name: str | None = None,
    train_time_s: float | None = None,
):
    from jinja2 import Template

    log_entries = [json.dumps(e) for e in log_history]
    packages, key_names = dump_environment(output_dir)

    template_path = Path(__file__).resolve().parent / "MODEL_DATACARD_TEMPLATE.md"
    template = Template(template_path.read_text())

    readme = template.render(
        base_model=args.model,
        dataset=args.dataset,
        subpopulation=args.subpopulation,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        dora=args.dora,
        lr=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        max_seq_length=args.max_seq_length,
        warmup_ratio=args.warmup_ratio,
        dtype=args.dtype,
        log_history=log_entries,
        packages=packages,
        key_names=key_names,
        hf_collection=args.hf_collection or "",
        gpu_name=gpu_name or "",
        train_time_s=train_time_s or 0,
    )

    readme_path = output_dir / "README.md"
    readme_path.write_text(readme)
    log.info("    README -> %s", readme_path)


def main():
    # Set up stdout logging immediately
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler()],
        force=True,
    )
    # Silence noisy HTTP loggers from HF libraries. "HTTP Request: ..."
    # lines come from httpx (huggingface_hub's HTTP client) at INFO level;
    # httpcore is httpx's connection layer.
    for noisy in (
        "datasets",
        "huggingface_hub",
        "httpx",
        "httpcore",
        "urllib3",
        "filelock",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from dotenv import load_dotenv

    load_dotenv()
    args = parse_args()

    # Note: CUDA_VISIBLE_DEVICES is set at module load by _apply_gpu_env()
    # (before the transformers import), because CUDA is initialised during
    # that import and the variable is only honoured at CUDA init time.

    if args.hf_token is None:
        args.hf_token = os.environ.get("HF_TOKEN")
    if args.hf_token is None:
        log.warning("[warn] HF_TOKEN not set — skipping Hugging Face upload")
    if args.hf_collection is None:
        args.hf_collection = os.environ.get("HF_COLLECTION")

    hf_org = os.environ.get("HF_ORG", "")
    if args.hf_repo is None and hf_org:
        model_slug = args.model.split("/")[-1]
        args.hf_repo = (
            f"{hf_org}/{model_slug}-nz-wvs-{args.dataset}-{args.subpopulation}"
        )
    if args.hf_repo:
        log.info("[upload] target repo: %s", args.hf_repo)

    log.info("=" * 60)
    log.info("Fine-tuning configuration")
    log.info("=" * 60)
    for k, v in sorted(vars(args).items()):
        log.info("  %s: %s", k, v)
    log.info("=" * 60)

    ds = load_dataset(args)

    model_path = Path(args.output_dir) / args.dataset / args.subpopulation
    model_path.mkdir(parents=True, exist_ok=True)

    # Now add file logging to the output dir
    setup_logging(model_path)

    if args.dataset in ("modal_response", "sampled_response"):
        _, log_history, gpu_name, train_time_s = run_sft(args, ds, model_path)
    elif args.dataset == "full_string_distribution":
        _, log_history, gpu_name, train_time_s = run_distributional(
            args, ds, model_path, variant="full_string"
        )
    elif args.dataset == "first_token_distribution":
        _, log_history, gpu_name, train_time_s = run_distributional(
            args, ds, model_path, variant="first_token"
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    save_run_metadata(model_path, args, gpu_name)
    generate_readme(
        model_path, args, log_history, gpu_name=gpu_name, train_time_s=train_time_s
    )

    if args.hf_token and args.upload_to_hf and args.hf_repo:
        uploaded_path = upload_to_hf(args, model_path)
        log.info("\n[done] uploaded to %s", uploaded_path)
    elif args.upload_to_hf and not args.hf_repo:
        log.warning("[upload] skipping — set --hf-repo or HF_ORG env var")


if __name__ == "__main__":
    main()
