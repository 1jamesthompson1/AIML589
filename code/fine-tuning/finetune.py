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
"""
Fine-tune an LLM on the WVS-NZ value alignment dataset using LoRA.

Uploads the adapter to its own HF Hub repo at
``{HF_ORG}/{model_slug}-nz-wvs-{dataset}-{subpopulation}``
for use with the LoRA-aware vLLM server (see ``serve.py``).

Usage:
    uv run finetune.py --model Qwen/Qwen3.6-27B \\
        --dataset distributional --subpopulation overall \\
        --upload-to-hf --hf-collection "1jamesthompson1/wvs-nz-lora-adapters"

    # SCP to a GPU machine and run:
    scp finetune.py user@host:~
    ssh user@host ./finetune.py --model Qwen/Qwen3.6-27B \\
        --dataset distributional --subpopulation overall \\
        --hf-token "$HF_TOKEN" --upload-to-hf

    uv run code/fine-tuning/finetune.py --help
"""

import argparse
import json
import logging
import os
from pathlib import Path

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
        "--dataset",
        default="single_modal",
        choices=["single_modal", "single_sample", "distributional"],
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
        "--save-steps", type=int, default=500, help="Save checkpoint every N steps"
    )
    p.add_argument("--eval-steps", type=int, default=500, help="Evaluate every N steps")

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


def load_dataset(args):
    from datasets import load_dataset

    ds = load_dataset(
        "1jamesthompson1/wvs-nz-value-alignment",
        args.dataset,
        split="train",
        token=args.hf_token,
    )
    ds = ds.filter(lambda x: x["subpopulation"] == args.subpopulation)
    log.info(
        "[data] loaded %d train examples (config=%s, subpop=%s)",
        len(ds),
        args.dataset,
        args.subpopulation,
    )
    return ds


def format_chat(system_prompt: str, user_prompt: str, response: str | None):
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if response is not None:
        msgs.append({"role": "assistant", "content": response})
    return msgs


def format_chat_distribution(
    system_prompt: str, user_prompt: str, categories: list[str]
):
    prompt = (
        user_prompt
        + "\n\nOptions:\n"
        + "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(categories))
    )
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
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


def run_sft(args, ds, model_path):
    import time
    from trl import SFTConfig, SFTTrainer

    log.info("[sft] loading tokenizer...")
    tokenizer = load_tokenizer(args)

    log.info("[sft] loading model...")
    model = load_base_model(args)

    peft_config = make_peft_config(args)

    def formatting_func(example):
        response = example.get("expected_text", "")
        msgs = format_chat(example["system_prompt"], example["user_prompt"], response)
        return tokenizer.apply_chat_template(msgs, tokenize=False)

    training_args = SFTConfig(
        output_dir=str(model_path),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="no",
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
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        peft_config=peft_config,
        formatting_func=formatting_func,
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


def run_distributional(args, ds, model_path):
    log.warning("[dist] distributional training not yet implemented — skipping")
    log.info("[dist] saving empty adapter for structure...")
    from peft import get_peft_model

    model = get_peft_model(load_base_model(args), make_peft_config(args))
    tokenizer = load_tokenizer(args)
    model.save_pretrained(str(model_path))
    tokenizer.save_pretrained(str(model_path))
    save_training_log(model_path, [])
    return model_path, [], _get_gpu_info(), 0


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


def save_run_metadata(output_dir: Path, args):
    import json
    from datetime import datetime

    metadata = {
        "model": args.model,
        "dataset": args.dataset,
        "subpopulation": args.subpopulation,
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
    """Parse package names from this script's PEP 723 dependency block."""
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
    # Silence noisy HTTP loggers from HF libraries
    for noisy in ("datasets", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from dotenv import load_dotenv

    load_dotenv()
    args = parse_args()

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

    if args.dataset.startswith("single_"):
        _, log_history, gpu_name, train_time_s = run_sft(args, ds, model_path)
    elif args.dataset == "distributional":
        _, log_history, gpu_name, train_time_s = run_distributional(
            args, ds, model_path
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    save_run_metadata(model_path, args)
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
