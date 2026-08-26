"""Single-sourced prompt construction for fine-tuning and evaluation.

Both ``finetune.py`` (training) and ``evaluate.py`` (served-model eval) MUST
build the *exact* same prompt, or the LoRA learned during training will not
fire at eval time.  A tiny difference in the chat-template suffix — e.g.
whether the ``<think>`` block is left open or closed — is enough to wipe out
the entire fine-tuning effect (the model gets the right task in the wrong
context and its learned shift perturbs answers instead of steering them).

This module is the one place that defines the format, so the two sides
cannot drift apart.

``THINKING_ENABLED`` is the default; both ``finetune.py`` and ``evaluate.py``
expose a CLI flag (``--reasoning``) that overrides it, and every rendering
helper accepts an ``enable_thinking`` argument so the value flows from the
command line through to the model unambiguously.
"""

from __future__ import annotations

# Default thinking behaviour.  ``False`` means the generation prompt ends
# with a *closed* empty think block
# (``<|im_start|>assistant\n<think>\n\n</think>\n\n``); vLLM eval passes the
# same value via ``chat_template_kwargs={"enable_thinking": ...}`` so the two
# match.  Change this in ONE place; the CLI flags override it per run.
THINKING_ENABLED = False


def build_messages(system_prompt: str, user_prompt: str, response: str | None = None):
    """Build the ``[system, user, (assistant)]`` message list."""
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if response is not None:
        msgs.append({"role": "assistant", "content": response})
    return msgs


def render_prompt_text(
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    *,
    enable_thinking: bool = THINKING_ENABLED,
):
    """Render the prompt (system + user) through the chat template with a
    generation prompt, returning the **string** (used for logging / inspection
    of exactly what the model is trained or evaluated on)."""
    return tokenizer.apply_chat_template(
        build_messages(system_prompt, user_prompt),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def render_prompt(
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    *,
    max_length: int,
    enable_thinking: bool = THINKING_ENABLED,
):
    """Render the prompt and return token ids, truncated from the left to
    ``max_length`` so the answer position stays at the end."""
    text = render_prompt_text(
        tokenizer, system_prompt, user_prompt, enable_thinking=enable_thinking
    )
    ids = tokenizer(text)["input_ids"]
    return ids[-max_length:] if len(ids) > max_length else ids


def chat_template_kwargs(enable_thinking: bool = THINKING_ENABLED):
    """Extra-body kwargs passed to the OpenAI-compatible server so the eval
    prompt matches the training prompt exactly."""
    return {"enable_thinking": enable_thinking}


def prompt_suffix(
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    *,
    n: int = 60,
    enable_thinking: bool = THINKING_ENABLED,
) -> str:
    """Return just the tail of the rendered prompt (the generation-prompt
    suffix).  Used for logging/verification so we can confirm training and
    eval use the same format."""
    return render_prompt_text(
        tokenizer, system_prompt, user_prompt, enable_thinking=enable_thinking
    )[-n:]
