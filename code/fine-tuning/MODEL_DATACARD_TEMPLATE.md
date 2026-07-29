---
base_model: {{ base_model }}
library_name: peft
model_name: {{ subpopulation }}
tags:
  - base_model:adapter:{{ base_model }}
  - lora
  - sft
  - transformers
  - trl
  - {{ dataset }}
  - {{ subpopulation }}
license: cc-by-sa-4.0
pipeline_tag: text-generation
datasets:
  - 1jamesthompson1/wvs-nz-value-alignment
---

{% set slug = base_model.split('/')[-1] %}
{% set dataset_label = dataset.replace('_', ' ').title() %}
{% set subpop_label = subpopulation.replace('_', ' ').title() %}

# {{ slug }} LoRA — {{ dataset_label }}, {{ subpop_label }}

This model is a LoRA fine-tune of [{{ base_model }}](https://huggingface.co/{{ base_model }})  as part of the
[AIML589 project](https://github.com/1jamesthompson1/AIML589).

This adapter is licensed under CC BY-SA 4.0.

## Dataset

Fine-tuned on the **{{ dataset }}** config of the
[wvs-nz-value-alignment](https://huggingface.co/datasets/1jamesthompson1/wvs-nz-value-alignment)
dataset, **{{ subpopulation }}** subpopulation.

{% if hf_collection %}
Part of the [{{ hf_collection }}](https://huggingface.co/collections/{{ hf_collection }}) collection.
{% endif %}
{% if gpu_name %}
**GPU:** {{ gpu_name }}{% if train_time_s %} · **Training time:** {{ (train_time_s // 60)|int }}m {{ (train_time_s % 60)|int }}s{% endif %}
{% endif %}

## Training hyperparameters

| Parameter | Value |
|-----------|-------|
| LoRA rank | {{ lora_r }} |
| LoRA alpha | {{ lora_alpha }} |
| LoRA dropout | {{ lora_dropout }} |
| DoRA | {{ dora }} |
| Learning rate | {{ lr }} |
| Batch size | {{ batch_size }} |
| Gradient accumulation | {{ gradient_accumulation_steps }} |
| Epochs | {{ num_epochs }} |
| Max seq length | {{ max_seq_length }} |
| Warmup ratio | {{ warmup_ratio }} |
| Dtype | {{ dtype }} |

## Training log

{% if log_history %}
```
{% for entry in log_history %}{{ entry }}
{% endfor %}```
{% endif %}

## Environment

| Package | Version |
|---------|---------|
{% for name in key_names %}| {{ name }} | {{ packages.get(name, '?') }} |
{% endfor %}

## Intended use

This adapter is intended for **research purposes only** as part of the
[AIML589 project](https://github.com/1jamesthompson1/AIML589), which
investigates value alignment of LLMs with New Zealand population
distributions from the World Values Survey.

### Out-of-scope

This model has not been safety-tuned for general-purpose deployment.
It should not be used in production systems, for making decisions about
people, or in contexts where reliability and safety are critical.

## Limitations and biases

- Fine-tuned on a single WVS wave (Wave 7) for New Zealand only.
- The training data reflects the values of those who responded to the
  survey and may not represent all New Zealanders.
- LoRA adapters are subject to the limitations and biases of the base
  model ([{{ base_model }}](https://huggingface.co/{{ base_model }})).
