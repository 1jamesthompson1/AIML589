# Aligning LLM to NZ public values and evaluation through public consultation

> [!NOTE]  
> **Work in Progress**<br>This project is a 4 month research project. Starting July 2026 ending in November 2026. It is being conducted by James Thompson as a Master project under the supervision of Dr Andrew Lensen. This repo will not be considered 'presentation' ready until the end. Regardless though it is public as that seem like the right way to do things.

A simple introduction of the project can be found on the [website](https://nz-llm.sjhl.nz). This project is a continuation of the work done over summer 25/26 in [AIML501](https://raw.githubusercontent.com/1jamesthompson1/AIML501/main/output/AIML501_James_Thompson.pdf).

## User guide

This repo is a mono repo containing several systems. Each top-level directory has a specific purpose:

| Directory | Purpose |
|---|---|
| `code/` | Python code for training datasets, fine-tuning models, and running behavioural simulations. Each phase has its own subdirectory with a README. Written as [Marimo](https://docs.marimo.io/) notebooks. |
| `docs/` | LaTeX documents (report, funding requests, ethics application). Compiled with `make`. Output goes to `docs/output/`. |
| `survey/` | Source files for the public consultation survey: participant information sheet, survey mockup, recruitment flyer/post, and the VUW logo. |
| `website/` | Public-facing project site ([nz-llm.sjhl.nz](https://nz-llm.sjhl.nz)). Built with Astro + React, deployed via GitHub Pages. |
| `workbench/` | Scratch scripts, experiments, and draft files — not part of the main pipeline. |
| `artifacts/` | Heavy pipeline outputs (fine-tuning evals, behavioural-simulation runs) mirrored locally and synced to a public HF storage bucket — **gitignored**. See [HF bucket artifacts](#hf-bucket-artifacts). |

### Setting up

Requires [`uv`](https://docs.astral.sh/uv/getting-started/installation/) and [`make`](https://www.gnu.org/software/make/)[^1].

[^1]: On Windows use [chocolatey](https://chocolatey.org/install) or [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install).

```bash
git clone https://github.com/1jamesthompson1/AIML589.git
cd AIML589
make setup          # also copies .env.example -> .env if missing
# edit .env (HF_TOKEN is required only when syncing the bucket)
make website        # local dev; reads artifacts/ directly
```

### HF bucket artifacts

Heavy outputs (`code/fine-tuning/output` → `artifacts/ft`, `code/behavioural-simulations/output` → `artifacts/bs`, both symlinks) live in a public Hugging Face **storage bucket** — plain object storage, no git overhead:

- **`make artifacts-sync`** — upload `artifacts/` to the bucket (skips unchanged files; runs best-effort from the pre-commit hook and does **not** delete remote-only history).
Remote bucket inspection is intentionally kept outside the Makefile. Use the
HF CLI directly:

```bash
hf buckets list 1jamesthompson1/wvs-nz-value-alignment-evals -R --tree
hf buckets info 1jamesthompson1/wvs-nz-value-alignment-evals
```

Remote cleanup is intentionally not automated while the local mirror may be
incomplete; review a prefix-specific plan after a replacement batch is validated.
- **`make artifacts-backup`** — Borg backup (unencrypted, auto-initialised) of `artifacts/` into `$HOME/grid-directory/AIML589-evals-backup`; also runs in pre-commit when `RUN_BORG_BACKUP=true`.
- **`make artifacts-pull`** — download the bucket back to `artifacts/` (fresh machines). The website dev server does not call this.

The website dev server reads the repository's `artifacts/ft` and `artifacts/bs`
directories directly, so local iteration does not require a download or a
sync. `?local=0` switches a dev session to the bucket for comparison.

Configured via `.env` (see `.env.example`): `HF_TOKEN`, `HF_BUCKET` (default `1jamesthompson1/wvs-nz-value-alignment-evals`), `RUN_BORG_BACKUP`, `BORG_REPO`. Directory paths of empty dirs are not stored in the bucket.

### Docs

LaTeX documents in `docs/` compile with `make all` or `make watch FILE=[name]`. Output lands in `docs/output/`. Each file inherits from `docs/common.tex` and sets its output name with `\docname{}` The main file is `docs/report.tex` which makes the final [university report](https://raw.githubusercontent.com/1jamesthompson1/AIML589/main/docs/output/AIML589-James-Thompson-LLM-NZ-Value-Alignment.pdf).

The report has a 15,000 word limit (excluding references and appendices). The working draft page shows the live word count, computed automatically by `texcount` on every compile (via `\write18`, see `.latexmkrc`). Check the count manually with `make wordcount`.


### Code

Separated into pipeline phases:

- **`training-dataset/`** — Wrangles WVS data, clusters respondents via LCA, builds training datasets (modal, sampled, distributional).
- **`fine-tuning/`** — LoRA fine-tunes open-weight LLMs on the dataset and evaluates distributional alignment.
- **`behavioural-simulations/`** — Runs agentic simulations and generates comparisons for the survey.

Each has its own README. Code is written as Marimo notebooks for reproducibility and bash scripts for automation.

### Survey

The `survey/` directory holds the source LaTeX for all ethics-application documents: participant information sheet, survey mockup, recruitment flyer, and recruitment post text. These compile via `make` like other docs.

### Website

Static site at [nz-llm.sjhl.nz](https://nz-llm.sjhl.nz). See `website/README.md` for commands and deployment. It is a React + Astro site deployed via GitHub Pages.

## Usage of AI models

AI models are used in this project to help with the research. It is generally used as a "research assistant" and is used in a few areas:

- A code assistant to help with code generation and debugging.
- A devops assistant to help with setting up the project and managing the repo.
- A literature review assistant to help with finding relevant papers.
- A proofreading assistant to help with grammar and spelling.

To do these jobs LLM are used in both the [Opencode](https://opencode.net/) and online chatbot interfaces (i.e ChatGPT).

All the work remains fully under the control of the researcher (James Thompson). With complete understanding of the code and the results.
