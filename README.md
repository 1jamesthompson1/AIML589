# Aligning LLM to NZ public values and evaluation through public consultation

> [!NOTE] Project status
> This project is a 4 month research project. Starting July 2026 ending in November 2026. It is being conducted by James Thompson as a Master project under the supervision of Dr Andrew Lensen.
> This repo will not be considered 'presentation' ready until the end. Regardless though it is public as that seem like the right way to do things.

A simple introduction of the project can be found in the [two-pager](docs/output/James-Thompson-LLM-NZ-Value-Alignment-Funding-Request.pdf). This project is a continuation of the work done over summer 25/26 in [AIML501](https://raw.githubusercontent.com/1jamesthompson1/AIML501/main/output/AIML501_James_Thompson.pdf).

## User guide

This repo is a  mono repo and so contains a few diffierent systems to help work with this project. This is the documentation for these different system.

### Setting up

The two tools that are used to manage things is [`uv`](https://docs.astral.sh/uv/getting-started/installation/) and [`make`](https://www.gnu.org/software/make/)[^1]. These should already be installed on your system.

[^1]: Which on windows can be used with [chocolatey](https://chocolatey.org/install) or with [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)

Then to get the repo setup simply run:

```bash
git clone https://github.com/1jamesthompson1/AIML589.git
cd AIML589
make setup
```

### Docs

There are various documents made in the `docs` folder. These are written in LaTeX and can be compiled using `make all` or `make watch FILE=[.tex file name]` to compile a specific file. The output will be in the `docs/output` folder. Each latex file should inherit the `docs/common.tex` file and define its output file name using the `\docname{}` command.

### Code

This project has a fair amount of code. The code is separated into phases and each phase has its own folder. The code is written using [Marimo](https://docs.marimo.io/) to help with reproducibility. In each subdir of code you will find a readme explaining what the code does and how to run it.