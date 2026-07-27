# WVS NZ value alignment Research Assistant Agent instructions

## Introduction

This directory is a mono-repo for a research project to fine tune open weight LLMs on the World Value Survey data and then run behavioural experiments and using public consultation to evaluate the alignment of the model with the New Zealand population.

## Dev ops basics

### Latex files

Latex files are mainly in the `docs/` directory. All latex files can be compiled with `make` from the root of the repo. If you want to wathc file for changes and recompile automatically, you can run `make watch FILE=[filename no extension]` from the root of the repo. For example, to watch `docs/two-pager.tex`, run:

```bash
make watch FILE=two-pager
```

All pdf files are found in a `*/output/` directory. For example, the output of `docs/two-pager.tex` is found in `docs/output/two-pager.pdf`.

### Code

Project uses `uv` so to run python scripts etc.

```bash
uv run <script.py>
```

Adding new packages (note this should only be donew ith user consent and remember most of the GPU code is never run in this repo so it should not be added to project pyproject.toml).

```bash
uv add <package>
```

## Documentation

This project follows the principle of documentation as close to code as possible. Almost all subdirs will have a README.md file that explains the purpose of the directory and how to use it. The top-level README.md file is a high-level overview of the project and its goals.

These should be updated as the project evolves and new features are added.