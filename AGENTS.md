# WVS NZ value alignment Research Assistant Agent instructions

This directory is a mono-repo for a research project to fine tune open weight LLMs on the World Value Survey data and then run behavioural experiments and using public consultation to evaluate the alignment of the model with the New Zealand population.

## Documentation

This project follows the principle of documentation as close to code as possible. Almost all subdirs will have a README.md file that explains the purpose of the directory and how to use it. The top-level README.md file is a high-level overview of the project and its goals.

These should be updated as the project evolves and new features are added.

Start with README.md in the root of the repo and for any repo you are interested in, read the README.md in that repo. If you have any questions ask. Also if readmes are stale or inaccurate alert user and provide suggested fix.

Each readme should be short and concise. It should contain:

- A description of the purpose of the dir (i.e basic intro, what it does  including outputs etc)
- A description of the workflows to use the dir
  THis should help guide a future user  to reproduce what was done. If relevant include information and what needs to be done to get data needed to redo the experiments.
- Any extra descriptions needed to understand the dir and its output.

Each of the README.md should be treated a bit like more agent instructions.

## Latex files

Latex files are mainly in the `docs/` directory. All latex files can be compiled with `make` from the root of the repo. If you want to watch file for changes and recompile automatically, you can run `make watch FILE=[filename no extension]` from the root of the repo. For example, to watch `docs/two-pager.tex`, run:

```bash
make watch FILE=two-pager
```

Generally speaking the user will have this command running when talking with you therefore you needn't constantly recompile the latex files. So changes to the latex files would be sufficient.

All pdf files are found in a `*/output/` directory. For example, the output of `docs/two-pager.tex` is found in `docs/output/two-pager.pdf`.

## Code

An important goal of this project is transparency and reproducibility. All code should be sufficiently documented and commented to allow a future user to understand what is being done and why. Keep it short and concise as verbose explanations will be found in the final report.

All code will be found in a `code/` subdir. Any scratch scripts can be written in the `workbench/` subdir.

### Scripts

All scripts should be `marimo` scripts. This is a modern altenrative to jupyter and increases reproducibility. A key catch with this is that variables can only be assigned once. Any overwritting assignment causes and error. Therefore functions should be used as much as possible to keep the global namespace clean.

### Package management

Project uses `uv` so to run python scripts etc.

```bash
uv run <script.py>
```

Adding new packages (note this should only be done with user consent and remember most of the GPU code is never run in this repo so it should not be added to project pyproject.toml).

```bash
uv add <package>
```

To run a package from the project dependencies you can do it with

```bash
uv run <package> ...
```

e.g. to run `marimo` you can do `uv run marimo edit --help`