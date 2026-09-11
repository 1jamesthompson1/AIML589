# WVS NZ value alignment Research Assistant Agent instructions

This directory is a mono-repo for a research project to fine tune open weight LLMs on the World Value Survey data and then run behavioural experiments and using public consultation to evaluate the alignment of the model with the New Zealand population.

Your primary role is to assist with the research project. You should complete the tasks given to you while avoiding making key design decisions. If something is ambiguous or unclear, you should ask for clarification or for a judgement to be made.

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

Keep the readme and inline documentation very concise and to the point.

## Latex files

Latex files are mainly in the `docs/` directory. All latex files can be compiled with `make` from the root of the repo. If you want to watch file for changes and recompile automatically, you can run `make watch FILE=[filename no extension]` from the root of the repo. For example, to watch `docs/two-pager.tex`, run:

```bash
make watch FILE=two-pager
```

Generally speaking the user will have this command running when talking with you therefore you needn't constantly recompile the latex files. So changes to the latex files would be sufficient.

All pdf files are found in a `*/output/` directory. For example, the output of `docs/two-pager.tex` is found in `docs/output/two-pager.pdf`.

Besides when explicitly asked to edit the reports please don't edit the latex files. All prose will be written by hand. You job will likely involve fixing bugs in the latex and adding tables and figures.

## Code

An important goal of this project is transparency and reproducibility. All code should be sufficiently documented and commented to allow a future user to understand what is being done and why. Keep it short and concise as verbose explanations will be found in the final report.

All code will be found in a `code/` subdir. Any scratch scripts can be written in the `workbench/` subdir.

### Scripts

There are two sorts of scripts in this repo. The first are scripts that may be ran interactively. These should be `marimo` scripts. A key catch with this is that variables can only be assigned once. Any overwritting assignment causes and error. Therefore functions should be used as much as possible to keep the global namespace clean. Other scripts that are not meant to be ran interactively should be simple python scripts.

**Important** marimo has special rules in which a global variable can only be assigned once. This is to help with reproducibility. Therefore, to prevent the polluting of the global namespace as much code as possible should be put into functions. The rough flow is that each cell defines the work it is doing in a function and at the end of the cell calls the function to run it.

Imports should be at the top of the scripts in the import cell unless there is very good reason to do otherwise.

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