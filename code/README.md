# Main collection of code for the WVS NZ value alignment research project

The code is split into pipeline stages (and subdirs):

1. `training-dataset` Creating a training-dataset from the WVS NZ data.
2. `fine-tuning` Fine-tuning open-weight LLMs on the created dataset.
3. `behavioural-simulations` Running behavioural simulations and generating vignettes to be used by the public consultation survey.

4. `dilemmas` Running handwritten moral dilemmas as single-response evaluations; see its README.

Each directory will have a `output` subdir which contains the output of the code (e.g. eval results, figures, etc). Data heavy outputs (e.g. trained models etc) will not be stored in the repo but instead will be uploaded to HuggingFace. Manually written datasets (i.e. survey question mappings) should be added to `output` as it is an output of that stage of the pipeline. There optionally may be an `input` directory which contains input data for the code.

There is one more dir at `code/figures` which contains figures which are used in the final report. Only  figures that are needed for report should be present.

## OpenRouter attribution

`openrouter_attribution.py` is the single source for OpenRouter
`HTTP-Referer` and `X-OpenRouter-Title` headers. Current app names are
**Simulations**, **Simulations Comparisons**, and **WVS Value Map**, so
OpenRouter activity and cost can be separated by workflow. New OpenRouter
clients should import the relevant headers or helper from this module rather
than defining attribution locally.

## LaTeX tables

`latex_tables.py` is the shared helper for writing pandas DataFrames as
booktabs LaTeX tables into `code/figures/`, wrapped in
`\resizebox{\textwidth}{!}{...}`. The report imports them with
`\ctable{<name>}{<caption>}`. Analysis notebooks in `code/<stage>/` should
import it (after adding `code/` to `sys.path`) rather than defining their own
escaping and rendering helpers.
