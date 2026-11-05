#!/usr/bin/env Rscript
# Fit LMM/GLMM variants with lme4 for survey/analyze.py.
#
# Usage:
#   Rscript survey/fit_models.R <spec.json> <out.json>
#
# The spec is a JSON object:
#   {
#     "data": "/path/to/data.csv",
#     "family": "gaussian" | "binomial",
#     "models": [{"name": "...", "formula": "y ~ order_effect + (1|g0)"}, ...]
#   }
#
# The output is a JSON list with one entry per model: coefficient names,
# estimates, standard errors, the fixed-effect covariance matrix, the random
# effect standard deviations and the residual SD (Gaussian only). This keeps
# all numerical fitting in lme4; Python only prepares the data and formats
# the tables.
#
# lme4 is used with REML for Gaussian models and Laplace (the glmer default)
# for binomial models. The formula strings are built by r_models.py from
# safe column names (g0, g1, ... for the grouping factors).

suppressPackageStartupMessages({
  library(lme4)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("usage: fit_models.R <spec.json> <out.json>")
}

spec <- fromJSON(args[1], simplifyVector = FALSE)
data <- read.csv(spec$data, stringsAsFactors = FALSE)
family <- spec$family
has_re <- function(formula) grepl("\\|", formula)

fit_one <- function(model) {
  formula <- as.formula(model$formula)
  if (family == "gaussian") {
    fit <- if (has_re(model$formula)) {
      lmer(formula, data = data, REML = TRUE)
    } else {
      lm(formula, data = data)
    }
  } else {
    fit <- if (has_re(model$formula)) {
      glmer(
        formula,
        data = data,
        family = binomial,
        control = glmerControl(
          optimizer = "bobyqa",
          optCtrl = list(maxfun = 2e5)
        )
      )
    } else {
      glm(formula, data = data, family = binomial)
    }
  }

  is_mer <- inherits(fit, "merMod")
  fixed <- if (is_mer) lme4::fixef(fit) else coef(fit)
  covariance <- vcov(fit)
  variance <- if (is_mer) as.data.frame(VarCorr(fit)) else NULL
  if (!is.null(variance)) {
    variance <- variance[variance$grp != "Residual", , drop = FALSE]
  }
  random_sd <- if (!is.null(variance) && nrow(variance) > 0) {
    setNames(as.list(variance$sdcor), variance$grp)
  } else {
    list()
  }

  list(
    name = model$name,
    n = nobs(fit),
    coef_names = names(fixed),
    coef_estimate = as.numeric(fixed),
    coef_se = as.numeric(sqrt(diag(covariance))),
    cov = lapply(seq_len(nrow(covariance)), function(i) {
      as.numeric(covariance[i, ])
    }),
    sigma = random_sd,
    sigma_resid = if (family == "gaussian") as.numeric(sigma(fit)) else NULL,
    loglik = as.numeric(logLik(fit))
  )
}

models <- lapply(spec$models, fit_one)
writeLines(
  toJSON(models, auto_unbox = TRUE, digits = 12, null = "null"),
  args[2]
)
