# Clusters output

This directory contains outputs from latent class analysis (StepMix, k=2) on
survey response data.

| File | Description |
|---|---|
| `cluster_assignments.csv` | Each row is a respondent with their predicted cluster (0 or 1), plus the posterior probability of belonging to each cluster (`prob_cluster_0`, `prob_cluster_1`). High probability indicates a clean assignment. |
| `cluster_response_distributions.json` | Human-readable JSON detailing each cluster's response patterns. Contains per-cluster item-level response probabilities over the original WVS codes (with word labels) and non-response rates. Consumed by `build_dataset.py`. |
| `model_selection_sweep.csv` | Model fit statistics (log-likelihood, AIC, BIC) for k = 1–6. Lower AIC/BIC indicate better fit; used to justify the chosen number of clusters. |
| `question_informativeness.csv` | Unified ranking of value survey + demographic questions by informativeness. Columns: `column`, `informativeness`, `type` (`"value"` or `"demographic"`), `question_text`. Use this to select the top-N questions for the consultation survey. |
| `stepmix_model_k{K}.joblib` | Serialized fitted StepMix model, saved via `joblib.dump()`. The filename includes the chosen K (`k2`, `k3`, etc.). Load with `joblib.load()` to reproduce assignments or apply to new data without refitting. |
