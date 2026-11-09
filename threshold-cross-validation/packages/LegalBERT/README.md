# OPP-115 five-fold classification-threshold experiment

This package selects category thresholds without using the 50-policy evaluation set.

- The original 65 training policies are assigned to five disjoint folds of 13 policies.
- Each fold model trains on 52 policies and predicts the held-out 13 policies.
- The five sets of predictions form out-of-fold probabilities for all 2,352 training segments.
- A global threshold and ten per-category thresholds are selected on a fixed 0.05–0.95 grid in steps of 0.01.
- The global threshold maximises category Macro-F1. Each category threshold maximises that category's positive-class F1.
- Ties prefer the threshold closest to 0.50, then the lower threshold.
- The global threshold is the primary sensitivity analysis; per-category thresholds are secondary.
- Only after selection are the fixed thresholds applied to the untouched 50-policy evaluation set.

Full execution requires a Colab GPU and the completed five-seed checkpoints under:

`.../matched_five_seeds_v1/full/MODEL/seed_SEED/checkpoint`

Each model notebook resumes at fold boundaries and writes its results to Google Drive. After all four finish, the final notebook cell creates `threshold_comparison.md`, which states directly whether the model ranking changes and reports Data Retention results.

No test labels or probabilities are used for fold construction, model fitting, threshold selection, or tie-breaking.
