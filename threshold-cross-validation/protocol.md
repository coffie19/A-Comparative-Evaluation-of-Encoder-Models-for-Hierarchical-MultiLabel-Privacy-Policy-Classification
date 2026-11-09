# Classification-threshold experiment: prepared protocol

## Purpose

Select category thresholds without consulting the 50-policy evaluation set, then test whether the category ranking and Data Retention findings survive on that untouched set.

## Design

- Split the original 65 training policies into five fixed, disjoint folds of 13 policies.
- Train each fold model on 52 policies and predict the remaining 13.
- Use the same fold assignments and fold seeds (17011, 29023, 41047, 53069 and 65089) for all four models.
- Pool exactly one out-of-fold probability for each of the 2,352 training segments.
- Search thresholds from 0.05 to 0.95 in steps of 0.01.
- Select one global threshold by out-of-fold category Macro-F1. Select ten per-category thresholds by the corresponding positive-class F1.
- Break ties by choosing the value closest to 0.50, then the lower value.
- Treat the global threshold as primary and per-category thresholds as secondary because the latter select ten parameters.
- Freeze thresholds before evaluating the original 50-policy set.
- Apply all three rules—fixed 0.50, global CV and per-category CV—to every available main-study seed checkpoint.
- Report category Macro-F1 and Data Retention precision, recall and F1 as mean ± sample standard deviation across five seeds.

## Fold validation support

| Fold | Policies | Segments | Data Retention | Do Not Track | Policy Change |
|---:|---:|---:|---:|---:|---:|
| 1 | 13 | 490 | 27 | 4 | 23 |
| 2 | 13 | 463 | 18 | 4 | 28 |
| 3 | 13 | 468 | 20 | 5 | 26 |
| 4 | 13 | 456 | 20 | 5 | 28 |
| 5 | 13 | 475 | 6 | 3 | 23 |

Every category has positive validation support in every fold. Data Retention support ranges from 6 to 27 segments, so the threshold estimate may still vary across folds. The final report retains fold-level outputs and does not treat the selected threshold as a causal explanation.

## Leakage controls

Fold construction uses only the 65 training policies. The 50 evaluation policies do not influence fold assignment, class weights, training, threshold selection, the threshold grid or tie-breaking. A policy and all its segments remain in one fold.

## Execution status

**The design, code, notebooks, archives and local preparation checks are complete. The twenty full GPU fold fits have not been executed on this computer. Therefore, no selected threshold or revised held-out score is reported yet.**

Each notebook resumes after completed folds, saves out-of-fold probabilities and thresholds, evaluates the five saved main-study checkpoints, and produces a four-model comparison after all model notebooks finish.
