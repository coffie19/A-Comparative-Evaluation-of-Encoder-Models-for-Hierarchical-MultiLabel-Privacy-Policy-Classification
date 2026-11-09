# Five-fold threshold selection: results visible in supplied screenshots

## Scope

These are training-side out-of-fold results from five policy-grouped folds. They establish the thresholds without using the 50-policy evaluation set. They do not by themselves establish whether held-out model rankings or Data Retention conclusions survive threshold tuning because the held-out results were below the cropped area.

## Global thresholds

| Model | Selected global threshold | Macro-F1 at 0.50 | Macro-F1 at selected threshold | Absolute change |
|---|---:|---:|---:|---:|
| BERT | 0.51 | 0.736958 | 0.737604 | +0.000646 |
| LegalBERT | 0.50 | 0.752129 | 0.752129 | 0.000000 |
| RoBERTa | 0.55 | 0.753620 | 0.757702 | +0.004082 |
| PrivBERT | 0.55 | 0.792053 | 0.793778 | +0.001725 |

The training-side ranking is unchanged after global threshold selection: PrivBERT ranks first, followed by RoBERTa, LegalBERT and BERT. Improvements over 0.50 are small on the out-of-fold training predictions, and LegalBERT selects 0.50 exactly.

## Data Retention

| Model | Selected Data Retention threshold | Out-of-fold Data Retention F1 |
|---|---:|---:|
| BERT | 0.51 | 0.447552 |
| LegalBERT | 0.53 | 0.604027 |
| RoBERTa | 0.55 | 0.558659 |
| PrivBERT | 0.55 | 0.651163 |

For Data Retention, PrivBERT has the highest out-of-fold F1, followed by LegalBERT, RoBERTa and BERT. This is training-side evidence only. The survival of the main Data Retention conclusion must be assessed from the held-out results produced after applying these frozen thresholds.

## Interpretation boundary

The screenshots demonstrate that threshold selection no longer uses the evaluation set. They support reporting the selected thresholds and training-side out-of-fold performance. Final claims about model rankings and Data Retention require `heldout_results.json` for all four models or uncropped screenshots of the held-out evaluation tables.
