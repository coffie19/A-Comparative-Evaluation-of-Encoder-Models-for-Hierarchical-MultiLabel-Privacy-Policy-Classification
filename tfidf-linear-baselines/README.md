# TF-IDF + logistic regression baseline

Open **results/report.html** for the complete comparison with the four rematch transformer checkpoints and the original Harkous et al. Polisis paper. Results are real trained-model results, not examples.

Original 1,440-segment evaluation: category Macro-F1 0.7281876381; fine-grained Macro-F1 0.4498786393. Selected C=1 using three policy-grouped folds drawn only from the 65 training policies. Word and character TF-IDF features are fitted separately inside every fold. The final model is results/baseline.joblib. A roundtrip check reproduced every probability and independent sklearn scoring verified both headline metrics.

The baseline exceeds the supplied transformer checkpoints on fine-grained prediction but not category Macro-F1. This compares complete implementations with different weighting/training procedures; it is not a causal proof of architectural superiority. No multi-seed uncertainty is claimed..

## Run

Use a Python environment compatible with the pinned dependencies (the completed run used the Python version recorded in package_manifest.json). Open a terminal in this folder:

```powershell
python -m pip install -r requirements.txt
python run_baseline.py --data-dir data/opp115 --output-dir rerun_results
python verify_saved.py --results-dir rerun_results --schema hierarchy_schema.json
python compare_results.py --baseline-dir rerun_results --transformer-predictions transformer_predictions --exclusions exclusions.json --output-dir rerun_results/comparison --schema hierarchy_schema.json
python build_report.py --results-dir rerun_results
```

The last command regenerates the canonical artifact.json and report.md. The already delivered results/report.html is self-contained. Creating a new HTML from an updated artifact requires the Data Analytics portable report builder used in the original workspace; that renderer is not part of this model package. The baseline training and metric comparison do not require that renderer, PyTorch, transformers, a GPU or network access after installing dependencies.

The run trains 148 independent binary classifiers (ten categories and 138 attribute values), with gold-applicable rows for attribute training and predicted-parent gating for evaluation. Fixed threshold 0.5; balanced binary class weights; L2 regularization; solver seed 1729. Full segment texts are used. TF-IDF has two separately normalized feature branches. CV uses a fixed outer-training-derived label vocabulary, while feature vocabularies/IDF are fitted on inner training folds only. Inspect results/results.json for full parameters, fold scores, timing and sklearn version.

## Comparison files

- results/comparison/matched_comparison.csv: identical-cohort five-model scores.
- results/comparison/category_comparison.csv and attribute_comparison.csv: full per-category and per-head results.
- results/comparison/transformer_gains_over_baseline.csv: signed gains, including duplicate-filtered evaluation.
- results/comparison/paper_category_results.csv and paper_attribute_results.csv: original paper values with PDF page numbers.
- results/comparison/paper_metric_aligned_comparison.csv: presence/absence F1 diagnostic with mismatched supports explicitly shown.
- transformer_predictions/: saved local checkpoint predictions needed to regenerate the comparison; original encoder weights are not included.
- exclusions.json: fixed audit exclusion lists. The filtered comparison removes 22 segments including one within-evaluation normalized duplicate and lexical near candidates.

The paper reports 0.84 for a nine-category query classifier and 0.81 for a twelve-category segment classifier. Our rematch headlines are ten-category positive-label Macro-F1, so those values are not directly comparable. The appendix gives fifteen selected attribute results, not a global score over our 138 values. No exact Polisis CNN reproduction is claimed. The supplied source PDF is included in references/original_paper.pdf.

HTML validation and structural checks passed. Browser layout and source-dialog verification were unavailable; semantic chart tables remain included. The original rematch codebases and checkpoints were not modified.
