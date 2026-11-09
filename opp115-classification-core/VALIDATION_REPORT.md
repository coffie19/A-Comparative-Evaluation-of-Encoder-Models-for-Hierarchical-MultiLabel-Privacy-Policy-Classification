# Four-model evaluation parity validation

Overall result: **PASSED**

- Passed checks: 29
- Failed checks: 0
- Skipped checks: 0

## Evidence

- Shared evaluator, hierarchy, CSV loader, and policy split files are compared byte-for-byte by SHA-256.
- Core threshold, weighting, loss, attribute metric, gating, and whole-model evaluation functions are compared by exact extracted source and normalized Python AST.
- Numerical checks run in isolated processes against the same deterministic adversarial fixture using real PyTorch.
- The fixture covers category errors, a shared-parent gate, gold applicability masks, empty attribute heads, threshold equality, and undefined balanced accuracy.

## Checks

| Check | Status |
|---|---|
| shared bytes: evaluation.py | passed |
| shared bytes: opp115_hierarchy.py | passed |
| shared bytes: opp115_csv_loader.py | passed |
| shared bytes: opp115_policy_split.py | passed |
| core exact source: threshold_probabilities | passed |
| core normalized AST: threshold_probabilities | passed |
| core exact source: compute_positive_class_weights | passed |
| core normalized AST: compute_positive_class_weights | passed |
| core exact source: _hierarchical_loss | passed |
| core normalized AST: _hierarchical_loss | passed |
| core exact source: _empty_attribute_metrics | passed |
| core normalized AST: _empty_attribute_metrics | passed |
| core exact source: compute_attribute_metrics | passed |
| core normalized AST: compute_attribute_metrics | passed |
| core exact source: gate_attribute_predictions | passed |
| core normalized AST: gate_attribute_predictions | passed |
| core exact source: _evaluate_model | passed |
| core normalized AST: _evaluate_model | passed |
| real PyTorch fixture executes: PrivBERT | passed |
| adversarial fixture semantics: PrivBERT | passed |
| real PyTorch fixture executes: BERT | passed |
| adversarial fixture semantics: BERT | passed |
| real PyTorch fixture executes: RoBERTa | passed |
| adversarial fixture semantics: RoBERTa | passed |
| real PyTorch fixture executes: LegalBERT | passed |
| adversarial fixture semantics: LegalBERT | passed |
| numerical parity: BERT | passed |
| numerical parity: RoBERTa | passed |
| numerical parity: LegalBERT | passed |

The full hashes, runtime versions, metrics, losses, weights, and worker errors (if any) are in `VALIDATION_REPORT.json`.
