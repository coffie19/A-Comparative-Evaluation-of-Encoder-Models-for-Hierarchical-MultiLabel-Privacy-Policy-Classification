# Validation of the new multi-seed codebases

The edited packages are prepared and tested. **No full GPU training runs were executed.**
The required experiment comprises 20 runs: five seeds for each of four models.

| Model | Unique unit tests verified | New multi-seed tests | Dataset and command checks | Archive integrity |
|---|---:|---:|---|---|
| BERT | 52 | 11 | Passed | Passed |
| PrivBERT | 52 | 11 | Passed | Passed |
| RoBERTa | 52 | 11 | Passed | Passed |
| LegalBERT | 52 | 11 | Passed | Passed |

The original applicable 41 source/data tests and 11 new multi-seed tests passed for every model
(208 distinct model/test combinations). The first suite invocation included nine new tests; the
expanded 11-test multi-seed suite was subsequently run for every model. Superseded single-seed
notebook-bundle tests were replaced by direct new-notebook parsing and ZIP/manifest checks.

The new tests verify sample standard deviation (ddof=1), rejection of duplicate/single seeds,
rejection of shortened training, incorrect seeds/revisions/hyperparameters, invalid metric values,
missing completions, modified summaries, mismatched cross-model seeds, and smoke/full mixing.
They also exercise combined reports and paired differences, explicit seed/checkpoint arguments,
all-five-seed scheduling, completed-seed reuse, and failure without writing a completion marker.
Neural training in orchestration tests is simulated using temporary fixtures. Those synthetic
scores are never shipped as research results.

Independent checks confirmed:

- All four real-data preparation commands completed successfully without training.
- The study specifications have identical seed lists, ordered policy splits, instance/text/label
  fingerprints, schema, hyperparameters, expected optimizer counts, and metric contracts.
- Every seed's generated command parses successfully using the corresponding real trainer's parser.
- All Python files and notebook code cells parse; notebook outputs are empty.
- Every portable archive passes CRC checks, and every manifest payload hash/size matches.
- All 231 bundled data files remain byte-for-byte identical to the corresponding original archive.
- Existing classification/pretraining source changes are limited to seed defaults from 42 to 17011;
  the multi-run driver supplies each of the five study seeds explicitly.

The runtime used for tests provides Python and a review-local scikit-learn installation. No pretrained
weights were downloaded, and no new PyTorch/Transformers forward/backward, Colab GPU execution, or
checkpoint roundtrip was performed. The code now supports the experiment; actual mean ± standard
deviation values will be produced only after the full GPU runs finish and validate.

See `VALIDATION_REPORT.json` for archive hashes and per-model checks, and `EXPERIMENT_STATUS.json`
for the pending training status.
