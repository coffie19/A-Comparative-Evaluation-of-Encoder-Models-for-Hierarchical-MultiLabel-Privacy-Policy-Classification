# Four-model, five-seed OPP-115 experiment

This is a new copy of the four model codebases, prepared for **20 full fine-tuning runs**.
The original `rematch` folder is unchanged. **Full training has not been performed for this delivery.**
No experimental mean or standard deviation is pre-filled.

Use these identical seeds for every model: **17011, 29023, 41047, 53069, 65089**.
Each seed starts independently from that model's pinned original pretrained checkpoint. Seeds are
passed explicitly to Python, NumPy, PyTorch, CUDA, and the training-data shuffle generator.
The driver also sets `PYTHONHASHSEED` and `CUBLAS_WORKSPACE_CONFIG` before each child process starts.
Deterministic settings from the original trainer are retained; exact bitwise reproducibility across
hardware/library versions is not guaranteed. Equal seed numbers do not mean equal tensors across architectures.

## Run in Colab

1. Open the notebook for each model:
   - `BERT/BERT_multiseed_colab.ipynb`
   - `PrivBERT/PrivBERT_multiseed_colab.ipynb`
   - `RoBERTa/RoBERTa_multiseed_colab.ipynb`
   - `LegalBERT/LegalBERT_multiseed_colab.ipynb`
2. Select a GPU runtime. Run all cells and upload that model's `*_multiseed_workspace.zip` when prompted.
3. Keep `MODE = 'full'` and the same `STUDY_NAME = 'matched_five_seeds_v1'` in all four notebooks.
4. Each notebook runs five seeds in sequence. Finish all four notebooks; they can be run in separate GPU sessions.
5. Run the combined-report cell after all four models complete. It generates `comparison.md`, `comparison.csv`, and `comparison.json`.

All artifacts persist under:

`MyDrive/OPP115_MultiSeed/studies/matched_five_seeds_v1/full/`

The layout is `MODEL/seed_SEED/` for checkpoints, logs, and raw summaries, followed by per-model
`per_seed.csv`, `completed_report.json`, and `report.md`. `comparison.*` files appear at the `full/` level.
Training logs are written to each seed folder's `training.log`. Completed seeds are validated and reused
after a Colab restart. Interrupted seeds restart from their original pretrained checkpoint.
Do not run two copies of the same model against the same study name concurrently.

## What gets reported

| Model | Category Macro-F1 | Hierarchical Macro-F1 | Gated fine-grained Macro-F1 | Required full seeds |
|---|---|---|---|---:|
| BERT | Pending | Pending | Pending | 5 |
| PrivBERT | Pending | Pending | Pending | 5 |
| RoBERTa | Pending | Pending | Pending | 5 |
| LegalBERT | Pending | Pending | Pending | 5 |

Every score is reported as **mean ± sample standard deviation**, with `ddof=1`, on the 0–1 scale.
The denominator for variance is `n - 1`; this measures variation across independent fine-tuning seeds
on one fixed test set. It is not a standard error, confidence interval, or significance test.

All five seeds must finish before a full aggregate is generated. The original primary outcome remains
hierarchical Macro-F1; category Macro-F1 is a separately labeled outcome. The combined JSON also gives
paired PrivBERT-minus-LegalBERT differences for every seed. The software does not declare a model superior
from a single run or automatically equate a larger mean with a statistically reliable improvement.

## Fixed experimental conditions

- Same whole-policy 65/50 split, 2,352 training / 1,440 held-out segments, and training-derived hierarchy schema.
- Same five seed identifiers across all four models, five epochs per seed, 370 optimizer updates per full run.
- Same maximum length 384, batch size 8, gradient accumulation 4, learning rate 2.5e-5, weight decay .01,
  warmup ratio .1, dropout .15, threshold .5, and original positive-class weighting.
- Extra domain-adaptive MLM pretraining is disabled for this experiment.
- Original custom metric, gold applicability masks, OOV handling, and ten repeated cross-split evaluation texts are retained.
- Original batch-dependent evaluation loss is not used as an aggregated comparison outcome.

The runner validates the actual saved seed, checkpoint revision, hyperparameters, dataset coverage,
schema, and completed optimizer steps. It rejects one-step results labeled as full runs. Resume receipts
bind each summary to its study specification and hash. Comparison requires matching seed lists, data
fingerprints, shared implementation hashes, and experiment settings across all four models.

`--mode smoke` uses tiny checkpoints for five one-step checks and writes to a separate `smoke/` tree.
Its reports are marked non-reportable and are never included in the full comparison.

## Local use and rebuilding

In each model folder, install its requirements in a compatible CUDA Python environment and run:

```sh
python multiseed.py --output-root /path/to/shared-study
```

Use the same absolute output root across all model folders. After all four complete, from any model folder or the top-level folder:

```sh
python compare_models.py --output-root /path/to/shared-study
```

`python multiseed.py --output-root /path/to/shared-study --prepare-only` checks the real data without training.
`--aggregate-only` reconstructs a report from all completed, verified seed summaries.
Use a new output root if code, data, seeds, or settings change. After source changes, rebuild the corresponding
portable ZIP with `python build_workspace.py` in that model folder.

`SOURCE_PROVENANCE.json` records the original archives. `VALIDATION_REPORT.md` describes the checks performed
for this edited delivery. `EXPERIMENT_STATUS.json` records that the actual full GPU experiments remain pending.
