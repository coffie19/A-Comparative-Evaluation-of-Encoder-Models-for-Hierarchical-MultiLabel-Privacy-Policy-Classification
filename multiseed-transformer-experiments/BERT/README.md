# BERT: matched five-seed OPP-115 study

Open `BERT_multiseed_colab.ipynb` in Google Colab, choose a GPU runtime, and run all cells.
Upload `BERT_multiseed_workspace.zip` when prompted. Use the same study name across all models.

Seeds: [17011, 29023, 41047, 53069, 65089]. Each fine-tuning run starts from `google-bert/bert-base-uncased` at revision `86b5e0934494bd15c9632b12f734a8a67f723594`.
Optional additional MLM pretraining is **not** part of this matched experiment.
The standalone trainer now defaults to seed 17011; the study driver explicitly passes every requested seed.

Each seed uses the original full recipe: 384 tokens, batch size 8, accumulation 4, five epochs,
learning rate 2.5e-5, weight decay .01, warmup .1, dropout .15, fixed threshold .5.
The fixed 65/50 policy split has 2,352 training and 1,440 held-out segments.
All five seeds must complete 370 optimizer steps each before a full report is generated.

Local CUDA use (install `requirements_bert_opp115.txt` or the provided Colab requirements plus compatible PyTorch):

```sh
python multiseed.py --output-root /path/to/shared-study
python multiseed.py --output-root /path/to/shared-study --aggregate-only
python compare_models.py --output-root /path/to/shared-study
```

`--prepare-only` validates data and saves the study specification without training.
`--mode smoke` runs five one-step checks using a tiny checkpoint; scores are explicitly non-reportable.
An interrupted seed restarts from its original checkpoint. Completed seeds are validated and reused.
Use a new output root when changing seed lists, data, source code, or training settings.
Do not run two concurrent copies of this model's notebook against the same study directory.

Results: `full/BERT/seed_<seed>/summary.json`, `training.log`, and `checkpoint/`; aggregate
`per_seed.csv`, `completed_report.json`, and `report.md`. The combined report is `full/comparison.md`.
Mean ± sample standard deviation (ddof=1) is reported separately for category Macro-F1,
hierarchical Macro-F1, and gated fine-grained Macro-F1. These are descriptive seed-variation
statistics, not a confidence interval or a significance claim. No experimental results are included.

The original custom metric, gold applicability masking, training-only attribute vocabulary,
and ten cross-split repeated evaluation texts are retained for comparison continuity.
The reported evaluation loss remains the original batch-dependent statistic; it is not aggregated
as an experiment outcome. Identical seed numbers do not imply identical tensors across architectures.

Tests: `python -m unittest discover -s tests -p "test_*.py"`.
Rebuild this archive after source edits with `python build_workspace.py`.
