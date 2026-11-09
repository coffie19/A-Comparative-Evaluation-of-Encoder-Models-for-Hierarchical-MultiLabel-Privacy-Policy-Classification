# RoBERTa + OPP-115 hierarchical classification

This package fine-tunes [`FacebookAI/roberta-base`](https://huggingface.co/FacebookAI/roberta-base)
on the two-level annotation structure of the
[OPP-115 corpus](https://aclanthology.org/P16-1126/). Paragraph-sized privacy
policy segments can have several high-level data-practice categories and
several fine-grained attribute values.

The full workflow starts from the RoBERTa-base checkpoint and pins Hugging Face revision
`e2da8e2f811d1448a5b465c236feacd80ffbac7b` and verifies that the loaded
checkpoint reports `model_type="roberta"`.

See the linked model card for checkpoint provenance and licensing.

## Task and model

This is not a flat 20-label classifier. One shared RoBERTa encoder feeds:

- one multi-label head for the 10 canonical OPP-115 categories; and
- 20 category-scoped, multi-label attribute heads.

Each attribute head predicts its own value vocabulary. On the fixed 65-policy
training split, those vocabularies contain 138 value dimensions in total.
Thus, `20` is the number of attribute classifiers, not the number of
fine-grained output labels.

The classifier takes RoBERTa's contextual first-token (`<s>`) representation,
then applies a shared dense/tanh projection and 0.15 classifier dropout before
the category and attribute heads. It intentionally does not use a pooler that
was not trained by RoBERTa's masked-language-model objective.

The hierarchy follows the task described by
[Polisis](https://www.usenix.org/system/files/conference/usenixsecurity18/sec18-harkous.pdf),
but this package uses a RoBERTa encoder and loss-masked hierarchy heads rather
than reproducing the original Polisis architecture.

## Label and split contract

- Annotation rows and annotators are unioned by
  `(annotation filename stem, segment_id)`, producing one multi-label example
  per policy segment.
- The split is deterministic and uses whole policies: 65 for training and 50
  for final held-out evaluation. No segment from an evaluation policy enters
  training.
- Attribute value vocabularies are derived from the 65 training policies only
  and then frozen. The held-out policies cannot add output dimensions.
- `not-selected` is ignored, while `Unspecified` remains a normalized value.
- On the bundled split, the schema contains 138 fine-grained value dimensions.
  The held-out-only value `do_not_track_policy=honored` is reported as out of
  vocabulary and does not resize a head.
- The prediction threshold is fixed at 0.5. It is not tuned on the final
  50-policy evaluation set.
- Positive-class weights are computed from training examples only. Attribute
  weights use only rows where the corresponding head applies; square-root
  inverse prevalence is capped at 10.

The saved `hierarchy_schema.json` records ordered categories, attribute heads,
per-head training vocabularies, training policy IDs, and a SHA-256 fingerprint.
Reloading rejects stale or incompatible schemas.

## Category conditioning and metrics

During training and evaluation, an attribute head contributes loss and metrics
only when its annotated parent category applies. At inference time, a head is
suppressed unless at least one predicted parent category applies. Reports also
retain independent, ungated attribute-head metrics so category errors remain
visible.

The primary result is `hierarchical_macro_f1`, calculated across the 10
category outputs and category-gated fine-grained value outputs. Reports also
include category macro-F1, gated and independent fine-grained macro-F1,
per-label results, and balanced accuracy where both outcomes are defined.

## The 0.96 target

`0.96` is an aspirational held-out hierarchical macro-F1 target, not a promised
or pre-filled result. A full GPU run must produce any claimed score.

The bundle contains data-validation and one-step smoke evidence only. Smoke
metrics are not research results. Do not repeatedly tune against the final 50
policies to chase 0.96; report the observed score even when it is lower.

## Google Colab workflow

1. Upload `roberta_opp115_colab_workspace.zip` to
   `MyDrive/ROBERTA_OPP115_Colab/`, or select it when prompted.
2. Open `roberta_opp115_colab.ipynb` in Google Colab.
3. Select **Runtime > Change runtime type > GPU**.
4. Leave `MODE = "full"`, choose a new stable `RUN_NAME`, and run all cells.

Full mode starts directly from pinned RoBERTa. `RUN_PRETRAINING = False`
preserves the reference experiment's training procedure. Optional
continued MLM training uses only the 65 training policies and can be enabled as
a controlled comparison.

Smoke mode uses pinned `hf-internal-testing/tiny-random-roberta` for one
optimizer step and must not be reported as a model result.

The full classifier uses 384-token maximum inputs with dynamic batch padding,
batch size 8, four-step gradient accumulation (effective batch 32), five
epochs, learning rate `2.5e-5`, weight decay `0.01`, warmup ratio `0.1`, and
classifier dropout `0.15`. Encoder and hierarchy-head checkpoints are saved as
safetensors.

Persistent run artifacts are written below
`MyDrive/ROBERTA_OPP115_Colab/runs/<RUN_NAME>/`. The classifier directory is named
`roberta-opp115-hierarchy-classifier`.

## Local validation

From this package directory, in a separate environment with
`requirements_roberta_opp115.txt` installed:

```powershell
python roberta_opp115_pretraining.py --data-dir data/opp115 --validate-data-only
python roberta_opp115_classification.py --data-dir data/opp115 --validate-data-only
python -m unittest discover -s tests -p "test_*.py"
```

The bundled `data/opp115` directory contains 115 annotation CSVs, 115 sanitized
policy HTML files, and the original OPP-115 attribution file. Both model stages
use the same deterministic whole-policy split.

## Main files

- `roberta_opp115_pretraining.py`: optional continued MLM training on the 65
  training policies.
- `roberta_opp115_classification.py`: shared RoBERTa encoder, 10-category head,
  20 masked attribute heads, gated evaluation, saving, and reloading.
- `opp115_hierarchy.py`: annotation normalization, aggregation, hierarchy
  applicability, training-only schema derivation, encoding, and OOV reporting.
- `roberta_opp115_colab.ipynb`: end-to-end Colab workflow.
- `roberta_opp115_colab_workspace.zip`: portable code, tests, data, and manifest.
- `tests/`: taxonomy, leakage, schema, model-family, metric, and bundle checks.

Completion markers support reuse after a Colab restart. They are stage-level
markers, not mid-epoch checkpoints; rerun an interrupted stage if no completion
marker was written.
