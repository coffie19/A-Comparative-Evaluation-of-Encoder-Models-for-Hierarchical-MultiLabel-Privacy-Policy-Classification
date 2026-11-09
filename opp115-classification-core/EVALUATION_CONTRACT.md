# Shared category and hierarchical evaluation contract

PrivBERT, BERT, RoBERTa, and LegalBERT use the supplied PrivBERT evaluation
definition. Only the encoder and tokenizer differ between the four full runs.

## Dataset and targets

- Corpus: OPP-115, aggregated by `(policy filename stem, segment_id)`
- Split: 65 complete policies for training and 50 complete policies for final
  held-out evaluation
- Segments: 2,352 training and 1,440 evaluation
- Category outputs: 10 canonical OPP-115 categories
- Fine-grained outputs: 20 category-scoped attribute heads containing 138
  training-derived attribute values
- Hierarchy schema SHA-256:
  `90627a25efe47a217e03a2f09431b3256bc254867cd8f9aee675a1eafbb93b9b`
- Prediction threshold: probability greater than or equal to 0.5

The attribute vocabulary is derived from the 65 training policies and then
frozen. The held-out-only value `do_not_track_policy=honored` is reported as
out of vocabulary and does not change the output dimensions.

## Training loss

Positive-class weights are computed from training examples only:

`sqrt((negative + 1) / (positive + 1))`, clipped to `[1, 10]`.

Attribute weights and attribute losses use only rows where the annotated parent
category makes that head applicable. The total classifier loss is category BCE
plus `1.0` times the mean BCE across active attribute heads. The final reported
evaluation loss is unweighted, matching the supplied reference.

## Category evaluation

The category block reports macro-F1, mean label-wise balanced accuracy, and
per-category precision, recall, F1, balanced accuracy, positive support, and
negative support for all 10 categories.

Balanced accuracy for a label is `(sensitivity + specificity) / 2`. A label
without both positive and negative ground-truth examples has undefined balanced
accuracy and is excluded from the aggregate balanced-accuracy mean.

## Hierarchical evaluation

The evaluator reports both independent attribute predictions and predictions
gated by the predicted parent categories. The primary
`hierarchical_macro_f1` is the equal-weight mean of:

1. the individual F1 scores of all 10 category labels; and
2. the individual F1 scores of every gated attribute value whose head has at
   least one applicable held-out example.

This is the supplied PrivBERT project's custom hierarchical macro-F1. It is not
a path-based hierarchical precision/recall metric and is not the arithmetic
mean of category macro-F1 and fine-grained macro-F1.

Fine-grained evaluation uses annotated parent-category applicability masks,
including after predicted-category gating. Predictions made on rows where a
head is not annotated as applicable do not enter that head's metrics.

## Required full-run evidence

A reportable run must use every segment from the fixed 65/50 policy split,
complete the configured full training schedule, and save category, gated
fine-grained, independent-head, and hierarchical metrics. One-step smoke
outputs are execution checks and cannot be reported as experimental results.
