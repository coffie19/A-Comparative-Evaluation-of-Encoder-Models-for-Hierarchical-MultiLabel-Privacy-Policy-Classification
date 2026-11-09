#!/usr/bin/env python
"""Fine-tune RoBERTa for hierarchical, fine-grained OPP-115 classification.

The classifier follows the two levels described by Polisis: one multi-label
head predicts the ten OPP-115 categories and twenty category-scoped,
loss-masked heads predict the values of the genuine privacy attributes.  All
annotation rows and annotators are unioned into one example per policy segment.
The output vocabulary is frozen from the 65 training policies before the 50
held-out policies are encoded, preventing label-space leakage.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from evaluation import compute_multilabel_classification_metrics
from opp115_csv_loader import PolicyTextExtractor
from opp115_hierarchy import (
    ATTRIBUTE_HEADS,
    CATEGORY_TO_ATTRIBUTES,
    HIGH_LEVEL_CATEGORIES,
    EncodedSegmentLabels,
    HierarchySchema,
    SegmentAnnotation,
    attribute_mask_for_categories,
    derive_hierarchy_schema,
    encode_segment_labels,
    fixed_policy_split,
    load_segment_annotations,
)
from opp115_policy_split import DEFAULT_EVAL_POLICIES, DEFAULT_TRAIN_POLICIES


LOGGER = logging.getLogger("roberta_opp115_classification")
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OPP115_DIR = PROJECT_ROOT / "data" / "opp115"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "roberta-opp115-hierarchy-classifier"
DEFAULT_MODEL_NAME = "FacebookAI/roberta-base"
EXPECTED_MODEL_TYPE = "roberta"
CLASSIFIER_FORMAT_VERSION = 3
POOLING_STRATEGY = "roberta_first_token_dense_tanh"


@dataclass(frozen=True)
class SegmentExample:
    policy_id: str
    segment_id: str
    text: str
    category_labels: tuple[int, ...]
    attribute_labels: tuple[int, ...]
    attribute_mask: tuple[int, ...]
    unknown_attribute_values: tuple[str, ...] = ()

    @property
    def labels(self) -> tuple[int, ...]:
        """Backward-compatible name for the category targets."""

        return self.category_labels


def _segment_sort_key(example: SegmentExample) -> tuple[str, int, int | str]:
    try:
        return example.policy_id.casefold(), 0, int(example.segment_id)
    except ValueError:
        return example.policy_id.casefold(), 1, example.segment_id.casefold()


def _unknown_labels(encoded: EncodedSegmentLabels) -> tuple[str, ...]:
    return tuple(
        f"{head}={value}"
        for head in ATTRIBUTE_HEADS
        for value in encoded.unknown_attribute_values.get(head, ())
    )


def _make_example(
    segment: SegmentAnnotation,
    text: str,
    schema: HierarchySchema,
) -> SegmentExample | None:
    cleaned_text = re.sub(r"\s+", " ", str(text)).strip()
    if not cleaned_text or not segment.categories:
        return None
    encoded = encode_segment_labels(segment, schema, unknown_values="ignore")
    return SegmentExample(
        policy_id=segment.policy_id,
        segment_id=segment.segment_id,
        text=cleaned_text,
        category_labels=encoded.category_targets,
        attribute_labels=encoded.flattened_attribute_targets,
        attribute_mask=encoded.mask_vector,
        unknown_attribute_values=_unknown_labels(encoded),
    )


def group_annotation_samples(
    samples: Iterable[Any], schema: HierarchySchema
) -> list[SegmentExample]:
    """Union practice-like objects into one encoded example per segment.

    This small compatibility helper is useful for synthetic tests.  Real data
    loading uses :func:`load_segment_examples`, which reads the authoritative
    raw JSON attributes through ``opp115_hierarchy.py``.
    """

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for sample in samples:
        key = (str(sample.policy_id), str(sample.segment_id))
        text = re.sub(r"\s+", " ", str(sample.text)).strip()
        if not text:
            continue
        record = grouped.setdefault(
            key,
            {"text": text, "categories": set(), "attributes": {}},
        )
        if len(text) > len(record["text"]):
            record["text"] = text
        record["categories"].update(
            category
            for category in getattr(sample, "categories", ())
            if category in HIGH_LEVEL_CATEGORIES
        )
        for head, values in getattr(sample, "attributes", {}).items():
            if head not in ATTRIBUTE_HEADS:
                continue
            record["attributes"].setdefault(head, set()).update(values)

    examples: list[SegmentExample] = []
    for (policy_id, segment_id), record in grouped.items():
        segment = SegmentAnnotation(
            policy_id=policy_id,
            segment_id=segment_id,
            categories=tuple(record["categories"]),
            attributes={
                head: tuple(values)
                for head, values in record["attributes"].items()
            },
            attribute_mask=attribute_mask_for_categories(record["categories"]),
        )
        example = _make_example(segment, record["text"], schema)
        if example is not None:
            examples.append(example)
    return sorted(examples, key=_segment_sort_key)


def load_segment_examples(
    data_dir: Path, schema: HierarchySchema
) -> list[SegmentExample]:
    """Load raw OPP-115 annotations and aligned paragraph-sized text."""

    data_dir = data_dir.expanduser().resolve()
    annotations_dir = data_dir / "annotations"
    policies_dir = data_dir / "sanitized_policies"
    if not annotations_dir.is_dir() or not policies_dir.is_dir():
        raise FileNotFoundError(
            f"Expected annotations/ and sanitized_policies/ below {data_dir}"
        )

    segments = load_segment_annotations(annotations_dir, strict=True)
    examples: list[SegmentExample] = []
    for segment in segments.values():
        policy_path = policies_dir / f"{segment.policy_id}.html"
        if not policy_path.is_file():
            raise FileNotFoundError(f"Missing sanitized policy: {policy_path}")
        try:
            segment_index = int(segment.segment_id)
        except ValueError as exc:
            raise ValueError(
                f"OPP-115 segment ID is not an integer: {segment.key!r}"
            ) from exc
        text = PolicyTextExtractor.get_segment_text(policy_path, segment_index)
        example = _make_example(segment, text, schema)
        if example is None:
            raise ValueError(f"Empty labeled OPP-115 segment: {segment.key!r}")
        examples.append(example)
    if not examples:
        raise RuntimeError(f"No labeled OPP-115 segments loaded from {data_dir}")
    return sorted(examples, key=_segment_sort_key)


def split_segment_examples(
    examples: Sequence[SegmentExample],
    *,
    train_count: int = DEFAULT_TRAIN_POLICIES,
    eval_count: int = DEFAULT_EVAL_POLICIES,
) -> tuple[list[SegmentExample], list[SegmentExample], tuple[str, ...], tuple[str, ...]]:
    """Apply the exact deterministic whole-policy split."""

    policy_ids = tuple(
        sorted(
            {example.policy_id for example in examples},
            key=lambda value: (value.casefold(), value),
        )
    )
    if len(policy_ids) != train_count + eval_count:
        raise ValueError(
            f"Exact {train_count}/{eval_count} split requires "
            f"{train_count + eval_count} policies; found {len(policy_ids)}."
        )
    train_ids = policy_ids[:train_count]
    eval_ids = policy_ids[train_count:]
    train_set = set(train_ids)
    eval_set = set(eval_ids)
    train_examples = [e for e in examples if e.policy_id in train_set]
    eval_examples = [e for e in examples if e.policy_id in eval_set]
    if {e.policy_id for e in train_examples} & {e.policy_id for e in eval_examples}:
        raise AssertionError("Policy leakage between train and evaluation examples")
    return train_examples, eval_examples, train_ids, eval_ids


def threshold_probabilities(
    probabilities: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    probabilities = np.asarray(probabilities)
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    return (probabilities >= threshold).astype(np.int64)


def _sample_examples(
    examples: Sequence[SegmentExample], maximum: int, seed: int
) -> list[SegmentExample]:
    if maximum <= 0 or maximum >= len(examples):
        return list(examples)
    selected = random.Random(seed).sample(list(examples), maximum)
    return sorted(selected, key=_segment_sort_key)


def _category_support(examples: Sequence[SegmentExample]) -> dict[str, int]:
    support = np.asarray([example.category_labels for example in examples]).sum(axis=0)
    return {
        category: int(support[index])
        for index, category in enumerate(HIGH_LEVEL_CATEGORIES)
    }


def _attribute_support(
    examples: Sequence[SegmentExample], schema: HierarchySchema
) -> dict[str, dict[str, int]]:
    matrix = np.asarray([example.attribute_labels for example in examples])
    output: dict[str, dict[str, int]] = {}
    for head in ATTRIBUTE_HEADS:
        start, stop = schema.attribute_slices[head]
        support = matrix[:, start:stop].sum(axis=0)
        output[head] = {
            value: int(support[index])
            for index, value in enumerate(schema.value_vocabulary[head])
        }
    return output


def _attribute_mask_support(
    examples: Sequence[SegmentExample],
) -> dict[str, int]:
    mask = np.asarray([example.attribute_mask for example in examples])
    return {
        head: int(mask[:, index].sum())
        for index, head in enumerate(ATTRIBUTE_HEADS)
    }


def _unknown_value_support(
    examples: Sequence[SegmentExample],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in examples:
        for label in example.unknown_attribute_values:
            counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def dataset_summary(
    examples: Sequence[SegmentExample],
    train_examples: Sequence[SegmentExample],
    eval_examples: Sequence[SegmentExample],
    train_ids: Sequence[str],
    eval_ids: Sequence[str],
    schema: HierarchySchema,
) -> dict[str, Any]:
    if tuple(train_ids) != schema.training_policy_ids:
        raise ValueError("The hierarchy schema was not derived from this training split")
    train_texts = {
        re.sub(r"\s+", " ", example.text).strip().casefold()
        for example in train_examples
    }
    duplicate_eval_segments = sum(
        re.sub(r"\s+", " ", example.text).strip().casefold() in train_texts
        for example in eval_examples
    )
    return {
        "corpus": "OPP-115",
        "task": "hierarchical fine-grained segment classification",
        "category_count": len(HIGH_LEVEL_CATEGORIES),
        "attribute_head_count": len(ATTRIBUTE_HEADS),
        "fine_grained_label_count": schema.flattened_value_count,
        "categories": list(HIGH_LEVEL_CATEGORIES),
        "attribute_heads": list(ATTRIBUTE_HEADS),
        "label_schema": schema.to_manifest(),
        "segments": len(examples),
        "multi_label_segments": sum(
            sum(example.category_labels) > 1 for example in examples
        ),
        "train_segments": len(train_examples),
        "eval_segments": len(eval_examples),
        "train_policies": len(train_ids),
        "eval_policies": len(eval_ids),
        "train_policy_ids": list(train_ids),
        "eval_policy_ids": list(eval_ids),
        "train_eval_policy_ids_disjoint": not (set(train_ids) & set(eval_ids)),
        "split_scope": "whole policy IDs",
        "cross_split_duplicate_eval_segments": duplicate_eval_segments,
        "category_support": _category_support(examples),
        "train_category_support": _category_support(train_examples),
        "eval_category_support": _category_support(eval_examples),
        "attribute_support": _attribute_support(examples, schema),
        "train_attribute_support": _attribute_support(train_examples, schema),
        "eval_attribute_support": _attribute_support(eval_examples, schema),
        "attribute_applicable_segments": _attribute_mask_support(examples),
        "train_attribute_applicable_segments": _attribute_mask_support(train_examples),
        "eval_attribute_applicable_segments": _attribute_mask_support(eval_examples),
        "held_out_unknown_attribute_values": _unknown_value_support(eval_examples),
    }


class EncodedSegmentDataset:
    def __init__(
        self,
        encodings: Mapping[str, Any],
        category_labels: Any,
        attribute_labels: Any,
        attribute_masks: Any,
    ) -> None:
        self.encodings = dict(encodings)
        self.category_labels = category_labels
        self.attribute_labels = attribute_labels
        self.attribute_masks = attribute_masks

    def __len__(self) -> int:
        return int(self.category_labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["category_labels"] = self.category_labels[index]
        item["attribute_labels"] = self.attribute_labels[index]
        item["attribute_masks"] = self.attribute_masks[index]
        return item


def _choose_device(torch_module: Any, requested: str) -> Any:
    if requested == "auto":
        requested = "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA GPU")
    return torch_module.device(requested)


def validate_roberta_model_type(model: Any) -> None:
    config = getattr(model, "config", None)
    actual_model_type = getattr(config, "model_type", None)
    if actual_model_type != EXPECTED_MODEL_TYPE:
        raise ValueError(
            "Expected RoBERTa with model_type='roberta', but the loaded "
            f"checkpoint reports model_type={actual_model_type!r}."
        )


def _build_model(
    torch_module: Any,
    encoder: Any,
    schema: HierarchySchema,
    classifier_dropout: float = 0.15,
) -> Any:
    class RoBERTaOPP115Hierarchy(torch_module.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            hidden_size = int(encoder.config.hidden_size)
            self.dropout = torch_module.nn.Dropout(classifier_dropout)
            self.pre_classifier = torch_module.nn.Linear(hidden_size, hidden_size)
            self.activation = torch_module.nn.Tanh()
            self.category_head = torch_module.nn.Linear(
                hidden_size, len(HIGH_LEVEL_CATEGORIES)
            )
            self.attribute_heads = torch_module.nn.ModuleDict(
                {
                    head: torch_module.nn.Linear(
                        hidden_size, len(schema.value_vocabulary[head])
                    )
                    for head in ATTRIBUTE_HEADS
                }
            )

        def forward(self, **model_inputs: Any) -> tuple[Any, Any]:
            outputs = self.encoder(**model_inputs)
            # RoBERTa is a RoBERTa masked-language-model checkpoint.  Use its
            # contextual first-token (<s>) state rather than a pooler that was
            # not trained by the MLM objective, then apply the standard
            # RoBERTa-style dense/tanh classification projection.
            pooled = outputs.last_hidden_state[:, 0]
            pooled = self.dropout(pooled)
            pooled = self.pre_classifier(pooled)
            pooled = self.activation(pooled)
            pooled = self.dropout(pooled)
            category_logits = self.category_head(pooled)
            attribute_logits = torch_module.cat(
                [self.attribute_heads[head](pooled) for head in ATTRIBUTE_HEADS],
                dim=-1,
            )
            return category_logits, attribute_logits

    return RoBERTaOPP115Hierarchy()


def compute_positive_class_weights(
    examples: Sequence[SegmentExample],
    schema: HierarchySchema,
    *,
    power: float = 0.5,
    maximum: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return smoothed training-only positive weights for masked BCE.

    Attribute weights are calculated only over examples on which their head
    applies.  Square-root inverse prevalence improves rare-label recall while
    the cap prevents one- or two-example labels from destabilizing training.
    """

    if not examples:
        raise ValueError("positive class weights require training examples")
    if not 0.0 <= power <= 1.0:
        raise ValueError("positive class-weight power must be in [0, 1]")
    if maximum < 1.0:
        raise ValueError("maximum positive class weight must be at least 1")

    def weights_for(targets: np.ndarray) -> np.ndarray:
        targets = np.asarray(targets, dtype=np.float64)
        if targets.ndim != 2 or targets.shape[0] == 0:
            raise ValueError("positive class-weight targets must be a non-empty matrix")
        positives = targets.sum(axis=0)
        negatives = targets.shape[0] - positives
        smoothed_ratio = (negatives + 1.0) / (positives + 1.0)
        return np.clip(smoothed_ratio**power, 1.0, maximum).astype(np.float32)

    category_targets = np.asarray(
        [example.category_labels for example in examples], dtype=np.float32
    )
    category_weights = weights_for(category_targets)

    attribute_weights = np.ones(schema.flattened_value_count, dtype=np.float32)
    for head_index, head in enumerate(ATTRIBUTE_HEADS):
        active_examples = [
            example for example in examples if example.attribute_mask[head_index]
        ]
        if not active_examples:
            continue
        start, stop = schema.attribute_slices[head]
        active_targets = np.asarray(
            [example.attribute_labels[start:stop] for example in active_examples],
            dtype=np.float32,
        )
        attribute_weights[start:stop] = weights_for(active_targets)
    return category_weights, attribute_weights


def _hierarchical_loss(
    category_logits: Any,
    attribute_logits: Any,
    category_labels: Any,
    attribute_labels: Any,
    attribute_masks: Any,
    schema: HierarchySchema,
    torch_module: Any,
    attribute_loss_weight: float,
    category_positive_weights: Any | None = None,
    attribute_positive_weights: Any | None = None,
) -> tuple[Any, Any, Any]:
    binary_cross_entropy = torch_module.nn.functional.binary_cross_entropy_with_logits
    category_loss = binary_cross_entropy(
        category_logits,
        category_labels,
        pos_weight=category_positive_weights,
    )
    head_losses: list[Any] = []
    for head_index, head in enumerate(ATTRIBUTE_HEADS):
        active = attribute_masks[:, head_index].bool()
        if not bool(active.any()):
            continue
        start, stop = schema.attribute_slices[head]
        head_losses.append(
            binary_cross_entropy(
                attribute_logits[active, start:stop],
                attribute_labels[active, start:stop],
                pos_weight=(
                    attribute_positive_weights[start:stop]
                    if attribute_positive_weights is not None
                    else None
                ),
            )
        )
    attribute_loss = (
        torch_module.stack(head_losses).mean()
        if head_losses
        else attribute_logits.sum() * 0.0
    )
    total_loss = category_loss + attribute_loss_weight * attribute_loss
    return total_loss, category_loss, attribute_loss


def _empty_attribute_metrics(head: str, schema: HierarchySchema) -> dict[str, Any]:
    per_label = {
        value: {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "balanced_accuracy": None,
            "positive_support": 0,
            "negative_support": 0,
            "evaluated_examples": 0,
        }
        for value in schema.value_vocabulary[head]
    }
    return {
        "primary_metric": "macro_f1",
        "macro_f1": None,
        "balanced_accuracy": None,
        "balanced_accuracy_definition": (
            "mean over values of (sensitivity + specificity) / 2"
        ),
        "balanced_accuracy_labels_defined": 0,
        "label_count": len(schema.value_vocabulary[head]),
        "evaluated_examples": 0,
        "per_label": per_label,
    }


def compute_attribute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    head_masks: np.ndarray,
    schema: HierarchySchema,
) -> dict[str, Any]:
    """Compute per-head and aggregate fine-grained metrics on applicable rows."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    head_masks = np.asarray(head_masks)
    if y_true.shape != y_pred.shape:
        raise ValueError("Fine-grained truth/prediction shapes differ")
    if y_true.ndim != 2 or y_true.shape[1] != schema.flattened_value_count:
        raise ValueError("Fine-grained matrices do not match the frozen schema")
    if head_masks.shape != (y_true.shape[0], len(ATTRIBUTE_HEADS)):
        raise ValueError("Attribute mask matrix has the wrong shape")

    per_attribute: dict[str, Any] = {}
    per_value: dict[str, Any] = {}
    f1_values: list[float] = []
    balanced_values: list[float] = []
    evaluated_head_instances = 0
    for head_index, head in enumerate(ATTRIBUTE_HEADS):
        active = head_masks[:, head_index].astype(bool)
        evaluated_head_instances += int(active.sum())
        if not active.any():
            metrics = _empty_attribute_metrics(head, schema)
            per_attribute[head] = metrics
            for value, value_metrics in metrics["per_label"].items():
                per_value[f"{head}={value}"] = dict(value_metrics)
            continue
        start, stop = schema.attribute_slices[head]
        values = list(schema.value_vocabulary[head])
        metrics = compute_multilabel_classification_metrics(
            y_true[active, start:stop], y_pred[active, start:stop], values
        )
        metrics["evaluated_examples"] = int(active.sum())
        per_attribute[head] = metrics
        for value in values:
            label = f"{head}={value}"
            value_metrics = dict(metrics["per_label"][value])
            value_metrics["evaluated_examples"] = int(active.sum())
            per_value[label] = value_metrics
            f1_values.append(float(value_metrics["f1"]))
            if value_metrics["balanced_accuracy"] is not None:
                balanced_values.append(float(value_metrics["balanced_accuracy"]))

    return {
        "primary_metric": "fine_grained_macro_f1",
        "fine_grained_macro_f1": float(np.mean(f1_values)) if f1_values else None,
        "balanced_accuracy": (
            float(np.mean(balanced_values)) if balanced_values else None
        ),
        "balanced_accuracy_definition": (
            "mean over fine-grained values of (sensitivity + specificity) / 2"
        ),
        "balanced_accuracy_values_defined": len(balanced_values),
        "attribute_head_count": len(ATTRIBUTE_HEADS),
        "fine_grained_label_count": schema.flattened_value_count,
        "evaluated_head_instances": evaluated_head_instances,
        "per_attribute": per_attribute,
        "per_value": per_value,
    }


def gate_attribute_predictions(
    attribute_predictions: np.ndarray,
    category_predictions: np.ndarray,
    schema: HierarchySchema,
) -> np.ndarray:
    """Suppress a head unless at least one predicted parent category applies."""

    gated = np.asarray(attribute_predictions).copy()
    category_predictions = np.asarray(category_predictions)
    for head in ATTRIBUTE_HEADS:
        parent_indices = [
            index
            for index, category in enumerate(HIGH_LEVEL_CATEGORIES)
            if head in CATEGORY_TO_ATTRIBUTES[category]
        ]
        applicable = (
            category_predictions[:, parent_indices].any(axis=1)
            if parent_indices
            else np.zeros(category_predictions.shape[0], dtype=bool)
        )
        start, stop = schema.attribute_slices[head]
        gated[~applicable, start:stop] = 0
    return gated


def _evaluate_model(
    model: Any,
    data_loader: Any,
    device: Any,
    torch_module: Any,
    schema: HierarchySchema,
    threshold: float,
    attribute_loss_weight: float,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    loss_sum = 0.0
    example_count = 0
    category_probabilities: list[np.ndarray] = []
    attribute_probabilities: list[np.ndarray] = []
    category_truth: list[np.ndarray] = []
    attribute_truth: list[np.ndarray] = []
    attribute_masks: list[np.ndarray] = []
    with torch_module.no_grad():
        for batch in data_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            category_labels = batch.pop("category_labels")
            attribute_labels = batch.pop("attribute_labels")
            head_masks = batch.pop("attribute_masks")
            category_logits, attribute_logits = model(**batch)
            loss, _category_loss, _attribute_loss = _hierarchical_loss(
                category_logits,
                attribute_logits,
                category_labels,
                attribute_labels,
                head_masks,
                schema,
                torch_module,
                attribute_loss_weight,
            )
            batch_size = int(category_labels.shape[0])
            loss_sum += float(loss.detach().cpu()) * batch_size
            example_count += batch_size
            category_probabilities.append(
                torch_module.sigmoid(category_logits).cpu().numpy()
            )
            attribute_probabilities.append(
                torch_module.sigmoid(attribute_logits).cpu().numpy()
            )
            category_truth.append(category_labels.cpu().numpy())
            attribute_truth.append(attribute_labels.cpu().numpy())
            attribute_masks.append(head_masks.cpu().numpy())
    if not example_count:
        raise RuntimeError("Evaluation loader produced no examples")

    category_probability_matrix = np.concatenate(category_probabilities)
    attribute_probability_matrix = np.concatenate(attribute_probabilities)
    category_true = np.concatenate(category_truth).astype(np.int64)
    attribute_true = np.concatenate(attribute_truth).astype(np.int64)
    head_mask_matrix = np.concatenate(attribute_masks).astype(np.int64)
    category_pred = threshold_probabilities(category_probability_matrix, threshold)
    attribute_pred = threshold_probabilities(attribute_probability_matrix, threshold)
    gated_attribute_pred = gate_attribute_predictions(
        attribute_pred, category_pred, schema
    )

    category_metrics = compute_multilabel_classification_metrics(
        category_true, category_pred, list(HIGH_LEVEL_CATEGORIES)
    )
    category_metrics["threshold"] = threshold
    category_metrics["evaluated_examples"] = example_count
    independent_attributes = compute_attribute_metrics(
        attribute_true, attribute_pred, head_mask_matrix, schema
    )
    gated_attributes = compute_attribute_metrics(
        attribute_true, gated_attribute_pred, head_mask_matrix, schema
    )

    hierarchy_f1_values = [
        float(values["f1"]) for values in category_metrics["per_label"].values()
    ] + [
        float(values["f1"])
        for values in gated_attributes["per_value"].values()
        if values.get("evaluated_examples", 0) > 0
    ]
    hierarchy_balanced_values = [
        float(values["balanced_accuracy"])
        for values in category_metrics["per_label"].values()
        if values["balanced_accuracy"] is not None
    ] + [
        float(values["balanced_accuracy"])
        for values in gated_attributes["per_value"].values()
        if values.get("evaluated_examples", 0) > 0
        and values["balanced_accuracy"] is not None
    ]
    results = {
        "primary_metric": "hierarchical_macro_f1",
        "hierarchical_macro_f1": float(np.mean(hierarchy_f1_values)),
        "balanced_accuracy": (
            float(np.mean(hierarchy_balanced_values))
            if hierarchy_balanced_values
            else None
        ),
        "threshold": threshold,
        "evaluated_examples": example_count,
        "category": category_metrics,
        "fine_grained": gated_attributes,
        "independent_attribute_heads": independent_attributes,
    }
    return loss_sum / example_count, results


def classifier_metadata(
    schema: HierarchySchema,
    *,
    threshold: float,
    source_model: str,
    source_revision: str | None,
    classifier_dropout: float = 0.15,
) -> dict[str, Any]:
    return {
        "classifier_format_version": CLASSIFIER_FORMAT_VERSION,
        "model_type": EXPECTED_MODEL_TYPE,
        "task": "opp115_hierarchical_fine_grained_classification",
        "threshold": threshold,
        "pooling_strategy": POOLING_STRATEGY,
        "classifier_dropout": classifier_dropout,
        "checkpoint_serialization": "safetensors",
        "source_model": source_model,
        "source_revision": source_revision,
        "category_labels": list(HIGH_LEVEL_CATEGORIES),
        "attribute_heads": list(ATTRIBUTE_HEADS),
        "hierarchy_schema": schema.to_manifest(),
    }


def validate_classifier_metadata(
    metadata: Mapping[str, Any],
    schema: HierarchySchema,
    threshold: float = 0.5,
) -> None:
    if metadata.get("classifier_format_version") != CLASSIFIER_FORMAT_VERSION:
        raise ValueError("Unsupported or missing hierarchy classifier format")
    if metadata.get("model_type") != EXPECTED_MODEL_TYPE:
        raise ValueError("Saved hierarchy classifier is not backed by RoBERTa")
    if metadata.get("pooling_strategy") != POOLING_STRATEGY:
        raise ValueError("Saved hierarchy classifier uses an unsupported pooling strategy")
    if metadata.get("checkpoint_serialization") != "safetensors":
        raise ValueError("Saved hierarchy classifier is not safely serialized")
    classifier_dropout = float(metadata.get("classifier_dropout", float("nan")))
    if not 0.0 <= classifier_dropout < 1.0:
        raise ValueError("Saved classifier dropout is invalid")
    if metadata.get("category_labels") != list(HIGH_LEVEL_CATEGORIES):
        raise ValueError("Saved category label order is stale")
    if metadata.get("attribute_heads") != list(ATTRIBUTE_HEADS):
        raise ValueError("Saved attribute-head order is stale")
    saved_schema = metadata.get("hierarchy_schema")
    if not isinstance(saved_schema, Mapping):
        raise ValueError("Saved hierarchy schema is missing")
    if saved_schema.get("schema_sha256") != schema.schema_sha256:
        raise ValueError("Saved hierarchy schema hash is stale")
    if not math.isclose(float(metadata.get("threshold", float("nan"))), threshold):
        raise ValueError("Saved classification threshold is stale")


def saved_classifier_directory_valid(
    directory: str | Path,
    schema: HierarchySchema,
    threshold: float = 0.5,
) -> bool:
    path = Path(directory)
    required = (
        path / "classifier_config.json",
        path / "hierarchy_schema.json",
        path / "hierarchy_heads.safetensors",
        path / "encoder" / "config.json",
    )
    if not all(item.is_file() for item in required):
        return False
    if not any(
        (path / "encoder" / name).is_file()
        for name in ("model.safetensors", "pytorch_model.bin")
    ):
        return False
    if not (
        (path / "tokenizer.json").is_file()
        or (path / "vocab.txt").is_file()
        or ((path / "vocab.json").is_file() and (path / "merges.txt").is_file())
    ):
        return False
    try:
        metadata = json.loads(required[0].read_text(encoding="utf-8"))
        validate_classifier_metadata(metadata, schema, threshold)
        encoder_config = json.loads(required[3].read_text(encoding="utf-8"))
        return encoder_config.get("model_type") == EXPECTED_MODEL_TYPE
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _save_classifier(
    model: Any,
    tokenizer: Any,
    torch_module: Any,
    output_dir: Path,
    schema: HierarchySchema,
    metadata: Mapping[str, Any],
    summary: Mapping[str, Any],
    overwrite: bool,
) -> None:
    from safetensors.torch import save_file

    output_dir = output_dir.expanduser().resolve()
    temporary = output_dir.with_name(output_dir.name + ".incomplete")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.encoder.save_pretrained(temporary / "encoder", safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    head_tensors = {
        **{
            f"pre_classifier.{name}": tensor.detach().cpu().contiguous()
            for name, tensor in model.pre_classifier.state_dict().items()
        },
        **{
            f"category_head.{name}": tensor.detach().cpu().contiguous()
            for name, tensor in model.category_head.state_dict().items()
        },
        **{
            f"attribute_heads.{name}": tensor.detach().cpu().contiguous()
            for name, tensor in model.attribute_heads.state_dict().items()
        },
    }
    save_file(head_tensors, temporary / "hierarchy_heads.safetensors")
    (temporary / "classifier_config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (temporary / "hierarchy_schema.json").write_text(
        json.dumps(schema.to_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (temporary / "roberta_opp115_hierarchy_classification_results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if output_dir.exists():
        if any(output_dir.iterdir()) and not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    temporary.replace(output_dir)


def load_saved_classifier(
    directory: str | Path, *, device: str = "cpu"
) -> tuple[Any, Any, HierarchySchema, Mapping[str, Any]]:
    """Reload a saved hierarchy classifier for inference."""

    try:
        import torch
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer
    except (ImportError, OSError) as exc:
        raise RuntimeError("Install requirements_roberta_opp115.txt first") from exc
    path = Path(directory).expanduser().resolve()
    metadata = json.loads((path / "classifier_config.json").read_text(encoding="utf-8"))
    schema = HierarchySchema.from_manifest(metadata["hierarchy_schema"])
    validate_classifier_metadata(metadata, schema, float(metadata["threshold"]))
    encoder = AutoModel.from_pretrained(path / "encoder")
    validate_roberta_model_type(encoder)
    model = _build_model(
        torch,
        encoder,
        schema,
        classifier_dropout=float(metadata.get("classifier_dropout", 0.15)),
    )
    state = load_file(path / "hierarchy_heads.safetensors", device="cpu")
    model.pre_classifier.load_state_dict(
        {
            name.removeprefix("pre_classifier."): tensor
            for name, tensor in state.items()
            if name.startswith("pre_classifier.")
        }
    )
    model.category_head.load_state_dict(
        {
            name.removeprefix("category_head."): tensor
            for name, tensor in state.items()
            if name.startswith("category_head.")
        }
    )
    model.attribute_heads.load_state_dict(
        {
            name.removeprefix("attribute_heads."): tensor
            for name, tensor in state.items()
            if name.startswith("attribute_heads.")
        }
    )
    model.to(torch.device(device))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    return model, tokenizer, schema, MappingProxyType(metadata)


def run_classification(
    args: argparse.Namespace,
    all_examples: Sequence[SegmentExample],
    train_examples: Sequence[SegmentExample],
    eval_examples: Sequence[SegmentExample],
    train_ids: Sequence[str],
    eval_ids: Sequence[str],
    schema: HierarchySchema,
) -> dict[str, Any]:
    if not args.no_save:
        output_dir = args.output_dir.expanduser().resolve()
        if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite_output_dir:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Choose another "
                "--output-dir or pass --overwrite-output-dir."
            )

    try:
        import torch
        import transformers
        from torch.utils.data import DataLoader
        from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Install requirements_roberta_opp115.txt in a compatible environment."
        ) from exc

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    used_train_examples = _sample_examples(
        train_examples, args.max_train_examples, args.seed
    )
    used_eval_examples = _sample_examples(
        eval_examples, args.max_eval_examples, args.seed + 1
    )
    LOGGER.info("Loading RoBERTa encoder checkpoint: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, revision=args.model_revision, use_fast=True
    )
    encoder = AutoModel.from_pretrained(
        args.model_name, revision=args.model_revision
    )
    validate_roberta_model_type(encoder)
    tokenizer_limit = getattr(tokenizer, "model_max_length", args.max_length)
    if tokenizer_limit < 1_000_000 and args.max_length > tokenizer_limit:
        raise ValueError(
            f"max_length={args.max_length} exceeds tokenizer limit {tokenizer_limit}"
        )
    model = _build_model(
        torch,
        encoder,
        schema,
        classifier_dropout=args.classifier_dropout,
    )

    def encode(examples: Sequence[SegmentExample]) -> EncodedSegmentDataset:
        encodings = tokenizer(
            [example.text for example in examples],
            truncation=True,
            padding=False,
            max_length=args.max_length,
        )
        return EncodedSegmentDataset(
            encodings,
            torch.tensor(
                [example.category_labels for example in examples],
                dtype=torch.float32,
            ),
            torch.tensor(
                [example.attribute_labels for example in examples],
                dtype=torch.float32,
            ),
            torch.tensor(
                [example.attribute_mask for example in examples],
                dtype=torch.float32,
            ),
        )

    train_dataset = encode(used_train_examples)
    eval_dataset = encode(used_eval_examples)

    def collate_batch(features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        model_features: list[dict[str, Any]] = []
        category_labels: list[Any] = []
        attribute_labels: list[Any] = []
        attribute_masks: list[Any] = []
        for feature in features:
            item = dict(feature)
            category_labels.append(item.pop("category_labels"))
            attribute_labels.append(item.pop("attribute_labels"))
            attribute_masks.append(item.pop("attribute_masks"))
            model_features.append(item)
        batch = tokenizer.pad(
            model_features,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        batch["category_labels"] = torch.stack(category_labels)
        batch["attribute_labels"] = torch.stack(attribute_labels)
        batch["attribute_masks"] = torch.stack(attribute_masks)
        return batch

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    device = _choose_device(torch, args.device)
    model.to(device)
    category_weight_values, attribute_weight_values = compute_positive_class_weights(
        used_train_examples,
        schema,
        power=args.positive_weight_power,
        maximum=args.max_positive_weight,
    )
    category_positive_weights = torch.as_tensor(
        category_weight_values, dtype=torch.float32, device=device
    )
    attribute_positive_weights = torch.as_tensor(
        attribute_weight_values, dtype=torch.float32, device=device
    )
    decay_parameters: list[Any] = []
    no_decay_parameters: list[Any] = []
    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter_name.endswith("bias") or parameter_name.endswith(
            ("LayerNorm.weight", "layer_norm.weight")
        ):
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_parameters, "weight_decay": args.weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
    )
    updates_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    total_steps = args.max_steps or args.epochs * updates_per_epoch
    epochs_to_run = (
        math.ceil(total_steps / updates_per_epoch) if args.max_steps else args.epochs
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    train_losses: list[float] = []
    for epoch_index in range(epochs_to_run):
        model.train()
        accumulated_batches = 0
        accumulated_loss = 0.0
        for batch_index, batch in enumerate(train_loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            category_labels = batch.pop("category_labels")
            attribute_labels = batch.pop("attribute_labels")
            attribute_masks = batch.pop("attribute_masks")
            category_logits, attribute_logits = model(**batch)
            loss, _category_loss, _attribute_loss = _hierarchical_loss(
                category_logits,
                attribute_logits,
                category_labels,
                attribute_labels,
                attribute_masks,
                schema,
                torch,
                args.attribute_loss_weight,
                category_positive_weights=category_positive_weights,
                attribute_positive_weights=attribute_positive_weights,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Training produced a non-finite loss")
            loss.backward()
            accumulated_batches += 1
            accumulated_loss += float(loss.detach().cpu())
            update_now = (
                accumulated_batches == args.gradient_accumulation_steps
                or batch_index == len(train_loader) - 1
            )
            if not update_now:
                continue
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated_batches)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            step_loss = accumulated_loss / accumulated_batches
            train_losses.append(step_loss)
            accumulated_batches = 0
            accumulated_loss = 0.0
            if global_step == 1 or global_step % args.log_every_steps == 0:
                LOGGER.info(
                    "epoch=%d step=%d/%d train_loss=%.6f",
                    epoch_index + 1,
                    global_step,
                    total_steps,
                    step_loss,
                )
            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    eval_loss, metrics = _evaluate_model(
        model,
        eval_loader,
        device,
        torch,
        schema,
        args.threshold,
        args.attribute_loss_weight,
    )
    LOGGER.info(
        "Hierarchical macro-F1=%.6f; category macro-F1=%.6f; fine-grained macro-F1=%s",
        metrics["hierarchical_macro_f1"],
        metrics["category"]["macro_f1"],
        metrics["fine_grained"]["fine_grained_macro_f1"],
    )

    summary = {
        **dataset_summary(
            all_examples,
            train_examples,
            eval_examples,
            train_ids,
            eval_ids,
            schema,
        ),
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "model_type": EXPECTED_MODEL_TYPE,
        "device": str(device),
        "seed": args.seed,
        "library_versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "training_configuration": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "threshold": args.threshold,
            "classifier_dropout": args.classifier_dropout,
            "padding": "dynamic_to_batch_multiple_of_8",
            "effective_batch_size": (
                args.batch_size * args.gradient_accumulation_steps
            ),
            "attribute_loss_weight": args.attribute_loss_weight,
            "positive_weight_power": args.positive_weight_power,
            "max_positive_weight": args.max_positive_weight,
            "max_train_examples": args.max_train_examples,
            "max_eval_examples": args.max_eval_examples,
        },
        "pooling_strategy": POOLING_STRATEGY,
        "category_positive_weights": {
            label: float(weight)
            for label, weight in zip(HIGH_LEVEL_CATEGORIES, category_weight_values)
        },
        "attribute_positive_weights": {
            label: float(weight)
            for label, weight in zip(
                schema.flattened_attribute_labels, attribute_weight_values
            )
        },
        "target_hierarchical_macro_f1": args.target_hierarchical_macro_f1,
        "target_reached": bool(
            metrics["hierarchical_macro_f1"] >= args.target_hierarchical_macro_f1
        ),
        "max_length": args.max_length,
        "training_examples_used": len(used_train_examples),
        "evaluation_examples_used": len(used_eval_examples),
        "training_policies_used": len({e.policy_id for e in used_train_examples}),
        "evaluation_policies_used": len({e.policy_id for e in used_eval_examples}),
        "training_category_support_used": _category_support(used_train_examples),
        "evaluation_category_support_used": _category_support(used_eval_examples),
        "training_attribute_support_used": _attribute_support(
            used_train_examples, schema
        ),
        "evaluation_attribute_support_used": _attribute_support(
            used_eval_examples, schema
        ),
        "sampling_applied": (
            len(used_train_examples) != len(train_examples)
            or len(used_eval_examples) != len(eval_examples)
        ),
        "completed_optimizer_steps": global_step,
        "mean_train_loss": float(np.mean(train_losses)),
        "eval_loss": eval_loss,
        "results": metrics,
    }

    if not args.no_save:
        metadata = classifier_metadata(
            schema,
            threshold=args.threshold,
            source_model=args.model_name,
            source_revision=args.model_revision,
            classifier_dropout=args.classifier_dropout,
        )
        _save_classifier(
            model,
            tokenizer,
            torch,
            args.output_dir,
            schema,
            metadata,
            summary,
            args.overwrite_output_dir,
        )
        LOGGER.info("Saved hierarchy classifier and results to %s", args.output_dir)
    return summary


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune RoBERTa on the OPP-115 hierarchy: ten category labels and "
            "twenty category-scoped, loss-masked attribute heads."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_OPP115_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--model-revision",
        help="Optional Hugging Face commit/tag revision; recorded in the summary.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-file", type=Path)
    parser.add_argument("--train-policies", type=positive_int, default=65)
    parser.add_argument("--eval-policies", type=positive_int, default=50)
    parser.add_argument("--max-length", type=positive_int, default=384)
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--epochs", type=positive_int, default=5)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-eval-examples", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2.5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-accumulation-steps", type=positive_int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--classifier-dropout", type=float, default=0.15)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--attribute-loss-weight", type=float, default=1.0)
    parser.add_argument("--positive-weight-power", type=float, default=0.5)
    parser.add_argument("--max-positive-weight", type=float, default=10.0)
    parser.add_argument("--target-hierarchical-macro-f1", type=float, default=0.96)
    parser.add_argument("--seed", type=int, default=17011)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--log-every-steps", type=positive_int, default=20)
    parser.add_argument("--validate-data-only", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--overwrite-output-dir", action="store_true")
    return parser


def validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for name in ("max_steps", "max_train_examples", "max_eval_examples"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.weight_decay < 0:
        parser.error("--weight-decay cannot be negative")
    if not 0.0 <= args.warmup_ratio < 1.0:
        parser.error("--warmup-ratio must be in [0, 1)")
    if args.max_grad_norm <= 0:
        parser.error("--max-grad-norm must be positive")
    if not 0.0 <= args.classifier_dropout < 1.0:
        parser.error("--classifier-dropout must be in [0, 1)")
    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be between 0 and 1")
    if args.attribute_loss_weight <= 0:
        parser.error("--attribute-loss-weight must be positive")
    if not 0.0 <= args.positive_weight_power <= 1.0:
        parser.error("--positive-weight-power must be in [0, 1]")
    if args.max_positive_weight < 1.0:
        parser.error("--max-positive-weight must be at least 1")
    if not 0.0 < args.target_hierarchical_macro_f1 <= 1.0:
        parser.error("--target-hierarchical-macro-f1 must be in (0, 1]")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")


def write_summary(summary: Mapping[str, Any], path: Path) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    LOGGER.info("Saved summary to %s", path)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    validate_arguments(parser, args)

    annotations_dir = args.data_dir.expanduser().resolve() / "annotations"
    train_ids, eval_ids = fixed_policy_split(
        annotations_dir,
        train_count=args.train_policies,
        eval_count=args.eval_policies,
    )
    schema = derive_hierarchy_schema(annotations_dir, train_ids)
    all_examples = load_segment_examples(args.data_dir, schema)
    train_examples, eval_examples, actual_train_ids, actual_eval_ids = (
        split_segment_examples(
            all_examples,
            train_count=args.train_policies,
            eval_count=args.eval_policies,
        )
    )
    if actual_train_ids != train_ids or actual_eval_ids != eval_ids:
        raise AssertionError("Schema and example policy splits differ")
    summary = dataset_summary(
        all_examples,
        train_examples,
        eval_examples,
        train_ids,
        eval_ids,
        schema,
    )
    LOGGER.info(
        "Loaded %d segments; exact split is %d train / %d eval policies; "
        "%d category labels + %d attribute heads / %d values",
        summary["segments"],
        summary["train_policies"],
        summary["eval_policies"],
        summary["category_count"],
        summary["attribute_head_count"],
        summary["fine_grained_label_count"],
    )
    if args.validate_data_only:
        if args.summary_file:
            write_summary(summary, args.summary_file)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    result = run_classification(
        args,
        all_examples,
        train_examples,
        eval_examples,
        train_ids,
        eval_ids,
        schema,
    )
    if args.summary_file:
        write_summary(result, args.summary_file)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
