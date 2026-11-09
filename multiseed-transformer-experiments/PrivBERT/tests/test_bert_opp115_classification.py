"""Tests for the hierarchical PrivBERT OPP-115 segment classifier."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bert_opp115_classification import (
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    EncodedSegmentDataset,
    SegmentExample,
    _build_model,
    build_argument_parser,
    classifier_metadata,
    compute_attribute_metrics,
    compute_positive_class_weights,
    dataset_summary,
    gate_attribute_predictions,
    group_annotation_samples,
    load_segment_examples,
    split_segment_examples,
    threshold_probabilities,
    validate_roberta_model_type,
    validate_classifier_metadata,
)
from opp115_hierarchy import (
    ATTRIBUTE_HEADS,
    HIGH_LEVEL_CATEGORIES,
    HierarchySchema,
    derive_hierarchy_schema,
    fixed_policy_split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CANDIDATES = (
    PROJECT_ROOT / "data" / "OPP-115_v1_0" / "OPP-115",
    PROJECT_ROOT / "data" / "opp115",
)
OPP115_DATA_DIR = next(
    (
        path
        for path in DATA_CANDIDATES
        if (path / "annotations").is_dir()
        and (path / "sanitized_policies").is_dir()
    ),
    None,
)


def _schema_with_one_value_per_head(
    *, training_policy_ids: tuple[str, ...] = ("policy-a",),
) -> HierarchySchema:
    return HierarchySchema(
        training_policy_ids=training_policy_ids,
        value_vocabulary={
            head: (f"{head}_value",) for head in ATTRIBUTE_HEADS
        },
    )


class _FakeModule:
    def __init__(self) -> None:
        pass


class _FakeLinear:
    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features = in_features
        self.out_features = out_features


class _FakeDropout:
    def __init__(self, probability: float) -> None:
        self.p = probability


class _FakeTanh:
    pass


class _FakeModuleDict(dict):
    pass


class _FakeNN:
    Module = _FakeModule
    Linear = _FakeLinear
    Dropout = _FakeDropout
    Tanh = _FakeTanh
    ModuleDict = _FakeModuleDict


class _FakeTorch:
    nn = _FakeNN()


class PrivBERTOPP115ClassificationUnitTests(unittest.TestCase):
    def test_defaults_select_privbert_and_hierarchy_output(self) -> None:
        args = build_argument_parser().parse_args([])
        self.assertEqual(DEFAULT_MODEL_NAME, "mukund/privbert")
        self.assertEqual(args.model_name, DEFAULT_MODEL_NAME)
        self.assertEqual(
            DEFAULT_OUTPUT_DIR.name, "privbert-opp115-hierarchy-classifier"
        )
        self.assertEqual(args.max_length, 384)
        self.assertEqual(args.epochs, 5)
        self.assertEqual(args.gradient_accumulation_steps, 4)
        self.assertEqual(args.learning_rate, 2.5e-5)
        self.assertEqual(args.classifier_dropout, 0.15)
        self.assertEqual(args.attribute_loss_weight, 1.0)
        self.assertEqual(args.positive_weight_power, 0.5)
        self.assertEqual(args.max_positive_weight, 10.0)
        self.assertEqual(args.target_hierarchical_macro_f1, 0.96)
        self.assertEqual(args.threshold, 0.5)

    def test_positive_class_weights_are_smoothed_capped_and_head_masked(self) -> None:
        schema = _schema_with_one_value_per_head()
        purpose_start, _ = schema.attribute_slices["purpose"]
        security_start, _ = schema.attribute_slices["security_measure"]
        purpose_head = ATTRIBUTE_HEADS.index("purpose")
        security_head = ATTRIBUTE_HEADS.index("security_measure")

        def example(
            index: int,
            *,
            categories: tuple[int, ...],
            purpose: int = 0,
            purpose_active: int = 0,
            security_active: int = 0,
        ) -> SegmentExample:
            category_labels = [0] * len(HIGH_LEVEL_CATEGORIES)
            for category_index in categories:
                category_labels[category_index] = 1
            attribute_labels = [0] * schema.flattened_value_count
            attribute_labels[purpose_start] = purpose
            attribute_mask = [0] * len(ATTRIBUTE_HEADS)
            attribute_mask[purpose_head] = purpose_active
            attribute_mask[security_head] = security_active
            return SegmentExample(
                policy_id=f"policy-{index}",
                segment_id=str(index),
                text=f"Segment {index}",
                category_labels=tuple(category_labels),
                attribute_labels=tuple(attribute_labels),
                attribute_mask=tuple(attribute_mask),
            )

        examples = [
            example(
                1,
                categories=(0, 1),
                purpose=1,
                purpose_active=1,
                security_active=1,
            ),
            example(
                2,
                categories=(1,),
                purpose_active=1,
                security_active=1,
            ),
            example(3, categories=(1,)),
        ]

        category_weights, attribute_weights = compute_positive_class_weights(
            examples, schema, power=0.5, maximum=1.5
        )

        self.assertEqual(category_weights.shape, (len(HIGH_LEVEL_CATEGORIES),))
        self.assertEqual(attribute_weights.shape, (schema.flattened_value_count,))
        self.assertEqual(category_weights.dtype, np.float32)
        self.assertEqual(attribute_weights.dtype, np.float32)
        np.testing.assert_allclose(category_weights[0], np.sqrt(3.0 / 2.0))
        self.assertEqual(category_weights[1], 1.0)
        self.assertEqual(category_weights[2], 1.5)
        self.assertEqual(attribute_weights[purpose_start], 1.0)
        self.assertEqual(attribute_weights[security_start], 1.5)
        retention_start, _ = schema.attribute_slices["retention_period"]
        self.assertEqual(attribute_weights[retention_start], 1.0)

    def test_model_type_guard_accepts_only_roberta(self) -> None:
        validate_roberta_model_type(
            SimpleNamespace(config=SimpleNamespace(model_type="roberta"))
        )
        with self.assertRaisesRegex(ValueError, "model_type='roberta'"):
            validate_roberta_model_type(
                SimpleNamespace(config=SimpleNamespace(model_type="bert"))
            )

    def test_one_shared_encoder_feeds_category_and_twenty_attribute_heads(self) -> None:
        schema = _schema_with_one_value_per_head()
        schema = HierarchySchema(
            training_policy_ids=schema.training_policy_ids,
            value_vocabulary={
                **schema.value_vocabulary,
                "purpose": ("advertising", "analytics_research"),
            },
        )
        encoder = SimpleNamespace(
            config=SimpleNamespace(hidden_size=16, hidden_dropout_prob=0.2)
        )

        model = _build_model(_FakeTorch, encoder, schema, classifier_dropout=0.15)

        self.assertIs(model.encoder, encoder)
        self.assertEqual(model.dropout.p, 0.15)
        self.assertEqual(model.pre_classifier.in_features, 16)
        self.assertEqual(model.pre_classifier.out_features, 16)
        self.assertEqual(model.category_head.in_features, 16)
        self.assertEqual(model.category_head.out_features, 10)
        self.assertEqual(tuple(model.attribute_heads), ATTRIBUTE_HEADS)
        self.assertEqual(len(model.attribute_heads), 20)
        self.assertEqual(model.attribute_heads["purpose"].out_features, 2)
        self.assertEqual(
            sum(head.out_features for head in model.attribute_heads.values()),
            schema.flattened_value_count,
        )

    def test_encoded_dataset_keeps_separate_targets_and_roberta_inputs(self) -> None:
        encodings = {
            "input_ids": np.array([[0, 2000, 2, 1]]),
            "attention_mask": np.array([[1, 1, 1, 0]]),
        }
        category_labels = np.zeros((1, 10), dtype=np.float32)
        attribute_labels = np.zeros((1, 20), dtype=np.float32)
        attribute_masks = np.zeros((1, 20), dtype=np.float32)

        item = EncodedSegmentDataset(
            encodings, category_labels, attribute_labels, attribute_masks
        )[0]

        self.assertNotIn("token_type_ids", item)
        self.assertEqual(item["category_labels"].shape, (10,))
        self.assertEqual(item["attribute_labels"].shape, (20,))
        self.assertEqual(item["attribute_masks"].shape, (20,))

    def test_samples_union_categories_attributes_and_applicability_masks(self) -> None:
        schema = _schema_with_one_value_per_head()
        samples = [
            SimpleNamespace(
                policy_id="policy-a",
                segment_id="0",
                text="We collect information.",
                categories=("First Party Collection/Use",),
                attributes={"purpose": {"purpose_value"}},
            ),
            SimpleNamespace(
                policy_id="policy-a",
                segment_id="0",
                text="We collect information and protect it.",
                categories=("Data Security",),
                attributes={"security_measure": {"security_measure_value"}},
            ),
        ]

        examples = group_annotation_samples(samples, schema)

        self.assertEqual(len(examples), 1)
        example = examples[0]
        self.assertEqual(example.text, "We collect information and protect it.")
        self.assertEqual(
            example.category_labels[
                HIGH_LEVEL_CATEGORIES.index("First Party Collection/Use")
            ],
            1,
        )
        self.assertEqual(
            example.category_labels[HIGH_LEVEL_CATEGORIES.index("Data Security")],
            1,
        )
        purpose_start, _ = schema.attribute_slices["purpose"]
        security_start, _ = schema.attribute_slices["security_measure"]
        self.assertEqual(example.attribute_labels[purpose_start], 1)
        self.assertEqual(example.attribute_labels[security_start], 1)
        self.assertEqual(
            example.attribute_mask[ATTRIBUTE_HEADS.index("purpose")], 1
        )
        self.assertEqual(
            example.attribute_mask[ATTRIBUTE_HEADS.index("security_measure")], 1
        )
        self.assertEqual(
            example.attribute_mask[ATTRIBUTE_HEADS.index("retention_period")], 0
        )

    def test_frozen_schema_reports_held_out_value_without_resizing(self) -> None:
        schema = _schema_with_one_value_per_head()
        sample = SimpleNamespace(
            policy_id="policy-eval",
            segment_id="3",
            text="This site honors browser tracking preferences.",
            categories=("Do Not Track",),
            attributes={"do_not_track_policy": {"honored"}},
        )

        example = group_annotation_samples([sample], schema)[0]

        start, stop = schema.attribute_slices["do_not_track_policy"]
        self.assertEqual(example.attribute_labels[start:stop], (0,))
        self.assertEqual(
            example.unknown_attribute_values,
            ("do_not_track_policy=honored",),
        )
        self.assertEqual(len(example.attribute_labels), schema.flattened_value_count)
        self.assertEqual(
            example.attribute_mask[ATTRIBUTE_HEADS.index("do_not_track_policy")],
            1,
        )

    def test_category_predictions_gate_attribute_heads(self) -> None:
        schema = _schema_with_one_value_per_head()
        attribute_predictions = np.ones(
            (2, schema.flattened_value_count), dtype=np.int64
        )
        category_predictions = np.zeros(
            (2, len(HIGH_LEVEL_CATEGORIES)), dtype=np.int64
        )
        category_predictions[
            0, HIGH_LEVEL_CATEGORIES.index("First Party Collection/Use")
        ] = 1
        category_predictions[1, HIGH_LEVEL_CATEGORIES.index("Other")] = 1

        gated = gate_attribute_predictions(
            attribute_predictions, category_predictions, schema
        )

        purpose_start, _ = schema.attribute_slices["purpose"]
        security_start, _ = schema.attribute_slices["security_measure"]
        self.assertEqual(gated[0, purpose_start], 1)
        self.assertEqual(gated[0, security_start], 0)
        self.assertEqual(int(gated[1].sum()), 0)

    def test_attribute_metrics_ignore_non_applicable_rows(self) -> None:
        schema = _schema_with_one_value_per_head()
        truth = np.zeros((2, schema.flattened_value_count), dtype=np.int64)
        predictions = np.zeros_like(truth)
        masks = np.zeros((2, len(ATTRIBUTE_HEADS)), dtype=np.int64)
        value_index, _ = schema.attribute_slices["security_measure"]
        head_index = ATTRIBUTE_HEADS.index("security_measure")
        truth[:, value_index] = 1
        predictions[0, value_index] = 1
        masks[0, head_index] = 1

        metrics = compute_attribute_metrics(truth, predictions, masks, schema)

        security = metrics["per_attribute"]["security_measure"]
        self.assertEqual(security["evaluated_examples"], 1)
        self.assertEqual(security["macro_f1"], 1.0)
        self.assertEqual(metrics["evaluated_head_instances"], 1)
        self.assertEqual(
            metrics["per_attribute"]["retention_period"]["evaluated_examples"],
            0,
        )
        self.assertEqual(
            len(metrics["per_value"]), schema.flattened_value_count
        )
        retention_start, _ = schema.attribute_slices["retention_period"]
        retention_value = schema.flattened_attribute_labels[retention_start]
        self.assertEqual(
            metrics["per_value"][retention_value]["evaluated_examples"], 0
        )

    def test_classifier_metadata_is_bound_to_head_order_schema_and_threshold(self) -> None:
        schema = _schema_with_one_value_per_head()
        metadata = classifier_metadata(
            schema,
            threshold=0.5,
            source_model=DEFAULT_MODEL_NAME,
            source_revision="test-revision",
        )

        validate_classifier_metadata(metadata, schema)
        self.assertEqual(metadata["model_type"], "roberta")
        self.assertEqual(metadata["checkpoint_serialization"], "safetensors")
        self.assertEqual(metadata["category_labels"], list(HIGH_LEVEL_CATEGORIES))
        self.assertEqual(metadata["attribute_heads"], list(ATTRIBUTE_HEADS))
        self.assertEqual(
            metadata["hierarchy_schema"]["schema_sha256"], schema.schema_sha256
        )

        stale = {**metadata, "attribute_heads": list(reversed(ATTRIBUTE_HEADS))}
        with self.assertRaisesRegex(ValueError, "attribute-head order"):
            validate_classifier_metadata(stale, schema)
        with self.assertRaisesRegex(ValueError, "threshold"):
            validate_classifier_metadata(metadata, schema, threshold=0.6)

    def test_whole_policy_split_is_exact_and_disjoint(self) -> None:
        zero_categories = (0,) * len(HIGH_LEVEL_CATEGORIES)
        zero_attributes = (0,) * len(ATTRIBUTE_HEADS)
        examples = [
            SegmentExample(
                policy_id=f"policy-{policy_index:03d}",
                segment_id=str(segment_index),
                text=f"Policy {policy_index}, segment {segment_index}",
                category_labels=zero_categories,
                attribute_labels=zero_attributes,
                attribute_mask=zero_attributes,
            )
            for policy_index in range(115)
            for segment_index in range(2)
        ]

        train, evaluation, train_ids, eval_ids = split_segment_examples(examples)

        self.assertEqual(len(train_ids), 65)
        self.assertEqual(len(eval_ids), 50)
        self.assertEqual(len(train), 130)
        self.assertEqual(len(evaluation), 100)
        self.assertFalse(set(train_ids) & set(eval_ids))

    def test_threshold_includes_exactly_point_five(self) -> None:
        probabilities = np.array([[0.49, 0.5, 0.51]])
        np.testing.assert_array_equal(
            threshold_probabilities(probabilities), np.array([[0, 1, 1]])
        )


@unittest.skipUnless(OPP115_DATA_DIR is not None, "real OPP-115 data is unavailable")
class PrivBERTOPP115ClassificationRealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert OPP115_DATA_DIR is not None
        annotations = OPP115_DATA_DIR / "annotations"
        cls.train_ids, cls.eval_ids = fixed_policy_split(annotations)
        cls.schema = derive_hierarchy_schema(annotations, cls.train_ids)
        cls.examples = load_segment_examples(OPP115_DATA_DIR, cls.schema)
        (
            cls.train_examples,
            cls.eval_examples,
            actual_train_ids,
            actual_eval_ids,
        ) = split_segment_examples(cls.examples)
        if actual_train_ids != cls.train_ids or actual_eval_ids != cls.eval_ids:
            raise AssertionError("Classifier and schema splits differ")
        cls.summary = dataset_summary(
            cls.examples,
            cls.train_examples,
            cls.eval_examples,
            cls.train_ids,
            cls.eval_ids,
            cls.schema,
        )

    def test_real_segments_and_policy_split_are_stable(self) -> None:
        self.assertEqual(len(self.examples), 3792)
        self.assertEqual(len(self.train_examples), 2352)
        self.assertEqual(len(self.eval_examples), 1440)
        self.assertEqual(self.summary["train_policies"], 65)
        self.assertEqual(self.summary["eval_policies"], 50)
        self.assertTrue(self.summary["train_eval_policy_ids_disjoint"])

    def test_real_training_schema_has_20_heads_and_138_value_dimensions(self) -> None:
        self.assertEqual(self.summary["category_count"], 10)
        self.assertEqual(self.summary["attribute_head_count"], 20)
        self.assertEqual(self.summary["fine_grained_label_count"], 138)
        self.assertEqual(self.schema.training_policy_ids, self.train_ids)
        self.assertEqual(
            set(self.summary["held_out_unknown_attribute_values"]),
            {"do_not_track_policy=honored"},
        )


if __name__ == "__main__":
    unittest.main()
