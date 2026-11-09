"""Tests for the faithful OPP-115/Polisis hierarchical target schema."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from opp115_hierarchy import (
    ATTRIBUTE_HEADS,
    CATEGORY_TO_ATTRIBUTES,
    HIGH_LEVEL_CATEGORIES,
    HierarchySchema,
    derive_hierarchy_schema,
    encode_segment_labels,
    extract_attribute_values,
    fixed_policy_split,
    load_opp115_annotations,
    load_segment_annotations,
    normalize_attribute_name,
    normalize_attribute_value,
    normalize_category,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CANDIDATES = (
    PROJECT_ROOT / "data" / "OPP-115_v1_0" / "OPP-115" / "annotations",
    PROJECT_ROOT / "data" / "opp115" / "annotations",
)
ANNOTATION_DIR = next((path for path in DATA_CANDIDATES if path.is_dir()), None)

EXPECTED_FLATTENED_TRAIN_VALUE_COUNT = 138
EXPECTED_HELD_OUT_ONLY_VALUES = {"do_not_track_policy": {"honored"}}


def _write_row(
    path: Path,
    *,
    annotation_id: int,
    annotator_id: int,
    segment_id: object,
    category: str,
    attributes: dict[str, object],
) -> None:
    with path.open("a", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow(
            [
                annotation_id,
                "batch",
                annotator_id,
                "101",
                segment_id,
                category,
                json.dumps(attributes),
                "1/1/15",
                "https://example.test/privacy",
            ]
        )


class OPP115HierarchySyntheticTests(unittest.TestCase):
    def test_published_category_and_attribute_contract(self) -> None:
        self.assertEqual(len(HIGH_LEVEL_CATEGORIES), 10)
        self.assertEqual(len(ATTRIBUTE_HEADS), 20)
        self.assertNotIn("other_type", ATTRIBUTE_HEADS)
        self.assertNotIn("user_choice", ATTRIBUTE_HEADS)
        self.assertIn("user_type", ATTRIBUTE_HEADS)
        self.assertNotIn("access_rights", ATTRIBUTE_HEADS)
        self.assertEqual(set(CATEGORY_TO_ATTRIBUTES), set(HIGH_LEVEL_CATEGORIES))

    def test_robust_raw_normalization_preserves_unspecified(self) -> None:
        self.assertEqual(
            normalize_category("  USER access, edit & deletion  "),
            "Access, Edit and Deletion",
        )
        self.assertEqual(
            normalize_attribute_name(" Do-Not-Track POLICY "),
            "do_not_track_policy",
        )
        self.assertEqual(
            normalize_attribute_value("  Service Operation & Security "),
            "service_operation_and_security",
        )
        self.assertIsNone(normalize_attribute_value("not-selected"))
        self.assertEqual(normalize_attribute_value("Unspecified"), "unspecified")

        values = extract_attribute_values(
            {
                "Choice Type": {"value": "not-selected"},
                "Personal Information Type": {"value": "Unspecified"},
                "Other Type": {"value": "Introductory/Generic"},
                "User Choice": {"value": "Opt-out"},
            }
        )
        self.assertEqual(
            values, {"personal_information_type": {"unspecified"}}
        )

    def test_headerless_rows_union_annotators_practices_and_masks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            annotations = Path(temporary)
            csv_path = annotations / "101_example.com.csv"
            _write_row(
                csv_path,
                annotation_id=1,
                annotator_id=0,
                segment_id=7,
                category="First Party Collection/Use",
                attributes={
                    "Purpose": {"value": "Advertising"},
                    "Collection Mode": {"value": "not-selected"},
                    "Personal Information Type": {"value": "Contact"},
                },
            )
            _write_row(
                csv_path,
                annotation_id=2,
                annotator_id=1,
                segment_id=7,
                category="First Party Collection/Use",
                attributes={
                    "Purpose": {"value": "Analytics/Research"},
                    "Personal Information Type": {"value": "Unspecified"},
                },
            )
            _write_row(
                csv_path,
                annotation_id=3,
                annotator_id=2,
                segment_id=7,
                category="Data Security",
                attributes={"Security Measure": {"value": "Generic"}},
            )

            segments = load_opp115_annotations(annotations, strict=True)

        self.assertEqual(len(segments), 1)
        segment = segments[0]
        self.assertEqual(segment.key, ("101_example.com", "7"))
        self.assertEqual(
            segment.categories,
            ("First Party Collection/Use", "Data Security"),
        )
        self.assertEqual(
            segment.attributes["purpose"],
            frozenset(("advertising", "analytics_research")),
        )
        self.assertEqual(
            segment.attributes["personal_information_type"],
            frozenset(("contact", "unspecified")),
        )
        self.assertNotIn("collection_mode", segment.attributes)
        self.assertTrue(segment.attribute_mask["collection_mode"])
        self.assertTrue(segment.attribute_mask["security_measure"])
        self.assertFalse(segment.attribute_mask["retention_period"])
        self.assertEqual(len(segment.mask_vector), 20)

    def test_schema_derivation_filters_held_out_values_and_encoder_reports_them(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            annotations = Path(temporary)
            train_path = annotations / "001_train.test.csv"
            eval_path = annotations / "002_eval.test.csv"
            _write_row(
                train_path,
                annotation_id=1,
                annotator_id=0,
                segment_id=0,
                category="Do Not Track",
                attributes={"Do Not Track policy": {"value": "Not honored"}},
            )
            _write_row(
                eval_path,
                annotation_id=2,
                annotator_id=0,
                segment_id=0,
                category="Do Not Track",
                attributes={"Do Not Track policy": {"value": "Honored"}},
            )
            all_segments = load_opp115_annotations(annotations, strict=True)
            schema = derive_hierarchy_schema(
                all_segments,
                ["001_train.test"],
                require_values_for_all_heads=False,
            )
            keyed_encoded = load_segment_annotations(
                annotations, schema, strict=True
            )

        self.assertEqual(schema.value_vocabulary["do_not_track_policy"], ("not_honored",))
        self.assertNotIn("honored", schema.flattened_attribute_labels)
        encoded = encode_segment_labels(all_segments[1], schema)
        self.assertEqual(encoded.category_targets[HIGH_LEVEL_CATEGORIES.index("Do Not Track")], 1)
        self.assertEqual(encoded.attribute_targets["do_not_track_policy"], (0,))
        self.assertEqual(
            encoded.unknown_attribute_values,
            {"do_not_track_policy": ("honored",)},
        )
        self.assertEqual(
            len(encoded.flattened_attribute_targets), schema.flattened_value_count
        )
        self.assertEqual(len(encoded.mask_vector), 20)
        self.assertTrue(encoded.attribute_mask["do_not_track_policy"])
        self.assertEqual(
            keyed_encoded[("002_eval.test", "0")].unknown_values,
            {"do_not_track_policy": ("honored",)},
        )

    def test_manifest_round_trip_and_tamper_detection(self) -> None:
        schema = HierarchySchema(
            training_policy_ids=("train-b", "train-a"),
            value_vocabulary={
                "purpose": ("unspecified", "advertising"),
                "do_not_track_policy": ("not_honored",),
            },
        )
        manifest = schema.to_manifest()
        restored = HierarchySchema.from_manifest(manifest)
        self.assertEqual(restored.schema_hash, schema.schema_hash)
        self.assertEqual(restored.attribute_slices, schema.attribute_slices)
        self.assertEqual(manifest["attribute_head_count"], 20)
        self.assertEqual(manifest["flattened_attribute_value_count"], 3)

        tampered = dict(manifest)
        tampered["training_policy_ids"] = ["different-policy"]
        with self.assertRaisesRegex(ValueError, "hash verification"):
            HierarchySchema.from_manifest(tampered)


@unittest.skipUnless(ANNOTATION_DIR is not None, "real OPP-115 data is unavailable")
class OPP115HierarchyRealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert ANNOTATION_DIR is not None
        cls.train_ids, cls.eval_ids = fixed_policy_split(ANNOTATION_DIR)
        cls.schema = derive_hierarchy_schema(ANNOTATION_DIR, cls.train_ids)

    def test_fixed_sorted_whole_policy_split(self) -> None:
        self.assertEqual(len(self.train_ids), 65)
        self.assertEqual(len(self.eval_ids), 50)
        self.assertFalse(set(self.train_ids) & set(self.eval_ids))
        self.assertEqual(self.train_ids[0], "1017_sci-news.com")
        self.assertEqual(self.train_ids[-1], "414_washingtonian.com")
        self.assertEqual(self.eval_ids[0], "453_barnesandnoble.com")

    def test_real_training_schema_has_twenty_heads_and_138_values(self) -> None:
        self.assertEqual(self.schema.attribute_head_count, 20)
        self.assertEqual(
            self.schema.flattened_value_count,
            EXPECTED_FLATTENED_TRAIN_VALUE_COUNT,
        )
        self.assertEqual(len(self.schema.flattened_attribute_labels), 138)
        self.assertEqual(self.schema.value_vocabulary["user_type"], (
            "other",
            "unspecified",
            "user_with_account",
            "user_without_account",
        ))

    def test_held_out_values_do_not_define_output_vocabulary(self) -> None:
        assert ANNOTATION_DIR is not None
        evaluation = load_opp115_annotations(ANNOTATION_DIR, self.eval_ids, strict=True)
        held_out_only = {head: set() for head in ATTRIBUTE_HEADS}
        for segment in evaluation:
            for head, values in segment.attributes.items():
                held_out_only[head].update(
                    set(values) - set(self.schema.value_vocabulary[head])
                )
        held_out_only = {
            head: values for head, values in held_out_only.items() if values
        }
        self.assertEqual(held_out_only, EXPECTED_HELD_OUT_ONLY_VALUES)
        self.assertNotIn("honored", self.schema.value_vocabulary["do_not_track_policy"])

    def test_real_rows_aggregate_to_known_segment_count(self) -> None:
        assert ANNOTATION_DIR is not None
        segments = load_opp115_annotations(ANNOTATION_DIR, strict=True)
        self.assertEqual(len(segments), 3792)
        self.assertEqual(len({segment.key for segment in segments}), 3792)
        self.assertTrue(
            all(len(segment.mask_vector) == len(ATTRIBUTE_HEADS) for segment in segments)
        )


if __name__ == "__main__":
    unittest.main()
