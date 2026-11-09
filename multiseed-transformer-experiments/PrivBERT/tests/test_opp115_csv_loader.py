"""Regression tests for the public-release OPP-115 CSV parser."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from opp115_csv_loader import OPP115CSVParser, PolicyTextExtractor


class OPP115CSVParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.annotations = root / "annotations"
        self.policies = root / "sanitized_policies"
        self.annotations.mkdir()
        self.policies.mkdir()
        self.parser = OPP115CSVParser(self.annotations, self.policies)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_public_headerless_row_and_taxonomy_normalization(self) -> None:
        attributes = {
            "Purpose": {"value": "Basic service/feature"},
            "Personal Information Type": {
                "value": "IP address and device IDs"
            },
            "Collection Mode": {"value": "Unspecified"},
        }
        csv_path = self.annotations / "101_example.com.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerow(
                [
                    "1",
                    "batch",
                    "7",
                    "101",
                    "0",
                    "First Party Collection/Use",
                    json.dumps(attributes),
                    "Not specified",
                    "https://example.com/privacy",
                ]
            )
        (self.policies / "101_example.com.html").write_text(
            "<html><body><p>We collect information to provide the service.</p>"
            "</body></html>",
            encoding="utf-8",
        )

        samples = self.parser.parse_all()

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].categories, ["First Party Collection/Use"])
        self.assertEqual(
            samples[0].attributes["purpose"], ["basic-service-feature"]
        )
        self.assertEqual(
            samples[0].attributes["personal_information_type"],
            ["ip-address-and-device-ids"],
        )
        self.assertEqual(samples[0].attributes["collection_mode"], ["unspecified"])

    def test_standalone_attribute_columns_use_header_as_name(self) -> None:
        csv_path = self.annotations / "202_example.org.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "segment_id",
                    "category_name",
                    "segment_text",
                    "purpose",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "segment_id": "3",
                    "category_name": "First Party Collection/Use",
                    "segment_text": "We use data for advertising purposes.",
                    "purpose": "Advertising",
                }
            )

        samples = self.parser.parse_all()

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].attributes["purpose"], ["advertising"])

    def test_public_policy_delimiter_preserves_segment_indexes(self) -> None:
        policy_path = self.policies / "303_example.net.html"
        policy_path.write_text(
            "First <b>segment</b><br>text|||Second &amp; final segment",
            encoding="utf-8",
        )

        segments = PolicyTextExtractor.get_segments(policy_path)

        self.assertEqual(segments, ["First segment text", "Second & final segment"])
        self.assertEqual(
            PolicyTextExtractor.get_segment_text(policy_path, 1),
            "Second & final segment",
        )

    def test_plain_text_release_policy_is_a_single_segment(self) -> None:
        policy_path = self.policies / "404_plain.example.html"
        policy_path.write_text(
            "Privacy statement. We use cookies for analytics.",
            encoding="utf-8",
        )

        self.assertEqual(
            PolicyTextExtractor.get_segments(policy_path),
            ["Privacy statement. We use cookies for analytics."],
        )
        self.assertEqual(
            PolicyTextExtractor.get_segment_text(policy_path, 0),
            "Privacy statement. We use cookies for analytics.",
        )


if __name__ == "__main__":
    unittest.main()
