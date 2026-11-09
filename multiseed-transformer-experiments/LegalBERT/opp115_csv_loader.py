"""
opp115_csv_loader.py
====================
Drop-in replacement for the OPP115Dataset class inside ml_layer_embeddings.py.

WHY THIS FILE EXISTS
--------------------
The original loader expected:
    annotations/<policy_id>/<segment_id>.json

The actual OPP-115 public release (usableprivacy.org) ships as:
    annotations/<policy_id>.csv          ← one file per policy
    sanitized_policies/<policy_id>.html  ← segment text is here

This module provides a compatibility layer for both layouts.  The canonical
``OPP115Dataset`` class remains in ``ml_layer_embeddings.py``; this module owns
the public-release CSV parser and split validation used by that class.

HOW TO USE
----------
Replace the import in ml_layer_embeddings.py:

    # Old (JSON-only):
    # class OPP115Dataset: ...

    # New (CSV + JSON):
    from opp115_csv_loader import OPP115Dataset, OPP115Sample

Or patch at the top of train.py / main.py:

    from opp115_csv_loader import OPP115Dataset
    import ml_layer_embeddings
    ml_layer_embeddings.OPP115Dataset = OPP115Dataset

OPP-115 ANNOTATION CSV FORMAT
------------------------------
Each row in annotations/<policy_id>.csv represents ONE annotator's annotation
of ONE policy segment.  The public release uses these columns:

    annotation_id   – unique integer ID for this annotation row
    batch_id        – crowdsourcing batch
    annotator_id    – which of the three annotators (0-2)
    policy_id       – integer; matches the numeric prefix of the filename
    segment_id      – integer; identifies a segment within the policy
    category_name   – high-level privacy category label (see CATEGORY_MAP)
    a1_name         – attribute 1 name  (e.g. "user_type")
    a1_value        – attribute 1 value (e.g. "unspecified")
    a2_name         – attribute 2 name
    a2_value        – attribute 2 value
    ... up to a10_name / a10_value

Segment text is NOT stored in the CSV.  It is extracted from the HTML policy
file in sanitized_policies/<policy_id>.html by matching segment_id to the
order of paragraph/list-item blocks.
"""

import csv
import html
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from config import (
    OPP115_DIR,
    OPP115_TRAIN_POLICIES,
    OPP115_TEST_POLICIES,
    HIGH_LEVEL_CATEGORIES,
    ATTRIBUTES,
)

# OPP115Sample is still defined here so the rest of the codebase can import
# from this single module instead of ml_layer_embeddings.
from ml_layer_embeddings import OPP115Sample

logger = logging.getLogger(__name__)


def validate_opp115_split(
    policy_ids: Sequence[str], *, logger_obj: logging.Logger = logger
) -> None:
    """Validate that the current policy set can support the 65/50 split."""
    required_policies = OPP115_TRAIN_POLICIES + OPP115_TEST_POLICIES
    logger_obj.info(
        "Validating OPP-115 65/50 split prerequisites: %d policies available, %d required.",
        len(policy_ids),
        required_policies,
    )

    if len(policy_ids) != required_policies:
        logger_obj.error(
            "Cannot create exact 65/50 split: %d policies available, %d required.",
            len(policy_ids),
            required_policies,
        )
        raise ValueError(
            f"Exact 65/50 split requires {required_policies} policies; "
            f"found {len(policy_ids)}."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Category name normalisation
# The CSV uses slightly different strings from our taxonomy in config.py.
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_MAP: Dict[str, str] = {
    # CSV value                               → HIGH_LEVEL_CATEGORIES entry
    "First Party Collection/Use"            : "First Party Collection/Use",
    "Third Party Sharing/Collection"        : "Third Party Sharing/Collection",
    "User Choice/Control"                   : "User Choice/Control",
    "Data Retention"                        : "Data Retention",
    "Data Security"                         : "Data Security",
    "International and Specific Audiences"  : "International and Specific Audiences",
    "Do Not Track"                          : "Do Not Track",
    "Policy Change"                         : "Policy Change",
   
   
     # OPP-115 uses "User Access..." in its CSV files.
     "User Access, Edit and Deletion"        : "Access, Edit and Deletion",
    "Access, Edit and Deletion"             : "Access, Edit and Deletion",
    "Other"                                 : "Other",







    # Common alternate spellings in some releases:
    "First Party Collection"                : "First Party Collection/Use",
    "Third Party Sharing"                   : "Third Party Sharing/Collection",
    "User Choice"                           : "User Choice/Control",
    "Access/Edit/Delete"                    : "Access, Edit and Deletion",
    "Introductory/Generic"                  : "Other",
    "Practice Not Covered"                  : "Other",
    "Privacy Contact Information"           : "Other",
}

# ─────────────────────────────────────────────────────────────────────────────
# Attribute name normalisation
# CSV attribute names → keys used in config.ATTRIBUTES
# ─────────────────────────────────────────────────────────────────────────────

ATTR_NAME_MAP: Dict[str, str] = {
    # CSV column name           → config.ATTRIBUTES key
    "user_type"               : "audience_type",
    "audience_type"           : "audience_type",
    "information_type"        : "personal_information_type",
    "personal_information_type": "personal_information_type",
    "purpose"                 : "purpose",
    "identifiability"         : "identifiability",
    "does_does_not"           : "does_does_not",
    "does/does not"           : "does_does_not",
    "action"                  : "action_first_party",
    "action first-party"      : "action_first_party",
    "action_first_party"      : "action_first_party",
    "action third-party"      : "action_third_party",
    "action_third_party"      : "action_third_party",
    "third_party_entity"      : "third_party_entity",
    "third-party entity"      : "third_party_entity",
    "choice_type"             : "choice_type",
    "choice type"             : "choice_type",
    "choice_scope"            : "choice_scope",
    "choice scope"            : "choice_scope",
    "retention_period"        : "retention_period",
    "retention period"        : "retention_period",
    "retention_purpose"       : "retention_purpose",
    "retention purpose"       : "retention_purpose",
    "security_measure"        : "security_measure",
    "security measure"        : "security_measure",
    "do_not_track_policy"     : "do_not_track_policy",
    "do not track policy"     : "do_not_track_policy",
    "change_type"             : "change_type",
    "change type"             : "change_type",
    "notification_type"       : "notification_type",
    "notification type"       : "notification_type",
    "access_scope"            : "access_scope",
    "access scope"            : "access_scope",
    "access_type"             : "access_type",
    "access type"             : "access_type",
    "access_rights"           : "access_rights",
    "access rights"           : "access_rights",
    "collection_mode"         : "collection_mode",
    "collection mode"         : "collection_mode",
    "audience type"           : "audience_type",
}

# ─────────────────────────────────────────────────────────────────────────────
# Segment-text extraction from sanitised HTML
# ─────────────────────────────────────────────────────────────────────────────

class PolicyTextExtractor:
    """
    Extracts ordered paragraph/segment texts from an OPP-115 sanitised HTML
    policy file.  The OPP-115 annotators used sequential segment_id integers
    (0-based) that correspond to the order of <p> / list-item blocks in the
    HTML after stripping structural tags.
    """

    _cache: Dict[Path, List[str]] = {}

    @classmethod
    def get_segments(cls, html_path: str | Path) -> List[str]:
        """
        Return a list of segment strings indexed by segment_id.
        Results are cached so each policy file is read at most once.
        """
        path = Path(html_path).resolve()
        if path in cls._cache:
            return cls._cache[path]

        segments: List[str] = []
        raw = path.read_text(encoding="utf-8", errors="strict")

        # The public OPP-115 release stores one policy as a minimal HTML stream
        # with literal ||| segment boundaries. Split first so segment_id remains
        # aligned even when the fragments contain only <br> tags.
        if "|||" in raw:
            for fragment in raw.split("|||"):
                text = re.sub(r"<br\s*/?>", " ", fragment, flags=re.IGNORECASE)
                text = re.sub(r"<[^>]+>", " ", text)
                text = html.unescape(text).replace("\xa0", " ")
                segments.append(re.sub(r"\s+", " ", text).strip())
            cls._cache[path] = segments
            return segments

        try:
            from bs4 import BeautifulSoup  # type: ignore
            soup = BeautifulSoup(raw, "lxml")
            # Remove non-content tags
            for tag in soup.find_all(["script", "style", "header",
                                      "footer", "nav", "aside"]):
                tag.decompose()
            # Collect leaf text blocks in document order
            for elem in soup.find_all(["p", "li", "dt", "dd"]):
                text = elem.get_text(" ", strip=True)
                if len(text.split()) >= 3:
                    segments.append(text)
        except ImportError:
            # Fallback: plain regex extraction
            for m in re.finditer(
                r"<(?:p|li)[^>]*>(.*?)</(?:p|li)>", raw,
                flags=re.DOTALL | re.IGNORECASE,
            ):
                text = re.sub(r"<[^>]+>", " ", m.group(1)).strip()
                if len(text.split()) >= 3:
                    segments.append(text)
        except Exception as exc:
            logger.debug("Could not parse %s: %s", path, exc)

        # A small number of release files are plain-text documents rather than
        # HTML block streams (for example, 988_solarviews.com).  They still
        # represent a valid single segment with segment_id=0.
        if not segments:
            text = re.sub(r"<br\s*/?>", " ", raw, flags=re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = html.unescape(text).replace("\xa0", " ")
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                segments.append(text)

        cls._cache[path] = segments
        return segments

    @classmethod
    def get_segment_text(cls, html_path: str | Path, segment_id: int) -> str:
        """
        Return the text for a specific segment_id.
        Falls back to an empty string when the index is out of range.
        """
        segs = cls.get_segments(html_path)
        if 0 <= segment_id < len(segs):
            return segs[segment_id]
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# CSV parser
# ─────────────────────────────────────────────────────────────────────────────

class OPP115CSVParser:
    """
    Parses the real OPP-115 annotation CSV files (usableprivacy.org release).

    One CSV row represents one annotated data practice.  Multiple rows may
    legitimately refer to the same policy segment, so they remain separate
    training examples.  This preserves the 23,177-practice public release and
    avoids silently combining potentially different practices or annotators.
    """

    def __init__(
        self,
        ann_dir: str | Path,
        policy_dir: str | Path,
    ):
        self.ann_dir = Path(ann_dir)
        self.policy_dir = Path(policy_dir)

    # ── Public entry point ───────────────────────────────────────────────────

    def parse_all(self) -> List[OPP115Sample]:
        """
        Parse every .csv file in ann_dir and return a flat list of
        OPP115Sample objects.
        """
        samples: List[OPP115Sample] = []
        csv_files = sorted(self.ann_dir.glob("*.csv"))

        if not csv_files:
            logger.warning(
                "No CSV files found in %s. "
                "Verify the OPP-115 dataset path.", self.ann_dir
            )
            return samples

        for csv_path in csv_files:
            policy_samples = self._parse_csv(csv_path)
            samples.extend(policy_samples)
            logger.debug("  %s -> %d segments", csv_path.name, len(policy_samples))

        logger.info(
            "OPP-115 CSV loader: parsed %d samples from %d policy files.",
            len(samples), len(csv_files),
        )
        return samples

    # ── Per-file parsing ─────────────────────────────────────────────────────

    def _parse_csv(self, csv_path: Path) -> List[OPP115Sample]:
        """
        Parse one <policy_id>.csv annotation file.

        Steps
        -----
        1. Read all rows and detect the column layout.
        2. For each annotation row (data practice):
           a. Extract category and attributes.
           b. Look up segment text from the sanitised policy HTML.
           c. Create one OPP115Sample per practice.
        3. Return all OPP115Sample objects (one per annotated practice).
        """
        policy_id = csv_path.stem   # e.g. "1017_sci-news.com"

        # Read raw rows
        rows = self._read_csv(csv_path)
        if not rows:
            return []

        # Detect column layout from the first row's keys
        headers = list(rows[0].keys())
        seg_id_col  = self._find_col(headers, ["segment_id", "segmentid", "seg_id"])
        cat_col     = self._find_col(headers, ["category_name", "category",
                                               "high_level_category"])
        attr_pairs  = self._detect_attr_pairs(headers)
        json_col    = self._find_col(headers, ["attributes_json", "attributes",
                                               "annotation_attrs", "raw_attributes"])

        if seg_id_col is None or cat_col is None:
            logger.warning("  Cannot identify required columns in %s (headers: %s)",
                           csv_path.name, headers[:10])
            return []

        # Find the sanitised HTML for segment text
        html_path = self._find_html(policy_id)

        samples: List[OPP115Sample] = []

        for row in rows:
            try:
                seg_id = int(row.get(seg_id_col, -1))
            except (ValueError, TypeError):
                continue
            if seg_id < 0:
                continue

            raw_cat = (row.get(cat_col) or "").strip()
            norm_cat = CATEGORY_MAP.get(raw_cat, "")
            if not norm_cat:
                continue

            attributes: Dict[str, List[str]] = {}
            if attr_pairs:
                for name_col, val_col in attr_pairs:
                    # Standalone columns use the header as the attribute name;
                    # paired columns store the name in a separate cell.
                    raw_name = (
                        name_col if name_col == val_col else row.get(name_col)
                    ) or ""
                    raw_val = row.get(val_col) or ""
                    raw_name = str(raw_name).strip().lower()
                    raw_val = str(raw_val).strip().lower()
                    if not raw_name or not raw_val or raw_val == "n/a":
                        continue
                    self._add_attribute(attributes, raw_name, raw_val)
            elif json_col:
                for raw_name, raw_val in self._parse_json_attributes(
                    row.get(json_col, "")
                ):
                    if not raw_name or not raw_val:
                        continue
                    self._add_attribute(attributes, raw_name, raw_val)

            if not attributes:
                attributes = {}

            if html_path:
                text = PolicyTextExtractor.get_segment_text(html_path, seg_id)
            else:
                text = ""

            if not text:
                text = self._inline_text(row, headers)

            if not text and json_col:
                text = self._segment_text_from_annotations([row], json_col)

            if not text:
                continue

            samples.append(OPP115Sample(
                policy_id  = policy_id,
                segment_id = str(seg_id),
                text       = text,
                categories = [norm_cat],
                attributes = attributes,
            ))

        return samples

    # ── Helper methods ────────────────────────────────────────────────────────

    @staticmethod
    def _read_csv(csv_path: Path) -> List[Dict[str, str]]:
        """Read a CSV with automatic dialect sniffing. Returns list of dicts."""
        rows: List[Dict[str, str]] = []
        try:
            with csv_path.open(encoding="utf-8-sig", errors="replace", newline="") as f:
                sample = f.read(4096)
                f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",\t|")
                except csv.Error:
                    dialect = csv.excel  # type: ignore[assignment]

                f.seek(0)
                reader = csv.reader(f, dialect=dialect)
                all_rows = [row for row in reader if row]

            if not all_rows:
                return []

            first_row = [cell.strip().lower() for cell in all_rows[0]]
            header_candidates = {
                "segment_id", "segmentid", "seg_id",
                "annotator_id", "policy_id",
                "category_name", "category", "high_level_category",
                "a1_name", "information_type", "purpose",
            }
            if any(h in header_candidates for h in first_row):
                with csv_path.open(encoding="utf-8-sig", errors="replace", newline="") as f:
                    reader = csv.DictReader(f, dialect=dialect)
                    rows = [dict(row) for row in reader]
            else:
                fieldnames = [
                    "annotation_id", "batch_id", "annotator_id", "policy_id",
                    "segment_id", "category_name", "attributes_json",
                    "policy_url", "date",
                ]
                for row in all_rows:
                    row = row + [""] * (len(fieldnames) - len(row))
                    row = row[: len(fieldnames)]
                    rows.append(dict(zip(fieldnames, row)))
        except Exception as exc:
            logger.warning("  Failed to read %s: %s", csv_path, exc)
        return rows

    @staticmethod
    def _find_col(headers: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
        """Return the first header name that matches any candidate (case-insensitive)."""
        h_lower = [h.lower().strip() for h in headers]
        for c in candidates:
            if c.lower() in h_lower:
                return headers[h_lower.index(c.lower())]
        return None

    @staticmethod
    def _detect_attr_pairs(headers: Sequence[str]) -> List[Tuple[str, str]]:
        """
        Detect (a1_name, a1_value) column pairs.

        Supports two layouts used by different OPP-115 releases:
          Layout A: a1_name / a1_value, a2_name / a2_value, ...  (integer suffix)
          Layout B: explicit attribute columns like information_type, purpose, ...
        """
        pairs: List[Tuple[str, str]] = []

        # ── Layout A: suffixed pairs ─────────────────────────────────────────
        for i in range(1, 16):
            name_col = f"a{i}_name"
            val_col  = f"a{i}_value"
            # Case-insensitive check
            h_lower  = [h.lower().strip() for h in headers]
            if name_col in h_lower and val_col in h_lower:
                pairs.append((
                    headers[h_lower.index(name_col)],
                    headers[h_lower.index(val_col)],
                ))

        if pairs:
            return pairs

        # ── Layout B: standalone attribute columns ───────────────────────────
        # Each attribute column is treated as both name and value:
        #   column name → attribute name, cell value → attribute value
        known_attr_cols = set(ATTR_NAME_MAP.keys())
        for h in headers:
            if h.lower().strip() in known_attr_cols:
                # synthetic "self-pair": (header, header) — parsed specially below
                pairs.append((h, h))

        return pairs

    def _find_html(self, policy_id: str) -> Optional[Path]:
        """
        Locate the sanitised HTML file for *policy_id*.

        Tries several filename conventions used across OPP-115 releases:
          <policy_dir>/<policy_id>.html
          <policy_dir>/<numeric_prefix>.html
        """
        if not self.policy_dir.is_dir():
            return None

        # Direct match
        direct = self.policy_dir / f"{policy_id}.html"
        if direct.is_file():
            return direct

        # Numeric-prefix match (e.g. "1017_sci-news.com" → "1017.html")
        prefix = policy_id.split("_")[0]
        by_prefix = self.policy_dir / f"{prefix}.html"
        if by_prefix.is_file():
            return by_prefix

        # Glob fallback
        matches = sorted(self.policy_dir.glob(f"{prefix}*.html"))
        if matches:
            return matches[0]

        return None

    @staticmethod
    def _inline_text(row: Mapping[str, str], headers: Sequence[str]) -> str:
        """
        Extract inline segment text from a CSV row if the dataset embeds it.
        Looks for common column names: 'segment_text', 'text', 'content'.
        """
        for col in ["segment_text", "text", "content", "segment"]:
            for h in headers:
                if h.lower().strip() == col:
                    val = (row.get(h) or "").strip()
                    if val:
                        return val
        return ""

    @staticmethod
    def _add_attribute(
        attributes: Dict[str, List[str]], raw_name: str, raw_val: str
    ) -> None:
        """Normalize and add one attribute/value pair into the attribute dict."""
        name_key = re.sub(r"[^a-z0-9]+", "_", raw_name.lower()).strip("_")
        norm_name = ATTR_NAME_MAP.get(raw_name, ATTR_NAME_MAP.get(name_key, name_key))
        if norm_name not in ATTRIBUTES:
            return

        invalid_values = {
            "", "n/a", "not-selected", "not selected",
            "null", "not specified", "not-specified",
        }
        raw_val = raw_val.strip().lower()
        if not raw_val or raw_val in invalid_values:
            return

        # OPP-115 uses values such as "Basic service/feature" and
        # "IP address and device IDs".  Canonical slugging maps these directly
        # to the vocabulary in config.ATTRIBUTES.
        norm_val = re.sub(r"[^a-z0-9]+", "-", raw_val).strip("-")
        if norm_val not in ATTRIBUTES[norm_name]:
            norm_val = OPP115CSVParser._fuzzy_match_value(
                norm_val, ATTRIBUTES[norm_name]
            )
        if not norm_val:
            return

        if norm_name not in attributes:
            attributes[norm_name] = []
        if norm_val not in attributes[norm_name]:
            attributes[norm_name].append(norm_val)

    @staticmethod
    def _parse_json_attributes(raw_json: str) -> List[Tuple[str, str]]:
        try:
            data = json.loads(raw_json)
        except (json.JSONDecodeError, TypeError):
            return []

        if not isinstance(data, dict):
            return []

        attributes: List[Tuple[str, str]] = []
        for raw_name, value in data.items():
            if isinstance(value, dict):
                val = value.get("value") or value.get("selectedText") or ""
            else:
                val = str(value)
            attributes.append((str(raw_name).strip().lower(), str(val).strip().lower()))
        return attributes

    @staticmethod
    def _extract_selected_text_from_json(raw_json: str) -> str:
        try:
            data = json.loads(raw_json)
        except (json.JSONDecodeError, TypeError):
            return ""

        if not isinstance(data, dict):
            return ""

        values: List[str] = []
        for value in data.values():
            if isinstance(value, dict):
                selected = value.get("selectedText")
                if isinstance(selected, str):
                    selected = selected.strip()
                    if selected and selected.lower() != "null":
                        values.append(selected)
        return max(values, key=len, default="")

    def _segment_text_from_annotations(
        self, ann_rows: List[Dict], json_col: str
    ) -> str:
        selected_texts: List[str] = []
        for row in ann_rows:
            selected = self._extract_selected_text_from_json(row.get(json_col, ""))
            if selected:
                selected_texts.append(selected)
        return max(selected_texts, key=len, default="")

    @staticmethod
    def _fuzzy_match_value(raw: str, allowed: List[str]) -> str:
        """
        Try substring / prefix matching when the raw value does not exactly
        match any allowed value.  Returns the best match or empty string.
        """
        # Exact match after hyphen normalisation
        raw_hyph = raw.replace(" ", "-").replace("_", "-")
        if raw_hyph in allowed:
            return raw_hyph

        # Substring match
        for a in allowed:
            if raw_hyph in a or a in raw_hyph:
                return a

        # First-token match (e.g. "opt-out ..." → "opt-out-link")
        first_tok = raw_hyph.split("-")[0]
        candidates = [a for a in allowed if a.startswith(first_tok)]
        if len(candidates) == 1:
            return candidates[0]

        return ""


# ─────────────────────────────────────────────────────────────────────────────
# JSON-to-CSV / CSV-to-JSON conversion utilities
# ─────────────────────────────────────────────────────────────────────────────

def convert_csv_to_json(opp115_dir: str, output_dir: str) -> int:
    """
    Convert the real OPP-115 CSV annotations into the JSON subdirectory
    layout originally expected by ml_layer_embeddings.OPP115Dataset.

    Output structure:
        <output_dir>/annotations/<policy_id>/<segment_id>.json

    This lets you use the original JSON loader without changing any other code.
    Returns the number of JSON files written.
    """
    source = Path(opp115_dir)
    destination = Path(output_dir)
    parser = OPP115CSVParser(
        source / "annotations", source / "sanitized_policies"
    )
    samples = parser.parse_all()

    # Several annotation rows can describe the same segment.  Group them
    # before writing so later practices do not overwrite earlier ones.
    grouped: Dict[Tuple[str, str], Dict[str, object]] = {}
    for sample in samples:
        key = (sample.policy_id, sample.segment_id)
        entry = grouped.setdefault(
            key,
            {
                "segment_text": sample.text,
                "annotations": [],
            },
        )
        annotations = entry["annotations"]
        assert isinstance(annotations, list)
        annotations.extend(
            {"category": category, "attributes": sample.attributes}
            for category in sample.categories
        )

    written = 0
    for (policy_id, segment_id), payload in grouped.items():
        policy_output = destination / "annotations" / policy_id
        policy_output.mkdir(parents=True, exist_ok=True)
        output_file = policy_output / f"{segment_id}.json"
        output_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written += 1

    logger.info("Converted %d samples to JSON at %s", written, output_dir)
    return written


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible OPP115Dataset (auto-detects CSV vs JSON)
# ─────────────────────────────────────────────────────────────────────────────

class OPP115Dataset:
    """
    Backward-compatible wrapper that auto-detects whether the OPP-115
    directory contains CSV files (real public dataset) or JSON subdirectories
    (synthetic / pre-converted data) and picks the correct parser.

    This class is a drop-in replacement for the original OPP115Dataset in
    ml_layer_embeddings.py.  All downstream code that calls:

        dataset = OPP115Dataset(OPP115_DIR).load()
        train, test = dataset.train_test_split()

    will work without modification.
    """

    def __init__(self, opp115_dir: str = OPP115_DIR):
        self.opp115_dir = opp115_dir
        self.ann_dir    = os.path.join(opp115_dir, "annotations")
        self.policy_dir = os.path.join(opp115_dir, "sanitized_policies")
        self._samples: List[OPP115Sample] = []
        self._loaded = False

    # ── Public API ───────────────────────────────────────────────────────────

    def load(self) -> "OPP115Dataset":
        if self._loaded:
            return self

        if not os.path.isdir(self.ann_dir):
            raise FileNotFoundError(
                f"OPP-115 annotations not found at {self.ann_dir}.\n"
                "Download from: https://usableprivacy.org/data"
            )

        if self._has_csv_files():
            logger.info("Detected CSV format — using OPP115CSVParser.")
            parser = OPP115CSVParser(self.ann_dir, self.policy_dir)
            self._samples = parser.parse_all()
        else:
            logger.info("Detected JSON format — using original JSON loader.")
            self._samples = self._load_json()

        if not self._samples:
            raise RuntimeError(
                f"OPP115Dataset loaded 0 samples from {self.opp115_dir}.\n"
                "Possible causes:\n"
                "  1. Segment text could not be extracted — confirm that\n"
                f"     {self.policy_dir}\n"
                "     contains the sanitized HTML policy files.\n"
                "  2. CSV column names differ from expected — check the first\n"
                "     few rows of your annotation CSV and compare to ATTR_NAME_MAP\n"
                "     in opp115_csv_loader.py.\n"
                "  3. Run the conversion utility to produce JSON and use that:\n"
                "       from opp115_csv_loader import convert_csv_to_json\n"
                f"       convert_csv_to_json('{self.opp115_dir}', 'data/opp115_json')\n"
            )

        logger.info("OPP-115 loaded: %d annotated segments.", len(self._samples))
        self._loaded = True
        return self

    def train_test_split(
        self,
    ) -> Tuple[List[OPP115Sample], List[OPP115Sample]]:
        """65 / 50 policy-level split, replicating the paper."""
        self.load()
        policy_ids = sorted({s.policy_id for s in self._samples})
        validate_opp115_split(policy_ids)

        from opp115_policy_split import split_policy_ids
        train_id_order, test_id_order = split_policy_ids(
            policy_ids,
            train_count=OPP115_TRAIN_POLICIES,
            eval_count=OPP115_TEST_POLICIES,
        )
        train_ids = set(train_id_order)
        test_ids = set(test_id_order)
        train = [s for s in self._samples if s.policy_id in train_ids]
        test  = [s for s in self._samples if s.policy_id in test_ids]
        logger.info(
            "Split → train: %d segments (%d policies), "
            "test: %d segments (%d policies).",
            len(train), len(train_ids), len(test), len(test_ids),
        )
        return train, test

    # ── Encoding helpers (identical to original) ─────────────────────────────

    @staticmethod
    def encode_categories(sample: OPP115Sample):
        import numpy as np
        vec = np.zeros(len(HIGH_LEVEL_CATEGORIES), dtype=np.float32)
        for cat in sample.categories:
            if cat in HIGH_LEVEL_CATEGORIES:
                vec[HIGH_LEVEL_CATEGORIES.index(cat)] = 1.0
        return vec

    @staticmethod
    def encode_attribute(sample: OPP115Sample, attr_name: str):
        import numpy as np
        values = ATTRIBUTES.get(attr_name, [])
        vec = np.zeros(len(values), dtype=np.float32)
        for val in sample.attributes.get(attr_name, []):
            if val in values:
                vec[values.index(val)] = 1.0
        return vec

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _has_csv_files(self) -> bool:
        """Return True if the annotations directory contains .csv files."""
        return any(Path(self.ann_dir).glob("*.csv"))

    def _load_json(self) -> List[OPP115Sample]:
        """
        Original JSON loader (kept for backward compatibility with
        synthetic / pre-converted datasets).
        """
        import json as _json
        samples: List[OPP115Sample] = []

        for policy_dir in sorted(Path(self.ann_dir).iterdir()):
            if not policy_dir.is_dir():
                continue
            policy_id = policy_dir.name
            for ann_file in sorted(policy_dir.glob("*.json")):
                try:
                    with open(ann_file, encoding="utf-8") as f:
                        data = _json.load(f)
                    text = data.get("segment_text", "").strip()
                    if not text:
                        continue
                    categories: List[str] = []
                    attributes: Dict[str, List[str]] = {}
                    for ann in data.get("annotations", []):
                        cat = ann.get("category", "")
                        if cat and cat not in categories:
                            categories.append(cat)
                        for k, vals in ann.get("attributes", {}).items():
                            if k not in attributes:
                                attributes[k] = []
                            for v in (vals if isinstance(vals, list) else [vals]):
                                if v and v not in attributes[k]:
                                    attributes[k].append(v)
                    if not categories:
                        continue
                    samples.append(OPP115Sample(
                        policy_id  = policy_id,
                        segment_id = ann_file.stem,
                        text       = text,
                        categories = categories,
                        attributes = attributes,
                    ))
                except Exception as exc:
                    logger.debug("Skipping %s: %s", ann_file, exc)

        return samples


# ─────────────────────────────────────────────────────────────────────────────
# Quick diagnostic  (run directly to verify your dataset path)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        stream = sys.stdout,
    )

    opp115_dir = sys.argv[1] if len(sys.argv) > 1 else OPP115_DIR
    print(f"\nDiagnosing OPP-115 at: {opp115_dir}")
    print("-" * 60)

    # Show what's in annotations/
    ann_dir = os.path.join(opp115_dir, "annotations")
    if os.path.isdir(ann_dir):
        entries = list(Path(ann_dir).iterdir())
        csv_files  = [e for e in entries if e.suffix == ".csv"]
        json_dirs  = [e for e in entries if e.is_dir()]
        print(f"  annotations/ contains: {len(csv_files)} CSV files, "
              f"{len(json_dirs)} subdirectories")
        if csv_files:
            print(f"  First CSV: {csv_files[0].name}")
            with open(csv_files[0], newline="", encoding="utf-8") as f:
                header_line = f.readline().strip()
            print(f"  Headers:   {header_line[:200]}")
    else:
        print(f"  annotations/ NOT FOUND at {ann_dir}")

    # Show what's in sanitized_policies/
    pol_dir = os.path.join(opp115_dir, "sanitized_policies")
    if os.path.isdir(pol_dir):
        html_files = list(Path(pol_dir).glob("*.html"))
        print(f"  sanitized_policies/ contains: {len(html_files)} HTML files")
    else:
        print(f"  sanitized_policies/ NOT FOUND at {pol_dir}")

    # Try loading
    print("\nAttempting to load …")
    try:
        ds = OPP115Dataset(opp115_dir).load()
        train, test = ds.train_test_split()
        print(f"\n  RESULT:")
        print(f"    Total segments : {len(ds._samples)}")
        print(f"    Train segments : {len(train)}")
        print(f"    Test  segments : {len(test)}")
        print(f"    Unique policies: {len({s.policy_id for s in ds._samples})}")
        if ds._samples:
            s0 = ds._samples[0]
            print(f"\n  Sample[0]:")
            print(f"    policy_id  : {s0.policy_id}")
            print(f"    segment_id : {s0.segment_id}")
            print(f"    text[:80]  : {s0.text[:80]}")
            print(f"    categories : {s0.categories}")
            print(f"    attributes : {dict(list(s0.attributes.items())[:3])}")
    except Exception as exc:
        print(f"\n  ERROR: {exc}")
        print("\nSuggested fix — convert to JSON and re-run:")
        print("  from opp115_csv_loader import convert_csv_to_json")
        print(f"  convert_csv_to_json('{opp115_dir}', 'data/opp115_json')")
        print("  Then set OPP115_DIR = 'data/opp115_json' in config.py")
