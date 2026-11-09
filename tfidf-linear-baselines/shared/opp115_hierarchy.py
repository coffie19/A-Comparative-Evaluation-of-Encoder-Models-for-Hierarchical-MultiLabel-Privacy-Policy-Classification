"""Faithful OPP-115/Polisis hierarchical labels and CSV aggregation.

The public OPP-115 release stores one *practice* annotation per headerless CSV
row.  Several annotators and several practices may refer to the same policy
segment, so training examples must be formed by taking their union.  This
module keeps the ten published categories and the twenty genuine attribute
types in ``categories-july30.xml``.  ``Other Type`` and the Policy Change
``User Choice`` attribute are deliberately excluded from the attribute heads.

Attribute output vocabularies are data dependent.  They must be derived from
training policies only; otherwise a label that occurs solely in a held-out
policy silently leaks into the classifier output space.  The resulting
``HierarchySchema`` is immutable, deterministic, hashable through its manifest
and directly usable to encode category and per-attribute multi-hot targets.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any


SCHEMA_VERSION = "opp115-polisis-hierarchy-v1"
NORMALIZATION_VERSION = "opp115-slug-v1"
EXPECTED_POLICY_COUNT = 115
DEFAULT_TRAIN_POLICY_COUNT = 65
DEFAULT_EVAL_POLICY_COUNT = 50

# Public names used by OPP-115 and by the segment classifier.  The order is a
# model contract and must not be inferred from filesystem or dictionary order.
HIGH_LEVEL_CATEGORIES: tuple[str, ...] = (
    "First Party Collection/Use",
    "Third Party Sharing/Collection",
    "User Choice/Control",
    "Data Retention",
    "Data Security",
    "International and Specific Audiences",
    "Do Not Track",
    "Policy Change",
    "Access, Edit and Deletion",
    "Other",
)

# First-occurrence order in the published taxonomy.  These are the twenty
# genuine raw attribute types after excluding ``Other Type`` and the separate
# Policy Change attribute named ``User Choice``.
ATTRIBUTE_HEADS: tuple[str, ...] = (
    "does_does_not",
    "collection_mode",
    "action_first_party",
    "identifiability",
    "personal_information_type",
    "purpose",
    "user_type",
    "choice_type",
    "choice_scope",
    "third_party_entity",
    "action_third_party",
    "access_type",
    "access_scope",
    "retention_period",
    "retention_purpose",
    "security_measure",
    "change_type",
    "notification_type",
    "do_not_track_policy",
    "audience_type",
)

# Backward-friendly name for classifier code.  This is an alias for the head
# names only; values live exclusively in a training-derived HierarchySchema.
ATTRIBUTES: tuple[str, ...] = ATTRIBUTE_HEADS

RAW_ATTRIBUTE_NAMES: Mapping[str, str] = MappingProxyType(
    {
        "does_does_not": "Does/Does Not",
        "collection_mode": "Collection Mode",
        "action_first_party": "Action First-Party",
        "identifiability": "Identifiability",
        "personal_information_type": "Personal Information Type",
        "purpose": "Purpose",
        "user_type": "User Type",
        "choice_type": "Choice Type",
        "choice_scope": "Choice Scope",
        "third_party_entity": "Third Party Entity",
        "action_third_party": "Action Third Party",
        "access_type": "Access Type",
        "access_scope": "Access Scope",
        "retention_period": "Retention Period",
        "retention_purpose": "Retention Purpose",
        "security_measure": "Security Measure",
        "change_type": "Change Type",
        "notification_type": "Notification Type",
        "do_not_track_policy": "Do Not Track policy",
        "audience_type": "Audience Type",
    }
)

EXCLUDED_ATTRIBUTE_HEADS: tuple[str, ...] = ("other_type", "user_choice")


def _ordered_heads(*heads: str) -> tuple[str, ...]:
    selected = set(heads)
    return tuple(head for head in ATTRIBUTE_HEADS if head in selected)


# Applicability is taxonomy-defined, not inferred from which values happened
# to be observed.  Optional attributes are still applicable and therefore have
# a true loss mask when their parent category is present.
CATEGORY_TO_ATTRIBUTES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "First Party Collection/Use": _ordered_heads(
            "does_does_not",
            "collection_mode",
            "action_first_party",
            "identifiability",
            "personal_information_type",
            "purpose",
            "user_type",
            "choice_type",
            "choice_scope",
        ),
        "Third Party Sharing/Collection": _ordered_heads(
            "third_party_entity",
            "does_does_not",
            "action_third_party",
            "identifiability",
            "personal_information_type",
            "purpose",
            "user_type",
            "choice_type",
            "choice_scope",
        ),
        "User Choice/Control": _ordered_heads(
            "choice_type",
            "choice_scope",
            "personal_information_type",
            "purpose",
            "user_type",
        ),
        "Data Retention": _ordered_heads(
            "retention_period", "retention_purpose", "personal_information_type"
        ),
        "Data Security": _ordered_heads("security_measure"),
        "International and Specific Audiences": _ordered_heads("audience_type"),
        "Do Not Track": _ordered_heads("do_not_track_policy"),
        "Policy Change": _ordered_heads("change_type", "notification_type"),
        "Access, Edit and Deletion": _ordered_heads(
            "access_type", "access_scope", "user_type"
        ),
        "Other": (),
    }
)


def _normalization_key(value: object) -> str:
    """Return a conservative ASCII-like key for taxonomy matching.

    Apostrophes are removed (rather than treated as word separators), making
    ``Don't`` and the release's ``Dont`` equivalent.  Other punctuation,
    whitespace, slashes and hyphens become a single underscore.
    """

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    text = re.sub(r"['\u2018\u2019\u02bc`]", "", text)
    text = text.replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


_CATEGORY_ALIASES: dict[str, str] = {
    _normalization_key(category): category for category in HIGH_LEVEL_CATEGORIES
}
_CATEGORY_ALIASES.update(
    {
        "user_access_edit_and_deletion": "Access, Edit and Deletion",
        "user_access_edit_deletion": "Access, Edit and Deletion",
        "access_edit_delete": "Access, Edit and Deletion",
        "first_party_collection": "First Party Collection/Use",
        "third_party_sharing": "Third Party Sharing/Collection",
        "user_choice": "User Choice/Control",
        "introductory_generic": "Other",
        "practice_not_covered": "Other",
        "privacy_contact_information": "Other",
    }
)

_ATTRIBUTE_ALIASES: dict[str, str] = {
    _normalization_key(raw_name): head for head, raw_name in RAW_ATTRIBUTE_NAMES.items()
}
_ATTRIBUTE_ALIASES.update({head: head for head in ATTRIBUTE_HEADS})
_ATTRIBUTE_ALIASES.update(
    {
        "action_firstparty": "action_first_party",
        "action_thirdparty": "action_third_party",
        "do_not_track": "do_not_track_policy",
        "dnt_policy": "do_not_track_policy",
        "information_type": "personal_information_type",
    }
)
_ATTRIBUTE_ALIASES.update(
    {
        _normalization_key("Other Type"): "other_type",
        _normalization_key("User Choice"): "user_choice",
    }
)


def normalize_category(raw_category: object, *, strict: bool = False) -> str | None:
    """Normalize an OPP-115 category to its canonical display name."""

    normalized = _CATEGORY_ALIASES.get(_normalization_key(raw_category))
    if normalized is None and strict:
        raise ValueError(f"Unknown OPP-115 category: {raw_category!r}")
    return normalized


def normalize_attribute_name(
    raw_name: object, *, strict: bool = False
) -> str | None:
    """Normalize a raw attribute name to a stable snake-case head name.

    The two deliberately excluded names normalize successfully so callers can
    distinguish an intentional exclusion from a truly unknown attribute.
    """

    normalized = _ATTRIBUTE_ALIASES.get(_normalization_key(raw_name))
    if normalized is None and strict:
        raise ValueError(f"Unknown OPP-115 attribute: {raw_name!r}")
    return normalized


def normalize_attribute_value(raw_value: object) -> str | None:
    """Normalize a raw value; ignore only the ``not-selected`` sentinel.

    In particular, ``Unspecified``, ``Other`` and ``None`` (the published
    Access Type value) remain real labels.
    """

    if raw_value is None:
        return None
    normalized = _normalization_key(raw_value)
    if not normalized or normalized == "not_selected":
        return None
    return normalized


def _iter_payload_values(payload: object) -> Iterator[object]:
    if isinstance(payload, Mapping):
        value = payload.get("value")
    else:
        value = payload
    if isinstance(value, (list, tuple, set, frozenset)):
        yield from value
    else:
        yield value


def extract_attribute_values(
    raw_attributes: object, *, strict: bool = False
) -> dict[str, set[str]]:
    """Parse and normalize the attribute JSON from an annotation row.

    Unknown and deliberately excluded attributes are omitted.  Malformed JSON
    produces an empty mapping unless ``strict=True``.
    """

    if isinstance(raw_attributes, Mapping):
        parsed: object = raw_attributes
    else:
        try:
            parsed = json.loads(str(raw_attributes))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            if strict:
                raise ValueError("Malformed OPP-115 attribute JSON") from exc
            return {}

    if not isinstance(parsed, Mapping):
        if strict:
            raise ValueError("OPP-115 attribute JSON must contain an object")
        return {}

    extracted: dict[str, set[str]] = defaultdict(set)
    for raw_name, payload in parsed.items():
        head = normalize_attribute_name(raw_name, strict=strict)
        if head is None or head in EXCLUDED_ATTRIBUTE_HEADS:
            continue
        for raw_value in _iter_payload_values(payload):
            value = normalize_attribute_value(raw_value)
            if value is not None:
                extracted[head].add(value)
    return dict(extracted)


@dataclass(frozen=True)
class AnnotationPractice:
    """One normalized practice row from a public OPP-115 CSV file."""

    policy_id: str
    segment_id: str
    category: str
    attributes: Mapping[str, frozenset[str]]
    annotation_id: str = ""
    annotator_id: str = ""
    source_line: int = 0

    def __post_init__(self) -> None:
        frozen = {
            head: frozenset(values)
            for head, values in self.attributes.items()
            if head in ATTRIBUTE_HEADS
        }
        object.__setattr__(self, "attributes", MappingProxyType(frozen))


@dataclass(frozen=True)
class SegmentAnnotation:
    """Union of all practices/annotators for one policy segment."""

    policy_id: str
    segment_id: str
    categories: tuple[str, ...]
    attributes: Mapping[str, frozenset[str]]
    attribute_mask: Mapping[str, bool]

    def __post_init__(self) -> None:
        categories = tuple(
            category
            for category in HIGH_LEVEL_CATEGORIES
            if category in set(self.categories)
        )
        attributes = {
            head: frozenset(values)
            for head, values in self.attributes.items()
            if head in ATTRIBUTE_HEADS and values
        }
        mask = {head: bool(self.attribute_mask.get(head, False)) for head in ATTRIBUTE_HEADS}
        object.__setattr__(self, "categories", categories)
        object.__setattr__(self, "attributes", MappingProxyType(attributes))
        object.__setattr__(self, "attribute_mask", MappingProxyType(mask))

    @property
    def key(self) -> tuple[str, str]:
        return self.policy_id, self.segment_id

    @property
    def mask_vector(self) -> tuple[int, ...]:
        return tuple(int(self.attribute_mask[head]) for head in ATTRIBUTE_HEADS)

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "segment_id": self.segment_id,
            "categories": list(self.categories),
            "attributes": {
                head: sorted(values) for head, values in self.attributes.items()
            },
            "attribute_mask": dict(self.attribute_mask),
        }


def _annotation_directory(path: str | Path) -> Path:
    candidate = Path(path)
    nested = candidate / "annotations"
    return nested if nested.is_dir() else candidate


def _policy_stem(policy_id: object) -> str:
    text = str(policy_id).strip()
    if not text:
        raise ValueError("Policy identifiers must not be empty")
    # Policy identifiers themselves contain domain suffixes (``105_amazon.com``),
    # so generic ``Path.stem`` would incorrectly turn that into ``105_amazon``.
    # Strip only the annotation file's known suffix.
    name = Path(text).name
    return name[:-4] if name.casefold().endswith(".csv") else name


def list_policy_ids(annotation_dir: str | Path) -> tuple[str, ...]:
    """List annotation filename stems in the fixed split's stable order."""

    directory = _annotation_directory(annotation_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"OPP-115 annotation directory not found: {directory}")
    return tuple(
        sorted(
            (path.stem for path in directory.glob("*.csv")),
            key=lambda value: (value.casefold(), value),
        )
    )


def fixed_policy_split(
    annotation_dir: str | Path,
    *,
    train_count: int = DEFAULT_TRAIN_POLICY_COUNT,
    eval_count: int = DEFAULT_EVAL_POLICY_COUNT,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the deterministic whole-policy filename split used by the project."""

    if train_count <= 0 or eval_count <= 0:
        raise ValueError("train_count and eval_count must both be positive")
    policy_ids = list_policy_ids(annotation_dir)
    required = train_count + eval_count
    if len(policy_ids) != required:
        raise ValueError(
            f"Exact {train_count}/{eval_count} split requires {required} policies; "
            f"found {len(policy_ids)}."
        )
    return policy_ids[:train_count], policy_ids[train_count:]


def _selected_csv_paths(
    annotation_dir: str | Path, policy_ids: Iterable[str] | None
) -> tuple[Path, ...]:
    directory = _annotation_directory(annotation_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"OPP-115 annotation directory not found: {directory}")

    available = {
        path.stem.casefold(): path for path in directory.glob("*.csv") if path.is_file()
    }
    if policy_ids is None:
        paths = tuple(available.values())
    else:
        requested_stems = tuple(_policy_stem(policy_id) for policy_id in policy_ids)
        if len({stem.casefold() for stem in requested_stems}) != len(requested_stems):
            raise ValueError("policy_ids contains duplicate identifiers")
        missing = [stem for stem in requested_stems if stem.casefold() not in available]
        if missing:
            raise FileNotFoundError(
                "No annotation CSV for policy identifiers: " + ", ".join(missing)
            )
        paths = tuple(available[stem.casefold()] for stem in requested_stems)
    return tuple(sorted(paths, key=lambda path: (path.stem.casefold(), path.stem)))


def iter_annotation_rows(
    csv_path: str | Path, *, strict: bool = False
) -> Iterator[AnnotationPractice]:
    """Yield normalized rows from one public, headerless annotation CSV."""

    path = Path(csv_path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, row in enumerate(csv.reader(stream), start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            # Be tolerant of an accidental header in a synthetic/exported file,
            # while treating the release itself as the documented headerless form.
            if row[0].strip().casefold() in {"annotation_id", "annotation id"}:
                continue
            if len(row) < 7:
                if strict:
                    raise ValueError(
                        f"{path}:{line_number}: expected at least 7 CSV columns; "
                        f"found {len(row)}"
                    )
                continue
            segment_id = row[4].strip()
            category = normalize_category(row[5], strict=strict)
            if not segment_id or category is None:
                if strict and not segment_id:
                    raise ValueError(f"{path}:{line_number}: empty segment_id")
                continue
            values = extract_attribute_values(row[6], strict=strict)
            yield AnnotationPractice(
                policy_id=path.stem,
                segment_id=segment_id,
                category=category,
                attributes={head: frozenset(items) for head, items in values.items()},
                annotation_id=row[0].strip(),
                annotator_id=row[2].strip() if len(row) > 2 else "",
                source_line=line_number,
            )


def load_raw_annotation_rows(
    annotation_dir: str | Path,
    policy_ids: Iterable[str] | None = None,
    *,
    strict: bool = False,
) -> tuple[AnnotationPractice, ...]:
    """Load normalized practice rows, optionally restricted to whole policies."""

    rows: list[AnnotationPractice] = []
    for path in _selected_csv_paths(annotation_dir, policy_ids):
        rows.extend(iter_annotation_rows(path, strict=strict))
    return tuple(rows)


def attribute_mask_for_categories(
    categories: Iterable[object],
) -> Mapping[str, bool]:
    """Return the taxonomy mask for the union of canonical parent categories."""

    applicable: set[str] = set()
    for raw_category in categories:
        category = normalize_category(raw_category)
        if category is not None:
            applicable.update(CATEGORY_TO_ATTRIBUTES[category])
    return MappingProxyType({head: head in applicable for head in ATTRIBUTE_HEADS})


def _segment_sort_key(segment_id: str) -> tuple[int, int | str, str]:
    stripped = segment_id.strip()
    if re.fullmatch(r"[+-]?\d+", stripped):
        return 0, int(stripped), stripped
    return 1, stripped.casefold(), stripped


def aggregate_annotation_rows(
    rows: Iterable[AnnotationPractice],
) -> tuple[SegmentAnnotation, ...]:
    """Union annotators and practices by ``(policy filename stem, segment_id)``."""

    grouped_categories: dict[tuple[str, str], set[str]] = defaultdict(set)
    grouped_attributes: dict[
        tuple[str, str], dict[str, set[str]]
    ] = defaultdict(lambda: defaultdict(set))

    for row in rows:
        key = row.policy_id, str(row.segment_id)
        grouped_categories[key].add(row.category)
        for head, values in row.attributes.items():
            if head in ATTRIBUTE_HEADS:
                grouped_attributes[key][head].update(values)

    keys = sorted(
        grouped_categories,
        key=lambda key: (
            key[0].casefold(),
            key[0],
            _segment_sort_key(key[1]),
        ),
    )
    segments: list[SegmentAnnotation] = []
    for policy_id, segment_id in keys:
        categories = tuple(
            category
            for category in HIGH_LEVEL_CATEGORIES
            if category in grouped_categories[(policy_id, segment_id)]
        )
        attributes = {
            head: tuple(sorted(grouped_attributes[(policy_id, segment_id)][head]))
            for head in ATTRIBUTE_HEADS
            if grouped_attributes[(policy_id, segment_id)].get(head)
        }
        segments.append(
            SegmentAnnotation(
                policy_id=policy_id,
                segment_id=segment_id,
                categories=categories,
                attributes=attributes,
                attribute_mask=attribute_mask_for_categories(categories),
            )
        )
    return tuple(segments)


def load_opp115_annotations(
    annotation_dir: str | Path,
    policy_ids: Iterable[str] | None = None,
    *,
    strict: bool = False,
) -> tuple[SegmentAnnotation, ...]:
    """Load and aggregate public OPP-115 annotations into segment examples."""

    return aggregate_annotation_rows(
        load_raw_annotation_rows(annotation_dir, policy_ids, strict=strict)
    )


def _schema_payload(
    training_policy_ids: Sequence[str],
    value_vocabulary: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "high_level_categories": list(HIGH_LEVEL_CATEGORIES),
        "attribute_heads": list(ATTRIBUTE_HEADS),
        "raw_attribute_names": dict(RAW_ATTRIBUTE_NAMES),
        "excluded_attribute_heads": list(EXCLUDED_ATTRIBUTE_HEADS),
        "category_to_attributes": {
            category: list(CATEGORY_TO_ATTRIBUTES[category])
            for category in HIGH_LEVEL_CATEGORIES
        },
        "training_policy_ids": list(training_policy_ids),
        "attribute_value_vocabulary": {
            head: list(value_vocabulary[head]) for head in ATTRIBUTE_HEADS
        },
    }


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class HierarchySchema:
    """Frozen training-derived output schema for the hierarchical classifier."""

    training_policy_ids: tuple[str, ...]
    value_vocabulary: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        policy_ids = tuple(
            sorted(
                (_policy_stem(policy_id) for policy_id in self.training_policy_ids),
                key=lambda value: (value.casefold(), value),
            )
        )
        if len({policy_id.casefold() for policy_id in policy_ids}) != len(policy_ids):
            raise ValueError("training_policy_ids contains duplicate identifiers")

        unknown_heads = set(self.value_vocabulary) - set(ATTRIBUTE_HEADS)
        if unknown_heads:
            raise ValueError(
                "Vocabulary contains unknown attribute heads: "
                + ", ".join(sorted(unknown_heads))
            )
        vocabulary: dict[str, tuple[str, ...]] = {}
        for head in ATTRIBUTE_HEADS:
            normalized_values = {
                normalized
                for value in self.value_vocabulary.get(head, ())
                if (normalized := normalize_attribute_value(value)) is not None
            }
            vocabulary[head] = tuple(sorted(normalized_values))

        object.__setattr__(self, "training_policy_ids", policy_ids)
        object.__setattr__(self, "value_vocabulary", MappingProxyType(vocabulary))

    @property
    def schema_version(self) -> str:
        return SCHEMA_VERSION

    @property
    def categories(self) -> tuple[str, ...]:
        return HIGH_LEVEL_CATEGORIES

    @property
    def attribute_heads(self) -> tuple[str, ...]:
        return ATTRIBUTE_HEADS

    @property
    def category_count(self) -> int:
        return len(HIGH_LEVEL_CATEGORIES)

    @property
    def attribute_head_count(self) -> int:
        return len(ATTRIBUTE_HEADS)

    @property
    def flattened_value_count(self) -> int:
        return sum(len(self.value_vocabulary[head]) for head in ATTRIBUTE_HEADS)

    @property
    def flattened_attribute_labels(self) -> tuple[str, ...]:
        return tuple(
            f"{head}={value}"
            for head in ATTRIBUTE_HEADS
            for value in self.value_vocabulary[head]
        )

    @property
    def attribute_slices(self) -> Mapping[str, tuple[int, int]]:
        start = 0
        slices: dict[str, tuple[int, int]] = {}
        for head in ATTRIBUTE_HEADS:
            stop = start + len(self.value_vocabulary[head])
            slices[head] = start, stop
            start = stop
        return MappingProxyType(slices)

    def _payload(self) -> dict[str, Any]:
        return _schema_payload(self.training_policy_ids, self.value_vocabulary)

    @property
    def schema_sha256(self) -> str:
        return _payload_sha256(self._payload())

    @property
    def schema_hash(self) -> str:
        return self.schema_sha256

    def to_manifest(self) -> dict[str, Any]:
        payload = self._payload()
        payload.update(
            {
                "schema_sha256": self.schema_sha256,
                "category_count": self.category_count,
                "attribute_head_count": self.attribute_head_count,
                "flattened_attribute_value_count": self.flattened_value_count,
                "attribute_slices": {
                    head: list(bounds) for head, bounds in self.attribute_slices.items()
                },
            }
        )
        return payload

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> "HierarchySchema":
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported hierarchy schema version: {manifest.get('schema_version')!r}"
            )
        if tuple(manifest.get("high_level_categories", ())) != HIGH_LEVEL_CATEGORIES:
            raise ValueError("Manifest high-level categories do not match this version")
        if tuple(manifest.get("attribute_heads", ())) != ATTRIBUTE_HEADS:
            raise ValueError("Manifest attribute heads do not match this version")
        schema = cls(
            training_policy_ids=tuple(manifest.get("training_policy_ids", ())),
            value_vocabulary=manifest.get("attribute_value_vocabulary", {}),
        )
        supplied_hash = manifest.get("schema_sha256")
        if supplied_hash is not None and supplied_hash != schema.schema_sha256:
            raise ValueError("Hierarchy manifest hash verification failed")
        return schema


def derive_hierarchy_schema(
    source: str | Path | Iterable[SegmentAnnotation],
    training_policy_ids: Iterable[str],
    *,
    require_values_for_all_heads: bool = True,
) -> HierarchySchema:
    """Derive and freeze value vocabularies from the named training policies.

    ``source`` may be an annotation directory (only the requested CSVs are
    opened) or a previously loaded collection of segments (which is filtered
    before any values are inspected).  Both paths enforce whole-policy scope.
    """

    requested = tuple(_policy_stem(policy_id) for policy_id in training_policy_ids)
    if not requested:
        raise ValueError("At least one training policy identifier is required")
    requested_folded = {policy_id.casefold() for policy_id in requested}
    if len(requested_folded) != len(requested):
        raise ValueError("training_policy_ids contains duplicate identifiers")

    if isinstance(source, (str, Path)):
        training_segments = load_opp115_annotations(source, requested, strict=True)
    else:
        all_segments = tuple(source)
        training_segments = tuple(
            segment
            for segment in all_segments
            if segment.policy_id.casefold() in requested_folded
        )
        found = {segment.policy_id.casefold() for segment in training_segments}
        missing = [policy_id for policy_id in requested if policy_id.casefold() not in found]
        if missing:
            raise ValueError(
                "No loaded segments for training policies: " + ", ".join(missing)
            )

    vocabulary: dict[str, set[str]] = {head: set() for head in ATTRIBUTE_HEADS}
    for segment in training_segments:
        # This assertion is the leakage guard when a caller supplied all data.
        if segment.policy_id.casefold() not in requested_folded:
            raise AssertionError("A held-out policy reached vocabulary derivation")
        for head, values in segment.attributes.items():
            if head in vocabulary:
                vocabulary[head].update(values)

    empty_heads = [head for head in ATTRIBUTE_HEADS if not vocabulary[head]]
    if require_values_for_all_heads and empty_heads:
        raise ValueError(
            "Training policies contain no values for attribute heads: "
            + ", ".join(empty_heads)
        )
    return HierarchySchema(
        training_policy_ids=requested,
        value_vocabulary={
            head: tuple(sorted(vocabulary[head])) for head in ATTRIBUTE_HEADS
        },
    )


# Intent-revealing aliases for notebook/training callers.
build_hierarchy_schema = derive_hierarchy_schema
derive_training_schema = derive_hierarchy_schema


def save_schema_manifest(schema: HierarchySchema, path: str | Path) -> Path:
    """Write a deterministic, self-verifying JSON schema manifest."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(schema.to_manifest(), indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return destination


def load_schema_manifest(path: str | Path) -> HierarchySchema:
    parsed = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(parsed, Mapping):
        raise ValueError("Hierarchy manifest must contain a JSON object")
    return HierarchySchema.from_manifest(parsed)


@dataclass(frozen=True)
class EncodedSegmentLabels:
    """Classifier-ready targets aligned to one ``HierarchySchema``."""

    policy_id: str
    segment_id: str
    category_targets: tuple[int, ...]
    attribute_targets: Mapping[str, tuple[int, ...]]
    attribute_mask: Mapping[str, int]
    unknown_attribute_values: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "attribute_targets", MappingProxyType(dict(self.attribute_targets))
        )
        object.__setattr__(
            self, "attribute_mask", MappingProxyType(dict(self.attribute_mask))
        )
        object.__setattr__(
            self,
            "unknown_attribute_values",
            MappingProxyType(dict(self.unknown_attribute_values)),
        )

    @property
    def flattened_attribute_targets(self) -> tuple[int, ...]:
        return tuple(
            value
            for head in ATTRIBUTE_HEADS
            for value in self.attribute_targets[head]
        )

    @property
    def mask_vector(self) -> tuple[int, ...]:
        return tuple(self.attribute_mask[head] for head in ATTRIBUTE_HEADS)

    @property
    def key(self) -> tuple[str, str]:
        return self.policy_id, self.segment_id

    # Short classifier-oriented aliases.  The longer names above remain more
    # descriptive in serialized reports.
    @property
    def category_vector(self) -> tuple[int, ...]:
        return self.category_targets

    @property
    def attribute_vector(self) -> tuple[int, ...]:
        return self.flattened_attribute_targets

    @property
    def head_mask(self) -> tuple[int, ...]:
        return self.mask_vector

    @property
    def unknown_values(self) -> Mapping[str, tuple[str, ...]]:
        return self.unknown_attribute_values


def encode_segment_labels(
    segment: SegmentAnnotation,
    schema: HierarchySchema,
    *,
    unknown_values: str = "ignore",
) -> EncodedSegmentLabels:
    """Encode a segment as category and per-head multi-hot targets.

    Held-out-only values cannot resize a frozen head.  With the default
    ``unknown_values='ignore'`` they are reported in ``unknown_attribute_values``
    and left zero in that head.  Use ``'error'`` for strict evaluation audits.
    """

    if unknown_values not in {"ignore", "error"}:
        raise ValueError("unknown_values must be either 'ignore' or 'error'")
    category_set = set(segment.categories)
    category_targets = tuple(
        int(category in category_set) for category in HIGH_LEVEL_CATEGORIES
    )
    targets: dict[str, tuple[int, ...]] = {}
    unknown: dict[str, tuple[str, ...]] = {}
    for head in ATTRIBUTE_HEADS:
        observed = set(segment.attributes.get(head, ()))
        vocabulary = schema.value_vocabulary[head]
        known = set(vocabulary)
        targets[head] = tuple(int(value in observed) for value in vocabulary)
        held_out = tuple(sorted(observed - known))
        if held_out:
            unknown[head] = held_out
    if unknown and unknown_values == "error":
        details = ", ".join(
            f"{head}={list(values)!r}" for head, values in unknown.items()
        )
        raise ValueError(f"Values are outside the frozen training schema: {details}")

    return EncodedSegmentLabels(
        policy_id=segment.policy_id,
        segment_id=segment.segment_id,
        category_targets=category_targets,
        attribute_targets=targets,
        attribute_mask={
            head: int(segment.attribute_mask[head]) for head in ATTRIBUTE_HEADS
        },
        unknown_attribute_values=unknown,
    )


def encode_segments(
    segments: Iterable[SegmentAnnotation],
    schema: HierarchySchema,
    *,
    unknown_values: str = "ignore",
) -> tuple[EncodedSegmentLabels, ...]:
    return tuple(
        encode_segment_labels(segment, schema, unknown_values=unknown_values)
        for segment in segments
    )


def load_segment_annotations(
    annotation_dir: str | Path,
    schema: HierarchySchema | None = None,
    policy_ids: Iterable[str] | None = None,
    *,
    strict: bool = False,
    unknown_values: str = "ignore",
) -> Mapping[
    tuple[str, str], SegmentAnnotation | EncodedSegmentLabels
]:
    """Load labels keyed by ``(policy filename stem, segment_id)``.

    Without ``schema`` the values are aggregated ``SegmentAnnotation`` records
    containing normalized raw value sets.  Supplying a frozen schema returns
    ``EncodedSegmentLabels`` records of fixed width and records held-out-only
    values without allowing them to alter that width.
    """

    segments = load_opp115_annotations(
        annotation_dir, policy_ids=policy_ids, strict=strict
    )
    records: Iterable[SegmentAnnotation | EncodedSegmentLabels]
    if schema is None:
        records = segments
    else:
        records = encode_segments(
            segments, schema, unknown_values=unknown_values
        )
    return MappingProxyType({record.key: record for record in records})


# A concise public surface for notebook imports and manifest tooling.
__all__ = [
    "ANNOTATION_CSV_COLUMNS",
    "ATTRIBUTES",
    "ATTRIBUTE_HEADS",
    "CATEGORY_TO_ATTRIBUTES",
    "DEFAULT_EVAL_POLICY_COUNT",
    "DEFAULT_TRAIN_POLICY_COUNT",
    "EXCLUDED_ATTRIBUTE_HEADS",
    "EXPECTED_POLICY_COUNT",
    "HIGH_LEVEL_CATEGORIES",
    "NORMALIZATION_VERSION",
    "RAW_ATTRIBUTE_NAMES",
    "SCHEMA_VERSION",
    "AnnotationPractice",
    "EncodedSegmentLabels",
    "HierarchySchema",
    "SegmentAnnotation",
    "aggregate_annotation_rows",
    "attribute_mask_for_categories",
    "build_hierarchy_schema",
    "derive_hierarchy_schema",
    "derive_training_schema",
    "encode_segment_labels",
    "encode_segments",
    "extract_attribute_values",
    "fixed_policy_split",
    "iter_annotation_rows",
    "list_policy_ids",
    "load_opp115_annotations",
    "load_raw_annotation_rows",
    "load_schema_manifest",
    "load_segment_annotations",
    "normalize_attribute_name",
    "normalize_attribute_value",
    "normalize_category",
    "save_schema_manifest",
]


# Documented positions in each public headerless row.  Kept at the end so it is
# visually close to ``__all__`` without being used as an inferred CSV header.
ANNOTATION_CSV_COLUMNS: tuple[str, ...] = (
    "annotation_id",
    "batch_id",
    "annotator_id",
    "policy_numeric_id",
    "segment_id",
    "category_name",
    "attributes_json",
    "policy_date",
    "policy_url",
)


if len(HIGH_LEVEL_CATEGORIES) != 10 or len(set(HIGH_LEVEL_CATEGORIES)) != 10:
    raise AssertionError("OPP-115 must expose exactly ten unique categories")
if len(ATTRIBUTE_HEADS) != 20 or len(set(ATTRIBUTE_HEADS)) != 20:
    raise AssertionError("OPP-115 must expose exactly twenty unique attribute heads")
if set(CATEGORY_TO_ATTRIBUTES) != set(HIGH_LEVEL_CATEGORIES):
    raise AssertionError("Every category must have an explicit applicability mapping")
if any(
    head not in ATTRIBUTE_HEADS
    for heads in CATEGORY_TO_ATTRIBUTES.values()
    for head in heads
):
    raise AssertionError("Category applicability references an unknown attribute head")
