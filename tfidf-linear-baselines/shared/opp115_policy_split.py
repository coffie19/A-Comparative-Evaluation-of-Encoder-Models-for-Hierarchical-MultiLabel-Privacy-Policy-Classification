"""Shared deterministic whole-policy split for every OPP-115 model stage."""

from __future__ import annotations

from collections.abc import Iterable


DEFAULT_TRAIN_POLICIES = 65
DEFAULT_EVAL_POLICIES = 50


def split_policy_ids(
    policy_ids: Iterable[str],
    *,
    train_count: int = DEFAULT_TRAIN_POLICIES,
    eval_count: int = DEFAULT_EVAL_POLICIES,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return an exact, input-order-independent whole-policy split.

    OPP-115 has exactly 115 policies.  Sorting their filename-based identifiers
    matches the rest of this codebase and produces the same membership on
    Windows, Linux, and Google Colab.
    """

    if train_count <= 0 or eval_count <= 0:
        raise ValueError("train_count and eval_count must both be positive")

    identifiers = [str(policy_id) for policy_id in policy_ids]
    unique_identifiers = sorted(set(identifiers), key=lambda value: value.casefold())
    if len(unique_identifiers) != len(identifiers):
        raise ValueError("policy_ids contains duplicate identifiers")

    required = train_count + eval_count
    if len(unique_identifiers) != required:
        raise ValueError(
            f"Exact {train_count}/{eval_count} policy split requires {required} "
            f"unique policies; found {len(unique_identifiers)}."
        )

    train_ids = tuple(unique_identifiers[:train_count])
    eval_ids = tuple(unique_identifiers[train_count:])
    if set(train_ids) & set(eval_ids):
        raise AssertionError("A policy appeared in both train and evaluation splits")
    return train_ids, eval_ids
