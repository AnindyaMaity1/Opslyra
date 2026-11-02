from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import numpy as np
import pandas as pd


def cohens_kappa(annotator_a: Iterable[str], annotator_b: Iterable[str]) -> dict[str, Any]:
    """Cohen's kappa for two categorical label sequences of equal length.

    Returns None-valued summary fields when inputs are unusable rather than
    inventing agreement. Callers must supply genuine independent annotations.
    """
    left = [str(value).strip() for value in annotator_a]
    right = [str(value).strip() for value in annotator_b]
    if len(left) != len(right):
        raise ValueError("Annotator sequences must have the same length.")
    n = len(left)
    if n == 0:
        return {
            "n": 0,
            "kappa": None,
            "observed_agreement": None,
            "expected_agreement": None,
            "labels": [],
            "confusion": {},
            "class_counts_a": {},
            "class_counts_b": {},
            "status": "empty",
        }
    labels = sorted(set(left) | set(right))
    confusion = {row: {col: 0 for col in labels} for row in labels}
    agree = 0
    for a_value, b_value in zip(left, right):
        confusion[a_value][b_value] += 1
        if a_value == b_value:
            agree += 1
    observed = agree / n
    counts_a = Counter(left)
    counts_b = Counter(right)
    expected = sum((counts_a[label] / n) * (counts_b[label] / n) for label in labels)
    denom = 1 - expected
    kappa = None if denom == 0 else (observed - expected) / denom
    return {
        "n": n,
        "kappa": None if kappa is None else float(kappa),
        "observed_agreement": float(observed),
        "expected_agreement": float(expected),
        "labels": labels,
        "confusion": confusion,
        "class_counts_a": dict(counts_a),
        "class_counts_b": dict(counts_b),
        "status": "ok",
    }


def kappa_from_frames(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    id_column: str = "Ticket ID",
    label_columns: tuple[str, ...] = ("Ticket Type", "Ticket Priority"),
) -> dict[str, Any]:
    if id_column not in frame_a.columns or id_column not in frame_b.columns:
        raise ValueError(f"Both annotation files must include {id_column!r}.")
    for frame, name in ((frame_a, "annotator_a"), (frame_b, "annotator_b")):
        if frame[id_column].duplicated().any():
            raise ValueError(f"{name} contains duplicate {id_column} values.")
    merged = frame_a.merge(frame_b, on=id_column, suffixes=("_a", "_b"), how="inner")
    if merged.empty:
        return {
            "status": "no_overlapping_ids",
            "n": 0,
            "labels": {},
            "note": "Independent dual annotation has not been completed for an overlapping ticket set.",
        }
    results = {}
    for column in label_columns:
        left_column = f"{column}_a"
        right_column = f"{column}_b"
        if left_column not in merged or right_column not in merged:
            raise ValueError(f"Both annotation files must include {column!r}.")
        valid = (
            merged[left_column].notna()
            & merged[right_column].notna()
            & merged[left_column].astype("string").str.strip().ne("")
            & merged[right_column].astype("string").str.strip().ne("")
        )
        result = cohens_kappa(
            merged.loc[valid, left_column],
            merged.loc[valid, right_column],
        )
        result["overlapping_rows"] = int(len(merged))
        result["unlabeled_pairs_excluded"] = int(len(merged) - valid.sum())
        results[column] = result
    return {"status": "ok", "n": int(len(merged)), "labels": results}


def annotation_status_note() -> dict[str, Any]:
    return {
        "dual_annotation": "not_yet_measured",
        "cohens_kappa": "Not measured",
        "adjudication": (
            "When two independent annotation files share Ticket IDs, run "
            "`python experiments.py --output instance/experiments kappa annotator_a.csv annotator_b.csv`. "
            "Do not invent annotator identities or agreement scores."
        ),
        "label_status": "No independently verified or dual-annotated labels are supplied.",
    }
