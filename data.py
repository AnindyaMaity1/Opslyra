from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from reproducibility import (
    EXCLUDED_PREDICTION_COLUMNS,
    PERMITTED_METADATA_COLUMNS,
    PERMITTED_TEXT_COLUMNS,
    dataset_fingerprint,
)


REQUIRED_COLUMNS = [
    "Ticket ID",
    "Customer Name",
    "Customer Email",
    "Customer Age",
    "Customer Gender",
    "Product Purchased",
    "Date of Purchase",
    "Ticket Type",
    "Ticket Subject",
    "Ticket Description",
    "Ticket Status",
    "Resolution",
    "Ticket Priority",
    "Ticket Channel",
    "First Response Time",
    "Time to Resolution",
    "Customer Satisfaction Rating",
]

KAGGLE_DATASET_PROVENANCE = {
    "source_kind": "kaggle",
    "source": "customer_support_tickets.csv",
    "label": "Kaggle Dataset",
    "note": (
        "The dataset is identified by the project owner as sourced from Kaggle. "
        "The exact Kaggle listing, citation, license, and label-verification evidence "
        "have not yet been recorded in this repository."
    ),
    "license": "Not recorded; verify the Kaggle listing's terms before redistribution or reuse.",
    "retrieval_date": None,
    "citation": None,
    "labels_verified": False,
    "timestamps_trustworthy": False,
}

EXTERNAL_DATASET_PROVENANCE_TEMPLATE = {
    "source_kind": "external",
    "label": "External dataset",
    "note": "Record source URL, license, retrieval date, and permitted use before research claims.",
    "labels_verified": False,
    "timestamps_trustworthy": False,
}

DEFAULT_DATASET_PATH = Path(__file__).resolve().with_name("customer_support_tickets.csv")

EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
REDACTION_PATTERNS = {
    "email": EMAIL_PATTERN,
    "username": re.compile(r"(?<![\w.])@[A-Za-z][A-Za-z0-9_.-]{1,30}\b"),
    "ip_address": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
    ),
    "secret": re.compile(
        r"\b(?:api[_-]?key|access[_-]?token|auth(?:orization)?|password|secret)"
        r"\s*[:=]\s*[^\s,;]+",
        re.IGNORECASE,
    ),
    "bearer_token": re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]+=*", re.IGNORECASE),
    "username_field": re.compile(
        r"\b(?:user(?:[_ -]?name)?|login)\s*[:=]\s*[A-Za-z0-9_.@-]+",
        re.IGNORECASE,
    ),
    "phone": PHONE_PATTERN,
}


class DatasetError(ValueError):
    """Raised when the configured ticket dataset cannot be used."""


def redact_sensitive_text(value: Any) -> str:
    """Redact common direct identifiers and credential-like strings."""
    redacted, _ = redact_with_audit(value)
    return redacted


def redact_with_audit(value: Any) -> tuple[str, dict[str, int]]:
    if pd.isna(value):
        return "", {category: 0 for category in REDACTION_PATTERNS}
    text = str(value)
    counts: dict[str, int] = {}
    for category, pattern in REDACTION_PATTERNS.items():
        text, count = pattern.subn(f"[redacted {category}]", text)
        counts[category] = count
    return text, counts


def _provenance_for_path(dataset_path: Path) -> dict[str, Any]:
    sidecar = dataset_path.with_suffix(dataset_path.suffix + ".provenance.json")
    if sidecar.is_file():
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise DatasetError(f"Could not read dataset provenance sidecar: {error}") from error
        if not isinstance(payload, dict):
            raise DatasetError("Dataset provenance sidecar must contain a JSON object.")
        for field in ("labels_verified", "timestamps_trustworthy"):
            if field in payload and not isinstance(payload[field], bool):
                raise DatasetError(f"Provenance field {field!r} must be a boolean.")
        provenance = EXTERNAL_DATASET_PROVENANCE_TEMPLATE.copy()
        provenance.update(payload)
        provenance.setdefault("source", dataset_path.name)
        return provenance
    if dataset_path.resolve() == DEFAULT_DATASET_PATH.resolve():
        provenance = KAGGLE_DATASET_PROVENANCE.copy()
        provenance["source"] = dataset_path.name
        return provenance
    provenance = EXTERNAL_DATASET_PROVENANCE_TEMPLATE.copy()
    provenance["source"] = dataset_path.name
    return provenance


def validate_raw_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """Validate schema and quality without mutating values."""
    issues: list[str] = []
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        issues.append("missing columns: " + ", ".join(missing_columns))
    duplicate_ids = (
        int(frame["Ticket ID"].duplicated().sum()) if "Ticket ID" in frame.columns else None
    )
    if duplicate_ids:
        issues.append(f"duplicate ticket IDs: {duplicate_ids}")
    if "Ticket ID" in frame.columns:
        missing_ids = int(frame["Ticket ID"].isna().sum())
        if missing_ids:
            issues.append(f"missing ticket IDs: {missing_ids}")
    for column, low, high in (
        ("Customer Age", None, None),
        ("Customer Satisfaction Rating", 1, 5),
    ):
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].notna() & values.isna()
        if low is not None and high is not None:
            invalid |= values.notna() & ~values.between(low, high)
        invalid_count = int(invalid.sum())
        if invalid_count:
            issues.append(f"invalid {column} values: {invalid_count}")
    for column in ("Date of Purchase", "First Response Time", "Time to Resolution"):
        if column not in frame.columns:
            continue
        present = frame[column].notna()
        parsed = pd.to_datetime(frame.loc[present, column], errors="coerce")
        invalid_count = int(parsed.isna().sum())
        if invalid_count:
            issues.append(f"unparseable {column} timestamps: {invalid_count}")
    return {
        "issues": issues,
        "duplicate_ticket_ids": duplicate_ids,
        "row_count": int(len(frame)),
        "missing_by_column": {
            column: int(frame[column].isna().sum())
            for column in frame.columns
        },
        "data_types": {column: str(dtype) for column, dtype in frame.dtypes.items()},
    }


def load_dataset(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise DatasetError(f"Dataset file was not found: {dataset_path}")

    try:
        frame = pd.read_csv(dataset_path, dtype={"Ticket ID": "string"})
    except (
        OSError,
        UnicodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        ValueError,
    ) as exc:
        raise DatasetError(f"Could not read the dataset: {exc}") from exc

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise DatasetError(
            "Dataset is missing required columns: " + ", ".join(missing_columns)
        )
    if frame.empty:
        raise DatasetError("Dataset contains the expected columns but no ticket rows.")

    # Keep a column-filtered copy. Blank strings become NA so missingness is
    # explicit. No invented label or numeric fill is applied.
    frame = frame[REQUIRED_COLUMNS].copy()
    frame = frame.replace(r"^\s*$", pd.NA, regex=True)
    provenance = _provenance_for_path(dataset_path)
    frame.attrs["dataset_label"] = provenance.get("label", "External dataset")
    frame.attrs["dataset_provenance"] = provenance
    frame.attrs["dataset_path"] = str(dataset_path.resolve())
    frame.attrs["dataset_fingerprint"] = dataset_fingerprint(frame, dataset_path)
    frame.attrs["representation"] = "raw"
    frame.attrs["excluded_prediction_columns"] = list(EXCLUDED_PREDICTION_COLUMNS)
    frame.attrs["permitted_text_columns"] = list(PERMITTED_TEXT_COLUMNS)
    frame.attrs["permitted_metadata_columns"] = list(PERMITTED_METADATA_COLUMNS)
    return frame


def analysis_copy(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy for exploration. Labels remain missing; no imputation."""
    copy = frame.copy()
    copy.attrs.update(dict(frame.attrs))
    copy.attrs["representation"] = "cleaned_for_analysis"
    return copy


def model_text_series(
    frame: pd.DataFrame, field: str = "subject_description"
) -> pd.Series:
    """Model-ready text. Missing subject/description become empty strings for vectorizers."""
    subjects = frame["Ticket Subject"].fillna("").map(redact_sensitive_text)
    descriptions = frame["Ticket Description"].fillna("").map(redact_sensitive_text)
    if field == "subject":
        return subjects.str.strip()
    if field == "description":
        return descriptions.str.strip()
    return (subjects + "\n" + descriptions).str.strip()


def dataset_profile(frame: pd.DataFrame) -> dict[str, Any]:
    missing_values = frame.isna().sum()
    rating_values = pd.to_numeric(
        frame["Customer Satisfaction Rating"], errors="coerce"
    )
    age_values = pd.to_numeric(frame["Customer Age"], errors="coerce")
    provenance = dict(frame.attrs.get("dataset_provenance", KAGGLE_DATASET_PROVENANCE))
    validation = validate_raw_frame(frame)
    profile: dict[str, Any] = {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "dataset_label": frame.attrs.get(
            "dataset_label", provenance.get("label", "External dataset")
        ),
        "dataset_provenance": provenance,
        "duplicate_ticket_ids": int(frame["Ticket ID"].duplicated().sum()),
        "missing_total": int(missing_values.sum()),
        "missing_values": {
            column: int(count)
            for column, count in missing_values.items()
            if count > 0
        },
        "targets": {},
        "date_fields": {},
        "data_types": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "numeric_checks": {
            "Customer Age": {
                "parseable": int(age_values.notna().sum()),
                "invalid": int((frame["Customer Age"].notna() & age_values.isna()).sum()),
            },
            "Customer Satisfaction Rating": {
                "parseable": int(rating_values.notna().sum()),
                "invalid": int(
                    (
                        frame["Customer Satisfaction Rating"].notna()
                        & (
                            rating_values.isna()
                            | ~rating_values.between(1, 5)
                        )
                    ).sum()
                ),
            },
        },
        "representation": frame.attrs.get("representation", "raw"),
        "fingerprint": frame.attrs.get("dataset_fingerprint"),
        "validation_issues": validation["issues"],
        "timestamps_trustworthy": provenance.get("timestamps_trustworthy") is True,
        "excluded_prediction_columns": list(
            frame.attrs.get("excluded_prediction_columns", EXCLUDED_PREDICTION_COLUMNS)
        ),
        "closed_count": int(
            frame["Ticket Status"].astype("string").str.casefold().eq("closed").sum()
        ),
        "average_satisfaction": (
            float(rating_values.mean())
            if rating_values.notna().any()
            else None
        ),
    }

    for column in ("Ticket Type", "Ticket Priority", "Ticket Status", "Ticket Channel"):
        labels = frame[column].dropna().astype(str)
        counts = labels.value_counts()
        profile["targets"][column] = [
            {"label": str(label), "count": int(count)}
            for label, count in counts.items()
        ]
        profile.setdefault("label_quality", {})[column] = {
            "distinct_labels": int(counts.size),
            "case_or_whitespace_variants": int(
                counts.size
                - labels.str.strip().str.casefold().nunique()
            ),
        }

    for column in ("Date of Purchase", "First Response Time", "Time to Resolution"):
        values = frame[column].dropna()
        parsed = pd.to_datetime(values, errors="coerce")
        profile["date_fields"][column] = {
            "present": int(values.size),
            "parseable": int(parsed.notna().sum()),
            "unparseable": int(parsed.isna().sum()),
        }

    return profile


def ticket_text(frame: pd.DataFrame) -> pd.Series:
    return model_text_series(frame, "subject_description")
