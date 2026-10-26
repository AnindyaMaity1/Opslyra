from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


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

    frame = frame[REQUIRED_COLUMNS].copy()
    frame = frame.replace(r"^\s*$", pd.NA, regex=True)
    return frame


def dataset_profile(frame: pd.DataFrame) -> dict[str, Any]:
    missing_values = frame.isna().sum()
    rating_values = pd.to_numeric(
        frame["Customer Satisfaction Rating"], errors="coerce"
    )
    age_values = pd.to_numeric(frame["Customer Age"], errors="coerce")
    profile: dict[str, Any] = {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
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
    subjects = frame["Ticket Subject"].fillna("").map(redact_sensitive_text)
    descriptions = frame["Ticket Description"].fillna("").map(redact_sensitive_text)
    return (subjects + "\n" + descriptions).str.strip()
