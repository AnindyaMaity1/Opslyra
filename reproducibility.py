from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


EXCLUDED_PREDICTION_COLUMNS = (
    "Ticket ID",
    "Customer Name",
    "Customer Email",
    "Customer Age",
    "Customer Gender",
    "Resolution",
    "Ticket Status",
    "Time to Resolution",
    "Customer Satisfaction Rating",
    "First Response Time",
)

PERMITTED_TEXT_COLUMNS = ("Ticket Subject", "Ticket Description")
PERMITTED_METADATA_COLUMNS = ("Ticket Channel", "Product Purchased")


def normalized_text_keys(texts: pd.Series) -> pd.Series:
    """Return normalized ticket text for equality checks; do not persist these values."""
    return (
        texts.fillna("")
        .astype("string")
        .str.casefold()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def exact_text_group_ids(texts: pd.Series) -> np.ndarray:
    """Return non-sensitive integer groups for normalized identical text."""
    normalized = normalized_text_keys(texts)
    return pd.factorize(normalized, sort=True)[0]


def stratified_group_holdout_indices(
    labels: pd.Series | np.ndarray,
    groups: np.ndarray,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Find a class-complete holdout while keeping exact-text groups intact."""
    label_values = np.asarray(labels, dtype=str)
    group_values = np.asarray(groups)
    if len(label_values) != len(group_values):
        raise ValueError("Labels and text groups must have the same length.")
    if not np.isfinite(test_fraction) or not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    n_splits = max(2, int(round(1.0 / test_fraction)))
    if len(np.unique(group_values)) < n_splits:
        return None

    classes = set(label_values)
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    candidates = []
    for train_idx, test_idx in splitter.split(
        np.zeros((len(label_values), 1)),
        label_values,
        group_values,
    ):
        train_classes = set(label_values[train_idx])
        test_classes = set(label_values[test_idx])
        if train_classes != classes or test_classes != classes:
            continue
        fraction_error = abs(len(test_idx) / len(label_values) - test_fraction)
        candidates.append((fraction_error, train_idx, test_idx))
    if not candidates:
        return None
    _, train_idx, test_idx = min(candidates, key=lambda item: item[0])
    return train_idx, test_idx


def dataset_fingerprint(frame: pd.DataFrame, path: str | Path | None = None) -> dict[str, Any]:
    payload = frame.astype("string").fillna("<NA>").to_csv(index=False).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    stat: dict[str, Any] = {"sha256": digest, "rows": int(len(frame)), "columns": list(frame.columns)}
    if path is not None:
        dataset_path = Path(path)
        if dataset_path.is_file():
            file_stat = dataset_path.stat()
            stat["path"] = str(dataset_path.resolve())
            stat["size_bytes"] = int(file_stat.st_size)
            stat["mtime_ns"] = int(file_stat.st_mtime_ns)
    return stat


def software_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    distributions = {
        "pandas": "pandas",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "scipy": "scipy",
        "flask": "Flask",
        "joblib": "joblib",
    }
    for name, distribution in distributions.items():
        try:
            versions[name] = version(distribution)
        except PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


def experiment_record(
    name: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    fingerprint: dict[str, Any],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    record = {
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config,
        "metrics": metrics,
        "dataset": fingerprint,
        "software": software_versions(),
        "output_path": str(output_path) if output_path else None,
    }
    return record


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def mean_std_ci(values: list[float], z: float = 1.96) -> dict[str, Any]:
    import math

    import numpy as np

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    n = int(array.size)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "ci95_low": None,
            "ci95_high": None,
            "warning": "No finite observations.",
        }
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if n > 1 else 0.0
    if n < 5:
        warning = (
            "Fewer than 5 observations; the 95% interval is a normal approximation "
            "and should not be treated as a well-powered confidence interval."
        )
    else:
        warning = None
    half = z * std / math.sqrt(n) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "warning": warning,
    }
