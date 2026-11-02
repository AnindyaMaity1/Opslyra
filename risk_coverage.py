from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from data import model_text_series
from reproducibility import exact_text_group_ids, stratified_group_holdout_indices


def selective_risk(correct: np.ndarray, accepted: np.ndarray) -> float | None:
    if not accepted.any():
        return None
    return float(1.0 - correct[accepted].mean())


def risk_coverage_curve(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_labels: np.ndarray | None = None,
    points: int = 15,
) -> list[dict[str, Any]]:
    """Risk-coverage using argmax class labels, not raw probability indices."""
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must be a 2D array with at least two classes.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite.")
    confidences = probabilities.max(axis=1)
    predicted_idx = probabilities.argmax(axis=1)
    if class_labels is None:
        y_pred = predicted_idx
    else:
        class_labels = np.asarray(class_labels)
        if len(class_labels) != probabilities.shape[1]:
            raise ValueError("class_labels must match the probability columns.")
        y_pred = class_labels[predicted_idx]
    return risk_coverage_from_labels(y_true, y_pred, confidences, points=points)


def risk_coverage_from_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    points: int = 12,
) -> list[dict[str, Any]]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    confidences = np.asarray(confidences, dtype=float)
    if not isinstance(points, int) or isinstance(points, bool) or points < 1:
        raise ValueError("points must be a positive integer.")
    if len(y_true) != len(y_pred) or len(y_true) != len(confidences):
        raise ValueError("Labels and confidence scores must have the same length.")
    if len(confidences) == 0:
        return []
    if not np.isfinite(confidences).all():
        raise ValueError("Confidence scores must be finite.")
    order = np.argsort(-confidences, kind="stable")
    n = len(confidences)
    correct = (y_true == y_pred).astype(float)
    curve = []
    seen_thresholds: set[float] = set()
    for fraction in np.linspace(1.0 / points, 1.0, points):
        k = max(1, int(round(n * fraction)))
        threshold = float(confidences[order[k - 1]])
        if threshold in seen_thresholds:
            continue
        seen_thresholds.add(threshold)
        chosen = confidences >= threshold
        accepted = int(chosen.sum())
        coverage = accepted / n
        risk = float(1.0 - correct[chosen].mean())
        curve.append(
            {
                "coverage": round(float(coverage), 4),
                "selective_risk": risk,
                "accepted": accepted,
                "abstained": int(n - accepted),
                "abstention_rate": round(float((n - accepted) / n), 4),
                "threshold": threshold,
            }
        )
    return curve


def threshold_for_target_coverage(curve: list[dict[str, Any]], target_coverage: float) -> dict[str, Any]:
    if not curve:
        return {"threshold": None, "coverage": None, "selective_risk": None}
    chosen = min(curve, key=lambda item: abs(item["coverage"] - target_coverage))
    return chosen


def threshold_for_target_risk(curve: list[dict[str, Any]], target_risk: float) -> dict[str, Any]:
    feasible = [item for item in curve if item["selective_risk"] is not None and item["selective_risk"] <= target_risk]
    if not feasible:
        if not curve:
            return {"threshold": None, "coverage": None, "selective_risk": None}
        return {
            "threshold": float(np.nextafter(max(item["threshold"] for item in curve), np.inf)),
            "coverage": 0.0,
            "selective_risk": None,
            "accepted": 0,
            "abstained": max(item["accepted"] + item["abstained"] for item in curve),
            "abstention_rate": 1.0,
        }
    return max(feasible, key=lambda item: item["coverage"])


def select_validation_threshold(
    frame: pd.DataFrame,
    target: str,
    selection_mode: str = "target_coverage",
    target_coverage: float = 0.6,
    target_risk: float = 0.1,
    max_rows: int | None = 800,
    seed: int = 0,
) -> dict[str, Any]:
    """Select a confidence threshold using only a held-in validation split."""
    if selection_mode not in {"target_coverage", "target_risk"}:
        raise ValueError("selection_mode must be target_coverage or target_risk.")
    if not np.isfinite(target_coverage) or not 0.1 <= target_coverage <= 1.0:
        raise ValueError("target_coverage must be between 0.1 and 1.0.")
    if not np.isfinite(target_risk) or not 0.0 <= target_risk <= 1.0:
        raise ValueError("target_risk must be between 0 and 1.")

    texts = model_text_series(frame)
    labels = frame[target]
    valid = labels.notna() & texts.ne("")
    x = texts.loc[valid].reset_index(drop=True).astype(str)
    y = labels.loc[valid].astype(str).reset_index(drop=True)
    counts = y.value_counts()
    if max_rows and len(x) > max_rows:
        if y.nunique() < 2 or int(counts.min()) < 2:
            return {
                "status": "insufficient_data",
                "warning": "Not enough examples per class for a stratified validation sample.",
            }
        x, _, y, _ = train_test_split(
            x,
            y,
            train_size=max_rows,
            random_state=seed,
            stratify=y,
        )
        x, y = x.reset_index(drop=True), y.reset_index(drop=True)
        counts = y.value_counts()
    if y.nunique() < 2 or int(counts.min()) < 4:
        return {
            "status": "insufficient_data",
            "warning": "Need at least four labeled examples per class to select a validation threshold.",
        }

    split = stratified_group_holdout_indices(
        y,
        exact_text_group_ids(x),
        0.25,
        seed,
    )
    if split is None:
        return {
            "status": "insufficient_data",
            "warning": "Could not form a class-complete exact-text-grouped validation split.",
        }
    train_idx, val_idx = split
    x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=6_000)
    classifier = LogisticRegression(max_iter=800, random_state=seed)
    classifier.fit(vectorizer.fit_transform(x_train), y_train)
    probabilities = classifier.predict_proba(vectorizer.transform(x_val))
    predictions = classifier.classes_[probabilities.argmax(axis=1)]
    curve = risk_coverage_from_labels(
        y_val.to_numpy(),
        predictions,
        probabilities.max(axis=1),
    )
    if selection_mode == "target_risk":
        selected = threshold_for_target_risk(curve, target_risk)
        selected_target = target_risk
        achieved = selected["selective_risk"] is not None and selected["selective_risk"] <= target_risk
    else:
        selected = threshold_for_target_coverage(curve, target_coverage)
        selected_target = target_coverage
        achieved = True
    return {
        "status": "ok",
        "threshold": selected["threshold"],
        "validation_coverage": selected["coverage"],
        "validation_selective_risk": selected["selective_risk"],
        "validation_curve": curve,
        "validation_rows": int(len(x_val)),
        "selection_mode": selection_mode,
        "target": selected_target,
        "target_met_on_validation": bool(achieved),
        "source": "validation split only",
    }


def simulate_risk_coverage(
    frame: pd.DataFrame,
    target: str = "Ticket Type",
    target_coverage: float = 0.6,
    max_rows: int | None = 800,
    seed: int = 0,
    selection_mode: str = "target_coverage",
    target_risk: float = 0.1,
) -> dict[str, Any]:
    if selection_mode not in {"target_coverage", "target_risk"}:
        raise ValueError("selection_mode must be target_coverage or target_risk.")
    if not np.isfinite(target_coverage) or not 0.1 <= target_coverage <= 1.0:
        raise ValueError("target_coverage must be between 0.1 and 1.0.")
    if not np.isfinite(target_risk) or not 0.0 <= target_risk <= 1.0:
        raise ValueError("target_risk must be between 0 and 1.")
    texts = model_text_series(frame)
    labels = frame[target]
    valid = labels.notna() & texts.ne("")
    x = texts.loc[valid]
    y = labels.loc[valid].astype(str)
    if max_rows and len(x) > max_rows:
        x, _, y, _ = train_test_split(x, y, train_size=max_rows, random_state=seed, stratify=y)
    if y.nunique() < 2 or y.value_counts().min() < 4:
        return {"status": "insufficient_data"}
    groups = exact_text_group_ids(x)
    test_split = stratified_group_holdout_indices(y, groups, 0.20, seed)
    if test_split is None:
        return {
            "status": "insufficient_data",
            "warning": "Could not form a class-complete exact-text-grouped holdout.",
        }
    dev_idx, test_idx = test_split
    x_dev, x_test = x.iloc[dev_idx], x.iloc[test_idx]
    y_dev, y_test = y.iloc[dev_idx], y.iloc[test_idx]
    val_split = stratified_group_holdout_indices(
        y_dev,
        groups[dev_idx],
        0.25,
        seed + 1,
    )
    if val_split is None:
        return {
            "status": "insufficient_data",
            "warning": "Could not form a class-complete exact-text-grouped validation split.",
        }
    train_idx, val_idx = val_split
    x_train, x_val = x_dev.iloc[train_idx], x_dev.iloc[val_idx]
    y_train, y_val = y_dev.iloc[train_idx], y_dev.iloc[val_idx]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=6_000)
    clf = LogisticRegression(max_iter=800, random_state=seed)
    clf.fit(vectorizer.fit_transform(x_train), y_train)
    val_proba = clf.predict_proba(vectorizer.transform(x_val))
    val_pred = clf.classes_[val_proba.argmax(axis=1)]
    test_proba = clf.predict_proba(vectorizer.transform(x_test))
    test_pred = clf.classes_[test_proba.argmax(axis=1)]
    val_curve = risk_coverage_from_labels(y_val.to_numpy(), val_pred, val_proba.max(axis=1))
    selected = (
        threshold_for_target_risk(val_curve, target_risk)
        if selection_mode == "target_risk"
        else threshold_for_target_coverage(val_curve, target_coverage)
    )
    threshold = selected["threshold"]
    test_conf = test_proba.max(axis=1)
    accepted = test_conf >= threshold
    test_curve = risk_coverage_from_labels(y_test.to_numpy(), test_pred, test_conf)
    correct = y_test.to_numpy() == test_pred
    return {
        "status": "ok",
        "selection_mode": selection_mode,
        "target_coverage": target_coverage,
        "target_risk": target_risk if selection_mode == "target_risk" else None,
        "threshold_from_validation": threshold,
        "selected_validation_point": selected,
        "validation_rows": int(len(x_val)),
        "validation_curve": val_curve,
        "holdout_curve": test_curve,
        "holdout": {
            "coverage": float(accepted.mean()) if len(accepted) else 0.0,
            "selective_risk": selective_risk(correct, accepted),
            "acceptance_rate": float(accepted.mean()) if len(accepted) else 0.0,
            "abstention_rate": float((~accepted).mean()) if len(accepted) else 0.0,
            "correction_rate": "Not yet measured",
            "human_acceptance_rate": "Not yet measured",
        },
        "definitions": {
            "coverage": "Fraction of tickets the policy does not abstain on.",
            "selective_risk": "Error rate among accepted tickets, using existing labels.",
            "acceptance_rate": "Same as coverage in this simulation (no human overlay).",
            "correction_rate": "Share of accepted suggestions later corrected by a human reviewer. Requires genuine review events.",
            "abstention_rate": "Fraction of tickets left for human handling.",
        },
        "simulation_mode": True,
        "note": (
            "Holdout selective risk uses dataset labels as an oracle. Human acceptance and "
            "correction rates stay 'Not yet measured' until genuine review feedback exists. "
            "Exact normalized text is kept within one data split."
        ),
    }


def review_feedback_from_activity(events: list[dict[str, Any]]) -> dict[str, Any]:
    events_by_id = {
        int(item["id"]): item
        for item in events
        if item.get("id") is not None
    }
    decisions = [
        item
        for item in events
        if item.get("event_type") in {"recommendation_approved", "recommendation_rejected"}
    ]
    if not decisions:
        return {
            "human_acceptance_rate": "Not yet measured",
            "correction_rate": "Not yet measured",
            "n_review_events": 0,
            "note": "No genuine approve/reject review events are stored yet.",
        }
    approved = [item for item in decisions if item.get("event_type") == "recommendation_approved"]
    rejected = [item for item in decisions if item.get("event_type") == "recommendation_rejected"]
    edited = 0
    for item in approved:
        details = item.get("details") or {}
        overlay = details.get("overlay") or {}
        parent = events_by_id.get(details.get("parent_activity_id"))
        predictions = (parent or {}).get("details", {}).get("predictions", {})
        if any(
            target in overlay
            and target in predictions
            and str(overlay[target]) != str(predictions[target].get("label"))
            for target in ("Ticket Type", "Ticket Priority")
        ):
            edited += 1
    n = len(decisions)
    return {
        "human_acceptance_rate": len(approved) / n,
        "n_review_events": n,
        "approved": len(approved),
        "rejected": len(rejected),
        "correction_rate": edited / len(approved) if approved else "Not yet measured",
        "note": "Rates use stored Copilot approve/reject events only; they are not simulated.",
    }
