from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from data import model_text_series, redact_sensitive_text
from reproducibility import (
    exact_text_group_ids,
    mean_std_ci,
    stratified_group_holdout_indices,
)


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def _fit_known_model(x_train: pd.Series, y_train: pd.Series):
    vectorizer = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=1, max_features=6_000
    )
    matrix = vectorizer.fit_transform(x_train)
    clf = LogisticRegression(max_iter=800, random_state=42)
    clf.fit(matrix, y_train)
    centroids = {}
    for label in clf.classes_:
        rows = matrix[(y_train.to_numpy() == label)]
        centroids[label] = np.asarray(rows.mean(axis=0)).ravel()
    return vectorizer, clf, centroids


def _scores(vectorizer, clf, centroids, texts: pd.Series) -> dict[str, np.ndarray]:
    matrix = vectorizer.transform(texts)
    probabilities = clf.predict_proba(matrix)
    predicted = clf.classes_[probabilities.argmax(axis=1)]
    msp = probabilities.max(axis=1)
    entropy = _entropy(probabilities)
    dense = np.asarray(normalize(matrix).todense())
    centroid_lookup = {
        label: normalize(centroid.reshape(1, -1)).ravel()
        for label, centroid in centroids.items()
    }
    cosine = np.array(
        [
            float(np.dot(dense[index], centroid_lookup[str(predicted[index])]))
            for index in range(len(predicted))
        ]
    )
    distance = 1.0 - np.clip(cosine, -1.0, 1.0)
    all_centroids = np.vstack([centroid_lookup[label] for label in clf.classes_])
    max_sim = dense @ all_centroids.T
    tfidf_novelty = 1.0 - np.clip(max_sim.max(axis=1), -1.0, 1.0)
    return {
        "msp": msp,
        "entropy": entropy,
        "centroid_distance": distance,
        "tfidf_novelty": tfidf_novelty,
        "predicted": predicted,
        "probabilities": probabilities,
    }


def _unknown_metrics(y_unknown: np.ndarray, score: np.ndarray, higher_means_unknown: bool) -> dict[str, Any]:
    ranking = score if higher_means_unknown else -score
    if len(np.unique(y_unknown)) < 2:
        return {
            "auroc": None,
            "aupr": None,
            "warning": "Need both known and unknown examples to compute AUROC/AUPR.",
        }
    auroc = float(roc_auc_score(y_unknown, ranking))
    aupr = float(average_precision_score(y_unknown, ranking))
    fpr, tpr, _ = roc_curve(y_unknown, ranking)
    precision, recall, _ = precision_recall_curve(y_unknown, ranking)
    return {
        "auroc": auroc,
        "aupr": aupr,
        "roc": {"fpr": fpr.tolist()[:: max(1, len(fpr) // 40)], "tpr": tpr.tolist()[:: max(1, len(tpr) // 40)]},
        "pr": {
            "precision": precision.tolist()[:: max(1, len(precision) // 40)],
            "recall": recall.tolist()[:: max(1, len(recall) // 40)],
        },
    }


def _threshold_from_validation(known_score: np.ndarray, target_fpr: float, higher_means_unknown: bool) -> float:
    scores = np.asarray(known_score, dtype=float)
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("Validation scores must be non-empty and finite.")
    allowed_false_positives = int(np.floor(target_fpr * len(scores)))
    ordered = np.sort(scores)
    if higher_means_unknown:
        if allowed_false_positives >= len(ordered):
            return float(np.nextafter(ordered[0], -np.inf))
        boundary = ordered[len(ordered) - allowed_false_positives - 1]
        return float(np.nextafter(boundary, np.inf))
    if allowed_false_positives >= len(ordered):
        return float(np.nextafter(ordered[-1], np.inf))
    boundary = ordered[allowed_false_positives]
    return float(np.nextafter(boundary, -np.inf))


def run_open_set_experiment(
    frame: pd.DataFrame,
    target: str = "Ticket Type",
    seeds: tuple[int, ...] = tuple(range(5)),
    unknown_classes: list[str] | None = None,
    max_rows: int | None = 900,
    target_known_fpr: float = 0.10,
) -> dict[str, Any]:
    if target not in frame.columns:
        raise ValueError(f"Unknown target column: {target}")
    if not np.isfinite(target_known_fpr) or not 0 <= target_known_fpr <= 1:
        raise ValueError("target_known_fpr must be between 0 and 1.")
    if not seeds:
        raise ValueError("At least one random seed is required.")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive when provided.")

    texts = model_text_series(frame)
    labels = frame[target]
    valid = labels.notna() & texts.ne("")
    working = pd.DataFrame({"text": texts.loc[valid], "label": labels.loc[valid].astype(str)})
    class_counts = working["label"].value_counts()
    if max_rows and len(working) > max_rows:
        if int(class_counts.min()) < 2 or max_rows < len(class_counts):
            return {
                "status": "insufficient_data",
                "message": (
                    "A stratified row cap must fit every class and requires at least "
                    "two records in every class."
                ),
            }
        working, _ = train_test_split(
            working,
            train_size=max_rows,
            random_state=0,
            stratify=working["label"],
        )
    class_counts = working["label"].value_counts()
    classes = list(class_counts.index)
    if len(classes) < 3:
        return {
            "status": "insufficient_classes",
            "message": "Open-set evaluation needs at least three labeled classes so one can be held out.",
        }
    requested_held_out = unknown_classes or [str(class_counts.index[-1])]
    missing_held_out = [label for label in requested_held_out if label not in classes]
    if missing_held_out:
        return {
            "status": "unknown_classes_not_found",
            "message": "Requested unknown classes were not found in the selected target labels.",
            "requested_unknown_classes": missing_held_out,
        }
    held_out = list(dict.fromkeys(requested_held_out))
    known_classes = [label for label in classes if label not in held_out]
    if len(known_classes) < 2:
        return {
            "status": "insufficient_known_classes",
            "message": "At least two known classes must remain after holding out unknown classes.",
        }

    working["text_group"] = exact_text_group_ids(working["text"])
    known_group_ids = set(
        working.loc[working["label"].isin(known_classes), "text_group"]
    )
    unknown_group_ids = set(
        working.loc[working["label"].isin(held_out), "text_group"]
    )
    overlapping_groups = known_group_ids & unknown_group_ids
    if overlapping_groups:
        working = working.loc[~working["text_group"].isin(overlapping_groups)]
    available_classes = set(working["label"])
    held_out = [label for label in held_out if label in available_classes]
    known_classes = [label for label in known_classes if label in available_classes]
    if len(known_classes) < 2 or not any(
        label in available_classes for label in held_out
    ):
        return {
            "status": "insufficient_data",
            "message": (
                "Removing exact-text groups shared by known and held-out labels left "
                "too little data for this experiment."
            ),
            "excluded_cross_partition_duplicate_groups": len(overlapping_groups),
        }

    methods = {
        "maximum_softmax_probability": {"higher_means_unknown": False, "key": "msp"},
        "entropy": {"higher_means_unknown": True, "key": "entropy"},
        "centroid_distance": {"higher_means_unknown": True, "key": "centroid_distance"},
        "tfidf_novelty": {"higher_means_unknown": True, "key": "tfidf_novelty"},
    }
    per_seed: list[dict[str, Any]] = []
    error_examples: list[dict[str, Any]] = []

    for seed in seeds:
        known = working[working["label"].isin(known_classes)]
        unknown = working[working["label"].isin(held_out)]
        if known.empty or unknown.empty:
            continue
        known_groups = known["text_group"].to_numpy()
        trainval_split = stratified_group_holdout_indices(
            known["label"],
            known_groups,
            0.20,
            seed,
        )
        if trainval_split is None:
            continue
        trainval_idx, test_idx = trainval_split
        trainval = known.iloc[trainval_idx]
        test_known = known.iloc[test_idx]
        val_split = stratified_group_holdout_indices(
            trainval["label"],
            trainval["text_group"].to_numpy(),
            0.25,
            seed + 1,
        )
        if val_split is None:
            continue
        train_idx, val_idx = val_split
        train = trainval.iloc[train_idx]
        val = trainval.iloc[val_idx]
        vectorizer, clf, centroids = _fit_known_model(train["text"], train["label"])
        val_scores = _scores(vectorizer, clf, centroids, val["text"])
        hold_texts = pd.concat([test_known["text"], unknown["text"]])
        hold_unknown = np.concatenate(
            [np.zeros(len(test_known), dtype=int), np.ones(len(unknown), dtype=int)]
        )
        hold_scores = _scores(vectorizer, clf, centroids, hold_texts)
        seed_result = {"seed": seed, "methods": {}}
        for method, spec in methods.items():
            key = spec["key"]
            higher = spec["higher_means_unknown"]
            threshold = _threshold_from_validation(
                val_scores[key][: len(val)], target_known_fpr, higher
            )
            ranking_metrics = _unknown_metrics(hold_unknown, hold_scores[key], higher)
            if higher:
                predicted_unknown = hold_scores[key] >= threshold
            else:
                predicted_unknown = hold_scores[key] <= threshold
            unknown_recall = (
                float(predicted_unknown[hold_unknown == 1].mean())
                if (hold_unknown == 1).any()
                else None
            )
            known_rejection = (
                float(predicted_unknown[hold_unknown == 0].mean())
                if (hold_unknown == 0).any()
                else None
            )
            known_pred = hold_scores["predicted"][hold_unknown == 0]
            known_true = test_known["label"].to_numpy()
            known_acc = float((known_pred == known_true).mean()) if len(known_true) else None
            known_report = classification_report(
                known_true,
                known_pred,
                labels=known_classes,
                output_dict=True,
                zero_division=0,
            )
            seed_result["methods"][method] = {
                "threshold": threshold,
                "unknown_recall": unknown_recall,
                "known_rejection_rate": known_rejection,
                "known_class_accuracy": known_acc,
                "known_classification": {
                    "per_class": {
                        label: known_report[label]
                        for label in known_classes
                        if label in known_report
                    },
                    "confusion_matrix": confusion_matrix(
                        known_true,
                        known_pred,
                        labels=known_classes,
                    ).tolist(),
                    "labels": known_classes,
                },
                **ranking_metrics,
            }
            if method == "maximum_softmax_probability" and len(error_examples) < 8:
                texts_hold = hold_texts.to_numpy()
                labels_hold = pd.concat(
                    [test_known["label"], unknown["label"]], ignore_index=True
                ).to_numpy()
                for index, is_unknown in enumerate(hold_unknown):
                    flagged = bool(predicted_unknown[index])
                    if flagged and not is_unknown and len(error_examples) < 8:
                        error_examples.append(
                            {
                                "kind": "false_unknown",
                                "text": redact_sensitive_text(texts_hold[index])[:220],
                                "true_label": str(labels_hold[index]),
                            }
                        )
                    if (not flagged) and is_unknown and len(error_examples) < 8:
                        error_examples.append(
                            {
                                "kind": "missed_unknown",
                                "text": redact_sensitive_text(texts_hold[index])[:220],
                                "true_label": str(labels_hold[index]),
                            }
                        )
        per_seed.append(seed_result)

    if not per_seed:
        return {
            "status": "insufficient_data",
            "message": "Could not form stratified known/unknown splits with the requested hold-out classes.",
            "held_out_classes": held_out,
        }

    summary = {}
    for method in methods:
        summary[method] = {
            "auroc": mean_std_ci(
                [row["methods"][method]["auroc"] for row in per_seed if row["methods"][method].get("auroc") is not None]
            ),
            "aupr": mean_std_ci(
                [row["methods"][method]["aupr"] for row in per_seed if row["methods"][method].get("aupr") is not None]
            ),
            "unknown_recall": mean_std_ci(
                [
                    row["methods"][method]["unknown_recall"]
                    for row in per_seed
                    if row["methods"][method].get("unknown_recall") is not None
                ]
            ),
            "known_rejection_rate": mean_std_ci(
                [
                    row["methods"][method]["known_rejection_rate"]
                    for row in per_seed
                    if row["methods"][method].get("known_rejection_rate") is not None
                ]
            ),
            "known_class_accuracy": mean_std_ci(
                [
                    row["methods"][method]["known_class_accuracy"]
                    for row in per_seed
                    if row["methods"][method].get("known_class_accuracy") is not None
                ]
            ),
        }

    return {
        "status": "ok",
        "target": target,
        "known_classes": known_classes,
        "held_out_classes": held_out,
        "seeds": [row["seed"] for row in per_seed],
        "target_known_fpr_for_threshold": target_known_fpr,
        "threshold_source": "validation split only",
        "split_grouping": "normalized identical text stays within a single known-class split",
        "excluded_cross_partition_duplicate_groups": len(overlapping_groups),
        "summary": summary,
        "per_seed": per_seed,
        "error_examples": error_examples,
        "limitations": [
            "Held-out known categories are a proxy for unseen incidents, not a guarantee of detecting every novel ticket.",
            "Real unseen issues can differ in language, product, and severity from any held-out training class.",
            "Thresholds are selected on validation data and then frozen for the untouched holdout.",
        ],
    }


def open_set_decision(
    probabilities: np.ndarray,
    classes: list[str],
    threshold: float,
    method: str = "maximum_softmax_probability",
) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(1, -1)
    msp = float(probabilities.max())
    entropy = float(_entropy(probabilities)[0])
    label = str(classes[int(probabilities.argmax())])
    if method == "entropy":
        unknown = entropy >= threshold
        score = entropy
    else:
        unknown = msp <= threshold
        score = msp
    return {
        "label": "UNKNOWN" if unknown else label,
        "predicted_class": label,
        "abstained": bool(unknown),
        "score": score,
        "msp": msp,
        "entropy": entropy,
        "method": method,
        "threshold": threshold,
    }
