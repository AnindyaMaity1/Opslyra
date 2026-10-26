from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data import redact_sensitive_text, ticket_text


TARGETS = ("Ticket Type", "Ticket Priority")


def train_baselines(
    frame: pd.DataFrame,
    corrections: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Pipeline], dict[str, dict[str, Any]]]:
    corrections = corrections or []
    texts = ticket_text(frame)
    estimators: dict[str, Pipeline] = {}
    reports: dict[str, dict[str, Any]] = {}
    correction_targets = {
        target: [
            item
            for item in corrections
            if item["target"] == target and str(item["corrected_label"]).strip()
        ]
        for target in (*TARGETS, "Team")
    }

    for target in (*TARGETS, "Team"):
        labels = (
            frame[target]
            if target in frame.columns
            else pd.Series(pd.NA, index=frame.index, dtype="string")
        )
        valid = labels.notna() & texts.ne("")
        x = texts.loc[valid]
        y = labels.loc[valid].astype(str)
        examples = correction_targets[target]
        corrected_ids = {str(item["ticket_id"]) for item in examples}
        if corrected_ids:
            uncorrected = frame.loc[valid, "Ticket ID"].astype(str).isin(corrected_ids)
            x = x.loc[~uncorrected]
            y = y.loc[~uncorrected]

        if target == "Team":
            if not examples:
                continue
            correction_texts = []
            correction_labels = []
            text_by_id = dict(zip(frame["Ticket ID"].astype(str), texts))
            for item in examples:
                ticket_id = str(item["ticket_id"])
                if ticket_id in text_by_id:
                    correction_texts.append(text_by_id[ticket_id])
                    correction_labels.append(str(item["corrected_label"]))
            if len(set(correction_labels)) < 2 or min(
                correction_labels.count(label) for label in set(correction_labels)
            ) < 2:
                continue
            pipeline = _text_classifier()
            pipeline.fit(pd.Series(correction_texts), correction_labels)
            estimators[target] = pipeline
            reports[target] = {
                "status": "reviewed examples only; no independent team ground truth",
                "training_rows": len(correction_labels),
                "test_rows": 0,
                "accuracy": None,
                "macro_f1": None,
                "weighted_f1": None,
                "classes": {},
            }
            continue

        if y.nunique() < 2 or y.value_counts().min() < 2:
            continue

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )
        if examples:
            text_by_id = dict(zip(frame["Ticket ID"].astype(str), texts))
            correction_texts = []
            correction_labels = []
            for item in examples:
                ticket_id = str(item["ticket_id"])
                if ticket_id in text_by_id:
                    correction_texts.append(text_by_id[ticket_id])
                    correction_labels.append(str(item["corrected_label"]))
            if correction_texts:
                x_train = pd.concat([x_train, pd.Series(correction_texts)], ignore_index=True)
                y_train = pd.concat(
                    [y_train.reset_index(drop=True), pd.Series(correction_labels)],
                    ignore_index=True,
                )
        pipeline = _text_classifier()
        pipeline.fit(x_train, y_train)
        predicted = pipeline.predict(x_test)
        probabilities = pipeline.predict_proba(x_test)
        max_probabilities = probabilities.max(axis=1)
        correct = (predicted == y_test.to_numpy()).astype(float)
        calibration_error = 0.0
        for lower in np.linspace(0, 0.9, 10):
            in_bin = (max_probabilities > lower) & (max_probabilities <= lower + 0.1)
            if in_bin.any():
                calibration_error += (
                    in_bin.mean()
                    * abs(correct[in_bin].mean() - max_probabilities[in_bin].mean())
                )
        class_indexes = {
            label: index for index, label in enumerate(pipeline.classes_)
        }
        true_indexes = np.asarray([class_indexes[label] for label in y_test])
        one_hot = np.eye(len(pipeline.classes_))[true_indexes]
        estimators[target] = pipeline
        reports[target] = {
            "training_rows": int(len(x_train)),
            "reviewed_examples": len(examples),
            "test_rows": int(len(x_test)),
            "accuracy": float(accuracy_score(y_test, predicted)),
            "macro_f1": float(f1_score(y_test, predicted, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_test, predicted, average="weighted", zero_division=0)),
            "expected_calibration_error": float(calibration_error),
            "multiclass_brier_score": float(
                np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
            ),
            "classes": classification_report(
                y_test,
                predicted,
                output_dict=True,
                zero_division=0,
            ),
        }

    return estimators, reports


def _text_classifier() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=30_000,
                    strip_accents="unicode",
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1_000, random_state=42)),
        ]
    )


def predict_ticket(
    estimators: dict[str, Pipeline], subject: str, description: str
) -> dict[str, dict[str, Any]]:
    text = pd.Series([redact_sensitive_text(f"{subject}\n{description}".strip())])
    results: dict[str, dict[str, Any]] = {}
    for target, estimator in estimators.items():
        probabilities = estimator.predict_proba(text)[0]
        best_index = int(probabilities.argmax())
        results[target] = {
            "label": str(estimator.classes_[best_index]),
            "score": float(probabilities[best_index]),
        }
    return results
