from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from data import redact_sensitive_text, ticket_text
from evaluation import chronological_split, expected_calibration_error
from openset import open_set_decision
from reproducibility import (
    exact_text_group_ids,
    mean_std_ci,
    normalized_text_keys,
)


TARGETS = ("Ticket Type", "Ticket Priority")
CV_SEEDS = (0, 1, 2)
CV_FOLDS = 3


def _correction_examples(
    frame: pd.DataFrame,
    texts: pd.Series,
    examples: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    text_by_id = dict(zip(frame["Ticket ID"].astype(str), texts))
    return [
        (str(text_by_id[str(item["ticket_id"])]), str(item["corrected_label"]))
        for item in examples
        if str(item["ticket_id"]) in text_by_id
        and str(text_by_id[str(item["ticket_id"])]).strip()
    ]


def _append_reviewed_examples(
    x_train: pd.Series,
    y_train: pd.Series,
    x_test: pd.Series,
    corrections: list[tuple[str, str]],
) -> tuple[pd.Series, pd.Series]:
    if not corrections:
        return x_train, y_train
    test_texts = set(normalized_text_keys(x_test))
    usable = [
        (text, label)
        for text, label in corrections
        if normalized_text_keys(pd.Series([text])).iloc[0] not in test_texts
    ]
    if not usable:
        return x_train, y_train
    correction_texts, correction_labels = zip(*usable)
    return (
        pd.concat([x_train, pd.Series(correction_texts)], ignore_index=True),
        pd.concat(
            [y_train.reset_index(drop=True), pd.Series(correction_labels)],
            ignore_index=True,
        ),
    )


def _classification_metrics(
    y_true: pd.Series | list[str],
    predictions: np.ndarray | list[str],
    probabilities: np.ndarray,
    classes: list[str],
) -> dict[str, float]:
    truth = [str(value) for value in y_true]
    predicted = [str(value) for value in predictions]
    all_classes = sorted(set(truth) | set(classes))
    class_indexes = {label: index for index, label in enumerate(classes)}
    one_hot = np.zeros((len(truth), len(all_classes)))
    probability_matrix = np.zeros_like(one_hot)
    all_class_indexes = {label: index for index, label in enumerate(all_classes)}
    for row_index, label in enumerate(truth):
        one_hot[row_index, all_class_indexes[label]] = 1.0
    for label, source_index in class_indexes.items():
        probability_matrix[:, all_class_indexes[label]] = probabilities[:, source_index]
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(
            f1_score(truth, predicted, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(truth, predicted, average="weighted", zero_division=0)
        ),
        "expected_calibration_error": expected_calibration_error(
            truth, probabilities, y_pred=predicted
        ),
        "multiclass_brier_score": float(
            np.mean(np.sum((probability_matrix - one_hot) ** 2, axis=1))
        ),
    }


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

        groups = exact_text_group_ids(x)
        support = pd.DataFrame({"label": y, "group": groups}).groupby("label")[
            "group"
        ].nunique()
        n_splits = min(CV_FOLDS, int(support.min()))
        cv_rows: list[dict[str, Any]] = []
        all_truth: list[str] = []
        all_predictions: list[str] = []
        all_corrections = _correction_examples(frame, texts, examples)

        if n_splits >= 2:
            for seed in CV_SEEDS:
                splitter = StratifiedGroupKFold(
                    n_splits=n_splits, shuffle=True, random_state=seed
                )
                for fold, (train_idx, test_idx) in enumerate(
                    splitter.split(x, y, groups), start=1
                ):
                    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    if set(y_train) != set(y):
                        continue
                    fold_x, fold_y = _append_reviewed_examples(
                        x_train,
                        y_train,
                        x_test,
                        all_corrections,
                    )
                    pipeline = _text_classifier()
                    pipeline.fit(fold_x, fold_y)
                    predicted = pipeline.predict(x_test)
                    probabilities = pipeline.predict_proba(x_test)
                    classes = [str(value) for value in pipeline.classes_]
                    scores = {
                        "seed": seed,
                        "fold": fold,
                        "test_rows": int(len(y_test)),
                        **_classification_metrics(
                            y_test, predicted, probabilities, classes
                        ),
                    }
                    cv_rows.append(scores)
                    all_truth.extend(str(value) for value in y_test)
                    all_predictions.extend(str(value) for value in predicted)

        cross_validation: dict[str, Any]
        if cv_rows:
            cross_validation = {
                "status": "ok",
                "folds": n_splits,
                "seeds": list(CV_SEEDS),
                "observations": len(cv_rows),
                "labeled_rows": int(len(x)),
                "scores": cv_rows,
                "metrics": {
                    metric: mean_std_ci([row[metric] for row in cv_rows])
                    for metric in (
                        "accuracy",
                        "macro_f1",
                        "weighted_f1",
                        "expected_calibration_error",
                        "multiclass_brier_score",
                    )
                },
                "note": (
                    "Intervals summarize fold scores across seeds; folds overlap across seeds, "
                    "so intervals are descriptive and not independent-sample inference."
                ),
            }
        else:
            cross_validation = {
                "status": "unavailable",
                "folds": n_splits,
                "seeds": list(CV_SEEDS),
                "observations": 0,
                "labeled_rows": int(len(x)),
                "metrics": {},
                "note": (
                    "Not enough distinct exact-text groups per class to form stratified "
                    "grouped folds with every class in training."
                ),
            }

        provenance = frame.attrs.get("dataset_provenance", {})
        chronological, chronological_meta = chronological_split(
            frame,
            target,
            timestamps_trustworthy=(
                provenance.get("timestamps_trustworthy") is True
            ),
        )
        chronological_report: dict[str, Any] = {
            "status": chronological_meta["status"],
            "detail": chronological_meta,
        }
        if chronological is not None:
            test_texts = set(normalized_text_keys(chronological["x_test"]))
            train_without_test_duplicates = ~normalized_text_keys(
                chronological["x_train"]
            ).isin(test_texts)
            chrono_train_x = chronological["x_train"].loc[
                train_without_test_duplicates
            ]
            chrono_train_y = chronological["y_train"].loc[
                train_without_test_duplicates
            ]
            if chrono_train_y.nunique() < 2:
                reason = (
                    "Too few training classes remain after removing exact-text "
                    "overlap with the later test period."
                )
                chronological_report.update(
                    {
                        "status": "unavailable",
                        "detail": {"status": "unavailable", "reason": reason},
                    }
                )
            else:
                chrono_x, chrono_y = _append_reviewed_examples(
                    chrono_train_x,
                    chrono_train_y,
                    chronological["x_test"],
                    all_corrections,
                )
                chrono_model = _text_classifier()
                chrono_model.fit(chrono_x, chrono_y)
                chrono_pred = chrono_model.predict(chronological["x_test"])
                chrono_probabilities = chrono_model.predict_proba(
                    chronological["x_test"]
                )
                chrono_classes = [str(value) for value in chrono_model.classes_]
                chronological_report.update(
                    {
                        "status": "ok",
                        "metrics": _classification_metrics(
                            chronological["y_test"],
                            chrono_pred,
                            chrono_probabilities,
                            chrono_classes,
                        ),
                        "train_rows": int(len(chrono_x)),
                        "test_rows": int(len(chronological["y_test"])),
                        "exact_text_overlap_rows_excluded": int(
                            (~train_without_test_duplicates).sum()
                        ),
                        "cutoff": chronological["cutoff"],
                        "classes": classification_report(
                            chronological["y_test"],
                            chrono_pred,
                            output_dict=True,
                            zero_division=0,
                        ),
                    }
                )

        final_x, final_y = _append_reviewed_examples(
            x,
            y,
            x.iloc[0:0],
            all_corrections,
        )
        pipeline = _text_classifier()
        pipeline.fit(final_x, final_y)
        estimators[target] = pipeline

        if cv_rows:
            cv_metrics = cross_validation["metrics"]
            reports[target] = {
                "status": "Repeated stratified grouped cross-validation",
                "training_rows": int(len(final_x)),
                "test_rows": 0,
                "labeled_rows": int(len(x)),
                "reviewed_examples": len(examples),
                "accuracy": cv_metrics["accuracy"]["mean"],
                "macro_f1": cv_metrics["macro_f1"]["mean"],
                "weighted_f1": cv_metrics["weighted_f1"]["mean"],
                "expected_calibration_error": cv_metrics[
                    "expected_calibration_error"
                ]["mean"],
                "multiclass_brier_score": cv_metrics[
                    "multiclass_brier_score"
                ]["mean"],
                "cross_validation": cross_validation,
                "chronological": chronological_report,
                "split_grouping": "normalized identical text stays within each fold",
                "classes": classification_report(
                    all_truth,
                    all_predictions,
                    labels=sorted(set(all_truth) | set(all_predictions)),
                    output_dict=True,
                    zero_division=0,
                ),
            }
        else:
            reports[target] = {
                "status": cross_validation["note"],
                "training_rows": int(len(final_x)),
                "test_rows": 0,
                "labeled_rows": int(len(x)),
                "reviewed_examples": len(examples),
                "accuracy": None,
                "macro_f1": None,
                "weighted_f1": None,
                "expected_calibration_error": None,
                "multiclass_brier_score": None,
                "cross_validation": cross_validation,
                "chronological": chronological_report,
                "split_grouping": "normalized identical text stays within each fold",
                "classes": {},
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


def top_contributing_terms(estimator: Pipeline, text: str, label: str, limit: int = 8) -> list[dict[str, Any]]:
    if "tfidf" not in estimator.named_steps or "classifier" not in estimator.named_steps:
        return []
    vectorizer = estimator.named_steps["tfidf"]
    classifier = estimator.named_steps["classifier"]
    if not hasattr(classifier, "coef_"):
        return []
    features = np.asarray(vectorizer.get_feature_names_out())
    transformed = vectorizer.transform([text])
    if transformed.nnz == 0:
        return []
    classes = list(classifier.classes_)
    class_index = classes.index(label)
    if classifier.coef_.shape[0] == 1 and len(classes) == 2:
        weights = classifier.coef_[0] * (1 if class_index == 1 else -1)
    else:
        weights = classifier.coef_[class_index]
    contributions = transformed.toarray().ravel() * weights
    order = np.argsort(np.abs(contributions))[::-1]
    terms = []
    for index in order:
        if contributions[index] == 0:
            continue
        terms.append(
            {
                "term": str(features[index]),
                "contribution": float(contributions[index]),
            }
        )
        if len(terms) >= limit:
            break
    return terms


def predict_ticket(
    estimators: dict[str, Pipeline],
    subject: str,
    description: str,
    open_set_threshold: float | None = None,
    open_set_method: str = "maximum_softmax_probability",
) -> dict[str, dict[str, Any]]:
    text_value = redact_sensitive_text(f"{subject}\n{description}".strip())
    text = pd.Series([text_value])
    results: dict[str, dict[str, Any]] = {}
    for target, estimator in estimators.items():
        if not hasattr(estimator, "predict_proba"):
            label = str(estimator.predict(text)[0])
            results[target] = {
                "label": label,
                "score": None,
                "probabilities": {},
                "probability_sum": None,
                "supports_probabilities": False,
                "top_terms": [],
                "abstained": False,
                "explanation_warning": (
                    "This estimator does not expose class probabilities. "
                    "Term contributions are model-specific and not causal."
                ),
            }
            continue
        probabilities = estimator.predict_proba(text)[0]
        classes = [str(value) for value in estimator.classes_]
        best_index = int(probabilities.argmax())
        label = classes[best_index]
        probability_map = {
            classes[index]: float(probabilities[index])
            for index in range(len(classes))
        }
        decision = None
        if open_set_threshold is not None and target == "Ticket Type":
            decision = open_set_decision(
                probabilities, classes, open_set_threshold, open_set_method
            )
        displayed_label = decision["label"] if decision and decision["abstained"] else label
        results[target] = {
            "label": displayed_label,
            "predicted_class": label,
            "score": float(probabilities[best_index]),
            "probabilities": probability_map,
            "probability_sum": float(sum(probability_map.values())),
            "supports_probabilities": True,
            "entropy": decision["entropy"] if decision else None,
            "abstained": bool(decision["abstained"]) if decision else False,
            "open_set": decision,
            "top_terms": top_contributing_terms(estimator, text_value, label),
            "explanation_warning": (
                "Highlighted terms are local contributions from this linear model. "
                "They are not causal explanations of the customer's issue."
            ),
        }
    return results
