from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from data import model_text_series
from reproducibility import (
    exact_text_group_ids,
    mean_std_ci,
    stratified_group_holdout_indices,
)


DEFAULT_SEEDS = (0, 1, 2)
HOLDOUT_FRACTION = 0.20
N_SPLITS = 3
MAX_FEATURES = 8_000


def expected_calibration_error(
    y_true,
    probabilities=None,
    n_bins: int = 10,
    y_pred=None,
) -> float:
    """Equal-width ECE on maximum predicted class probability.

    Bins are ``[0, 1/n_bins), ..., [(n_bins-1)/n_bins, 1]``. The last bin is
    closed on the right. ECE is the weighted absolute gap between accuracy and
    confidence in each bin. ``y_true``/``y_pred`` are compared as strings so
    class-index order of the probability matrix cannot silently invert ECE.
    """
    if probabilities is None:
        raise ValueError("probabilities are required for ECE")
    if n_bins < 1:
        raise ValueError("n_bins must be a positive integer.")
    probabilities = np.asarray(probabilities)
    confidences = probabilities.max(axis=1)
    if y_pred is None:
        y_pred = probabilities.argmax(axis=1)
    y_true = list(y_true)
    y_pred = list(y_pred)
    if len(y_true) != len(y_pred) or len(y_true) != len(confidences):
        raise ValueError("Labels and probability rows must have the same length.")
    correct = np.asarray([str(truth) == str(pred) for truth, pred in zip(y_true, y_pred)], dtype=float)
    return expected_calibration_error_from_confidence(correct, confidences, n_bins=n_bins)


def expected_calibration_error_from_confidence(
    correct, confidences, n_bins: int = 10
) -> float:
    correct = np.asarray(correct, dtype=float)
    confidences = np.asarray(confidences, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        left, right = edges[index], edges[index + 1]
        if index == n_bins - 1:
            in_bin = (confidences >= left) & (confidences <= right)
        else:
            in_bin = (confidences >= left) & (confidences < right)
        if not in_bin.any():
            continue
        ece += in_bin.mean() * abs(correct[in_bin].mean() - confidences[in_bin].mean())
    return float(ece)


def reliability_diagram_points(
    y_true,
    probabilities=None,
    n_bins: int = 10,
    y_pred=None,
    correct=None,
    confidences=None,
) -> list[dict[str, float]]:
    if n_bins < 1:
        raise ValueError("n_bins must be a positive integer.")
    if correct is None or confidences is None:
        if probabilities is None:
            return []
        probabilities = np.asarray(probabilities)
        confidences = probabilities.max(axis=1)
        if y_pred is None:
            y_pred = probabilities.argmax(axis=1)
        correct = np.asarray(
            [str(truth) == str(pred) for truth, pred in zip(y_true, y_pred)],
            dtype=float,
        )
    correct = np.asarray(correct, dtype=float)
    confidences = np.asarray(confidences, dtype=float)
    try:
        frac_pos, mean_pred = calibration_curve(
            correct, confidences, n_bins=n_bins, strategy="uniform"
        )
        return [
            {"mean_predicted": float(mp), "empirical": float(fp)}
            for mp, fp in zip(mean_pred, frac_pos)
        ]
    except ValueError:
        return []


def _tfidf(ngram_range=(1, 2), max_features=MAX_FEATURES) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=ngram_range,
        min_df=1,
        max_features=max_features,
        strip_accents="unicode",
    )


def _encode_split(x_train: pd.Series, x_test: pd.Series) -> tuple[np.ndarray, np.ndarray, str]:
    """Encode train/test without fitting transformers on the test split.

    Sentence-transformers are used only when already installed *and*
    ``OPSLYRA_EMBEDDING_MODEL`` is set. Otherwise a TF-IDF + TruncatedSVD
    surrogate is fitted on training text only.
    """
    import os

    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    x_train = x_train.astype(str)
    x_test = x_test.astype(str)
    model_name = os.environ.get("OPSLYRA_EMBEDDING_MODEL", "").strip()
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        SentenceTransformer = None  # type: ignore
    if SentenceTransformer is not None and model_name:
        encoder = SentenceTransformer(model_name)
        train_mat = np.asarray(encoder.encode(list(x_train), show_progress_bar=False))
        test_mat = np.asarray(encoder.encode(list(x_test), show_progress_bar=False))
        return train_mat, test_mat, model_name

    tfidf = TfidfVectorizer(min_df=1, ngram_range=(1, 2), max_features=6000)
    train_sparse = tfidf.fit_transform(x_train)
    test_sparse = tfidf.transform(x_test)
    n_features = train_sparse.shape[1]
    n_rows = train_sparse.shape[0]
    components = min(128, max(1, n_rows - 1), max(1, n_features - 1))
    if components < 2:
        return train_sparse.toarray(), test_sparse.toarray(), "tfidf_raw_fallback"
    svd = TruncatedSVD(n_components=components, random_state=42)
    train_mat = normalize(svd.fit_transform(train_sparse))
    test_mat = normalize(svd.transform(test_sparse))
    return train_mat, test_mat, "tfidf_svd_fallback"


def model_zoo() -> dict[str, Any]:
    return {
        "majority": DummyClassifier(strategy="most_frequent"),
        "complement_nb": Pipeline(
            [("tfidf", _tfidf((1, 1))), ("clf", ComplementNB())]
        ),
        "linear_svm": Pipeline(
            [
                ("tfidf", _tfidf((1, 2))),
                ("clf", LinearSVC(max_iter=2_000)),
            ]
        ),
        "tfidf_logreg": Pipeline(
            [
                ("tfidf", _tfidf((1, 2))),
                ("clf", LogisticRegression(max_iter=800, random_state=42)),
            ]
        ),
        "embeddings_logreg": "embeddings_logreg",
    }


def _score(y_true, y_pred, y_proba=None, labels=None) -> dict[str, Any]:
    labels = labels if labels is not None else sorted(set(map(str, y_true)) | set(map(str, y_pred)))
    y_true_list = [str(value) for value in y_true]
    y_pred_list = [str(value) for value in y_pred]
    report = classification_report(
        y_true_list, y_pred_list, labels=labels, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_true_list, y_pred_list, labels=labels).tolist()
    ece = None
    reliability = []
    if y_proba is not None:
        confidences = np.asarray(y_proba).max(axis=1)
        ece = expected_calibration_error(
            y_true_list, y_proba, y_pred=y_pred_list
        )
        reliability = reliability_diagram_points(
            y_true_list, y_proba, y_pred=y_pred_list
        )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class": {
            label: report[label]
            for label in labels
            if label in report
        },
        "confusion_matrix": matrix,
        "labels": labels,
        "ece": ece,
        "reliability_diagram": reliability,
        "support": {label: int(report[label]["support"]) for label in labels if label in report},
    }


def prepare_supervised_frame(
    frame: pd.DataFrame,
    target: str,
    text_field: str = "subject_description",
    max_rows: int | None = None,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    texts = model_text_series(frame, text_field)
    labels = frame[target]
    valid = labels.notna() & texts.ne("")
    x = texts.loc[valid].reset_index(drop=True)
    y = labels.loc[valid].astype(str).reset_index(drop=True)
    warning = None
    counts = y.value_counts()
    if max_rows and len(x) > max_rows:
        x, _, y, _ = train_test_split(
            x,
            y,
            train_size=max_rows,
            random_state=seed,
            stratify=y if counts.min() >= 2 and y.nunique() > 1 else None,
        )
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)
        counts = y.value_counts()
    if y.nunique() < 2:
        warning = "Target has fewer than two classes."
    elif int(counts.min()) < 2:
        warning = "At least one class has fewer than two examples; stratification is not possible."
    return x, y, {"warning": warning, "class_counts": counts.astype(int).to_dict(), "rows": int(len(x))}


def chronological_split(
    frame: pd.DataFrame,
    target: str,
    text_field: str = "subject_description",
    timestamps_trustworthy: bool = False,
):
    if not timestamps_trustworthy:
        return None, {
            "status": "unavailable",
            "reason": (
                "Timestamp provenance has not been verified. Set "
                "timestamps_trustworthy=true in the dataset provenance sidecar only "
                "after confirming collection semantics."
            ),
        }
    timestamps = pd.to_datetime(frame["First Response Time"], errors="coerce")
    valid_time = timestamps.notna()
    texts = model_text_series(frame, text_field)
    labels = frame[target]
    usable = valid_time & labels.notna() & texts.ne("")
    if int(usable.sum()) < 10:
        return None, {
            "status": "unavailable",
            "reason": "Too few parseable timestamps with labels for a chronological split.",
        }
    ordered = frame.loc[usable].copy()
    ordered["_ts"] = timestamps.loc[usable]
    ordered = ordered.sort_values("_ts", kind="stable")
    if ordered["_ts"].nunique() < 3:
        return None, {
            "status": "unavailable",
            "reason": "At least three distinct trusted timestamps are required for a chronological split.",
        }
    split_at = int(len(ordered) * 0.8)
    train = ordered.iloc[:split_at]
    test = ordered.iloc[split_at:]
    if train[target].nunique() < 2 or test[target].nunique() < 1:
        return None, {
            "status": "unavailable",
            "reason": "Chronological split does not retain enough labeled classes.",
        }
    return {
        "x_train": model_text_series(train, text_field),
        "y_train": train[target].astype(str),
        "x_test": model_text_series(test, text_field),
        "y_test": test[target].astype(str),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "cutoff": str(train["_ts"].max()),
    }, {"status": "ok"}


def _fit_predict(name: str, estimator, x_train, y_train, x_test):
    started = time.perf_counter()
    if name == "embeddings_logreg":
        train_mat, test_mat, backend = _encode_split(x_train, x_test)
        model = LogisticRegression(max_iter=800, random_state=42)
        model.fit(train_mat, y_train)
        train_time = time.perf_counter() - started
        infer_started = time.perf_counter()
        pred = model.predict(test_mat)
        proba = model.predict_proba(test_mat)
        infer_time = time.perf_counter() - infer_started
        return pred, proba, list(model.classes_), train_time, infer_time, backend
    model = clone(estimator) if estimator is not None else None
    model.fit(x_train, y_train)
    train_time = time.perf_counter() - started
    infer_started = time.perf_counter()
    pred = model.predict(x_test)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_test)
    elif hasattr(model, "named_steps"):
        last = list(model.named_steps.values())[-1]
        if hasattr(last, "predict_proba"):
            proba = last.predict_proba(model[:-1].transform(x_test) if hasattr(model[:-1], "transform") else x_test)
    infer_time = time.perf_counter() - infer_started
    classes = list(getattr(model, "classes_", []))
    if not classes and hasattr(model, "named_steps"):
        last = list(model.named_steps.values())[-1]
        classes = list(getattr(last, "classes_", []))
    return pred, proba, classes, train_time, infer_time, name


def evaluate_models(
    frame: pd.DataFrame,
    target: str = "Ticket Type",
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    n_splits: int = N_SPLITS,
    max_rows: int | None = 900,
    text_field: str = "subject_description",
) -> dict[str, Any]:
    x, y, meta = prepare_supervised_frame(frame, target, text_field, max_rows=max_rows)
    warnings = [meta["warning"]] if meta["warning"] else []
    counts = pd.Series(y).value_counts()
    can_stratify = y.nunique() >= 2 and int(counts.min()) >= 2
    if not can_stratify:
        return {
            "status": "insufficient_data",
            "target": target,
            "warnings": warnings or ["Cannot stratify or train with the available class support."],
            "meta": meta,
        }

    groups = exact_text_group_ids(x)
    group_support = pd.DataFrame({"label": y, "group": groups}).groupby("label")[
        "group"
    ].nunique()
    grouped_holdout = (
        stratified_group_holdout_indices(y, groups, HOLDOUT_FRACTION, seeds[0])
        if int(group_support.min()) >= 5
        else None
    )
    if grouped_holdout is not None:
        dev_idx, hold_idx = grouped_holdout
        x_dev, x_hold = x.iloc[dev_idx], x.iloc[hold_idx]
        y_dev, y_hold = y.iloc[dev_idx], y.iloc[hold_idx]
        groups_dev = groups[dev_idx]
    else:
        x_dev, y_dev = x, y
        x_hold = y_hold = None
        groups_dev = groups
        warnings.append(
            "No class-complete exact-text-grouped final holdout could be formed; "
            "CV-only results are reported."
        )

    grouped_support = pd.DataFrame(
        {"label": y_dev, "group": groups_dev}
    ).groupby("label")["group"].nunique()
    min_class_groups = int(grouped_support.min())
    splits = min(n_splits, min_class_groups)
    if splits < 2:
        warnings.append("Not enough class support for stratified k-fold.")
        splits = 0
    zoo = model_zoo()
    cv_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in zoo}
    if splits >= 2:
        for seed in seeds:
            folder = StratifiedGroupKFold(
                n_splits=splits, shuffle=True, random_state=seed
            )
            for fold_id, (train_idx, val_idx) in enumerate(
                folder.split(x_dev, y_dev, groups_dev)
            ):
                x_train, x_val = x_dev.iloc[train_idx], x_dev.iloc[val_idx]
                y_train, y_val = y_dev.iloc[train_idx], y_dev.iloc[val_idx]
                for name, estimator in zoo.items():
                    pred, proba, classes, train_time, infer_time, backend = _fit_predict(
                        name, estimator, x_train, y_train, x_val
                    )
                    scores = _score(y_val, pred, proba, labels=sorted(y_dev.unique()))
                    scores.update(
                        {
                            "seed": seed,
                            "fold": fold_id,
                            "train_seconds": train_time,
                            "infer_seconds": infer_time,
                            "backend": backend,
                            "predictions": [
                                {"true": str(truth), "predicted": str(prediction)}
                                for truth, prediction in zip(y_val, pred)
                            ],
                        }
                    )
                    cv_rows[name].append(scores)
    else:
        warnings.append(
            "Stratified group k-fold skipped because a class has fewer distinct text groups "
            "than the requested number of folds."
        )

    summary = {}
    for name, rows in cv_rows.items():
        if not rows:
            summary[name] = {"status": "not_run"}
            continue
        summary[name] = {
            "accuracy": mean_std_ci([row["accuracy"] for row in rows]),
            "macro_f1": mean_std_ci([row["macro_f1"] for row in rows]),
            "weighted_f1": mean_std_ci([row["weighted_f1"] for row in rows]),
            "balanced_accuracy": mean_std_ci([row["balanced_accuracy"] for row in rows]),
            "ece": mean_std_ci([row["ece"] for row in rows if row["ece"] is not None]),
            "mean_train_seconds": float(np.mean([row["train_seconds"] for row in rows])),
            "mean_infer_seconds": float(np.mean([row["infer_seconds"] for row in rows])),
            "backend": rows[0]["backend"],
            "folds_times_seeds": len(rows),
        }

    holdout_results = {}
    if x_hold is not None:
        for name, estimator in zoo.items():
            pred, proba, classes, train_time, infer_time, backend = _fit_predict(
                name, estimator, x_dev, y_dev, x_hold
            )
            scores = _score(y_hold, pred, proba, labels=sorted(y.unique()))
            scores.update(
                {
                    "train_seconds": train_time,
                    "infer_seconds": infer_time,
                    "backend": backend,
                    "split": "final_holdout",
                    "predictions": [
                        {"true": str(truth), "predicted": str(prediction)}
                        for truth, prediction in zip(y_hold, pred)
                    ],
                }
            )
            holdout_results[name] = scores

    provenance = frame.attrs.get("dataset_provenance", {})
    chrono, chrono_meta = chronological_split(
        frame,
        target,
        text_field,
        timestamps_trustworthy=provenance.get("timestamps_trustworthy") is True,
    )
    chronological = {"status": chrono_meta["status"], "detail": chrono_meta}
    if chrono:
        chronological_models = {}
        for name, estimator in zoo.items():
            pred, proba, classes, train_time, infer_time, backend = _fit_predict(
                name,
                estimator,
                chrono["x_train"],
                chrono["y_train"],
                chrono["x_test"],
            )
            scores = _score(
                chrono["y_test"],
                pred,
                proba,
                labels=sorted(set(chrono["y_train"]) | set(chrono["y_test"])),
            )
            scores.update(
                {
                    "train_seconds": train_time,
                    "infer_seconds": infer_time,
                    "backend": backend,
                    "train_rows": chrono["train_rows"],
                    "test_rows": chrono["test_rows"],
                    "cutoff": chrono["cutoff"],
                    "predictions": [
                        {"true": str(truth), "predicted": str(prediction)}
                        for truth, prediction in zip(chrono["y_test"], pred)
                    ],
                }
            )
            chronological_models[name] = scores
        chronological["models"] = chronological_models

    return {
        "status": "ok",
        "target": target,
        "text_field": text_field,
        "seeds": list(seeds),
        "n_splits": splits,
        "rows_used": meta["rows"],
        "class_counts": meta["class_counts"],
        "distinct_text_groups": int(len(np.unique(groups))),
        "split_grouping": "normalized identical text stays within a single split",
        "model_settings": {
            "majority": {"strategy": "most_frequent"},
            "complement_nb": {"representation": "TF-IDF unigrams", "alpha": "scikit-learn default"},
            "linear_svm": {
                "representation": "TF-IDF unigrams and bigrams",
                "max_iter": 2_000,
                "probability_scores": False,
            },
            "tfidf_logreg": {
                "representation": "TF-IDF unigrams and bigrams",
                "max_iter": 800,
                "random_state": 42,
            },
            "embeddings_logreg": {
                "configured_model": "OPSLYRA_EMBEDDING_MODEL",
                "fallback": "training-only TF-IDF + TruncatedSVD",
                "downloads_model_only_when_configured": True,
            },
        },
        "warnings": warnings,
        "cross_validation": summary,
        "cross_validation_runs": cv_rows,
        "final_holdout": holdout_results or {"status": "not_run"},
        "chronological": chronological,
        "notes": [
            "Cross-validation never uses the final holdout; the final holdout is not used for model selection.",
            "Normalized identical ticket text is kept within one split to reduce exact-duplicate leakage.",
            "Imputation is not applied to labels; unlabeled rows are dropped only for supervised training.",
            "Confidence intervals summarize repeated seed/fold observations and are descriptive; folds are not independent samples.",
            "Sentence embeddings download only when OPSLYRA_EMBEDDING_MODEL is set and sentence-transformers is installed; configure SENTENCE_TRANSFORMERS_HOME for caching.",
        ],
    }


def ablation_study(
    frame: pd.DataFrame,
    target: str = "Ticket Type",
    max_rows: int | None = 500,
    seed: int = 0,
) -> dict[str, Any]:
    configs = [
        {"name": "subject_unigram", "text_field": "subject", "ngram": (1, 1)},
        {"name": "description_unigram", "text_field": "description", "ngram": (1, 1)},
        {"name": "subject_description_unigram", "text_field": "subject_description", "ngram": (1, 1)},
        {"name": "subject_description_bigram", "text_field": "subject_description", "ngram": (1, 2)},
    ]
    feature_texts = {
        field: model_text_series(frame, field).astype(str)
        for field in ("subject", "description", "subject_description")
    }
    labels = frame[target]
    common_valid = (
        labels.notna()
        & feature_texts["subject"].ne("")
        & feature_texts["description"].ne("")
    )
    y = labels.loc[common_valid].astype(str).reset_index(drop=True)
    x_by_field = {
        field: values.loc[common_valid].reset_index(drop=True)
        for field, values in feature_texts.items()
    }
    permitted_metadata = pd.Series(
        [
            f"channel:{channel if pd.notna(channel) else 'unknown'} "
            f"product:{product if pd.notna(product) else 'unknown'}"
            for channel, product in zip(
                frame.loc[common_valid, "Ticket Channel"],
                frame.loc[common_valid, "Product Purchased"],
            )
        ],
        dtype="string",
    )
    if max_rows and len(y) > max_rows:
        sampled_idx, _ = train_test_split(
            np.arange(len(y)),
            train_size=max_rows,
            random_state=seed,
            stratify=y,
        )
        y = y.iloc[sampled_idx].reset_index(drop=True)
        x_by_field = {
            field: values.iloc[sampled_idx].reset_index(drop=True)
            for field, values in x_by_field.items()
        }
        permitted_metadata = permitted_metadata.iloc[sampled_idx].reset_index(drop=True)
    if y.nunique() < 2 or int(y.value_counts().min()) < 3:
        return {
            "target": target,
            "status": "insufficient_data",
            "warning": "Need at least three examples per class with both subject and description for a shared ablation split.",
            "results": [],
        }
    groups = exact_text_group_ids(x_by_field["subject_description"])
    split = stratified_group_holdout_indices(y, groups, 0.25, seed)
    if split is None:
        return {
            "target": target,
            "status": "insufficient_data",
            "warning": "Could not form a class-complete exact-text-grouped ablation split.",
            "results": [],
        }
    train_idx, test_idx = split
    results = []
    for config in configs:
        x = x_by_field[config["text_field"]]
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = Pipeline(
            [
                ("tfidf", _tfidf(config["ngram"], max_features=4000)),
                ("clf", LogisticRegression(max_iter=600, random_state=seed)),
            ]
        )
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        results.append(
            {
                **config,
                "status": "ok",
                "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
                "accuracy": float(accuracy_score(y_test, pred)),
                "rows": int(len(y)),
                "split": "shared stratified exact-text-grouped holdout",
                "exploratory": True,
            }
        )
    # embeddings vs tfidf on combined text
    x = x_by_field["subject_description"]
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    pred, _, _, _, _, backend = _fit_predict(
        "embeddings_logreg", None, x_train, y_train, x_test
    )
    results.append(
        {
            "name": "embeddings_logreg",
            "status": "ok",
            "backend": backend,
            "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y_test, pred)),
            "rows": int(len(y)),
            "split": "shared stratified exact-text-grouped holdout",
            "exploratory": True,
        }
    )
    meta_text = x + "\n" + permitted_metadata
    model = Pipeline(
        [
            ("tfidf", _tfidf((1, 2), max_features=4000)),
            ("clf", LogisticRegression(max_iter=600, random_state=seed)),
        ]
    )
    model.fit(meta_text.iloc[train_idx], y_train)
    pred_meta = model.predict(meta_text.iloc[test_idx])
    results.append(
        {
            "name": "subject_description_permitted_metadata",
            "status": "ok",
            "metadata": ["Ticket Channel", "Product Purchased"],
            "macro_f1": float(
                f1_score(y_test, pred_meta, average="macro", zero_division=0)
            ),
            "accuracy": float(accuracy_score(y_test, pred_meta)),
            "rows": int(len(y)),
            "split": "shared stratified holdout",
            "exploratory": True,
        }
    )
    return {
        "target": target,
        "results": results,
        "note": (
            "All ablations use the same stratified, exact-text-grouped train/test split, "
            "restricted to records with both subject and description. Results are exploratory "
            "single-split diagnostics."
        ),
    }


def calibration_study(
    frame: pd.DataFrame,
    target: str = "Ticket Type",
    max_rows: int | None = 500,
    seed: int = 0,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Fit isotonic/sigmoid calibrators on a validation fold; score ECE on holdout."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    x, y, meta = prepare_supervised_frame(frame, target, max_rows=max_rows, seed=seed)
    if y.nunique() < 2 or int(y.value_counts().min()) < 6:
        return {
            "status": "insufficient_data",
            "warning": meta["warning"] or "Need enough labeled rows for train/val/holdout calibration.",
        }
    groups = exact_text_group_ids(x)
    holdout = stratified_group_holdout_indices(y, groups, 0.25, seed)
    if holdout is None:
        return {
            "status": "insufficient_data",
            "warning": "Could not form a class-complete exact-text-grouped calibration holdout.",
        }
    dev_idx, hold_idx = holdout
    x_dev, x_hold = x.iloc[dev_idx], x.iloc[hold_idx]
    y_dev, y_hold = y.iloc[dev_idx], y.iloc[hold_idx]
    val_split = stratified_group_holdout_indices(
        y_dev,
        groups[dev_idx],
        0.25,
        seed + 1,
    )
    if val_split is None:
        return {
            "status": "insufficient_data",
            "warning": "Could not form a class-complete exact-text-grouped calibration validation split.",
        }
    train_idx, val_idx = val_split
    x_train, x_val = x_dev.iloc[train_idx], x_dev.iloc[val_idx]
    y_train, y_val = y_dev.iloc[train_idx], y_dev.iloc[val_idx]
    calibration_folds = min(5, int(y_val.value_counts().min()))
    if calibration_folds < 2:
        return {
            "status": "insufficient_data",
            "warning": (
                "The grouped calibration validation split needs at least two "
                "examples per class."
            ),
        }
    base = Pipeline(
        [
            ("tfidf", _tfidf((1, 2), max_features=4000)),
            ("clf", LogisticRegression(max_iter=600, random_state=seed)),
        ]
    )
    base.fit(x_train, y_train)
    uncal_proba = base.predict_proba(x_hold)
    uncal_pred = base.predict(x_hold)
    uncal_ece = expected_calibration_error(y_hold, uncal_proba, y_pred=uncal_pred, n_bins=n_bins)
    methods = {}
    for method in ("sigmoid", "isotonic"):
        calibrator = CalibratedClassifierCV(
            FrozenEstimator(base),
            method=method,
            cv=calibration_folds,
        )
        calibrator.fit(x_val, y_val)
        proba = calibrator.predict_proba(x_hold)
        pred = calibrator.predict(x_hold)
        methods[method] = {
            "ece": expected_calibration_error(y_hold, proba, y_pred=pred, n_bins=n_bins),
            "reliability_diagram": reliability_diagram_points(
                y_hold, proba, y_pred=pred, n_bins=n_bins
            ),
            "fitted_on": "validation split only",
            "evaluated_on": "untouched holdout",
        }
    return {
        "status": "ok",
        "target": target,
        "n_bins": n_bins,
        "calibration_folds": calibration_folds,
        "binning": "equal-width on [0, 1]",
        "uncalibrated_ece": uncal_ece,
        "uncalibrated_reliability": reliability_diagram_points(
            y_hold, uncal_proba, y_pred=uncal_pred, n_bins=n_bins
        ),
        "methods": methods,
        "holdout_rows": int(len(x_hold)),
        "note": (
            "Calibrators are fit on validation data only. Exact normalized text is kept "
            "within one split. Results are exploratory."
        ),
    }
