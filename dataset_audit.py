from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline

from data import model_text_series, redact_sensitive_text
from reproducibility import (
    EXCLUDED_PREDICTION_COLUMNS,
    PERMITTED_TEXT_COLUMNS,
    exact_text_group_ids,
)


def _class_table(series: pd.Series) -> list[dict[str, Any]]:
    counts = series.dropna().astype(str).value_counts()
    total = int(counts.sum())
    return [
        {
            "label": str(label),
            "count": int(count),
            "share": (int(count) / total) if total else 0.0,
        }
        for label, count in counts.items()
    ]


def _chi2_independence_proxy(texts: pd.Series, labels: pd.Series) -> dict[str, Any]:
    """Cheap signal check: length-quartile vs label Cramer's V-like association."""
    working = pd.DataFrame({"text": texts.fillna(""), "label": labels})
    working = working[working["label"].notna() & working["text"].ne("")]
    if working.empty or working["label"].nunique() < 2:
        return {"status": "insufficient_data", "association": None}
    lengths = working["text"].str.split().str.len().fillna(0)
    try:
        quartiles = pd.qcut(lengths, q=min(4, lengths.nunique()), duplicates="drop")
    except ValueError:
        return {"status": "insufficient_variation", "association": None}
    table = pd.crosstab(quartiles, working["label"].astype(str))
    if table.size == 0 or table.shape[0] < 2 or table.shape[1] < 2:
        return {"status": "insufficient_variation", "association": None}
    observed = table.to_numpy(dtype=float)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((observed - expected) ** 2 / np.where(expected == 0, np.nan, expected))
    k = min(table.shape) - 1
    n = observed.sum()
    cramer = float(np.sqrt(chi2 / (n * k))) if n and k else None
    return {"status": "ok", "length_label_cramers_v": cramer, "note": "Text-length proxy only."}


def _cv_signal(
    texts: pd.Series,
    labels: pd.Series,
    seed: int = 42,
    max_rows: int | None = 800,
) -> dict[str, Any]:
    valid = labels.notna() & texts.fillna("").ne("")
    x = texts.loc[valid].astype(str)
    y = labels.loc[valid].astype(str)
    counts = y.value_counts()
    if max_rows and len(x) > max_rows and y.nunique() >= 2 and int(counts.min()) >= 2:
        x, _, y, _ = train_test_split(
            x, y, train_size=max_rows, random_state=seed, stratify=y
        )
        counts = y.value_counts()
    groups = exact_text_group_ids(x)
    group_support = pd.DataFrame({"label": y, "group": groups}).groupby("label")[
        "group"
    ].nunique()
    if y.nunique() < 2 or counts.min() < 3 or group_support.min() < 3:
        return {
            "status": "insufficient_class_support",
            "majority_macro_f1": None,
            "model_macro_f1": None,
            "majority_accuracy": None,
            "tfidf_lr_accuracy": None,
            "accuracy_lift": None,
            "fold_accuracy": [],
            "warning": "Need at least three distinct text groups in every class for a 3-fold signal check.",
        }
    splits = min(3, int(group_support.min()))
    fold = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=seed)
    majority_scores = []
    model_scores = []
    balanced = []
    majority_accuracies = []
    model_accuracies = []
    fold_accuracy = []
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 1), min_df=1, max_features=4000)),
            ("clf", LogisticRegression(max_iter=400, random_state=seed)),
        ]
    )
    dummy = DummyClassifier(strategy="most_frequent")
    for train_idx, test_idx in fold.split(x, y, groups):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        dummy.fit(x_train, y_train)
        pipeline.fit(x_train, y_train)
        majority_pred = dummy.predict(x_test)
        model_pred = pipeline.predict(x_test)
        majority_scores.append(float(f1_score(y_test, majority_pred, average="macro", zero_division=0)))
        model_scores.append(float(f1_score(y_test, model_pred, average="macro", zero_division=0)))
        balanced.append(float(balanced_accuracy_score(y_test, model_pred)))
        majority_accuracy = float(accuracy_score(y_test, majority_pred))
        model_accuracy = float(accuracy_score(y_test, model_pred))
        majority_accuracies.append(majority_accuracy)
        model_accuracies.append(model_accuracy)
        fold_accuracy.append(
            {
                "fold": len(fold_accuracy) + 1,
                "test_rows": int(len(y_test)),
                "majority_accuracy": majority_accuracy,
                "tfidf_lr_accuracy": model_accuracy,
            }
        )
    majority_mean = float(np.mean(majority_scores))
    model_mean = float(np.mean(model_scores))
    majority_accuracy = float(np.mean(majority_accuracies))
    model_accuracy = float(np.mean(model_accuracies))
    macro_f1_lift = model_mean - majority_mean
    accuracy_baseline_met = model_accuracy >= majority_accuracy
    suitable = macro_f1_lift >= 0.05 and accuracy_baseline_met
    return {
        "status": "ok",
        "folds": splits,
        "labeled_rows": int(len(y)),
        "distinct_text_groups": int(len(np.unique(groups))),
        "split_grouping": "normalized identical text stays within a single fold",
        "majority_macro_f1": majority_mean,
        "tfidf_lr_macro_f1": model_mean,
        "majority_accuracy": majority_accuracy,
        "tfidf_lr_accuracy": model_accuracy,
        "accuracy_lift": model_accuracy - majority_accuracy,
        "fold_accuracy": fold_accuracy,
        "tfidf_lr_balanced_accuracy": float(np.mean(balanced)),
        "lift_over_majority": macro_f1_lift,
        "accuracy_baseline_met": accuracy_baseline_met,
        "appears_to_have_text_signal": suitable,
        "note": (
            "The text-signal heuristic requires at least a 5-point macro-F1 lift and accuracy "
            "no lower than the majority baseline. This is diagnostic, not proof of usefulness."
        ),
    }


def duplicate_text_analysis(frame: pd.DataFrame) -> dict[str, Any]:
    texts = model_text_series(frame)
    normalized = texts.str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()
    nonempty = normalized.ne("")
    exact_dupes = int(normalized.loc[nonempty].duplicated().sum())
    groups = normalized.loc[nonempty].groupby(normalized.loc[nonempty]).size()
    targets = ("Ticket Type", "Ticket Priority", "Ticket Status")
    conflicting_by_target = {target: 0 for target in targets}
    conflicting_groups = 0
    conflict_examples = []
    for text, count in groups.items():
        if count < 2:
            continue
        subset = frame.loc[normalized == text, list(targets)]
        label_conflicts = {
            target: subset[target].dropna().astype(str).nunique() > 1
            for target in targets
        }
        for target, has_conflict in label_conflicts.items():
            if has_conflict:
                conflicting_by_target[target] += 1
        if any(label_conflicts.values()):
            conflicting_groups += 1
            if len(conflict_examples) < 5:
                sample = frame.loc[normalized == text].iloc[0]
                conflict_examples.append(
                    {
                        "subject": redact_sensitive_text(sample["Ticket Subject"])[:160],
                        "type_labels": sorted(subset["Ticket Type"].dropna().astype(str).unique().tolist()),
                        "priority_labels": sorted(
                            subset["Ticket Priority"].dropna().astype(str).unique().tolist()
                        ),
                        "status_labels": sorted(
                            subset["Ticket Status"].dropna().astype(str).unique().tolist()
                        ),
                    }
                )
    near = _near_duplicates(normalized, frame)
    return {
        "exact_duplicate_rows": exact_dupes,
        "distinct_nonempty_text_groups": int(len(groups)),
        "conflicting_exact_text_groups": conflicting_groups,
        "conflicting_exact_text_groups_by_target": conflicting_by_target,
        "conflict_examples": conflict_examples,
        "near_duplicates": near,
    }


def _near_duplicates(normalized: pd.Series, frame: pd.DataFrame, limit: int = 400) -> dict[str, Any]:
    nonempty = normalized[normalized.ne("")]
    representatives = nonempty[~nonempty.duplicated()]
    if len(representatives) < 4:
        return {
            "status": "insufficient_data",
            "sample_size": int(len(representatives)),
            "available_distinct_texts": int(len(representatives)),
            "similarity_threshold": 0.92,
            "pairs_above_threshold": 0,
            "label_conflicts_by_target": {},
            "examples": [],
            "note": "Near-duplicate scan needs at least four distinct nonempty texts.",
        }
    sample = representatives
    if len(sample) > limit:
        sample = sample.sample(limit, random_state=42)
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 1), max_features=3000)
    matrix = vectorizer.fit_transform(sample)
    similar_pairs = 0
    targets = ("Ticket Type", "Ticket Priority", "Ticket Status")
    label_conflicts = {target: 0 for target in targets}
    examples = []
    similarities = (matrix @ matrix.T).toarray()
    for i in range(similarities.shape[0]):
        for j in range(i + 1, similarities.shape[1]):
            similarity = float(similarities[i, j])
            if similarity >= 0.92:
                similar_pairs += 1
                left_text = sample.iloc[i]
                right_text = sample.iloc[j]
                left_rows = frame.loc[normalized == left_text]
                right_rows = frame.loc[normalized == right_text]
                for target in targets:
                    left_labels = set(left_rows[target].dropna().astype(str))
                    right_labels = set(right_rows[target].dropna().astype(str))
                    if left_labels and right_labels and left_labels.isdisjoint(right_labels):
                        label_conflicts[target] += 1
                if len(examples) < 5:
                    examples.append({"similarity": similarity})
    return {
        "status": "sampled",
        "sample_size": int(len(sample)),
        "available_distinct_texts": int(len(representatives)),
        "similarity_threshold": 0.92,
        "pairs_above_threshold": similar_pairs,
        "high_similarity_pairs_in_block": similar_pairs,
        "label_conflicts_by_target": label_conflicts,
        "examples": examples,
        "note": (
            "TF-IDF word-unigram cosine similarity was computed for every pair in a deterministic "
            "sample of distinct nonempty texts; exact normalized matches are excluded."
        ),
    }


def accuracy_comparison_report(
    frame: pd.DataFrame,
    targets: tuple[str, ...] = ("Ticket Priority", "Ticket Status"),
) -> dict[str, Any]:
    """Compare text-model accuracy with a fold-local majority baseline."""
    texts = model_text_series(frame)
    return {
        target: _cv_signal(texts, frame[target], max_rows=None)
        for target in targets
    }


def leakage_review(frame: pd.DataFrame) -> dict[str, Any]:
    present_excluded = [column for column in EXCLUDED_PREDICTION_COLUMNS if column in frame.columns]
    return {
        "excluded_from_text_prediction": present_excluded,
        "permitted_text_columns": list(PERMITTED_TEXT_COLUMNS),
        "warning": (
            "Resolution, satisfaction, time-to-resolution, customer identity, and ticket ID "
            "must not be used as predictive features. Status and first-response time are also "
            "excluded because they can encode post-intake or operational outcome information."
        ),
    }


def audit_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    texts = model_text_series(frame)
    type_signal = _cv_signal(texts, frame["Ticket Type"])
    priority_signal = _cv_signal(texts, frame["Ticket Priority"])
    status_signal = _cv_signal(texts, frame["Ticket Status"])
    provenance = dict(frame.attrs.get("dataset_provenance", {}))
    labels_verified = provenance.get("labels_verified") is True
    suitability = {
        "Ticket Type": _suitability(type_signal, "ticket type", labels_verified),
        "Ticket Priority": _suitability(priority_signal, "priority", labels_verified),
        "Ticket Status": {
            **_suitability(status_signal, "status", labels_verified),
            "research_use": "not recommended as a text-prediction target in this prototype",
            "reason": "Status often reflects workflow progress rather than intake text.",
        },
    }
    return {
        "dataset_label": frame.attrs.get("dataset_label", "External dataset"),
        "source_kind": provenance.get("source_kind", "external"),
        "provenance": provenance,
        "row_count": int(len(frame)),
        "class_distributions": {
            "Ticket Type": _class_table(frame["Ticket Type"]),
            "Ticket Priority": _class_table(frame["Ticket Priority"]),
            "Ticket Status": _class_table(frame["Ticket Status"]),
        },
        "missing_labels": {
            "Ticket Type": int(frame["Ticket Type"].isna().sum()),
            "Ticket Priority": int(frame["Ticket Priority"].isna().sum()),
            "Ticket Status": int(frame["Ticket Status"].isna().sum()),
        },
        "text_length_association": {
            "Ticket Type": _chi2_independence_proxy(texts, frame["Ticket Type"]),
            "Ticket Priority": _chi2_independence_proxy(texts, frame["Ticket Priority"]),
        },
        "text_signal": {
            "Ticket Type": type_signal,
            "Ticket Priority": priority_signal,
            "Ticket Status": status_signal,
        },
        "duplicates": duplicate_text_analysis(frame),
        "leakage": leakage_review(frame),
        "target_suitability": suitability,
        "limitations": [
            "The exact Kaggle listing, license, and label-validation evidence are not recorded; do not make unsupported source, licensing, or operational-validity claims.",
            "Priority and status can be weakly related to intake text or assigned after outcome.",
            "Held-out known classes are only a proxy for genuinely unseen incident types.",
        ],
    }


def _suitability(
    signal: dict[str, Any],
    name: str,
    labels_verified: bool = False,
) -> dict[str, Any]:
    if signal.get("status") != "ok":
        return {
            "target": name,
            "text_signal_heuristic_passed": False,
            "labels_verified": labels_verified,
            "suitable_for_predictive_research": False,
            "reason": signal.get("warning") or signal.get("status"),
        }
    text_signal = bool(signal.get("appears_to_have_text_signal"))
    suitable = text_signal and labels_verified
    return {
        "target": name,
        "text_signal_heuristic_passed": text_signal,
        "labels_verified": labels_verified,
        "suitable_for_predictive_research": suitable,
        "reason": (
            "The text-signal heuristic passed and dataset provenance marks labels as verified."
            if suitable
            else (
                "The text-signal heuristic passed, but labels are not independently verified; "
                "treat this as exploratory only."
                if text_signal
                else "Text model did not clearly beat the majority-class baseline; labels may be weak, noisy, or unrelated to text."
            )
        ),
        "accuracy_baseline_met": signal.get("accuracy_baseline_met"),
        "majority_accuracy": signal.get("majority_accuracy"),
        "model_accuracy": signal.get("tfidf_lr_accuracy"),
        "majority_macro_f1": signal.get("majority_macro_f1"),
        "model_macro_f1": signal.get("tfidf_lr_macro_f1"),
    }
