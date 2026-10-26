from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.cluster import KMeans
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, f1_score, recall_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD

from data import redact_sensitive_text, ticket_text


ERROR_PATTERN = re.compile(
    r"\b(?:[A-Z]{2,}[-_ ]?\d{2,}|(?:0x)?[A-F0-9]{6,})\b", re.IGNORECASE
)
IMPACT_TERMS = (
    "outage",
    "production",
    "all users",
    "everyone",
    "data loss",
    "security",
    "cannot access",
    "not working",
)
TEAM_SKILLS = {
    "Technical Support": {"Technical issue", "Product inquiry"},
    "Billing & Payments": {"Billing inquiry", "Refund request"},
    "Retention": {"Cancellation request"},
}


def _text_model() -> tuple[TfidfVectorizer, LogisticRegression]:
    return (
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_features=15_000,
            strip_accents="unicode",
        ),
        LogisticRegression(max_iter=500, random_state=42),
    )


def _cost_minutes(texts: pd.Series) -> np.ndarray:
    words = texts.str.split().str.len().fillna(0).to_numpy(dtype=float)
    return np.clip(0.5 + words / 120, 0.5, 3.0)


def _impact_scores(texts: pd.Series) -> np.ndarray:
    impact = np.zeros(len(texts), dtype=float)
    for index, text in enumerate(texts.str.casefold()):
        impact[index] = min(1.0, sum(term in text for term in IMPACT_TERMS) / 3)
    return impact


def _sparse_unique_count(matrix) -> int:
    return len(
        {
            (
                tuple(matrix.getrow(index).indices),
                tuple(np.round(matrix.getrow(index).data, 6)),
            )
            for index in range(matrix.shape[0])
        }
    )


def _novelty_to_reference(query_matrix, reference_matrix) -> np.ndarray:
    if reference_matrix.shape[0] == 0:
        return np.ones(query_matrix.shape[0], dtype=float)
    neighbors = NearestNeighbors(
        n_neighbors=1,
        metric="cosine",
        algorithm="brute",
        n_jobs=1,
    ).fit(reference_matrix)
    distances, _ = neighbors.kneighbors(query_matrix)
    return np.clip(distances[:, 0], 0.0, 1.0)


def _select_batch(
    policy: str,
    pool_indexes: np.ndarray,
    probabilities: np.ndarray,
    matrix,
    labeled_indexes: np.ndarray,
    impacts: np.ndarray,
    costs: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    count = min(batch_size, len(pool_indexes))
    if policy == "random":
        return rng.choice(pool_indexes, size=count, replace=False)

    uncertainty = 1 - probabilities.max(axis=1)
    if policy == "uncertainty":
        return pool_indexes[np.argsort(uncertainty)[-count:][::-1]]

    initial_novelty = _novelty_to_reference(
        matrix[pool_indexes], matrix[labeled_indexes]
    )

    selected: list[int] = []
    remaining_positions = np.arange(len(pool_indexes))
    novelty = initial_novelty
    if policy == "diversity":
        for _ in range(count):
            best_position = int(novelty.argmax())
            selected.append(int(pool_indexes[remaining_positions[best_position]]))
            remaining = pool_indexes[remaining_positions]
            new_similarity = np.asarray(
                (matrix[remaining] @ matrix[selected[-1]].T).toarray()
            ).ravel()
            novelty = np.minimum(novelty, np.clip(1 - new_similarity, 0, 1))
            keep = np.ones(len(remaining_positions), dtype=bool)
            keep[best_position] = False
            remaining_positions = remaining_positions[keep]
            novelty = novelty[keep]
            if not len(remaining_positions):
                break
        return np.asarray(selected, dtype=int)
    for _ in range(count):
        remaining = pool_indexes[remaining_positions]
        raw = (
            0.35 * uncertainty[remaining_positions]
            + 0.30 * novelty
            + 0.25 * impacts[remaining]
        ) / costs[remaining]
        raw = raw / max(float(raw.max()), 1e-9)
        if selected:
            similarity = matrix[remaining] @ matrix[np.asarray(selected)].T
            max_similarity = np.asarray(similarity.max(axis=1).toarray()).ravel()
            diversity = np.clip(1 - max_similarity, 0, 1)
        else:
            diversity = novelty
        score = 0.8 * raw + 0.2 * diversity
        best_position = int(score.argmax())
        selected.append(int(remaining[best_position]))
        selected_text = matrix[selected[-1]]
        new_similarity = np.asarray(
            (matrix[remaining] @ selected_text.T).toarray()
        ).ravel()
        novelty = np.minimum(novelty, np.clip(1 - new_similarity, 0, 1))
        keep = np.ones(len(remaining_positions), dtype=bool)
        keep[best_position] = False
        remaining_positions = remaining_positions[keep]
        novelty = novelty[keep]
        if not len(remaining_positions):
            break
    return np.asarray(selected, dtype=int)


def active_learning_benchmark(
    frame: pd.DataFrame,
    budget: int = 60,
    seed: int = 42,
    max_rows: int = 1600,
) -> dict[str, Any]:
    target = "Ticket Type"
    texts = ticket_text(frame)
    valid = frame[target].notna() & texts.ne("")
    working = frame.loc[valid].copy()
    if len(working) > max_rows:
        working, _ = train_test_split(
            working,
            train_size=max_rows,
            random_state=seed,
            stratify=working[target],
        )
    labels = working[target].astype(str).to_numpy()
    if len(np.unique(labels)) < 2 or min(pd.Series(labels).value_counts()) < 4:
        raise ValueError("Active-learning comparison needs at least four examples per class.")

    all_indexes = np.arange(len(working))
    pool_indexes, test_indexes = train_test_split(
        all_indexes,
        test_size=0.25,
        random_state=seed,
        stratify=labels,
    )
    local_pool_indexes = np.arange(len(pool_indexes))
    initial_indexes, unlabeled_indexes = train_test_split(
        local_pool_indexes,
        train_size=0.15,
        random_state=seed + 1,
        stratify=labels[pool_indexes],
    )
    maximum_budget = min(max(1, int(budget)), len(unlabeled_indexes))
    checkpoint_sizes = sorted(
        set(
            int(value)
            for value in np.linspace(
                min(1, maximum_budget),
                maximum_budget,
                num=min(4, maximum_budget),
            )
        )
    )
    texts = texts.loc[working.index].reset_index(drop=True)
    vectorizer, _ = _text_model()
    pool_texts = texts.iloc[pool_indexes]
    matrix = vectorizer.fit_transform(pool_texts)
    test_matrix = vectorizer.transform(texts.iloc[test_indexes])
    pool_labels = labels[pool_indexes]
    impacts = _impact_scores(pool_texts.reset_index(drop=True))
    costs = _cost_minutes(pool_texts.reset_index(drop=True))
    class_counts = pd.Series(pool_labels).value_counts()
    rare_label = str(class_counts.idxmin())
    strategies = ("random", "uncertainty", "diversity", "proposed")
    curves: dict[str, list[dict[str, Any]]] = {strategy: [] for strategy in strategies}
    rng = np.random.default_rng(seed)

    for strategy in strategies:
        labeled = initial_indexes.copy()
        unlabeled = unlabeled_indexes.copy()
        acquired = 0
        for checkpoint in checkpoint_sizes:
            requested = checkpoint - acquired
            while requested > 0 and len(unlabeled):
                estimator = LogisticRegression(max_iter=500, random_state=seed)
                estimator.fit(matrix[labeled], pool_labels[labeled])
                probabilities = estimator.predict_proba(matrix[unlabeled])
                selected = _select_batch(
                    strategy,
                    unlabeled,
                    probabilities,
                    matrix,
                    labeled,
                    impacts,
                    costs,
                    requested,
                    rng,
                )
                if not len(selected):
                    break
                labeled = np.concatenate([labeled, selected])
                selected_set = set(selected.tolist())
                unlabeled = np.asarray(
                    [index for index in unlabeled if index not in selected_set],
                    dtype=int,
                )
                acquired += len(selected)
                requested -= len(selected)
            estimator = LogisticRegression(max_iter=500, random_state=seed)
            estimator.fit(matrix[labeled], pool_labels[labeled])
            predictions = estimator.predict(test_matrix)
            test_labels = labels[test_indexes]
            curves[strategy].append(
                {
                    "budget": acquired,
                    "annotation_minutes": float(costs[labeled[len(initial_indexes):]].sum()),
                    "macro_f1": float(
                        f1_score(test_labels, predictions, average="macro", zero_division=0)
                    ),
                    "rare_class": rare_label,
                    "rare_class_recall": float(
                        recall_score(
                            test_labels,
                            predictions,
                            labels=[rare_label],
                            average=None,
                            zero_division=0,
                        )[0]
                    ),
                    "test_rows": len(test_indexes),
                    "initial_labels": len(initial_indexes),
                }
            )
            if acquired >= maximum_budget:
                break

    return {
        "seed": seed,
        "rows_used": len(working),
        "test_rows": len(test_indexes),
        "initial_labels": len(initial_indexes),
        "requested_budget": maximum_budget,
        "rare_class": rare_label,
        "policies": {
            "random": "Random sampling",
            "uncertainty": "Uncertainty only",
            "diversity": "Diversity only",
            "proposed": "Uncertainty + novelty + impact + diversity, cost-adjusted",
        },
        "curves": curves,
        "metric_warning": (
            "Offline oracle simulation on existing labels; identical fixed holdout, "
            "seed, and annotation counts. It does not measure actual reviewer quality."
        ),
    }


def active_learning_queue(
    frame: pd.DataFrame,
    estimators: dict[str, Any],
    weights: dict[str, float] | None = None,
    limit: int = 12,
) -> list[dict[str, Any]]:
    estimator = estimators.get("Ticket Type")
    if estimator is None:
        return []
    weights = weights or {
        "uncertainty": 0.35,
        "novelty": 0.30,
        "impact": 0.25,
        "diversity": 0.10,
    }
    texts = ticket_text(frame).fillna("")
    valid_positions = np.flatnonzero(texts.ne("").to_numpy())
    if not len(valid_positions):
        return []
    if len(valid_positions) > 1800:
        valid_positions = np.random.default_rng(42).choice(
            valid_positions, size=1800, replace=False
        )
    matrix = estimator.named_steps["tfidf"].transform(texts.iloc[valid_positions])
    predictions = estimator.predict_proba(texts.iloc[valid_positions])
    uncertainty = 1 - predictions.max(axis=1)
    neighbors = NearestNeighbors(
        n_neighbors=min(2, len(valid_positions)),
        metric="cosine",
        algorithm="brute",
        n_jobs=1,
    ).fit(matrix)
    distances, _ = neighbors.kneighbors(matrix)
    novelty = distances[:, -1] if len(valid_positions) > 1 else np.ones(1)
    impact = _impact_scores(texts.iloc[valid_positions].reset_index(drop=True))
    costs = _cost_minutes(texts.iloc[valid_positions].reset_index(drop=True))
    normalized = [uncertainty, novelty, impact]
    normalized = [
        values / max(float(values.max()), 1e-9)
        for values in normalized
    ]
    base = (
        weights["uncertainty"] * normalized[0]
        + weights["novelty"] * normalized[1]
        + weights["impact"] * normalized[2]
    ) / costs
    selected: list[int] = []
    available = np.ones(len(valid_positions), dtype=bool)
    for _ in range(min(limit, len(valid_positions))):
        score = base.copy()
        if selected:
            similarity = matrix @ matrix[np.asarray(selected)].T
            max_similarity = np.asarray(similarity.max(axis=1).toarray()).ravel()
            diversity = np.clip(1 - max_similarity, 0, 1)
            score = score + weights["diversity"] * diversity
        score[~available] = -1
        best = int(score.argmax())
        selected.append(best)
        available[best] = False

    output = []
    records = frame.iloc[valid_positions]
    for index in selected:
        row = records.iloc[index]
        output.append(
            {
                "ticket_id": str(row["Ticket ID"]),
                "subject": redact_sensitive_text(row["Ticket Subject"]),
                "predicted_type": str(estimator.classes_[predictions[index].argmax()]),
                "uncertainty": float(uncertainty[index]),
                "novelty": float(novelty[index]),
                "impact": float(impact[index]),
                "estimated_annotation_minutes": float(costs[index]),
                "score": float(base[index]),
                "reasons": [
                    reason
                    for reason, present in (
                        ("uncertain prediction", uncertainty[index] >= 0.45),
                        ("unusual text", novelty[index] >= 0.65),
                        ("high-impact wording", impact[index] > 0),
                    )
                    if present
                ],
            }
        )
    return output


def constrained_routes(
    ticket_type: str,
    priority: str,
    workload: dict[str, int],
    capacity: dict[str, int],
    remaining_sla_hours: float,
) -> dict[str, Any]:
    eligible = [
        team for team, skills in TEAM_SKILLS.items() if ticket_type in skills
    ]
    valid = [
        team
        for team in eligible
        if workload.get(team, 0) < capacity.get(team, 0)
    ]
    route = (
        min(valid, key=lambda team: workload.get(team, 0) / max(capacity.get(team, 1), 1))
        if valid
        else None
    )
    baseline = eligible[0] if eligible else None
    priority_hours = {"low": 72, "medium": 24, "high": 8, "critical": 4}.get(
        priority.casefold(), 24
    )
    projected_hours = (
        1 + 4 * workload.get(route, 0) / max(capacity.get(route, 1), 1)
        if route
        else float("inf")
    )
    effective_sla_budget = min(remaining_sla_hours, float(priority_hours))
    constraint_valid = bool(
        route
        and route in eligible
        and workload.get(route, 0) < capacity.get(route, 0)
        and projected_hours <= effective_sla_budget
    )
    return {
        "eligible_teams": eligible,
        "baseline_team": baseline,
        "constrained_team": route,
        "capacity_feasible": bool(route),
        "projected_hours": projected_hours if np.isfinite(projected_hours) else None,
        "sla_budget_hours": effective_sla_budget,
        "sla_feasible": bool(projected_hours <= effective_sla_budget),
        "constraint_valid": constraint_valid,
        "priority_sla_hours_assumption": priority_hours,
        "effective_sla_budget_hours": effective_sla_budget,
    }


def counterfactual_routing_study(
    estimators: dict[str, Any],
    subject: str,
    description: str,
    workload: dict[str, int],
    capacity: dict[str, int],
    remaining_sla_hours: float,
) -> dict[str, Any]:
    from intelligence import counterfactual_comparison

    scenarios = counterfactual_comparison(estimators, subject, description)
    results = []
    for scenario in scenarios:
        predicted_type = scenario["predictions"].get("Ticket Type", {}).get("label", "")
        predicted_priority = scenario["predictions"].get(
            "Ticket Priority", {}
        ).get("label", "Medium")
        route = constrained_routes(
            predicted_type,
            predicted_priority,
            workload,
            capacity,
            remaining_sla_hours,
        )
        results.append({**scenario, "routing": route})
    original_team = results[0]["routing"]["constrained_team"]
    stable_runs = 0
    for variant in range(3):
        suffix = (
            "Production outage impacts all customers."
            if variant == 0
            else "A production-wide service outage is impacting every customer."
            if variant == 1
            else "All customers are unable to use the production service."
        )
        alternate = counterfactual_comparison(
            estimators, subject, f"{description}\n{suffix}"
        )[0]
        alternate_type = alternate["predictions"].get("Ticket Type", {}).get(
            "label", ""
        )
        alternate_priority = alternate["predictions"].get(
            "Ticket Priority", {}
        ).get("label", "Medium")
        alternate_route = constrained_routes(
            alternate_type,
            alternate_priority,
            workload,
            capacity,
            remaining_sla_hours,
        )
        stable_runs += alternate_route["constrained_team"] == original_team
    changed = any(
        item["routing"]["constrained_team"] != original_team
        for item in results[1:]
    )
    return {
        "scenarios": results,
        "explanation_stability": stable_runs / 3,
        "counterfactual_route_change": changed,
        "validity_rate": float(
            np.mean([item["routing"]["constraint_valid"] for item in results])
        ),
        "simulated_sla_feasibility": float(
            np.mean([item["routing"]["sla_feasible"] for item in results])
        ),
        "routing_accuracy": None,
        "accuracy_note": "Unavailable: the dataset has no assigned-team ground-truth labels.",
        "constraint_note": (
            "Team capabilities, capacity, SLA budget, and processing-time formula are "
            "configurable simulation assumptions; no assignment is executed."
        ),
    }


def _fingerprint_fields(frame: pd.DataFrame) -> tuple[list[dict[str, float]], pd.Series]:
    fields = []
    error_codes = []
    for _, row in frame.iterrows():
        subject = "" if pd.isna(row["Ticket Subject"]) else str(row["Ticket Subject"])
        description = (
            "" if pd.isna(row["Ticket Description"]) else str(row["Ticket Description"])
        )
        codes = sorted(set(ERROR_PATTERN.findall(f"{subject}\n{description}")))
        product = (
            "unknown"
            if pd.isna(row["Product Purchased"])
            else str(row["Product Purchased"])
        )
        feature = {f"product:{product.casefold()}": 1.0}
        feature.update({f"error:{code.casefold()}": 1.0 for code in codes})
        fields.append(feature)
        error_codes.append(codes)
    return fields, pd.Series(error_codes, index=frame.index)


def _time_aware_fingerprint_matrices(
    training_rows: pd.DataFrame,
    heldout_rows: pd.DataFrame,
):
    train_text = ticket_text(training_rows).fillna("")
    heldout_text = ticket_text(heldout_rows).fillna("")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=10_000,
        sublinear_tf=True,
    )
    train_lexical = vectorizer.fit_transform(train_text)
    heldout_lexical = vectorizer.transform(heldout_text)
    components = min(64, train_lexical.shape[0] - 1, train_lexical.shape[1] - 1)
    if components < 1:
        return None, None
    latent = TruncatedSVD(n_components=components, n_iter=5, random_state=42)
    train_text_vectors = normalize(csr_matrix(latent.fit_transform(train_lexical)))
    heldout_text_vectors = normalize(csr_matrix(latent.transform(heldout_lexical)))

    train_fields, _ = _fingerprint_fields(training_rows)
    heldout_fields, _ = _fingerprint_fields(heldout_rows)
    field_vectorizer = DictVectorizer(sparse=True)
    train_fields_matrix = field_vectorizer.fit_transform(train_fields)
    heldout_fields_matrix = field_vectorizer.transform(heldout_fields)

    train_times = pd.to_datetime(training_rows["First Response Time"], errors="coerce")
    heldout_times = pd.to_datetime(heldout_rows["First Response Time"], errors="coerce")
    origin = train_times.min()
    span = max(
        float((train_times.max() - origin).total_seconds() / 86400),
        1.0,
    )

    def temporal_matrix(times: pd.Series):
        hours = times.dt.hour.to_numpy(dtype=float) + times.dt.minute.to_numpy(
            dtype=float
        ) / 60
        angle = 2 * np.pi * hours / 24
        days_from_train_start = (times - origin).dt.total_seconds().to_numpy(
            dtype=float
        ) / 86400
        return csr_matrix(
            np.column_stack(
                [
                    np.sin(angle),
                    np.cos(angle),
                    days_from_train_start / span,
                ]
            )
        )

    train_features = normalize(
        hstack(
            [
                train_text_vectors * 0.75,
                train_fields_matrix * 0.20,
                temporal_matrix(train_times) * 0.05,
            ]
        ).tocsr()
    )
    heldout_features = normalize(
        hstack(
            [
                heldout_text_vectors * 0.75,
                heldout_fields_matrix * 0.20,
                temporal_matrix(heldout_times) * 0.05,
            ]
        ).tocsr()
    )
    return train_features, heldout_features


def temporal_fingerprint_study(
    frame: pd.DataFrame,
    max_rows: int = 1500,
    seed: int = 42,
) -> dict[str, Any]:
    timestamps = pd.to_datetime(frame["First Response Time"], errors="coerce")
    valid = timestamps.notna()
    working = frame.loc[valid].copy()
    working["_ticket_time"] = timestamps.loc[valid]
    if len(working) > max_rows:
        working = working.sample(max_rows, random_state=seed)
        working = working.sort_values("_ticket_time")
    else:
        working = working.sort_values("_ticket_time")
    if len(working) < 8:
        return {
            "status": "insufficient_data",
            "dated_tickets": len(working),
            "days_observed": int(working["_ticket_time"].dt.floor("D").nunique()),
            "fingerprints": [],
            "comparison": [],
            "forecast": None,
            "message": "Need at least eight timestamped tickets for an exploratory fingerprint.",
        }

    plain_text = ticket_text(working).fillna("")
    plain_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=1, max_features=10_000, sublinear_tf=True
    )
    plain_matrix = normalize(plain_vectorizer.fit_transform(plain_text))
    components = min(64, plain_matrix.shape[0] - 1, plain_matrix.shape[1] - 1)
    if components < 1:
        return {
            "status": "insufficient_features",
            "dated_tickets": len(working),
            "days_observed": int(working["_ticket_time"].dt.floor("D").nunique()),
            "fingerprints": [],
            "comparison": [],
            "forecast": None,
            "message": "Ticket text has too few distinct features for latent fingerprints.",
        }
    latent_text = TruncatedSVD(
        n_components=components,
        n_iter=5,
        random_state=seed,
    ).fit_transform(plain_matrix)
    latent_matrix = normalize(csr_matrix(latent_text))
    fingerprint_fields, error_codes = _fingerprint_fields(working)
    product_error_matrix = DictVectorizer(sparse=True).fit_transform(fingerprint_fields)
    time_values = working["_ticket_time"]
    hours = (
        time_values.dt.hour.to_numpy(dtype=float)
        + time_values.dt.minute.to_numpy(dtype=float) / 60
    )
    hour_angle = 2 * np.pi * hours / 24
    day_offsets = (
        (time_values - time_values.min()).dt.total_seconds().to_numpy(dtype=float)
        / 86400
    )
    time_features = np.column_stack(
        [
            np.sin(hour_angle),
            np.cos(hour_angle),
            day_offsets / max(float(day_offsets.max()), 1.0),
        ]
    )
    time_matrix = csr_matrix(time_features)
    augmented_matrix = normalize(
        hstack(
            [
                latent_matrix * 0.75,
                product_error_matrix * 0.20,
                time_matrix * 0.05,
            ]
        ).tocsr()
    )
    distinct_rows = min(
        _sparse_unique_count(plain_matrix),
        _sparse_unique_count(augmented_matrix),
    )
    if distinct_rows < 2:
        return {
            "status": "insufficient_distinct_fingerprints",
            "dated_tickets": len(working),
            "days_observed": int(working["_ticket_time"].dt.floor("D").nunique()),
            "fingerprints": [],
            "comparison": [],
            "forecast": None,
            "message": "The observed tickets do not contain enough distinct fingerprints to cluster.",
        }
    count = min(6, max(2, int(np.sqrt(len(working) / 2))), distinct_rows)
    count = min(count, len(working) - 1)
    plain_labels = KMeans(n_clusters=count, random_state=seed, n_init=5).fit_predict(
        plain_matrix
    )
    augmented_labels = KMeans(
        n_clusters=count, random_state=seed, n_init=5
    ).fit_predict(augmented_matrix)
    def bootstrap_stability(matrix, random_state: int) -> float | None:
        rng = np.random.default_rng(random_state)
        sample_size = max(4, int(len(working) * 0.8))
        first = rng.choice(len(working), size=sample_size, replace=False)
        second = rng.choice(len(working), size=sample_size, replace=False)
        common = np.intersect1d(first, second)
        first_matrix = matrix[first]
        second_matrix = matrix[second]
        first_clusters = min(count, _sparse_unique_count(first_matrix))
        second_clusters = min(count, _sparse_unique_count(second_matrix))
        if len(common) < 4 or first_clusters < 2 or second_clusters < 2:
            return None
        first_model = KMeans(
            n_clusters=first_clusters,
            random_state=random_state,
            n_init=5,
        ).fit(first_matrix)
        second_model = KMeans(
            n_clusters=second_clusters,
            random_state=random_state + 1,
            n_init=5,
        ).fit(second_matrix)
        return float(
            adjusted_rand_score(
                first_model.predict(matrix[common]),
                second_model.predict(matrix[common]),
            )
        )

    plain_stability = bootstrap_stability(plain_matrix, seed)
    combined_stability = bootstrap_stability(augmented_matrix, seed)

    def safe_silhouette(matrix, labels):
        if len(set(labels)) < 2 or len(set(labels)) >= len(labels):
            return None
        return float(silhouette_score(matrix, labels, metric="cosine"))

    comparison = [
        {
            "name": "TF-IDF text baseline",
            "silhouette": safe_silhouette(plain_matrix, plain_labels),
            "bootstrap_stability_ari": plain_stability,
        },
        {
            "name": "Temporal incident fingerprint",
            "silhouette": safe_silhouette(augmented_matrix, augmented_labels),
            "bootstrap_stability_ari": combined_stability,
        },
    ]
    daily = working.groupby(working["_ticket_time"].dt.floor("D")).size().sort_index()
    calendar_days = pd.date_range(
        daily.index.min(), daily.index.max(), freq="D"
    )
    daily = daily.reindex(calendar_days, fill_value=0)
    days = [
        {"date": date.strftime("%Y-%m-%d"), "count": int(number)}
        for date, number in daily.items()
    ]
    forecast = None
    recurrence_forecasts = []
    if len(days) >= 4:
        observed = [point["count"] for point in days]
        train, actual = observed[:-1], observed[-1]
        persistence = train[-1]
        expanding_mean = float(np.mean(train))
        forecast = {
            "held_out_date": days[-1]["date"],
            "actual": actual,
            "persistence_forecast": int(persistence),
            "persistence_absolute_error": float(abs(actual - persistence)),
            "historical_mean_forecast": expanding_mean,
            "historical_mean_absolute_error": float(abs(actual - expanding_mean)),
        }
        last_day = working["_ticket_time"].dt.floor("D").max()
        train_mask = working["_ticket_time"].dt.floor("D") < last_day
        holdout_mask = ~train_mask
        train_rows = working.loc[train_mask].copy()
        test_rows = working.loc[holdout_mask].copy()
        training_matrix, heldout_matrix = _time_aware_fingerprint_matrices(
            train_rows, test_rows
        )
        distinct_training = (
            _sparse_unique_count(training_matrix)
            if training_matrix is not None
            else 0
        )
        forecast_cluster_count = min(count, distinct_training)
        if (
            forecast_cluster_count >= 2
            and heldout_matrix is not None
            and heldout_matrix.shape[0]
        ):
            temporal_model = KMeans(
                n_clusters=forecast_cluster_count,
                random_state=seed,
                n_init=5,
            )
            train_cluster_labels = temporal_model.fit_predict(training_matrix)
            heldout_cluster_labels = temporal_model.predict(heldout_matrix)
            train_rows["_fingerprint_cluster"] = train_cluster_labels
            test_rows["_fingerprint_cluster"] = heldout_cluster_labels
            prior_days = pd.date_range(
                train_rows["_ticket_time"].dt.floor("D").min(),
                last_day - pd.Timedelta(days=1),
                freq="D",
            )
            for cluster_id in range(forecast_cluster_count):
                cluster_rows = train_rows[
                    train_rows["_fingerprint_cluster"] == cluster_id
                ]
                history = (
                    cluster_rows.groupby(cluster_rows["_ticket_time"].dt.floor("D"))
                    .size()
                    .reindex(prior_days, fill_value=0)
                    .to_numpy(dtype=float)
                )
                observed_count = int(
                    (test_rows["_fingerprint_cluster"] == cluster_id).sum()
                )
                recurrence_forecasts.append(
                    {
                        "cluster": cluster_id,
                        "latest_day": last_day.strftime("%Y-%m-%d"),
                        "observed": observed_count,
                        "persistence_forecast": float(history[-1]),
                        "persistence_absolute_error": float(
                            abs(observed_count - history[-1])
                        ),
                        "historical_mean_forecast": float(history.mean()),
                        "historical_mean_absolute_error": float(
                            abs(observed_count - history.mean())
                        ),
                        "training_days": len(prior_days),
                    }
                )
    fingerprints = []
    for cluster_id in range(count):
        positions = np.flatnonzero(augmented_labels == cluster_id)
        if not len(positions):
            continue
        rows = working.iloc[positions]
        codes = sorted(
            {
                code
                for values in error_codes.iloc[positions]
                for code in values
            }
        )
        mean_text_vector = np.asarray(plain_matrix[positions].mean(axis=0)).ravel()
        top_terms = np.asarray(plain_vectorizer.get_feature_names_out())[
            np.argsort(mean_text_vector)[-3:][::-1]
        ].tolist()
        fingerprints.append(
            {
                "cluster": cluster_id,
                "count": len(positions),
                "terms": top_terms,
                "product": str(rows["Product Purchased"].mode().iloc[0]),
                "start": rows["_ticket_time"].min().strftime("%Y-%m-%d %H:%M"),
                "end": rows["_ticket_time"].max().strftime("%Y-%m-%d %H:%M"),
                "error_codes": codes[:5],
                "sample_ids": [str(value) for value in rows["Ticket ID"].head(3)],
            }
        )
    return {
        "status": "ready",
        "dated_tickets": len(working),
        "days_observed": len(days),
        "cluster_count": count,
        "comparison": comparison,
        "fingerprints": fingerprints,
        "daily_volume": days,
        "forecast": forecast,
        "recurrence_forecasts": recurrence_forecasts,
        "event_precision_recall": None,
        "event_note": "Unavailable: no incident/event ground-truth labels exist.",
        "message": (
            "The fingerprint combines 64-or-fewer TF-IDF latent dimensions, product/error-code indicators, "
            "and response-time features. It is not a pretrained semantic embedding."
        ),
    }
