from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data import redact_sensitive_text, ticket_text


ERROR_CODE_PATTERN = re.compile(
    r"\b(?:[A-Z]{2,}[-_ ]?\d{2,}|(?:0x)?[A-F0-9]{6,})\b", re.IGNORECASE
)
TECHNICAL_TERMS = {
    "app",
    "application",
    "device",
    "firmware",
    "network",
    "payment",
    "portal",
    "printer",
    "server",
    "software",
    "website",
}


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=20_000,
        strip_accents="unicode",
    )


def _prepared_text(frame: pd.DataFrame) -> pd.Series:
    return ticket_text(frame).fillna("")


class TextIndex:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.texts = _prepared_text(frame)
        self.valid = self.texts.ne("")
        self.positions = np.flatnonzero(self.valid.to_numpy())
        self.vectorizer = make_vectorizer()
        self.matrix = (
            self.vectorizer.fit_transform(self.texts.loc[self.valid])
            if self.positions.size
            else None
        )

    def nearest(
        self,
        query: str,
        limit: int = 5,
        exclude_ticket_id: str | None = None,
        resolved_only: bool = False,
    ) -> list[dict[str, Any]]:
        query = redact_sensitive_text(query).strip()
        if not query or self.matrix is None:
            return []
        query_vector = self.vectorizer.transform([query])
        if query_vector.nnz == 0:
            return []
        scores = (self.matrix @ query_vector.T).toarray().ravel()
        candidate_frame = self.frame.iloc[self.positions]
        eligible = np.ones(len(candidate_frame), dtype=bool)
        if exclude_ticket_id is not None:
            eligible &= candidate_frame["Ticket ID"].astype(str).to_numpy() != str(
                exclude_ticket_id
            )
        if resolved_only:
            eligible &= candidate_frame["Resolution"].notna().to_numpy()
        scores[~eligible] = 0
        indexes = np.argsort(scores)[::-1][:limit]
        results = []
        for index in indexes:
            if scores[index] <= 0:
                continue
            record = candidate_frame.iloc[int(index)]
            resolution = record["Resolution"]
            results.append(
                {
                    "ticket_id": str(record["Ticket ID"]),
                    "subject": redact_sensitive_text(record["Ticket Subject"]),
                    "ticket_type": str(record["Ticket Type"]),
                    "priority": str(record["Ticket Priority"]),
                    "similarity": float(scores[index]),
                    "resolution": (
                        redact_sensitive_text(resolution)
                        if pd.notna(resolution) and str(resolution).strip()
                        else ""
                    ),
                }
            )
        return results


def nearest_tickets(
    frame: pd.DataFrame,
    query: str,
    limit: int = 5,
    exclude_ticket_id: str | None = None,
    resolved_only: bool = False,
    index: TextIndex | None = None,
) -> list[dict[str, Any]]:
    text_index = index or TextIndex(frame)
    return text_index.nearest(query, limit, exclude_ticket_id, resolved_only)


def novelty_assessment(
    frame: pd.DataFrame, query: str, index: TextIndex | None = None
) -> dict[str, Any]:
    text_index = index or TextIndex(frame)
    texts = text_index.texts[text_index.valid]
    safe_query = redact_sensitive_text(query).strip()
    if len(texts) < 3 or not safe_query:
        return {
            "status": "insufficient_data",
            "nearest_similarity": 0.0,
            "local_density": 0.0,
            "threshold": 0.0,
            "density_threshold": 0.0,
            "nearest": [],
        }
    vectorizer = text_index.vectorizer
    matrix = text_index.matrix
    if matrix is None:
        return {
            "status": "insufficient_data",
            "nearest_similarity": 0.0,
            "local_density": 0.0,
            "threshold": 0.0,
            "density_threshold": 0.0,
            "nearest": [],
        }
    query_vector = vectorizer.transform([safe_query])
    if query_vector.nnz == 0:
        nearest_similarity = 0.0
        density = 0.0
        calibration = []
        calibration_density = []
    else:
        scores = cosine_similarity(query_vector, matrix).ravel()
        nearest_similarity = float(scores.max(initial=0))
        density = float(np.sort(scores)[-min(5, len(scores)):].mean())
        calibration = []
        calibration_density = []
        sample_indexes = np.linspace(0, len(texts) - 1, min(len(texts), 250), dtype=int)
        sample = matrix[sample_indexes]
        self_similarity = cosine_similarity(sample, matrix)
        for row_index, source_index in enumerate(sample_indexes):
            self_similarity[row_index, source_index] = 0.0
            row_scores = self_similarity[row_index]
            calibration.append(float(row_scores.max(initial=0)))
            calibration_density.append(
                float(np.sort(row_scores)[-min(5, len(row_scores)):].mean())
            )
    threshold = float(np.quantile(calibration, 0.05)) if calibration else 0.15
    density_threshold = (
        float(np.quantile(calibration_density, 0.05))
        if calibration_density
        else 0.05
    )
    nearest = nearest_tickets(frame, safe_query, limit=3, index=text_index)
    return {
        "status": (
            "unfamiliar"
            if nearest_similarity < threshold or density < density_threshold
            else "known_like"
        ),
        "nearest_similarity": nearest_similarity,
        "local_density": density,
        "threshold": threshold,
        "density_threshold": density_threshold,
        "nearest": nearest,
        "validation_samples": len(calibration),
    }


def relationship_graph(
    frame: pd.DataFrame, ticket_id: str, limit: int = 6, index: TextIndex | None = None
) -> dict[str, Any]:
    rows = frame[frame["Ticket ID"].astype(str) == str(ticket_id)]
    if rows.empty:
        return {"ticket": None, "links": []}
    ticket = rows.iloc[0]
    own_text = f"{ticket['Ticket Subject'] or ''}\n{ticket['Ticket Description'] or ''}"
    matches = nearest_tickets(
        frame,
        own_text,
        limit=min(len(frame), 30),
        exclude_ticket_id=str(ticket_id),
        index=index,
    )
    own_errors = set(ERROR_CODE_PATTERN.findall(own_text))
    own_product = str(ticket["Product Purchased"])
    own_date = pd.to_datetime(ticket["First Response Time"], errors="coerce")
    for link in matches:
        candidate = frame[frame["Ticket ID"].astype(str) == link["ticket_id"]].iloc[0]
        candidate_text = f"{candidate['Ticket Subject'] or ''}\n{candidate['Ticket Description'] or ''}"
        candidate_errors = set(ERROR_CODE_PATTERN.findall(candidate_text))
        candidate_date = pd.to_datetime(candidate["First Response Time"], errors="coerce")
        link["shared_product"] = bool(
            own_product and own_product != "<NA>" and own_product == str(candidate["Product Purchased"])
        )
        link["shared_error_codes"] = sorted(own_errors & candidate_errors)
        link["days_apart"] = (
            abs((own_date - candidate_date).total_seconds()) / 86400
            if pd.notna(own_date) and pd.notna(candidate_date)
            else None
        )
        link["link_reasons"] = [
            reason
            for reason, present in (
                ("semantic text similarity", link["similarity"] >= 0.12),
                ("shared product", link["shared_product"]),
                ("shared error code", bool(link["shared_error_codes"])),
                ("within 7 days", link["days_apart"] is not None and link["days_apart"] <= 7),
            )
            if present
        ]
        link["link_score"] = min(
            1.0,
            link["similarity"] * 0.65
            + 0.18 * link["shared_product"]
            + 0.12 * bool(link["shared_error_codes"])
            + 0.05
            * (link["days_apart"] is not None and link["days_apart"] <= 7),
        )
    links = sorted(matches, key=lambda item: item["link_score"], reverse=True)[:limit]
    return {
        "ticket": {
            "ticket_id": str(ticket["Ticket ID"]),
            "subject": redact_sensitive_text(ticket["Ticket Subject"]),
            "product": str(ticket["Product Purchased"]),
            "error_codes": sorted(own_errors),
        },
        "links": links,
    }


def issue_clusters(frame: pd.DataFrame, max_rows: int = 1200) -> list[dict[str, Any]]:
    texts = _prepared_text(frame)
    valid = texts.ne("")
    working = frame.loc[valid].copy()
    if len(working) > max_rows:
        working = working.sample(max_rows, random_state=42)
    if len(working) < 8:
        return []
    vectorizer = make_vectorizer()
    matrix = vectorizer.fit_transform(texts.loc[working.index])
    cluster_count = min(6, max(2, int(np.sqrt(len(working) / 2))))
    clusterer = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=42,
        n_init=3,
        batch_size=256,
    )
    labels = clusterer.fit_predict(matrix)
    terms = np.asarray(vectorizer.get_feature_names_out())
    clusters = []
    for cluster_id in range(cluster_count):
        row_indexes = np.flatnonzero(labels == cluster_id)
        if row_indexes.size == 0:
            continue
        center = clusterer.cluster_centers_[cluster_id]
        top_terms = terms[np.argsort(center)[-4:][::-1]].tolist()
        clusters.append(
            {
                "label": " · ".join(top_terms[:3]) or f"Pattern {cluster_id + 1}",
                "count": int(row_indexes.size),
                "sample_ids": [
                    str(value)
                    for value in working.iloc[row_indexes[:3]]["Ticket ID"].tolist()
                ],
            }
        )
    return sorted(clusters, key=lambda cluster: cluster["count"], reverse=True)


def recurrence_radar(frame: pd.DataFrame) -> dict[str, Any]:
    timestamps = pd.to_datetime(frame["First Response Time"], errors="coerce")
    valid = timestamps.dropna()
    if valid.empty:
        return {"points": [], "days_observed": 0, "spikes": [], "source": None}
    per_day = valid.dt.floor("D").value_counts().sort_index()
    points = [
        {"date": timestamp.strftime("%Y-%m-%d"), "count": int(count)}
        for timestamp, count in per_day.items()
    ]
    counts = np.asarray([point["count"] for point in points], dtype=float)
    mean = float(counts.mean())
    std = float(counts.std())
    spikes = [
        point
        for point in points
        if std > 0 and (point["count"] - mean) / std >= 2
    ]
    return {
        "points": points,
        "days_observed": len(points),
        "spikes": spikes,
        "source": "First Response Time",
    }


def routing_simulation(
    frame: pd.DataFrame, capacity_factor: float = 1.0
) -> dict[str, Any]:
    category_counts = frame["Ticket Type"].dropna().astype(str).value_counts()
    teams = [
        {
            "name": "Technical Support",
            "skills": ["Technical issue", "Product inquiry"],
            "capacity": max(1, int(180 * capacity_factor)),
        },
        {
            "name": "Billing & Payments",
            "skills": ["Billing inquiry", "Refund request"],
            "capacity": max(1, int(160 * capacity_factor)),
        },
        {
            "name": "Retention",
            "skills": ["Cancellation request"],
            "capacity": max(1, int(130 * capacity_factor)),
        },
    ]
    routed: dict[str, int] = {team["name"]: 0 for team in teams}
    unassigned = 0
    for category, count in category_counts.items():
        eligible = [team for team in teams if category in team["skills"]]
        if not eligible:
            unassigned += int(count)
            continue
        team = min(
            eligible,
            key=lambda item: routed[item["name"]] / item["capacity"],
        )
        routed[team["name"]] += int(count)
    total_capacity = sum(team["capacity"] for team in teams)
    total = int(category_counts.sum())
    return {
        "teams": teams,
        "ticket_count": total,
        "unassigned": unassigned,
        "skill_match": [
            {
                "team": team["name"],
                "tickets": routed[team["name"]],
                "capacity": team["capacity"],
                "utilization": routed[team["name"]] / team["capacity"],
            }
            for team in teams
        ],
        "round_robin": {
            team["name"]: int(total / len(teams))
            for team in teams
        },
        "capacity_coverage": min(1.0, total_capacity / total) if total else 0.0,
    }


def sla_risk_sandbox(
    priority: str, elapsed_hours: float, target_hours: float, queue_load: int
) -> dict[str, Any]:
    if (
        not all(math.isfinite(value) for value in (elapsed_hours, target_hours))
        or target_hours <= 0
        or elapsed_hours < 0
        or queue_load < 0
    ):
        raise ValueError("SLA target must be positive; elapsed time and queue load cannot be negative.")
    priority_factor = {
        "low": 0.0,
        "medium": 0.08,
        "high": 0.16,
        "critical": 0.24,
    }.get(priority.casefold(), 0.0)
    elapsed_ratio = elapsed_hours / target_hours
    queue_factor = min(queue_load / 100, 0.2)
    risk = min(1.0, max(0.0, 0.78 * elapsed_ratio + priority_factor + queue_factor))
    return {
        "risk_index": risk,
        "elapsed_ratio": elapsed_ratio,
        "priority_factor": priority_factor,
        "queue_factor": queue_factor,
        "remaining_hours": max(0.0, target_hours - elapsed_hours),
    }


def drift_summary(frame: pd.DataFrame, reports: dict[str, Any] | None) -> dict[str, Any]:
    timestamp = pd.to_datetime(frame["First Response Time"], errors="coerce")
    valid = frame.loc[timestamp.notna()].copy()
    valid["_timestamp"] = timestamp[timestamp.notna()]
    days = sorted(valid["_timestamp"].dt.floor("D").unique())
    category_shift = []
    jensen_shannon = 0.0
    if len(days) >= 2:
        midpoint = len(days) // 2
        early = valid[valid["_timestamp"].dt.floor("D") <= days[midpoint - 1]]
        late = valid[valid["_timestamp"].dt.floor("D") >= days[midpoint]]
        early_dist = early["Ticket Type"].value_counts(normalize=True)
        late_dist = late["Ticket Type"].value_counts(normalize=True)
        labels = sorted(set(early_dist.index) | set(late_dist.index))
        early_values = np.asarray([early_dist.get(label, 0) for label in labels], dtype=float)
        late_values = np.asarray([late_dist.get(label, 0) for label in labels], dtype=float)
        early_values = (early_values + 1e-9) / (early_values.sum() + 1e-9 * len(labels))
        late_values = (late_values + 1e-9) / (late_values.sum() + 1e-9 * len(labels))
        midpoint_dist = (early_values + late_values) / 2
        jensen_shannon = float(
            0.5 * np.sum(early_values * np.log(early_values / midpoint_dist))
            + 0.5 * np.sum(late_values * np.log(late_values / midpoint_dist))
        )
        category_shift = [
            {
                "label": label,
                "early": float(early_dist.get(label, 0)),
                "late": float(late_dist.get(label, 0)),
            }
            for label in labels
        ]
    return {
        "days_observed": len(days),
        "category_shift": category_shift,
        "category_jensen_shannon": jensen_shannon,
        "distribution_shift_detected": jensen_shannon >= 0.1,
        "scores_available": bool(reports),
        "chronological_validation": False,
        "review_required": len(days) < 14 or jensen_shannon >= 0.1,
    }


def counterfactual_comparison(
    estimators: dict[str, Any], subject: str, description: str
) -> list[dict[str, Any]]:
    from models import predict_ticket

    scenarios = [
        {
            "name": "Original wording",
            "text": f"{subject}\n{description}".strip(),
            "note": "As entered",
        },
        {
            "name": "Production-wide outage wording",
            "text": f"{subject}\n{description}\nProduction-wide outage affecting all users.",
            "note": "Hypothetical phrase added",
        },
        {
            "name": "Single-user issue wording",
            "text": f"{subject}\n{description}\nOnly one user is affected; other users can continue working.",
            "note": "Hypothetical phrase added",
        },
    ]
    output = []
    for scenario in scenarios:
        prediction = predict_ticket(
            estimators,
            redact_sensitive_text(scenario["text"]),
            "",
        )
        output.append(
            {
                "name": scenario["name"],
                "note": scenario["note"],
                "predictions": prediction,
            }
        )
    return output
