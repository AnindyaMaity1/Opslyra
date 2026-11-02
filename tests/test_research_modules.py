from __future__ import annotations

import numpy as np
import pandas as pd

from annotation import cohens_kappa, kappa_from_frames
from dataset_audit import _cv_signal, duplicate_text_analysis
from openset import run_open_set_experiment
from reproducibility import exact_text_group_ids, stratified_group_holdout_indices
from risk_coverage import (
    risk_coverage_from_labels,
    review_feedback_from_activity,
    simulate_risk_coverage,
    threshold_for_target_risk,
)


def three_class_frame(rows_per_class: int = 30) -> pd.DataFrame:
    examples = {
        "Billing inquiry": ("invoice payment", "charge declined"),
        "Technical issue": ("device reboot", "software crash"),
        "Product inquiry": ("product features", "device compatibility"),
    }
    rows = []
    for label, (subject, description) in examples.items():
        for index in range(rows_per_class):
            rows.append(
                {
                    "Ticket Type": label,
                    "Ticket Subject": f"{subject} example {index}",
                    "Ticket Description": f"{description} report {index}",
                }
            )
    return pd.DataFrame(rows)


def test_grouped_split_keeps_normalized_duplicate_text_together():
    texts = pd.Series(
        ["Same   ticket", "same ticket", "billing one", "tech one", "billing two", "tech two"]
    )
    labels = np.array(["billing", "billing", "billing", "technical", "billing", "technical"])
    groups = exact_text_group_ids(texts)
    split = stratified_group_holdout_indices(labels, groups, 0.5, 5)
    assert split is not None
    train_idx, test_idx = split
    assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))


def test_text_accuracy_comparison_reports_grouped_majority_baseline():
    frame = three_class_frame(rows_per_class=15)
    result = _cv_signal(
        frame["Ticket Subject"] + " " + frame["Ticket Description"],
        frame["Ticket Type"],
        max_rows=None,
    )
    assert result["status"] == "ok"
    assert result["labeled_rows"] == len(frame)
    assert result["folds"] == 3
    assert result["tfidf_lr_accuracy"] > result["majority_accuracy"]
    assert result["accuracy_lift"] > 0
    assert len(result["fold_accuracy"]) == 3


def test_duplicate_audit_counts_priority_and_status_label_conflicts():
    frame = pd.DataFrame(
        {
            "Ticket Subject": [
                "Repeated ticket",
                "Repeated ticket",
                "Another ticket",
                "A third ticket",
                "A fourth ticket",
            ],
            "Ticket Description": [
                "same description",
                "same description",
                "different description one",
                "different description two",
                "different description three",
            ],
            "Ticket Type": ["Billing", "Billing", "Billing", "Technical", "Product"],
            "Ticket Priority": ["Low", "High", "Low", "Medium", "Critical"],
            "Ticket Status": ["Open", "Closed", "Open", "Closed", "Open"],
        }
    )
    result = duplicate_text_analysis(frame)
    assert result["exact_duplicate_rows"] == 1
    assert result["conflicting_exact_text_groups_by_target"]["Ticket Priority"] == 1
    assert result["conflicting_exact_text_groups_by_target"]["Ticket Status"] == 1
    assert result["near_duplicates"]["sample_size"] == 4
    assert "Ticket Status" in result["near_duplicates"]["label_conflicts_by_target"]


def test_risk_coverage_ties_match_threshold_acceptance():
    curve = risk_coverage_from_labels(
        np.array([0, 1, 1, 0]),
        np.array([0, 0, 1, 0]),
        np.array([0.9, 0.9, 0.4, 0.4]),
        points=4,
    )
    assert curve
    for point in curve:
        accepted = np.array([0.9, 0.9, 0.4, 0.4]) >= point["threshold"]
        assert point["accepted"] == int(accepted.sum())
        assert point["coverage"] == round(float(accepted.mean()), 4)


def test_unmet_validation_risk_selects_full_abstention():
    curve = risk_coverage_from_labels(
        np.array([0, 1, 0, 1]),
        np.array([1, 0, 1, 0]),
        np.array([0.9, 0.8, 0.7, 0.6]),
    )
    selected = threshold_for_target_risk(curve, target_risk=0)
    assert selected["coverage"] == 0
    assert selected["selective_risk"] is None
    assert selected["threshold"] > 0.9


def test_risk_coverage_simulation_executes_and_honors_unmet_target():
    result = simulate_risk_coverage(
        three_class_frame(),
        target_risk=0,
        selection_mode="target_risk",
        max_rows=30,
        seed=3,
    )
    assert result["status"] == "ok"
    assert result["threshold_from_validation"] is not None
    assert result["validation_curve"]
    assert result["validation_rows"] > 0
    assert 0 <= result["holdout"]["coverage"] <= 1


def test_open_set_stratified_row_cap_and_known_class_metrics():
    result = run_open_set_experiment(
        three_class_frame(),
        seeds=(0,),
        unknown_classes=["Product inquiry"],
        max_rows=18,
    )
    assert result["status"] == "ok"
    assert result["held_out_classes"] == ["Product inquiry"]
    method = result["per_seed"][0]["methods"]["maximum_softmax_probability"]
    assert method["known_classification"]["labels"] == result["known_classes"]
    assert len(method["known_classification"]["confusion_matrix"]) == len(
        result["known_classes"]
    )
    assert method["known_classification"]["per_class"]


def test_open_set_reports_invalid_requested_unknown_class():
    result = run_open_set_experiment(
        three_class_frame(),
        seeds=(0,),
        unknown_classes=["Unknown category"],
        max_rows=30,
    )
    assert result["status"] == "unknown_classes_not_found"
    assert result["requested_unknown_classes"] == ["Unknown category"]


def test_review_correction_rate_compares_overlay_with_parent_prediction():
    events = [
        {
            "id": 7,
            "event_type": "recommendation",
            "details": {
                "predictions": {
                    "Ticket Type": {"label": "Billing inquiry"},
                    "Ticket Priority": {"label": "Low"},
                }
            },
        },
        {
            "event_type": "recommendation_approved",
            "details": {
                "parent_activity_id": 7,
                "overlay": {
                    "Ticket Type": "Technical issue",
                    "Ticket Priority": "Low",
                },
            },
        },
        {"event_type": "recommendation_rejected", "details": {}},
    ]
    result = review_feedback_from_activity(events)
    assert result["human_acceptance_rate"] == 0.5
    assert result["correction_rate"] == 1.0
    assert result["n_review_events"] == 2


def test_annotation_agreement_excludes_unlabeled_and_unmatched_pairs():
    annotator_a = pd.DataFrame(
        {
            "Ticket ID": ["1", "2", "3"],
            "Ticket Type": ["Billing", "Technical", "Billing"],
            "Ticket Priority": ["Low", None, "High"],
        }
    )
    annotator_b = pd.DataFrame(
        {
            "Ticket ID": ["1", "2", "4"],
            "Ticket Type": ["Billing", "Technical", "Technical"],
            "Ticket Priority": ["Low", "High", "Medium"],
        }
    )
    result = kappa_from_frames(annotator_a, annotator_b)
    assert result["n"] == 2
    assert result["labels"]["Ticket Type"]["n"] == 2
    assert result["labels"]["Ticket Priority"]["n"] == 1
    assert result["labels"]["Ticket Priority"]["unlabeled_pairs_excluded"] == 1


def test_degenerate_perfect_agreement_does_not_invent_kappa():
    result = cohens_kappa(["Billing", "Billing"], ["Billing", "Billing"])
    assert result["observed_agreement"] == 1
    assert result["kappa"] is None
