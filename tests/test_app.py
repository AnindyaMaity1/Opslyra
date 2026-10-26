from __future__ import annotations

import csv

import pandas as pd
import pytest

from app import create_app
from adaptive import evaluate_ticket_workflow
from data import REQUIRED_COLUMNS, dataset_profile, load_dataset, redact_sensitive_text
from intelligence import (
    TextIndex,
    nearest_tickets,
    novelty_assessment,
    relationship_graph,
    sla_risk_sandbox,
)
from models import train_baselines
from research import (
    active_learning_benchmark,
    active_learning_queue,
    constrained_routes,
    counterfactual_routing_study,
    temporal_fingerprint_study,
)


@pytest.fixture()
def small_dataset(tmp_path):
    path = tmp_path / "tickets.csv"
    rows = []
    for index in range(48):
        rows.append(
            {
                "Ticket ID": str(index + 1),
                "Customer Name": f"Private Person {index}",
                "Customer Email": f"person{index}@example.com",
                "Customer Age": "30",
                "Customer Gender": "Other",
                "Product Purchased": "Widget",
                "Date of Purchase": "2024-01-01",
                "Ticket Type": "Billing inquiry" if index % 2 else "Technical issue",
                "Ticket Subject": "Payment problem" if index % 2 else "Device will not start",
                "Ticket Description": (
                    "The card payment was declined and invoice is incorrect."
                    if index % 2
                    else "The device does not turn on after charging."
                ),
                "Ticket Status": "Closed" if index % 3 else "Open",
                "Resolution": "Resolved" if index % 3 else "",
                "Ticket Priority": "High" if index % 2 else "Low",
                "Ticket Channel": "Email",
                "First Response Time": "2024-01-02 10:00:00",
                "Time to Resolution": "2024-01-02 12:00:00" if index % 3 else "",
                "Customer Satisfaction Rating": "4" if index % 3 else "",
            }
        )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture()
def client(small_dataset):
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "DATASET_PATH": str(small_dataset),
            "ITEMS_PER_PAGE": 10,
            "STATE_DATABASE": str(small_dataset.parent / "state.sqlite3"),
            "MODEL_ARTIFACTS": str(small_dataset.parent / "models"),
        }
    )
    return app.test_client()


def test_dataset_schema_and_profile(small_dataset):
    frame = load_dataset(small_dataset)
    profile = dataset_profile(frame)
    assert list(frame.columns) == REQUIRED_COLUMNS
    assert profile["row_count"] == 48
    assert profile["duplicate_ticket_ids"] == 0
    assert len(profile["targets"]["Ticket Type"]) == 2
    assert profile["date_fields"]["First Response Time"]["parseable"] == 48


def test_redacts_common_contact_details():
    result = redact_sensitive_text("Email me at help@example.com or call 555-123-4567")
    assert "help@example.com" not in result
    assert "555-123-4567" not in result
    assert "[redacted email]" in result
    assert "[redacted phone]" in result
    result = redact_sensitive_text(
        "username:alice @agent 192.168.2.5 api_key=secret-value Bearer token-value"
    )
    for value in ("alice", "@agent", "192.168.2.5", "secret-value", "token-value"):
        assert value not in result


def test_dashboard_uses_dataset_metrics(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Opslyra" in response.data
    assert b"From Every Ticket to Intelligent Action" in response.data
    assert b"TicketMind" not in response.data
    assert b"Your next ticket to act on, at a glance." in response.data
    assert b"<br>" not in response.data
    assert b"4.0" in response.data
    assert b"48" in response.data
    assert b"Priority review queue" in response.data
    assert b"Awaiting customer" in response.data
    assert b"Issue categories" in response.data
    assert b"overview-category-card" in response.data
    assert b"overview-priority-card" in response.data
    assert b"overview-status-ring" in response.data
    assert response.data.index(b"overview-category-orbit") < response.data.index(b"overview-category-copy")
    assert b"overview-category-track" not in response.data
    assert b"overview-priority-track" not in response.data
    assert b"overview-status-track" not in response.data
    assert b"not live SLA data" in response.data


def test_dashboard_prioritizes_open_urgent_tickets_and_redacts_subject(client, small_dataset):
    frame = load_dataset(small_dataset)
    frame.loc[0, "Ticket Priority"] = "Critical"
    frame.loc[0, "Ticket Subject"] = "VPN failure for person0@example.com"
    frame.to_csv(small_dataset, index=False)

    response = client.get("/")
    assert response.status_code == 200
    assert b"VPN failure for [redacted email]" in response.data
    assert b"person0@example.com" not in response.data
    assert b"Review impact and follow your incident escalation process" in response.data
    assert b"/tickets/1" in response.data


def test_ticket_search_and_details_do_not_expose_customer_identifiers(client):
    response = client.get("/tickets?q=Payment")
    assert response.status_code == 200
    assert b"Payment problem" in response.data
    assert b"Private Person" not in response.data
    assert b"person1@example.com" not in response.data

    detail = client.get("/tickets/2")
    assert detail.status_code == 200
    assert b"person1@example.com" not in detail.data
    assert b"Private Person" not in detail.data


def test_adaptive_workspace_modes_require_csrf_and_persist(client):
    page = client.get("/adaptive-ai")
    assert page.status_code == 200
    assert b"Fully Manual" in page.data
    assert b"Maximum Permitted Autonomy" not in page.data

    rejected = client.post("/api/adaptive-ai/mode", json={"mode": "copilot"})
    assert rejected.status_code == 403

    with client.session_transaction() as session:
        csrf_token = session["adaptive_csrf_token"]
    response = client.post(
        "/api/adaptive-ai/mode",
        json={"mode": "copilot"},
        headers={"X-CSRF-Token": csrf_token, "Idempotency-Key": "workspace-mode-001"},
    )
    assert response.status_code == 200
    assert response.json["mode"] == "copilot"
    assert client.get("/adaptive-ai").data.find(b"AI Copilot") >= 0


def test_copilot_recommendations_require_human_decision_and_are_idempotent(client):
    client.get("/adaptive-ai")
    with client.session_transaction() as session:
        csrf_token = session["adaptive_csrf_token"]
    headers = {
        "X-CSRF-Token": csrf_token,
        "Idempotency-Key": "recommendation-request-001",
    }
    client.post(
        "/api/adaptive-ai/mode",
        json={"mode": "copilot"},
        headers={**headers, "Idempotency-Key": "workspace-mode-002"},
    )

    first = client.post(
        "/api/tickets/2/ai/recommend", json={}, headers=headers
    )
    second = client.post(
        "/api/tickets/2/ai/recommend", json={}, headers=headers
    )
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json["activity"]["id"] == second.json["activity"]["id"]
    assert first.json["activity"]["status"] == "pending_approval"
    assert set(first.json["predictions"]) == {"Ticket Type", "Ticket Priority"}
    assert first.json["activity"]["details"]["score_note"].startswith("Model scores are uncalibrated")

    activity_id = first.json["activity"]["id"]
    approved = client.post(
        f"/api/tickets/2/ai/recommendations/{activity_id}/approve",
        json={"values": {"Ticket Type": "Technical issue", "Ticket Priority": "Low"}},
        headers={**headers, "Idempotency-Key": "approve-request-001"},
    )
    assert approved.status_code == 200
    detail = client.get("/tickets/2")
    assert b"Opslyra label overlay" in detail.data
    assert b"Category: Technical issue" in detail.data
    assert b"Priority: Low" in detail.data
    activity = client.application.extensions["ticketmind_state"].ai_activity("2")
    decision = next(item for item in activity if item["event_type"] == "recommendation_approved")
    assert decision["details"]["overlay"] == {
        "Ticket Type": "Technical issue",
        "Ticket Priority": "Low",
    }

    duplicate_approval = client.post(
        f"/api/tickets/2/ai/recommendations/{activity_id}/approve",
        json={},
        headers={**headers, "Idempotency-Key": "approve-request-002"},
    )
    assert duplicate_approval.status_code == 409


def test_human_takeover_blocks_new_ai_workflow(client):
    client.get("/adaptive-ai")
    with client.session_transaction() as session:
        csrf_token = session["adaptive_csrf_token"]
    headers = {"X-CSRF-Token": csrf_token, "Idempotency-Key": "takeover-mode-001"}
    client.post(
        "/api/tickets/2/adaptive-mode",
        json={"mode": "takeover"},
        headers=headers,
    )
    response = client.post(
        "/api/tickets/2/ai/recommend",
        json={},
        headers={**headers, "Idempotency-Key": "takeover-recommend-001"},
    )
    assert response.status_code == 409
    detail = client.get("/tickets/2")
    assert b"Human takeover active" in detail.data
    assert b"AI activity timeline" in detail.data


def test_selective_workflow_respects_enabled_actions_threshold_and_risk():
    predictions = {
        "Ticket Type": {"label": "Technical issue", "score": 0.91},
        "Ticket Priority": {"label": "High", "score": 0.72},
    }
    result = evaluate_ticket_workflow(
        "selective",
        predictions,
        {"confidence_threshold": 0.85, "enabled_actions": ["category"]},
        "The application will not start.",
    )
    assert result["status"] == "applied_to_workspace_overlay"
    assert [action["action"] for action in result["actions"]] == ["category"]

    risky = evaluate_ticket_workflow(
        "selective",
        predictions,
        {"confidence_threshold": 0.85, "enabled_actions": ["category", "priority"]},
        "We detected unauthorized access to an account.",
    )
    assert risky["status"] == "needs_review"
    assert risky["actions"] == []


def test_adaptive_policy_validates_actions_and_records_updates(client):
    client.get("/adaptive-ai")
    with client.session_transaction() as session:
        csrf_token = session["adaptive_csrf_token"]
    headers = {"X-CSRF-Token": csrf_token, "Idempotency-Key": "policy-save-001"}
    invalid = client.post(
        "/api/adaptive-ai/policy",
        json={"confidence_threshold": 0.4, "enabled_actions": ["category"]},
        headers=headers,
    )
    assert invalid.status_code == 400

    valid = client.post(
        "/api/adaptive-ai/policy",
        json={"confidence_threshold": 0.92, "enabled_actions": ["priority"]},
        headers=headers,
    )
    assert valid.status_code == 200
    assert valid.json["policy"]["confidence_threshold"] == 0.92
    assert valid.json["policy"]["enabled_actions"] == ["priority"]
    assert b"workspace policy changed" in client.get("/adaptive-ai").data.lower()


def test_takeover_race_stops_workflow_before_any_action_is_committed(client):
    client.get("/adaptive-ai")
    store = client.application.extensions["ticketmind_state"]
    store.set_workspace_ai_mode("copilot")
    store.set_ticket_ai_mode("2", "takeover")

    result = store.add_ticket_workflow_activity(
        "2",
        "copilot",
        "recommendation",
        "pending_approval",
        {"predictions": {}},
        "racing-workflow-001",
    )
    assert result is None
    assert store.ai_activity("2") == []


def test_prediction_trains_models_and_returns_two_targets(client):
    response = client.post(
        "/predict",
        data={
            "subject": "Payment problem",
            "description": "Please review the invoice and card payment.",
        },
    )
    assert response.status_code == 200
    assert b"Ticket Type" in response.data
    assert b"Ticket Priority" in response.data
    assert b"uncalibrated" in response.data


def test_missing_required_column_returns_clear_dataset_error(tmp_path):
    path = tmp_path / "incomplete.csv"
    pd.DataFrame({"Ticket ID": ["1"]}).to_csv(path, index=False)
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "DATASET_PATH": str(path),
        }
    )
    response = app.test_client().get("/")
    assert response.status_code == 503
    assert b"Dataset is missing required columns" in response.data


def test_missing_dataset_path_returns_clear_error(tmp_path):
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "DATASET_PATH": str(tmp_path / "missing.csv"),
            "STATE_DATABASE": str(tmp_path / "state.sqlite3"),
            "MODEL_ARTIFACTS": str(tmp_path / "models"),
        }
    )
    response = app.test_client().get("/")
    assert response.status_code == 503
    assert b"Dataset file is not accessible" in response.data


def test_empty_dataset_is_rejected_with_clear_error(tmp_path):
    path = tmp_path / "empty.csv"
    pd.DataFrame(columns=REQUIRED_COLUMNS).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no ticket rows"):
        load_dataset(path)


def test_quality_page_renders_schema_and_label_checks(client):
    response = client.get("/data-quality")
    assert response.status_code == 200
    assert b"Schema and numeric checks" in response.data
    assert b"case or whitespace" not in response.data
    assert b"Ticket Type" in response.data


def test_tickets_have_pagination(client):
    response = client.get("/tickets")
    assert response.status_code == 200
    assert b"Page 1 of 5" in response.data
    assert b"Next" in response.data


def test_intelligence_dashboard_exposes_all_ten_workflows(client):
    response = client.get("/intelligence")
    assert response.status_code == 200
    for heading in (
        b"Human-in-the-loop learning",
        b"Open-set incident discovery",
        b"Counterfactual text explorer",
        b"Incident relationship graph",
        b"SLA risk sandbox",
        b"Adaptive routing simulator",
        b"Recurrence &amp; early-warning radar",
        b"Privacy-aware text handling",
        b"Evidence-grounded resolution memory",
        b"Model reliability observatory",
    ):
        assert heading in response.data


def test_sla_sandbox_changes_with_user_inputs(client):
    lower = client.post(
        "/intelligence",
        data={
            "action": "sla",
            "priority": "Low",
            "elapsed_hours": "1",
            "target_hours": "8",
            "queue_load": "0",
        },
    )
    higher = client.post(
        "/intelligence",
        data={
            "action": "sla",
            "priority": "Critical",
            "elapsed_hours": "7",
            "target_hours": "8",
            "queue_load": "80",
        },
    )
    assert lower.status_code == higher.status_code == 200
    assert b">10<" in lower.data
    assert b">100<" in higher.data


def test_feedback_training_versions_and_rollback(client):
    initial = client.post("/intelligence", data={"action": "train"})
    assert initial.status_code == 302

    correction = client.post(
        "/intelligence",
        data={
            "action": "correction",
            "ticket_id": "1",
            "target": "Ticket Type",
            "corrected_label": "Billing inquiry",
        },
    )
    assert correction.status_code == 302

    retrained = client.post("/intelligence", data={"action": "train"})
    assert retrained.status_code == 302
    app = client.application
    store = app.extensions["ticketmind_state"]
    versions = store.model_versions()
    assert len(versions) == 2
    assert versions[0]["active"] == 1
    assert versions[0]["metrics"]["Ticket Type"]["reviewed_examples"] == 1

    rollback = client.post(
        "/intelligence", data={"action": "rollback", "version": versions[1]["version"]}
    )
    assert rollback.status_code == 302
    assert store.model_versions()[1]["active"] == 1


def test_privacy_scan_redacts_identifiers_and_logs_counts(client):
    response = client.post(
        "/intelligence",
        data={
            "action": "privacy_scan",
            "privacy_text": (
                "Email alice@example.com, mention @agent, connect to 192.168.1.7, "
                "api_key=very-secret-value and user_name:alice"
            ),
        },
    )
    assert response.status_code == 200
    for sensitive in (b"alice@example.com", b"192.168.1.7", b"very-secret-value"):
        assert sensitive not in response.data
    audit = client.application.extensions["ticketmind_state"].redaction_summary()
    assert audit["events"] == 1
    assert audit["counts"]["email"] == 1
    assert audit["counts"]["ip_address"] == 1
    assert audit["counts"]["secret"] == 1


def test_novelty_graph_and_resolution_memory_are_data_driven(small_dataset):
    frame = load_dataset(small_dataset)
    index = TextIndex(frame)
    novelty = novelty_assessment(
        frame,
        "Quantum reactor telemetry has failed with ZZZ-99887",
        index,
    )
    assert novelty["status"] == "unfamiliar"
    assert novelty["local_density"] < novelty["density_threshold"]

    graph = relationship_graph(frame, "1", index=index)
    assert graph["ticket"]["ticket_id"] == "1"
    assert graph["links"]
    assert any(link["shared_product"] for link in graph["links"])

    matches = nearest_tickets(
        frame,
        "Payment problem card payment invoice",
        limit=3,
        resolved_only=True,
        index=index,
    )
    assert matches
    assert all(item["resolution"] for item in matches)


def test_novelty_page_records_unknown_rate_for_observatory(client):
    response = client.post(
        "/intelligence",
        data={
            "action": "novelty",
            "subject": "Unseen quantum reactor failure ZZZ-99887",
            "description": "Facility telemetry has a new anomaly.",
        },
    )
    assert response.status_code == 200
    assert b"UNKNOWN" in response.data
    assert b"Unknown-rate monitor" in response.data
    summary = client.application.extensions["ticketmind_state"].novelty_summary()
    assert summary["total"] == 1
    assert summary["unknown"] == 1


def test_counterfactual_scenarios_show_changed_model_inputs(client):
    response = client.post(
        "/intelligence",
        data={
            "action": "counterfactual",
            "subject": "Login issue",
            "description": "One employee cannot access the portal.",
        },
    )
    assert response.status_code == 200
    assert b"Original wording" in response.data
    assert b"Production-wide outage wording" in response.data
    assert b"Single-user issue wording" in response.data
    assert b"not a causal guarantee" in response.data


def test_reviewed_team_examples_enable_team_model_without_fake_metrics(small_dataset):
    frame = load_dataset(small_dataset)
    corrections = [
        {"ticket_id": "1", "target": "Team", "corrected_label": "Technical"},
        {"ticket_id": "3", "target": "Team", "corrected_label": "Technical"},
        {"ticket_id": "2", "target": "Team", "corrected_label": "Billing"},
        {"ticket_id": "4", "target": "Team", "corrected_label": "Billing"},
    ]
    estimators, reports = train_baselines(frame, corrections)
    assert "Team" in estimators
    assert reports["Team"]["accuracy"] is None
    assert reports["Team"]["test_rows"] == 0


def test_sla_sandbox_rejects_non_finite_inputs():
    with pytest.raises(ValueError, match="target must be positive"):
        sla_risk_sandbox("High", float("nan"), 8, 4)


def test_research_page_renders_differentiator_modules(client):
    response = client.get("/research")
    assert response.status_code == 200
    assert b"UNCERTAINTY-GATED ACTIVE LEARNING" in response.data
    assert b"COUNTERFACTUAL INCIDENT ROUTING" in response.data
    assert b"TEMPORAL INCIDENT FINGERPRINTING" in response.data


def test_counterfactual_routing_form_returns_constraint_metrics(client):
    response = client.post(
        "/research",
        data={
            "action": "routing-study",
            "subject": "Production payment outage",
            "description": "All users cannot pay invoices.",
            "workload_0": "25",
            "capacity_0": "80",
            "workload_1": "20",
            "capacity_1": "70",
            "workload_2": "15",
            "capacity_2": "50",
            "remaining_sla_hours": "8",
        },
    )
    assert response.status_code == 200
    assert b"Explanation stability" in response.data
    assert b"Routing accuracy" in response.data
    assert b"assigned-team ground-truth labels" in response.data


def test_active_learning_benchmark_uses_equal_budget_policies(small_dataset):
    frame = load_dataset(small_dataset)
    result = active_learning_benchmark(frame, budget=8, max_rows=100)
    assert set(result["curves"]) == {"random", "uncertainty", "diversity", "proposed"}
    final_budgets = []
    for points in result["curves"].values():
        assert len(points) == 4
        final_budgets.append(points[-1]["budget"])
        assert points[-1]["rare_class"] in {"Billing inquiry", "Technical issue"}
        assert points[-1]["test_rows"] > 0
    assert len(set(final_budgets)) == 1
    assert final_budgets[0] == 8


def test_active_learning_queue_excludes_customer_fields(small_dataset):
    frame = load_dataset(small_dataset)
    estimators, _ = train_baselines(frame)
    queue = active_learning_queue(frame, estimators, limit=5)
    assert len(queue) == 5
    assert all("Customer Email" not in item for item in queue)
    assert all("uncertainty" in item and "novelty" in item for item in queue)


def test_constrained_routing_obeys_capacity_and_priority_sla():
    result = constrained_routes(
        "Technical issue",
        "Critical",
        {"Technical Support": 9},
        {"Technical Support": 10},
        12,
    )
    assert result["constrained_team"] == "Technical Support"
    assert result["sla_budget_hours"] == 4
    assert result["constraint_valid"] is False


def test_counterfactual_routing_study_marks_accuracy_unavailable(small_dataset):
    frame = load_dataset(small_dataset)
    estimators, _ = train_baselines(frame)
    result = counterfactual_routing_study(
        estimators,
        "Production outage",
        "All employees are affected.",
        {"Technical Support": 10, "Billing & Payments": 5, "Retention": 5},
        {"Technical Support": 50, "Billing & Payments": 30, "Retention": 20},
        8,
    )
    assert len(result["scenarios"]) == 3
    assert result["routing_accuracy"] is None
    assert 0 <= result["explanation_stability"] <= 1
    assert all("routing" in scenario for scenario in result["scenarios"])


def test_temporal_fingerprints_and_chronological_holdout(small_dataset):
    frame = load_dataset(small_dataset)
    for index in frame.index:
        frame.loc[index, "First Response Time"] = (
            f"2024-01-{1 + index // 10:02d} 10:00:00"
        )
    result = temporal_fingerprint_study(frame, max_rows=100)
    assert result["status"] == "ready"
    assert result["days_observed"] == 5
    assert len(result["comparison"]) == 2
    assert result["forecast"] is not None
    assert result["forecast"]["held_out_date"] == "2024-01-05"
    assert result["recurrence_forecasts"]
    assert result["event_precision_recall"] is None


def test_temporal_fingerprint_withholds_short_forecast(small_dataset):
    frame = load_dataset(small_dataset)
    result = temporal_fingerprint_study(frame, max_rows=100)
    assert result["status"] == "ready"
    assert result["days_observed"] == 1
    assert result["forecast"] is None
