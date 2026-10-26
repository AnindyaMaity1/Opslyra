from __future__ import annotations

from typing import Any


MODE_DETAILS: dict[str, dict[str, Any]] = {
    "manual": {
        "label": "Fully Manual",
        "description": "No AI recommendations or ticket actions run unless an employee changes the mode.",
        "allowed_actions": [],
        "approval": "All ticket decisions stay with the employee.",
        "escalation": "Not applicable; AI is paused.",
    },
    "copilot": {
        "label": "AI Copilot",
        "description": "AI prepares classification and evidence for review; nothing is applied without approval.",
        "allowed_actions": ["recommend category", "recommend priority", "retrieve similar resolved tickets"],
        "approval": "Human approval required before saving AI label suggestions.",
        "escalation": "Low model scores and sensitive wording are highlighted for human review.",
    },
    "selective": {
        "label": "Selective Automation",
        "description": "Only explicitly enabled category or priority suggestions may be saved to the local Opslyra overlay.",
        "allowed_actions": ["category (if enabled)", "priority (if enabled)", "retrieve similar resolved tickets"],
        "approval": "Enabled label suggestions can be applied only above the configured score threshold.",
        "escalation": "Sensitive wording or scores below threshold require human review.",
    },
    "autonomous": {
        "label": "Autonomous Workflow",
        "description": "Runs the permitted local workflow in sequence; unsupported or risky steps stop for review.",
        "allowed_actions": ["category (if enabled)", "priority (if enabled)", "retrieve similar resolved tickets"],
        "approval": "No customer-facing response or external action is permitted.",
        "escalation": "Sensitive wording or scores below threshold stop the workflow for human review.",
    },
    "takeover": {
        "label": "Human Takeover",
        "description": "AI processing is paused for this ticket while an employee handles it.",
        "allowed_actions": [],
        "approval": "No AI action can be proposed or applied during takeover.",
        "escalation": "Ticket remains with the human operator until they switch modes.",
    },
}

DEFAULT_POLICY: dict[str, Any] = {
    "confidence_threshold": 0.85,
    "enabled_actions": ["category", "priority"],
}

SENSITIVE_TERMS = (
    "security incident",
    "data breach",
    "unauthorized access",
    "account compromised",
    "data loss",
    "fraud",
)


def evaluate_ticket_workflow(
    mode: str,
    predictions: dict[str, dict[str, Any]],
    policy: dict[str, Any],
    ticket_text: str,
) -> dict[str, Any]:
    if mode not in MODE_DETAILS:
        raise ValueError("Unsupported adaptive AI mode.")

    text = ticket_text.casefold()
    risk_flags = [term for term in SENSITIVE_TERMS if term in text]
    priority_prediction = predictions.get("Ticket Priority", {})
    if str(priority_prediction.get("label", "")).casefold() == "critical":
        risk_flags.append("model_recommends_critical_priority")
    threshold = float(policy.get("confidence_threshold", DEFAULT_POLICY["confidence_threshold"]))

    if mode in {"manual", "takeover"}:
        return {
            "status": "paused",
            "actions": [],
            "risk_flags": risk_flags,
            "threshold": threshold,
            "message": MODE_DETAILS[mode]["description"],
        }

    if mode == "copilot":
        return {
            "status": "pending_approval",
            "actions": ["category", "priority"],
            "risk_flags": risk_flags,
            "threshold": threshold,
            "message": "Recommendations are ready for review. Nothing has been applied.",
        }

    enabled = set(policy.get("enabled_actions", []))
    actions: list[dict[str, Any]] = []
    below_threshold: list[str] = []
    for target, action_name in (
        ("Ticket Type", "category"),
        ("Ticket Priority", "priority"),
    ):
        if action_name not in enabled or target not in predictions:
            continue
        prediction = predictions[target]
        score = float(prediction["score"])
        if score < threshold:
            below_threshold.append(action_name)
        elif risk_flags:
            continue
        else:
            actions.append(
                {
                    "action": action_name,
                    "target": target,
                    "value": str(prediction["label"]),
                    "score": score,
                }
            )

    if risk_flags or below_threshold:
        status = "needs_review"
        message = "Automation paused; review the risk signals and low-scoring suggestions."
    elif actions:
        status = "applied_to_workspace_overlay"
        message = "Permitted label suggestions were saved to the local Opslyra overlay. The source CSV was not changed."
    else:
        status = "no_actions_enabled"
        message = "No automation actions are enabled for this mode."

    if mode == "autonomous" and not risk_flags and not below_threshold:
        message += " External routing and customer responses are not connected and were not attempted."

    return {
        "status": status,
        "actions": actions,
        "risk_flags": risk_flags,
        "below_threshold": below_threshold,
        "threshold": threshold,
        "message": message,
    }
