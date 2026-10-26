from __future__ import annotations

import math
import os
import secrets
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import pandas as pd
from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from adaptive import DEFAULT_POLICY, MODE_DETAILS, evaluate_ticket_workflow
from data import (
    DatasetError,
    dataset_profile,
    load_dataset,
    redact_sensitive_text,
    redact_with_audit,
)
from intelligence import (
    counterfactual_comparison,
    drift_summary,
    issue_clusters,
    nearest_tickets,
    novelty_assessment,
    recurrence_radar,
    relationship_graph,
    routing_simulation,
    sla_risk_sandbox,
)
from models import predict_ticket, train_baselines
from research import (
    active_learning_benchmark,
    active_learning_queue,
    counterfactual_routing_study,
    temporal_fingerprint_study,
)
from state import StateStore


ROOT = Path(__file__).resolve().parent


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_mapping(
        SECRET_KEY=(
            os.environ.get("OPSLYRA_SECRET_KEY")
            or os.environ.get("TICKETMIND_SECRET_KEY")
            or secrets.token_hex(32)
        ),
        DATASET_PATH=os.environ.get(
            "TICKETMIND_DATASET", str(ROOT / "customer_support_tickets.csv")
        ),
        ITEMS_PER_PAGE=20,
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,
        STATE_DATABASE=str(ROOT / "instance" / "ticketmind.sqlite3"),
        MODEL_ARTIFACTS=str(ROOT / "instance" / "models"),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
    )
    if test_config:
        app.config.update(test_config)

    models: dict[str, Any] = {
        "estimators": None,
        "reports": None,
        "fingerprint": None,
    }
    store = StateStore(app.config["STATE_DATABASE"], app.config["MODEL_ARTIFACTS"])
    app.extensions["ticketmind_state"] = store
    dataset_cache: dict[str, Any] = {"fingerprint": None, "frame": None}
    intelligence_cache: dict[str, Any] = {
        "fingerprint": None,
        "text_index": None,
        "clusters": None,
        "radar": None,
    }
    research_cache: dict[str, Any] = {
        "fingerprint": None,
        "temporal": None,
    }

    @app.context_processor
    def inject_adaptive_workspace():
        csrf_token = session.get("adaptive_csrf_token")
        if csrf_token is None:
            csrf_token = secrets.token_urlsafe(32)
            session["adaptive_csrf_token"] = csrf_token
        settings = store.adaptive_settings()
        return {
            "adaptive_csrf_token": csrf_token,
            "workspace_ai_mode": settings["mode"],
            "ai_mode_details": MODE_DETAILS,
        }

    def get_data():
        path = Path(app.config["DATASET_PATH"]).resolve()
        try:
            stat = path.stat()
        except OSError as error:
            raise DatasetError(f"Dataset file is not accessible: {error}") from error
        try:
            fingerprint = (str(path), stat.st_mtime_ns, stat.st_size)
            if dataset_cache["fingerprint"] != fingerprint:
                dataset_cache["frame"] = load_dataset(path)
                dataset_cache["fingerprint"] = fingerprint
            return dataset_cache["frame"]
        except DatasetError:
            app.logger.exception("Unable to load configured ticket dataset")
            raise

    def get_intelligence_index(frame: pd.DataFrame):
        path = Path(app.config["DATASET_PATH"]).resolve()
        stat = path.stat()
        fingerprint = (str(path), stat.st_mtime_ns, stat.st_size)
        if intelligence_cache["fingerprint"] != fingerprint:
            from intelligence import TextIndex

            intelligence_cache.update(
                {
                    "fingerprint": fingerprint,
                    "text_index": TextIndex(frame),
                    "clusters": None,
                    "radar": None,
                }
            )
        return intelligence_cache["text_index"]

    def get_temporal_study(frame: pd.DataFrame):
        path = Path(app.config["DATASET_PATH"]).resolve()
        stat = path.stat()
        fingerprint = (str(path), stat.st_mtime_ns, stat.st_size)
        if research_cache["fingerprint"] != fingerprint:
            research_cache["fingerprint"] = fingerprint
            research_cache["temporal"] = temporal_fingerprint_study(frame)
        return research_cache["temporal"]

    def ensure_models(force: bool = False):
        frame = get_data()
        path = Path(app.config["DATASET_PATH"]).resolve()
        stat = path.stat()
        fingerprint = (str(path), stat.st_mtime_ns, stat.st_size)
        if not force and models["fingerprint"] == fingerprint:
            return
        if not force and models["fingerprint"] is None:
            active_version = next(
                (item for item in store.model_versions() if item["active"]),
                None,
            )
            if active_version:
                artifact_path = Path(active_version["artifact_path"]).resolve()
                artifact_root = Path(app.config["MODEL_ARTIFACTS"]).resolve()
                if artifact_path.is_relative_to(artifact_root) and artifact_path.is_file():
                    artifact = joblib.load(artifact_path)
                    if artifact.get("dataset_fingerprint") == fingerprint:
                        models["estimators"] = artifact["estimators"]
                        models["reports"] = artifact["reports"]
                        models["fingerprint"] = fingerprint
                        return
        redacted_counts: dict[str, int] = {}
        for column in ("Ticket Subject", "Ticket Description"):
            for value in frame[column].dropna():
                _, counts = redact_with_audit(value)
                for category, count in counts.items():
                    redacted_counts[category] = (
                        redacted_counts.get(category, 0) + count
                    )
        store.record_redactions(redacted_counts)
        estimators, reports = train_baselines(frame, store.corrections())
        if not estimators:
            raise ValueError(
                "No target has enough labeled examples to train a model. "
                "Add reviewed corrections or provide additional labels."
            )
        artifact_path = (
            Path(app.config["MODEL_ARTIFACTS"]) / f"model-{uuid4().hex}.joblib"
        )
        joblib.dump(
            {
                "estimators": estimators,
                "reports": reports,
                "dataset_fingerprint": fingerprint,
            },
            artifact_path,
        )
        store.add_model_version(
            artifact_path,
            reports,
            sum(report["training_rows"] for report in reports.values()),
        )
        models["estimators"], models["reports"] = estimators, reports
        models["fingerprint"] = fingerprint

    def record_redactions(*texts: str) -> None:
        totals: dict[str, int] = {}
        for text in texts:
            _, counts = redact_with_audit(text)
            for category, count in counts.items():
                totals[category] = totals.get(category, 0) + count
        store.record_redactions(totals)

    def csrf_is_valid() -> bool:
        expected = session.get("adaptive_csrf_token", "")
        supplied = request.headers.get("X-CSRF-Token", "")
        return bool(expected and supplied and secrets.compare_digest(expected, supplied))

    def json_object() -> dict[str, Any] | None:
        payload = request.get_json(silent=True)
        return payload if isinstance(payload, dict) else None

    def ticket_row(frame: pd.DataFrame, ticket_id: str) -> pd.Series | None:
        matches = frame[frame["Ticket ID"].astype(str) == ticket_id]
        return None if matches.empty else matches.iloc[0]

    def effective_ticket_mode(ticket_id: str) -> str:
        return store.ticket_ai_mode(ticket_id) or store.adaptive_settings()["mode"]

    @app.errorhandler(DatasetError)
    def handle_dataset_error(error: DatasetError):
        app.logger.error("Dataset configuration error: %s", error)
        return render_template("dataset_error.html", message=str(error)), 503

    @app.route("/")
    def dashboard():
        frame = get_data()
        profile = dataset_profile(frame)
        statuses = frame["Ticket Status"].fillna("").astype(str).str.strip()
        priorities = frame["Ticket Priority"].fillna("").astype(str).str.strip()
        open_mask = statuses.str.casefold().eq("open")
        active_mask = statuses.str.casefold().isin(
            {"open", "pending customer response"}
        )
        urgent_mask = priorities.str.casefold().isin({"critical", "high"})
        attention_frame = frame.loc[open_mask & urgent_mask].copy()
        attention_frame["_priority_order"] = (
            attention_frame["Ticket Priority"]
            .fillna("")
            .astype(str)
            .str.casefold()
            .map({"critical": 0, "high": 1})
        )
        attention_frame = attention_frame.sort_values(
            ["_priority_order", "Ticket ID"], kind="stable"
        ).head(6)
        attention_subjects = [
            "" if pd.isna(value) else str(value)
            for value in attention_frame["Ticket Subject"]
        ]
        record_redactions(*attention_subjects)
        attention_tickets = []
        for (_, ticket), subject_text in zip(
            attention_frame.iterrows(), attention_subjects
        ):
            priority = str(ticket["Ticket Priority"])
            type_value = ticket["Ticket Type"]
            channel_value = ticket["Ticket Channel"]
            attention_tickets.append(
                {
                    "id": str(ticket["Ticket ID"]),
                    "subject": redact_sensitive_text(subject_text)
                    or "No subject provided",
                    "priority": priority,
                    "ticket_type": (
                        "Uncategorized" if pd.isna(type_value) else str(type_value)
                    ),
                    "channel": (
                        "Unknown channel" if pd.isna(channel_value) else str(channel_value)
                    ),
                    "next_step": (
                        "Review impact and follow your incident escalation process"
                        if priority.casefold() == "critical"
                        else "Confirm affected users and the next customer update"
                    ),
                }
            )
        active_count = int(active_mask.sum())
        return render_template(
            "dashboard.html",
            profile=profile,
            active_count=active_count,
            open_count=int(open_mask.sum()),
            urgent_open_count=int((open_mask & urgent_mask).sum()),
            waiting_count=int(
                statuses.str.casefold().eq("pending customer response").sum()
            ),
            active_percent=(
                round(active_count / len(frame) * 100) if len(frame) else 0
            ),
            attention_tickets=attention_tickets,
        )

    @app.route("/adaptive-ai")
    def adaptive_ai_page():
        settings = store.adaptive_settings()
        workspace_activity = store.ai_activity("__workspace__", limit=15)
        return render_template(
            "adaptive_ai.html",
            modes=MODE_DETAILS,
            settings=settings,
            workspace_activity=workspace_activity,
        )

    @app.route("/api/adaptive-ai/mode", methods=["POST"])
    def set_workspace_ai_mode():
        if not csrf_is_valid():
            return jsonify(error="Your session token expired. Refresh the page and try again."), 403
        payload = json_object()
        if payload is None:
            return jsonify(error="A JSON object is required."), 400
        mode = payload.get("mode")
        if not isinstance(mode, str) or mode not in {
            "manual",
            "copilot",
            "selective",
            "autonomous",
        }:
            return jsonify(error="Choose a valid workspace mode. Human takeover is ticket-specific."), 400
        store.set_workspace_ai_mode(mode)
        activity = store.add_ai_activity(
            "__workspace__",
            mode,
            "workspace_mode_changed",
            "saved",
            {"mode_label": MODE_DETAILS[mode]["label"]},
            request.headers.get("Idempotency-Key") or uuid4().hex,
        )
        return jsonify(
            mode=mode,
            label=MODE_DETAILS[mode]["label"],
            activity_id=activity["id"],
        )

    @app.route("/api/adaptive-ai/policy", methods=["POST"])
    def set_workspace_ai_policy():
        if not csrf_is_valid():
            return jsonify(error="Your session token expired. Refresh the page and try again."), 403
        payload = json_object()
        if payload is None:
            return jsonify(error="A JSON object is required."), 400
        try:
            threshold = float(payload.get("confidence_threshold"))
        except (TypeError, ValueError):
            return jsonify(error="Enter a numeric score threshold between 0.50 and 0.99."), 400
        if not math.isfinite(threshold) or not 0.50 <= threshold <= 0.99:
            return jsonify(error="Score threshold must be between 0.50 and 0.99."), 400
        enabled_actions = payload.get("enabled_actions")
        if not isinstance(enabled_actions, list) or any(
            not isinstance(action, str) or action not in {"category", "priority"}
            for action in enabled_actions
        ):
            return jsonify(error="Only category and priority actions can be enabled."), 400
        policy = {
            "confidence_threshold": threshold,
            "enabled_actions": sorted(set(enabled_actions)),
        }
        store.set_adaptive_policy(policy)
        settings = store.adaptive_settings()
        store.add_ai_activity(
            "__workspace__",
            settings["mode"],
            "workspace_policy_changed",
            "saved",
            policy,
            request.headers.get("Idempotency-Key") or uuid4().hex,
        )
        return jsonify(
            policy=policy,
            note="Scores are uncalibrated model outputs, not calibrated probabilities.",
        )

    @app.route("/api/tickets/<ticket_id>/adaptive-mode", methods=["POST"])
    def set_ticket_adaptive_mode(ticket_id: str):
        if not csrf_is_valid():
            return jsonify(error="Your session token expired. Refresh the page and try again."), 403
        frame = get_data()
        if ticket_row(frame, ticket_id) is None:
            return jsonify(error="Ticket was not found."), 404
        payload = json_object()
        if payload is None:
            return jsonify(error="A JSON object is required."), 400
        mode = payload.get("mode")
        if not isinstance(mode, str) or (
            mode != "workspace" and mode not in MODE_DETAILS
        ):
            return jsonify(error="Choose a valid ticket mode."), 400
        if mode == "workspace":
            store.clear_ticket_ai_mode(ticket_id)
            effective_mode = store.adaptive_settings()["mode"]
        else:
            store.set_ticket_ai_mode(ticket_id, mode)
            effective_mode = mode
        event_type = (
            "human_takeover" if effective_mode == "takeover" else "ticket_mode_changed"
        )
        store.add_ai_activity(
            ticket_id,
            effective_mode,
            event_type,
            "paused" if effective_mode == "takeover" else "saved",
            {"mode_label": MODE_DETAILS[effective_mode]["label"]},
            request.headers.get("Idempotency-Key") or uuid4().hex,
        )
        return jsonify(
            ticket_id=ticket_id,
            mode=effective_mode,
            label=MODE_DETAILS[effective_mode]["label"],
            message=(
                "AI processing is paused for this ticket."
                if effective_mode == "takeover"
                else "Ticket mode saved."
            ),
        )

    @app.route("/api/tickets/<ticket_id>/ai/recommend", methods=["POST"])
    def recommend_ticket_actions(ticket_id: str):
        if not csrf_is_valid():
            return jsonify(error="Your session token expired. Refresh the page and try again."), 403
        if json_object() is None:
            return jsonify(error="A JSON object is required."), 400
        frame = get_data()
        row = ticket_row(frame, ticket_id)
        if row is None:
            return jsonify(error="Ticket was not found."), 404
        mode = effective_ticket_mode(ticket_id)
        if mode in {"manual", "takeover"}:
            return jsonify(
                error=f"AI actions are paused in {MODE_DETAILS[mode]['label']} mode."
            ), 409
        idempotency_key = request.headers.get("Idempotency-Key", "")
        if not 8 <= len(idempotency_key) <= 100:
            return jsonify(error="A valid Idempotency-Key header is required."), 400
        try:
            ensure_models()
        except ValueError as error:
            app.logger.exception("Could not prepare models for adaptive ticket workflow")
            return jsonify(error=str(error)), 503
        subject = "" if pd.isna(row["Ticket Subject"]) else str(row["Ticket Subject"])
        description = (
            "" if pd.isna(row["Ticket Description"]) else str(row["Ticket Description"])
        )
        record_redactions(subject, description)
        safe_subject = redact_sensitive_text(subject)
        safe_description = redact_sensitive_text(description)
        safe_text = f"{safe_subject}\n{safe_description}".strip()
        predictions = predict_ticket(models["estimators"], safe_subject, safe_description)
        settings = store.adaptive_settings()
        workflow = evaluate_ticket_workflow(
            mode,
            predictions,
            settings["policy"],
            safe_text,
        )
        similar_cases = nearest_tickets(
            frame,
            safe_text,
            limit=3,
            exclude_ticket_id=ticket_id,
            resolved_only=True,
            index=get_intelligence_index(frame),
        )
        details = {
            "predictions": predictions,
            "workflow": workflow,
            "similar_cases": similar_cases,
            "extractive_preview": (
                safe_description.splitlines()[0].split(".")[0].strip()[:240]
                if safe_description.strip()
                else ""
            ),
            "score_note": "Model scores are uncalibrated and should not be read as probabilities.",
        }
        if mode == "copilot":
            activity_status = "pending_approval"
            event_type = "recommendation"
        else:
            activity_status = workflow["status"]
            event_type = "automation_run"
        activity = store.add_ticket_workflow_activity(
            ticket_id,
            mode,
            event_type,
            activity_status,
            details,
            idempotency_key,
        )
        if activity is None:
            return jsonify(
                error="The ticket mode changed while the workflow was running. No action was applied."
            ), 409
        return jsonify(
            activity=activity,
            mode=mode,
            mode_label=MODE_DETAILS[mode]["label"],
            workflow=workflow,
            predictions=predictions,
            similar_cases=similar_cases,
        )

    @app.route(
        "/api/tickets/<ticket_id>/ai/recommendations/<int:activity_id>/<resolution>",
        methods=["POST"],
    )
    def resolve_ticket_recommendation(ticket_id: str, activity_id: int, resolution: str):
        if not csrf_is_valid():
            return jsonify(error="Your session token expired. Refresh the page and try again."), 403
        payload = json_object()
        if payload is None:
            return jsonify(error="A JSON object is required."), 400
        if resolution not in {"approve", "reject"}:
            return jsonify(error="Choose approve or reject."), 400
        activity = next(
            (item for item in store.ai_activity(ticket_id) if item["id"] == activity_id),
            None,
        )
        if activity is None or activity["status"] != "pending_approval":
            return jsonify(error="This recommendation is no longer awaiting approval."), 409
        overlay: dict[str, str] = {}
        if resolution == "approve":
            predictions = activity["details"].get("predictions", {})
            requested_values = payload.get("values", {})
            if not isinstance(requested_values, dict) or any(
                key not in {"Ticket Type", "Ticket Priority"}
                or not isinstance(value, str)
                for key, value in requested_values.items()
            ):
                return jsonify(error="Edited values must contain valid category or priority labels."), 400
            valid_values = {
                "Ticket Type": set(get_data()["Ticket Type"].dropna().astype(str)),
                "Ticket Priority": set(get_data()["Ticket Priority"].dropna().astype(str)),
            }
            for target in ("Ticket Type", "Ticket Priority"):
                prediction = predictions.get(target)
                value = requested_values.get(
                    target, prediction.get("label") if prediction else None
                )
                if value is not None:
                    if value not in valid_values[target]:
                        return jsonify(error=f"Choose an observed {target} label."), 400
                    overlay[target] = value
            if not overlay:
                return jsonify(error="The recommendation did not contain a valid dataset label."), 409
        result = store.resolve_ai_activity(
            activity_id,
            ticket_id,
            "approved" if resolution == "approve" else "rejected",
            overlay if resolution == "approve" else None,
            {"overlay": overlay} if resolution == "approve" else None,
        )
        if result is None:
            return jsonify(error="This recommendation was already resolved."), 409
        return jsonify(
            activity=result,
            message=(
                "Suggestions saved to the local Opslyra overlay; the source CSV is unchanged."
                if resolution == "approve"
                else "Recommendation rejected and retained in the activity history."
            ),
        )

    @app.route("/tickets")
    def tickets():
        frame = get_data()
        search = request.args.get("q", "").strip()
        ticket_type = request.args.get("type", "").strip()
        priority = request.args.get("priority", "").strip()
        status = request.args.get("status", "").strip()

        filtered = frame.copy()
        for column, value in (
            ("Ticket Type", ticket_type),
            ("Ticket Priority", priority),
            ("Ticket Status", status),
        ):
            if value:
                filtered = filtered[filtered[column].astype("string").str.casefold() == value.casefold()]
        if search:
            searchable = (
                filtered["Ticket ID"].fillna("").astype(str)
                + " "
                + filtered["Ticket Subject"].fillna("").astype(str)
                + " "
                + filtered["Ticket Description"].fillna("").astype(str)
            )
            filtered = filtered[searchable.str.contains(search, case=False, regex=False, na=False)]

        page = max(1, request.args.get("page", 1, type=int))
        per_page = app.config["ITEMS_PER_PAGE"]
        total = len(filtered)
        pages = max(1, math.ceil(total / per_page))
        page = min(page, pages)
        start = (page - 1) * per_page
        records = filtered.iloc[start : start + per_page].to_dict(orient="records")
        records = [
            {
                key: ("" if pd.isna(value) else value)
                for key, value in record.items()
            }
            for record in records
        ]
        for record in records:
            record_redactions(
                record["Ticket Subject"],
                record["Ticket Description"],
            )
            record["Ticket Subject"] = redact_sensitive_text(record["Ticket Subject"])
            record["Ticket Description"] = redact_sensitive_text(record["Ticket Description"])

        options = {
            column: sorted(frame[column].dropna().astype(str).unique().tolist())
            for column in ("Ticket Type", "Ticket Priority", "Ticket Status")
        }
        return render_template(
            "tickets.html",
            tickets=records,
            total=total,
            page=page,
            pages=pages,
            query=search,
            selected_type=ticket_type,
            selected_priority=priority,
            selected_status=status,
            options=options,
        )

    @app.route("/tickets/<ticket_id>")
    def ticket_detail(ticket_id: str):
        frame = get_data()
        matches = frame[frame["Ticket ID"].astype(str) == ticket_id]
        if matches.empty:
            abort(404)
        record = matches.iloc[0].to_dict()
        record = {
            key: ("" if pd.isna(value) else value)
            for key, value in record.items()
        }
        record_redactions(
            record["Ticket Subject"],
            record["Ticket Description"],
            record["Resolution"],
        )
        for column in ("Ticket Subject", "Ticket Description", "Resolution"):
            record[column] = redact_sensitive_text(record[column])
        return render_template(
            "ticket_detail.html",
            ticket=record,
            ai_mode=effective_ticket_mode(ticket_id),
            ai_mode_override=store.ticket_ai_mode(ticket_id),
            ai_override=store.ticket_ai_override(ticket_id),
            ai_activity=store.ai_activity(ticket_id),
            ticket_types=sorted(frame["Ticket Type"].dropna().astype(str).unique()),
            ticket_priorities=sorted(
                frame["Ticket Priority"].dropna().astype(str).unique()
            ),
        )

    @app.route("/data-quality")
    def data_quality():
        profile = dataset_profile(get_data())
        return render_template("data_quality.html", profile=profile)

    @app.route("/models", methods=["GET", "POST"])
    def models_page():
        try:
            ensure_models(force=request.method == "POST")
        except ValueError as error:
            app.logger.exception("Could not train baseline models")
            flash(f"Model training failed: {error}", "error")
        return render_template(
            "models.html",
            reports=models["reports"],
            trained=models["estimators"] is not None,
        )

    @app.route("/predict", methods=["GET", "POST"])
    def predict():
        results = None
        subject = ""
        description = ""
        if request.method == "POST":
            subject = request.form.get("subject", "").strip()
            description = request.form.get("description", "").strip()
            if not subject and not description:
                flash("Enter a ticket subject or description to request a prediction.", "error")
            elif len(subject) > 500 or len(description) > 10_000:
                flash("Subject or description is longer than the supported limit.", "error")
            else:
                record_redactions(subject, description)
                try:
                    ensure_models()
                except ValueError as error:
                    app.logger.exception("Could not train models for prediction")
                    flash(f"Model training failed: {error}", "error")
                if models["estimators"]:
                    safe_subject = redact_sensitive_text(subject)
                    safe_description = redact_sensitive_text(description)
                    results = predict_ticket(
                        models["estimators"], safe_subject, safe_description
                    )
        return render_template(
            "predict.html",
            results=results,
            subject=redact_sensitive_text(subject),
            description=redact_sensitive_text(description),
        )

    @app.route("/intelligence", methods=["GET", "POST"])
    def intelligence_page():
        frame = get_data()
        feature_result: dict[str, Any] | None = None
        active_ticket_id = str(
            request.args.get("ticket_id")
            or (frame.iloc[0]["Ticket ID"] if not frame.empty else "")
        )
        if request.method == "POST":
            action = request.form.get("action", "")
            try:
                if action == "correction":
                    ticket_id = request.form.get("ticket_id", "").strip()
                    target = request.form.get("target", "")
                    label = request.form.get("corrected_label", "").strip()
                    if target not in ("Ticket Type", "Ticket Priority", "Team"):
                        raise ValueError("Choose a supported correction target.")
                    if not label or len(label) > 100:
                        raise ValueError("Enter a correction label up to 100 characters.")
                    matches = frame[frame["Ticket ID"].astype(str) == ticket_id]
                    if matches.empty:
                        raise ValueError("That ticket ID is not present in the dataset.")
                    record = matches.iloc[0]
                    text = (
                        f"{record['Ticket Subject'] or ''}\n"
                        f"{record['Ticket Description'] or ''}"
                    )
                    record_redactions(text)
                    original = (
                        str(record[target])
                        if target in record.index and pd.notna(record[target])
                        else None
                    )
                    information_score = 0.5
                    if target in (models["estimators"] or {}):
                        estimate = predict_ticket(
                            models["estimators"], record["Ticket Subject"], record["Ticket Description"]
                        ).get(target)
                        if estimate:
                            information_score = 1.0 - estimate["score"]
                    store.add_correction(
                        ticket_id,
                        target,
                        original,
                        label,
                        information_score,
                    )
                    flash(
                        "Correction saved as a reviewed example. Train a new version to include it.",
                        "success",
                    )
                    return redirect(url_for("intelligence_page"))
                if action == "train":
                    ensure_models(force=True)
                    flash("A new model version was trained and marked active.", "success")
                    return redirect(url_for("intelligence_page"))
                if action == "rollback":
                    version_number = request.form.get("version", type=int)
                    version = next(
                        (
                            item
                            for item in store.model_versions()
                            if item["version"] == version_number
                        ),
                        None,
                    )
                    if version is None:
                        raise ValueError("That model version does not exist.")
                    artifact_path = Path(version["artifact_path"]).resolve()
                    artifact_root = Path(app.config["MODEL_ARTIFACTS"]).resolve()
                    if not artifact_path.is_relative_to(artifact_root) or not artifact_path.is_file():
                        raise ValueError("The selected model artifact is unavailable.")
                    artifact = joblib.load(artifact_path)
                    store.activate_model_version(version_number)
                    models["estimators"] = artifact["estimators"]
                    models["reports"] = artifact["reports"]
                    current_path = Path(app.config["DATASET_PATH"]).resolve()
                    current_stat = current_path.stat()
                    models["fingerprint"] = (
                        str(current_path),
                        current_stat.st_mtime_ns,
                        current_stat.st_size,
                    )
                    flash(f"Rolled back to model version {version_number}.", "success")
                    return redirect(url_for("intelligence_page"))
                if action == "privacy_scan":
                    original_text = request.form.get("privacy_text", "")
                    if len(original_text) > 10_000:
                        raise ValueError("Privacy scan text exceeds 10,000 characters.")
                    redacted_text, counts = redact_with_audit(original_text)
                    record_redactions(original_text)
                    feature_result = {
                        "action": action,
                        "data": {"text": redacted_text, "counts": counts},
                    }
                elif action in ("novelty", "counterfactual", "memory"):
                    subject = request.form.get("subject", "").strip()
                    description = request.form.get("description", "").strip()
                    if not subject and not description:
                        raise ValueError("Enter a subject or description for this workflow.")
                    if len(subject) > 500 or len(description) > 10_000:
                        raise ValueError("Ticket text exceeds the supported limits.")
                    record_redactions(subject, description)
                    safe_text = redact_sensitive_text(f"{subject}\n{description}")
                    if action == "novelty":
                        text_index = get_intelligence_index(frame)
                        novelty = novelty_assessment(frame, safe_text, text_index)
                        store.record_novelty(
                            novelty["status"],
                            novelty["nearest_similarity"],
                            novelty["local_density"],
                        )
                        feature_result = {
                            "action": action,
                            "data": novelty,
                        }
                    elif action == "counterfactual":
                        ensure_models()
                        feature_result = {
                            "action": action,
                            "data": counterfactual_comparison(
                                models["estimators"], subject, description
                            ),
                        }
                    else:
                        text_index = get_intelligence_index(frame)
                        feature_result = {
                            "action": action,
                            "data": nearest_tickets(
                                frame,
                                safe_text,
                                limit=5,
                                resolved_only=True,
                                index=text_index,
                            ),
                        }
                elif action == "sla":
                    feature_result = {
                        "action": action,
                        "data": sla_risk_sandbox(
                            request.form.get("priority", "Medium"),
                            float(request.form.get("elapsed_hours", 0)),
                            float(request.form.get("target_hours", 8)),
                            int(request.form.get("queue_load", 0)),
                        ),
                    }
                elif action == "routing":
                    capacity_factor = float(request.form.get("capacity_factor", 1))
                    if not 0.5 <= capacity_factor <= 2:
                        raise ValueError("Capacity factor must be between 0.5 and 2.")
                    feature_result = {
                        "action": action,
                        "data": routing_simulation(frame, capacity_factor),
                    }
            except (ValueError, TypeError, OSError, KeyError) as error:
                app.logger.exception("Intelligence workflow failed")
                flash(str(error), "error")

        profile = dataset_profile(frame)
        if active_ticket_id not in set(frame["Ticket ID"].astype(str)):
            active_ticket_id = str(frame.iloc[0]["Ticket ID"])
        text_index = get_intelligence_index(frame)
        graph = relationship_graph(frame, active_ticket_id, index=text_index)
        resolved_memory = nearest_tickets(
            frame,
            (
                f"{graph['ticket']['subject']}\n{graph['ticket']['product']}"
                if graph["ticket"]
                else ""
            ),
            limit=5,
            exclude_ticket_id=active_ticket_id,
            resolved_only=True,
            index=text_index,
        )
        versions = store.model_versions()
        active_version = next((version for version in versions if version["active"]), None)
        previous_version = next(
            (version for version in versions if not version["active"]),
            None,
        )
        version_deltas = {}
        if active_version and previous_version:
            for target, current in active_version["metrics"].items():
                previous = previous_version["metrics"].get(target)
                if (
                    previous
                    and current.get("accuracy") is not None
                    and previous.get("accuracy") is not None
                ):
                    version_deltas[target] = (
                        current["accuracy"] - previous["accuracy"]
                    )
        performance_regression = any(
            delta < -0.05 for delta in version_deltas.values()
        )
        drift = drift_summary(frame, models["reports"])
        drift["performance_regression"] = performance_regression
        drift["review_required"] = (
            drift["review_required"] or performance_regression
        )
        corrections = store.corrections()
        if intelligence_cache["radar"] is None:
            intelligence_cache["radar"] = recurrence_radar(frame)
        if intelligence_cache["clusters"] is None:
            intelligence_cache["clusters"] = issue_clusters(frame)
        radar = intelligence_cache["radar"]
        return render_template(
            "intelligence.html",
            profile=profile,
            graph=graph,
            clusters=intelligence_cache["clusters"],
            radar=radar,
            drift=drift,
            version_deltas=version_deltas,
            novelty_summary=store.novelty_summary(),
            versions=versions,
            corrections=corrections,
            audit=store.redaction_summary(),
            feature_result=feature_result,
            resolved_memory=resolved_memory,
            active_ticket_id=active_ticket_id,
            ticket_options=frame.head(100)[["Ticket ID", "Ticket Subject"]].to_dict("records"),
            model_ready=models["estimators"] is not None,
        )

    @app.route("/research", methods=["GET", "POST"])
    def research_page():
        frame = get_data()
        form_result: dict[str, Any] | None = None
        queue_weights = {
            "uncertainty": 0.35,
            "novelty": 0.30,
            "impact": 0.25,
            "diversity": 0.10,
        }
        if request.method == "POST":
            try:
                action = request.form.get("action", "")
                if action == "queue":
                    raw_weights = {
                        key: float(request.form.get(key, default))
                        for key, default in queue_weights.items()
                    }
                    if any(value < 0 or value > 1 for value in raw_weights.values()):
                        raise ValueError("Sampling weights must be between 0 and 1.")
                    total_weight = sum(raw_weights.values())
                    if total_weight <= 0:
                        raise ValueError("At least one sampling weight must be positive.")
                    queue_weights = {
                        key: value / total_weight for key, value in raw_weights.items()
                    }
                    ensure_models()
                    form_result = {
                        "action": action,
                        "queue": active_learning_queue(
                            frame, models["estimators"], queue_weights, limit=12
                        ),
                    }
                elif action == "benchmark":
                    budget = request.form.get("budget", default=60, type=int)
                    if budget < 4 or budget > 120:
                        raise ValueError("Annotation budget must be between 4 and 120.")
                    form_result = {
                        "action": action,
                        "data": active_learning_benchmark(
                            frame, budget=budget, seed=42
                        ),
                    }
                elif action == "routing-study":
                    ensure_models()
                    defaults = {
                        "Technical Support": (30, 80),
                        "Billing & Payments": (25, 70),
                        "Retention": (18, 50),
                    }
                    workload = {}
                    capacity = {}
                    for index, team in enumerate(defaults):
                        workload[team] = request.form.get(
                            f"workload_{index}", defaults[team][0], type=int
                        )
                        capacity[team] = request.form.get(
                            f"capacity_{index}", defaults[team][1], type=int
                        )
                        if workload[team] < 0 or capacity[team] < 1:
                            raise ValueError("Workload cannot be negative and capacity must be positive.")
                    remaining_sla_hours = request.form.get(
                        "remaining_sla_hours", 8, type=float
                    )
                    if remaining_sla_hours <= 0 or not math.isfinite(remaining_sla_hours):
                        raise ValueError("Remaining SLA hours must be positive and finite.")
                    subject = request.form.get("subject", "").strip()
                    description = request.form.get("description", "").strip()
                    if not subject and not description:
                        raise ValueError("Enter a ticket subject or description.")
                    if len(subject) > 500 or len(description) > 10_000:
                        raise ValueError("Ticket text exceeds the supported limits.")
                    record_redactions(subject, description)
                    form_result = {
                        "action": action,
                        "data": counterfactual_routing_study(
                            models["estimators"],
                            redact_sensitive_text(subject),
                            redact_sensitive_text(description),
                            workload,
                            capacity,
                            remaining_sla_hours,
                        ),
                    }
            except (ValueError, TypeError) as error:
                app.logger.exception("Research workflow failed")
                flash(str(error), "error")

        try:
            ensure_models()
        except ValueError as error:
            app.logger.exception("Could not train models for research workflows")
            flash(str(error), "error")
        queue = (
            form_result["queue"]
            if form_result and form_result["action"] == "queue"
            else active_learning_queue(frame, models["estimators"] or {}, queue_weights)
        )
        return render_template(
            "research.html",
            active_learning=(
                form_result["data"]
                if form_result and form_result["action"] == "benchmark"
                else None
            ),
            routing_study=(
                form_result["data"]
                if form_result and form_result["action"] == "routing-study"
                else None
            ),
            queue=queue,
            queue_weights=queue_weights,
            temporal=get_temporal_study(frame),
            profile=dataset_profile(frame),
        )

    return app


app = create_app()

if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG", "").casefold() == "true",
    )
