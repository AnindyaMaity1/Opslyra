from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StateStore:
    def __init__(self, database_path: str | Path, artifact_directory: str | Path):
        self.database_path = Path(database_path)
        self.artifact_directory = Path(artifact_directory)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_directory.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id TEXT NOT NULL,
                    target TEXT NOT NULL,
                    original_label TEXT,
                    corrected_label TEXT NOT NULL,
                    informativeness REAL NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS model_versions (
                    version INTEGER PRIMARY KEY AUTOINCREMENT,
                    artifact_path TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    training_examples INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 0
                );
                CREATE UNIQUE INDEX IF NOT EXISTS correction_ticket_target
                    ON corrections (ticket_id, target);
                CREATE TABLE IF NOT EXISTS redaction_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    counts_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS novelty_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    nearest_similarity REAL NOT NULL,
                    local_density REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ai_workspace_settings (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    mode TEXT NOT NULL DEFAULT 'manual',
                    policy_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ai_ticket_modes (
                    ticket_id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ai_ticket_overrides (
                    ticket_id TEXT PRIMARY KEY,
                    ticket_type TEXT,
                    ticket_priority TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ai_activity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    idempotency_key TEXT UNIQUE,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ai_activity_ticket_created
                    ON ai_activity (ticket_id, created_at DESC);
                """
            )

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def add_correction(
        self,
        ticket_id: str,
        target: str,
        original_label: str | None,
        corrected_label: str,
        informativeness: float,
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO corrections
                    (ticket_id, target, original_label, corrected_label, informativeness, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticket_id, target) DO UPDATE SET
                    original_label = excluded.original_label,
                    corrected_label = excluded.corrected_label,
                    informativeness = excluded.informativeness,
                    created_at = excluded.created_at
                """,
                (
                    ticket_id,
                    target,
                    original_label,
                    corrected_label,
                    informativeness,
                    self.now(),
                ),
            )
            return int(cursor.lastrowid)

    def corrections(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM corrections
                ORDER BY informativeness DESC, created_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def add_model_version(
        self,
        artifact_path: str | Path,
        metrics: dict[str, Any],
        training_examples: int,
    ) -> int:
        with self.connect() as connection:
            connection.execute("UPDATE model_versions SET active = 0")
            cursor = connection.execute(
                """
                INSERT INTO model_versions
                    (artifact_path, metrics_json, training_examples, created_at, active)
                VALUES (?, ?, ?, ?, 1)
                """,
                (
                    str(Path(artifact_path).resolve()),
                    json.dumps(metrics),
                    training_examples,
                    self.now(),
                ),
            )
            return int(cursor.lastrowid)

    def model_versions(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM model_versions ORDER BY version DESC"
            ).fetchall()
        versions = []
        for row in rows:
            version = dict(row)
            version["metrics"] = json.loads(version.pop("metrics_json"))
            versions.append(version)
        return versions

    def activate_model_version(self, version: int) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM model_versions WHERE version = ?", (version,)
            ).fetchone()
            if row is None:
                return None
            connection.execute("UPDATE model_versions SET active = 0")
            connection.execute(
                "UPDATE model_versions SET active = 1 WHERE version = ?", (version,)
            )
        result = dict(row)
        result["metrics"] = json.loads(result.pop("metrics_json"))
        return result

    def record_redactions(self, counts: dict[str, int]) -> None:
        if not any(counts.values()):
            return
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO redaction_audit (created_at, counts_json) VALUES (?, ?)",
                (self.now(), json.dumps(counts)),
            )

    def redaction_summary(self) -> dict[str, Any]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT counts_json FROM redaction_audit"
            ).fetchall()
        totals: dict[str, int] = {}
        for row in rows:
            for category, count in json.loads(row["counts_json"]).items():
                totals[category] = totals.get(category, 0) + int(count)
        return {"events": len(rows), "counts": totals}

    def record_novelty(self, status: str, similarity: float, density: float) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO novelty_events
                    (created_at, status, nearest_similarity, local_density)
                VALUES (?, ?, ?, ?)
                """,
                (self.now(), status, similarity, density),
            )

    def novelty_summary(self) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS total,
                       SUM(CASE WHEN status = 'unfamiliar' THEN 1 ELSE 0 END) AS unknown
                FROM novelty_events
                """
            ).fetchone()
        total = int(row["total"] or 0)
        unknown = int(row["unknown"] or 0)
        return {
            "total": total,
            "unknown": unknown,
            "unknown_rate": unknown / total if total else 0.0,
        }

    def adaptive_settings(self) -> dict[str, Any]:
        from adaptive import DEFAULT_POLICY

        with self.connect() as connection:
            row = connection.execute(
                "SELECT mode, policy_json FROM ai_workspace_settings WHERE id = 1"
            ).fetchone()
            if row is None:
                policy = DEFAULT_POLICY.copy()
                connection.execute(
                    """
                    INSERT INTO ai_workspace_settings (id, mode, policy_json, updated_at)
                    VALUES (1, 'manual', ?, ?)
                    """,
                    (json.dumps(policy), self.now()),
                )
                return {"mode": "manual", "policy": policy}
        policy = DEFAULT_POLICY.copy()
        policy.update(json.loads(row["policy_json"]))
        return {"mode": str(row["mode"]), "policy": policy}

    def set_workspace_ai_mode(self, mode: str) -> None:
        settings = self.adaptive_settings()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO ai_workspace_settings (id, mode, policy_json, updated_at)
                VALUES (1, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    mode = excluded.mode,
                    updated_at = excluded.updated_at
                """,
                (mode, json.dumps(settings["policy"]), self.now()),
            )

    def set_adaptive_policy(self, policy: dict[str, Any]) -> None:
        settings = self.adaptive_settings()
        merged = settings["policy"].copy()
        merged.update(policy)
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO ai_workspace_settings (id, mode, policy_json, updated_at)
                VALUES (1, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    policy_json = excluded.policy_json,
                    updated_at = excluded.updated_at
                """,
                (settings["mode"], json.dumps(merged), self.now()),
            )

    def ticket_ai_mode(self, ticket_id: str) -> str | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT mode FROM ai_ticket_modes WHERE ticket_id = ?", (ticket_id,)
            ).fetchone()
        return str(row["mode"]) if row else None

    def set_ticket_ai_mode(self, ticket_id: str, mode: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO ai_ticket_modes (ticket_id, mode, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(ticket_id) DO UPDATE SET
                    mode = excluded.mode,
                    updated_at = excluded.updated_at
                """,
                (ticket_id, mode, self.now()),
            )

    def clear_ticket_ai_mode(self, ticket_id: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "DELETE FROM ai_ticket_modes WHERE ticket_id = ?", (ticket_id,)
            )

    def ticket_ai_override(self, ticket_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT ticket_type, ticket_priority FROM ai_ticket_overrides WHERE ticket_id = ?",
                (ticket_id,),
            ).fetchone()
        return dict(row) if row else None

    def _write_ticket_override(
        self, connection: sqlite3.Connection, ticket_id: str, values: dict[str, str]
    ) -> None:
        current = connection.execute(
            "SELECT ticket_type, ticket_priority FROM ai_ticket_overrides WHERE ticket_id = ?",
            (ticket_id,),
        ).fetchone()
        merged = dict(current) if current else {
            "ticket_type": None,
            "ticket_priority": None,
        }
        if "Ticket Type" in values or "ticket_type" in values:
            merged["ticket_type"] = values.get("Ticket Type", values.get("ticket_type"))
        if "Ticket Priority" in values or "ticket_priority" in values:
            merged["ticket_priority"] = values.get(
                "Ticket Priority", values.get("ticket_priority")
            )
        connection.execute(
            """
            INSERT INTO ai_ticket_overrides
                (ticket_id, ticket_type, ticket_priority, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(ticket_id) DO UPDATE SET
                ticket_type = excluded.ticket_type,
                ticket_priority = excluded.ticket_priority,
                updated_at = excluded.updated_at
            """,
            (ticket_id, merged["ticket_type"], merged["ticket_priority"], self.now()),
        )

    def set_ticket_ai_override(self, ticket_id: str, values: dict[str, str]) -> None:
        with self.connect() as connection:
            self._write_ticket_override(connection, ticket_id, values)

    @staticmethod
    def _activity_record(row: sqlite3.Row) -> dict[str, Any]:
        activity = dict(row)
        activity["details"] = json.loads(activity.pop("details_json"))
        return activity

    def add_ai_activity(
        self,
        ticket_id: str,
        mode: str,
        event_type: str,
        status: str,
        details: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO ai_activity
                    (ticket_id, mode, event_type, status, details_json, idempotency_key, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(idempotency_key) DO NOTHING
                """,
                (
                    ticket_id,
                    mode,
                    event_type,
                    status,
                    json.dumps(details),
                    idempotency_key,
                    self.now(),
                ),
            )
            if cursor.rowcount:
                activity_id = int(cursor.lastrowid)
            else:
                row = connection.execute(
                    "SELECT id FROM ai_activity WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                activity_id = int(row["id"])
            row = connection.execute(
                "SELECT * FROM ai_activity WHERE id = ?", (activity_id,)
            ).fetchone()
        return self._activity_record(row)

    def add_ticket_workflow_activity(
        self,
        ticket_id: str,
        mode: str,
        event_type: str,
        status: str,
        details: dict[str, Any],
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM ai_activity WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                return self._activity_record(existing)
            ticket_mode = connection.execute(
                "SELECT mode FROM ai_ticket_modes WHERE ticket_id = ?", (ticket_id,)
            ).fetchone()
            if ticket_mode is not None:
                current_mode = str(ticket_mode["mode"])
            else:
                workspace = connection.execute(
                    "SELECT mode FROM ai_workspace_settings WHERE id = 1"
                ).fetchone()
                current_mode = str(workspace["mode"]) if workspace else "manual"
            if current_mode != mode:
                return None
            cursor = connection.execute(
                """
                INSERT INTO ai_activity
                    (ticket_id, mode, event_type, status, details_json, idempotency_key, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticket_id,
                    mode,
                    event_type,
                    status,
                    json.dumps(details),
                    idempotency_key,
                    self.now(),
                ),
            )
            if status == "applied_to_workspace_overlay":
                overlay = {
                    action["target"]: action["value"]
                    for action in details.get("workflow", {}).get("actions", [])
                }
                if overlay:
                    self._write_ticket_override(connection, ticket_id, overlay)
            row = connection.execute(
                "SELECT * FROM ai_activity WHERE id = ?", (int(cursor.lastrowid),)
            ).fetchone()
        return self._activity_record(row)

    def ai_activity(self, ticket_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM ai_activity
                WHERE ticket_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (ticket_id, limit),
            ).fetchall()
        return [self._activity_record(row) for row in rows]

    def resolve_ai_activity(
        self,
        activity_id: int,
        ticket_id: str,
        resolution: str,
        overlay: dict[str, str] | None = None,
        resolution_details: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if resolution not in {"approved", "rejected"}:
            raise ValueError("Unsupported activity resolution.")
        event_details = {"parent_activity_id": activity_id}
        if resolution_details:
            event_details.update(resolution_details)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM ai_activity
                WHERE id = ? AND ticket_id = ? AND event_type = 'recommendation'
                """,
                (activity_id, ticket_id),
            ).fetchone()
            if row is None or row["status"] != "pending_approval":
                return None
            connection.execute(
                "UPDATE ai_activity SET status = ? WHERE id = ?",
                (resolution, activity_id),
            )
            if resolution == "approved" and overlay:
                self._write_ticket_override(connection, ticket_id, overlay)
            connection.execute(
                """
                INSERT INTO ai_activity
                    (ticket_id, mode, event_type, status, details_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    ticket_id,
                    row["mode"],
                    f"recommendation_{resolution}",
                    resolution,
                    json.dumps(event_details),
                    self.now(),
                ),
            )
            updated = connection.execute(
                "SELECT * FROM ai_activity WHERE id = ?", (activity_id,)
            ).fetchone()
        return self._activity_record(updated)
