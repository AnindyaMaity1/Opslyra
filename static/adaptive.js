(() => {
  const csrfToken = document.querySelector('meta[name="adaptive-csrf-token"]')?.content;

  function requestId() {
    return crypto.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  async function sendJson(url, payload, idempotencyKey) {
    const response = await fetch(url, {
      method: "POST",
      credentials: "same-origin",
      headers: {
        "Content-Type": "application/json",
        "X-CSRF-Token": csrfToken || "",
        "Idempotency-Key": idempotencyKey || requestId(),
      },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || `Request failed (${response.status}).`);
    }
    return result;
  }

  const workspaceMode = document.querySelector("#workspace-ai-mode");
  workspaceMode?.addEventListener("change", async () => {
    const feedback = document.querySelector("#adaptive-mode-feedback");
    workspaceMode.disabled = true;
    if (feedback) feedback.textContent = "Saving workspace mode…";
    try {
      const result = await sendJson("/api/adaptive-ai/mode", { mode: workspaceMode.value });
      if (feedback) feedback.textContent = `${result.label} is now the workspace default.`;
      document.querySelectorAll(".adaptive-set-workspace").forEach((button) => {
        const active = button.dataset.mode === result.mode;
        button.disabled = active;
        button.textContent = active ? "Active workspace mode" : "Set workspace mode";
        button.classList.toggle("button-primary", !active);
        button.classList.toggle("button-secondary", active);
      });
      document.querySelectorAll(".adaptive-mode-card[data-mode-card]").forEach((card) => {
        const active = card.dataset.modeCard === result.mode;
        card.classList.toggle("is-current", active);
        const state = card.querySelector(".adaptive-mode-state");
        if (state) state.textContent = active ? "CURRENT" : "MODE";
      });
      const label = document.querySelector("#adaptive-current-label");
      if (label) label.textContent = result.label;
    } catch (error) {
      if (feedback) feedback.textContent = error.message;
      workspaceMode.value = workspaceMode.dataset.previous || "manual";
    } finally {
      workspaceMode.disabled = false;
      workspaceMode.dataset.previous = workspaceMode.value;
    }
  });
  if (workspaceMode) workspaceMode.dataset.previous = workspaceMode.value;

  document.querySelectorAll(".adaptive-set-workspace").forEach((button) => {
    button.addEventListener("click", async () => {
      const selector = document.querySelector("#workspace-ai-mode");
      if (selector) selector.value = button.dataset.mode;
      selector?.dispatchEvent(new Event("change"));
    });
  });

  document.querySelector("#adaptive-policy-form")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const submit = form.querySelector('button[type="submit"]');
    const feedback = document.querySelector("#adaptive-mode-feedback");
    const formData = new FormData(form);
    submit.disabled = true;
    if (feedback) feedback.textContent = "Saving automation policy…";
    try {
      const result = await sendJson("/api/adaptive-ai/policy", {
        enabled_actions: formData.getAll("enabled_actions"),
        selection_mode: formData.get("selection_mode"),
        target_coverage: formData.get("target_coverage"),
        target_risk: formData.get("target_risk"),
      });
      if (feedback) feedback.textContent = `Policy saved. ${result.note} Refreshing the selected operating points…`;
      window.location.reload();
    } catch (error) {
      if (feedback) feedback.textContent = error.message;
    } finally {
      submit.disabled = false;
    }
  });

  const policyPreviewButton = document.querySelector("#policy-preview-button");
  const policyPreviewTicket = document.querySelector("#policy-preview-ticket");
  const policyPreviewResult = document.querySelector("#policy-preview-result");
  const policyPreviewFeedback = document.querySelector("#policy-preview-feedback");
  policyPreviewButton?.addEventListener("click", async () => {
    const ticketId = policyPreviewTicket?.value;
    if (!ticketId) {
      if (policyPreviewFeedback) policyPreviewFeedback.textContent = "Choose a ticket to preview.";
      return;
    }
    policyPreviewButton.disabled = true;
    policyPreviewResult.hidden = true;
    if (policyPreviewFeedback) policyPreviewFeedback.textContent = "Calculating a side-effect-free policy preview…";
    try {
      const result = await sendJson(
        `/api/tickets/${encodeURIComponent(ticketId)}/ai/policy-preview`,
        {},
      );
      const heading = document.createElement("strong");
      heading.textContent = `Ticket #${result.ticket_id} · ${result.workflow.decision.replace(/_/g, " ")}`;
      const intro = document.createElement("p");
      intro.textContent = result.message;
      const rows = document.createElement("div");
      rows.className = "adaptive-preview-predictions";
      for (const [target, prediction] of Object.entries(result.predictions)) {
        const row = document.createElement("div");
        const label = document.createElement("span");
        label.textContent = target;
        const value = document.createElement("strong");
        value.textContent = prediction.label;
        const score = document.createElement("small");
        const selection = result.thresholds[target];
        const threshold = selection?.status === "ok"
          ? selection.threshold
          : null;
        const confidence = Number(prediction.score);
        score.textContent = !Number.isFinite(confidence)
          ? "Model score unavailable — send to human review"
          : threshold == null
            ? `Score ${(confidence * 100).toFixed(1)}%; validation threshold unavailable — human review`
            : `Score ${(confidence * 100).toFixed(1)}% · validation cut-off ${(threshold * 100).toFixed(1)}% · ${confidence >= threshold ? "within selected coverage" : "send to human review"}`;
        row.append(label, value, score);
        rows.append(row);
      }
      const note = document.createElement("small");
      note.textContent = `Validation operating point: ${result.policy.selection_mode === "target_risk" ? `target risk ${(result.policy.target_risk * 100).toFixed(0)}%` : `target coverage ${(result.policy.target_coverage * 100).toFixed(0)}%`}. Scores are uncalibrated. No ticket, overlay, audit event, or external action was changed.`;
      policyPreviewResult.replaceChildren(heading, intro, rows, note);
      policyPreviewResult.hidden = false;
      if (policyPreviewFeedback) policyPreviewFeedback.textContent = "Preview ready. This was a dry run only.";
    } catch (error) {
      policyPreviewResult.hidden = true;
      if (policyPreviewFeedback) policyPreviewFeedback.textContent = error.message;
    } finally {
      policyPreviewButton.disabled = false;
    }
  });

  const ticketPanel = document.querySelector(".adaptive-ticket-control");
  if (ticketPanel) {
    const ticketId = ticketPanel.dataset.ticketId;
    const modeSelector = document.querySelector("#ticket-ai-mode");
    const feedback = ticketPanel.querySelector(".adaptive-inline-feedback");
    const previewButton = ticketPanel.querySelector(".adaptive-preview-ticket");
    const runButton = ticketPanel.querySelector(".adaptive-run-ticket");
    const previewPanel = ticketPanel.querySelector(".adaptive-dry-run-preview");
    let previewId = null;
    modeSelector?.addEventListener("change", async () => {
      modeSelector.disabled = true;
      if (feedback) feedback.textContent = "Saving ticket mode…";
      try {
        const result = await sendJson(`/api/tickets/${encodeURIComponent(ticketId)}/adaptive-mode`, {
          mode: modeSelector.value,
        });
        if (feedback) feedback.textContent = result.message;
        window.location.reload();
      } catch (error) {
        if (feedback) feedback.textContent = error.message;
        modeSelector.disabled = false;
      }
    });

    previewButton?.addEventListener("click", async () => {
      previewButton.disabled = true;
      runButton.disabled = true;
      previewId = null;
      if (feedback) feedback.textContent = "Preparing a side-effect-free workflow preview…";
      try {
        const result = await sendJson(`/api/tickets/${encodeURIComponent(ticketId)}/ai/dry-run`, {});
        previewId = result.preview_id;
        const title = document.createElement("strong");
        title.textContent = "Dry-run preview — no ticket, audit, or external action has been changed.";
        const summary = document.createElement("p");
        summary.textContent = `Decision: ${result.workflow.decision}. Routing suggestion: ${result.workflow.routing_suggestion}.`;
        const list = document.createElement("ul");
        for (const [target, prediction] of Object.entries(result.predictions)) {
          const item = document.createElement("li");
          const probability = prediction.score == null
            ? "score unavailable"
            : `score ${(prediction.score * 100).toFixed(1)}% (uncalibrated)`;
          item.textContent = `${target}: ${prediction.label} — ${probability}`;
          list.append(item);
        }
        const policy = document.createElement("small");
        policy.textContent = `Policy: ${result.mode_label}; target coverage ${result.policy.target_coverage ?? "not configured"}; selected actions ${(result.policy.enabled_actions || []).join(", ") || "none"}.`;
        previewPanel.replaceChildren(title, summary, list, policy);
        previewPanel.hidden = false;
        runButton.disabled = false;
        if (feedback) feedback.textContent = "Review the preview, then explicitly proceed or change the ticket mode.";
      } catch (error) {
        if (feedback) feedback.textContent = error.message;
        previewPanel.hidden = true;
      } finally {
        previewButton.disabled = false;
      }
    });

    runButton?.addEventListener("click", async () => {
      if (!previewId) {
        if (feedback) feedback.textContent = "Run a dry-run preview before proceeding.";
        runButton.disabled = true;
        return;
      }
      runButton.disabled = true;
      if (feedback) feedback.textContent = "Applying only the workflow actions shown in the preview…";
      try {
        await sendJson(
          `/api/tickets/${encodeURIComponent(ticketId)}/ai/recommend`,
          { preview_id: previewId },
          requestId(),
        );
        previewId = null;
        window.location.reload();
      } catch (error) {
        if (feedback) feedback.textContent = error.message;
        runButton.disabled = false;
      }
    });

  }

  const ticketId = document.querySelector(".adaptive-ticket-control")?.dataset.ticketId;
  if (ticketId) {
    document.querySelectorAll(".adaptive-toggle-edit").forEach((button) => {
      button.addEventListener("click", () => {
        const form = document.getElementById(button.getAttribute("aria-controls"));
        if (!form) return;
        const expanded = button.getAttribute("aria-expanded") === "true";
        button.setAttribute("aria-expanded", String(!expanded));
        form.hidden = expanded;
      });
    });

    document.querySelectorAll(".adaptive-edit-form").forEach((form) => {
      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const submit = form.querySelector('button[type="submit"]');
        const feedback = document.querySelector(".adaptive-inline-feedback");
        submit.disabled = true;
        if (feedback) feedback.textContent = "Saving your edited labels…";
        const values = Object.fromEntries(new FormData(form).entries());
        const url = `/api/tickets/${encodeURIComponent(ticketId)}/ai/recommendations/${form.dataset.activityId}/approve`;
        try {
          await sendJson(url, { values });
          window.location.reload();
        } catch (error) {
          if (feedback) feedback.textContent = error.message;
          submit.disabled = false;
        }
      });
    });

    document.querySelectorAll(".adaptive-resolve-recommendation").forEach((button) => {
      button.addEventListener("click", async () => {
        const feedback = document.querySelector(".adaptive-inline-feedback");
        button.disabled = true;
        if (feedback) feedback.textContent = "Recording your decision…";
        try {
          const url = `/api/tickets/${encodeURIComponent(ticketId)}/ai/recommendations/${button.dataset.activityId}/${button.dataset.resolution}`;
          await sendJson(url, {});
          window.location.reload();
        } catch (error) {
          if (feedback) feedback.textContent = error.message;
          button.disabled = false;
        }
      });
    });
  }
})();
