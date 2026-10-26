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
        confidence_threshold: formData.get("confidence_threshold"),
        enabled_actions: formData.getAll("enabled_actions"),
      });
      if (feedback) feedback.textContent = `Policy saved. ${result.note}`;
    } catch (error) {
      if (feedback) feedback.textContent = error.message;
    } finally {
      submit.disabled = false;
    }
  });

  const ticketPanel = document.querySelector(".adaptive-ticket-control");
  if (ticketPanel) {
    const ticketId = ticketPanel.dataset.ticketId;
    const modeSelector = document.querySelector("#ticket-ai-mode");
    const feedback = ticketPanel.querySelector(".adaptive-inline-feedback");
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

    ticketPanel.querySelector(".adaptive-run-ticket")?.addEventListener("click", async (event) => {
      const button = event.currentTarget;
      button.disabled = true;
      if (feedback) feedback.textContent = "Running the permitted ticket workflow…";
      try {
        await sendJson(
          `/api/tickets/${encodeURIComponent(ticketId)}/ai/recommend`,
          {},
          requestId(),
        );
        window.location.reload();
      } catch (error) {
        if (feedback) feedback.textContent = error.message;
        button.disabled = false;
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
