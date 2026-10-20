const labels = {
  quiz: "Generazione delle domande",
  guide: "Domanda guida del tutor",
  evaluate: "Valutazione della risposta",
  contest: "Riesame della valutazione",
  next: "Avanzamento della sessione",
  upload: "Caricamento e indicizzazione",
  "doubt-start": "Domanda guida del tutor",
  "doubt-answer": "Feedback del tutor",
};

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("form[data-operation]").forEach(form => {
    let running = false;
    form.addEventListener("submit", async event => {
      const label = labels[form.dataset.operation];
      if (!label || !form.reportValidity()) return;
      event.preventDefault();
      if (running) return;
      running = true;
      const button = event.submitter;
      if (button) button.disabled = true;
      let panel = form.querySelector("[data-operation-progress]");
      if (!panel) {
        panel = document.createElement("section");
        panel.className = "notes-progress operation-progress";
        panel.dataset.operationProgress = "";
        panel.innerHTML = '<div class="notes-progress-heading"><span class="notes-progress-indicator" aria-hidden="true"></span><strong></strong></div><p role="status" aria-live="polite"></p><ol aria-label="Fasi completate"></ol>';
        form.append(panel);
      }
      panel.hidden = false;
      panel.dataset.state = "running";
      panel.querySelector("strong").textContent = label;
      const message = panel.querySelector("p");
      const list = panel.querySelector("ol");
      list.replaceChildren();
      message.textContent = form.dataset.operation === "upload" ? "Invio i file al server…" : "Invio la richiesta al server…";
      panel.scrollIntoView({ block: "nearest", behavior: "smooth" });
      let finished = false;
      try {
        const response = await fetch(form.getAttribute("action"), {
          method: "POST", body: new FormData(form), credentials: "same-origin",
          headers: { Accept: "application/x-ndjson" },
        });
        if (!response.ok) throw new Error("Il server non ha accettato la richiesta. Riprova.");
        if (!response.body) throw new Error("Questo browser non supporta l'avanzamento in tempo reale.");
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let pending = "";
        while (true) {
          const { value, done } = await reader.read();
          pending += decoder.decode(value || new Uint8Array(), { stream: !done });
          let newline;
          while ((newline = pending.indexOf("\n")) >= 0) {
            const line = pending.slice(0, newline).trim();
            pending = pending.slice(newline + 1);
            if (!line) continue;
            const update = JSON.parse(line);
            if (update.type === "progress") {
              message.textContent = update.message;
              const item = document.createElement("li");
              item.textContent = update.message;
              list.append(item);
            } else if (update.type === "error") {
              finished = true;
              panel.dataset.state = "error";
              if (update.code === "api_limit") {
                panel.querySelector("strong").textContent = "Limite delle chiamate API raggiunto";
              }
              message.setAttribute("role", "alert");
              message.textContent = update.message;
            } else if (update.type === "complete") {
              finished = true;
              message.textContent = "Operazione completata. Apro il risultato…";
              window.location.assign(update.url);
            }
          }
          if (done) break;
        }
        if (!finished) throw new Error("Il collegamento si è interrotto. Controlla il risultato prima di riprovare.");
      } catch (error) {
        panel.dataset.state = "error";
        message.setAttribute("role", "alert");
        message.textContent = error instanceof Error ? error.message : "Operazione non riuscita. Riprova.";
      } finally {
        running = false;
        if (button) button.disabled = false;
      }
    });
  });
});
