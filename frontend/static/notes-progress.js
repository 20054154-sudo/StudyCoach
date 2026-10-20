// Browser-only progress for the existing Python notes generator.
document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector("[data-source-form]");
  if (!form) return;

  const button = form.querySelector("[data-notes-button]");
  const panel = form.querySelector("[data-notes-progress]");
  const status = form.querySelector("[data-notes-status]");
  const meter = form.querySelector("[data-notes-meter]");
  const steps = form.querySelector("[data-notes-steps]");
  const download = form.querySelector("[data-notes-download]");
  const markdownDownload = form.querySelector("[data-notes-markdown]");
  let downloadUrl = null;
  let markdownUrl = null;

  function describe(event) {
    const source = event.source || "fonte selezionata";
    const prefix = event.source_index ? `Fonte ${event.source_index}/${event.source_total} · ${source}` : source;
    const part = event.part ? `parte ${event.part}/${event.parts}` : "";
    switch (event.phase) {
      case "starting": return "Preparo le fonti selezionate…";
      case "reading": return `${prefix}: leggo i passaggi indicizzati…`;
      case "generating": return `${prefix}: genero gli appunti, ${part}…`;
      case "reviewing": return `${prefix}: verifico la fedeltà, ${part}…`;
      case "part_done": return `${prefix}: ${part} completata.`;
      case "source_done": return `${prefix}: completata.`;
      case "pdf": return "Impagino il PDF…";
      default: return "Generazione in corso…";
    }
  }

  function progressValue(event) {
    if (event.phase === "pdf") return 98;
    if (!event.source_index || !event.source_total) return 0;
    const sourceStart = (event.source_index - 1) / event.source_total;
    if (event.phase === "source_done") return Math.min(95, Math.round(95 * event.source_index / event.source_total));
    const parts = event.parts || 1;
    const part = event.part || 1;
    let within = 0;
    if (event.phase === "generating") within = (part - 1) / parts;
    if (event.phase === "reviewing") within = (part - 0.5) / parts;
    if (event.phase === "part_done") within = part / parts;
    return Math.min(95, Math.round(95 * (sourceStart + within / event.source_total)));
  }

  function showError(message) {
    panel.dataset.state = "error";
    status.setAttribute("role", "alert");
    status.textContent = message;
  }

  function handleEvent(event) {
    if (event.type === "heartbeat") return;
    if (event.type === "error") {
      if (event.code === "api_limit") {
        panel.querySelector(".notes-progress-heading strong").textContent = "Limite delle chiamate API raggiunto";
      }
      showError(event.message || "Impossibile generare gli appunti. Riprova.");
      return;
    }
    if (event.type === "complete") {
      const binary = atob(event.pdf_base64);
      const bytes = Uint8Array.from(binary, character => character.charCodeAt(0));
      downloadUrl = URL.createObjectURL(new Blob([bytes], { type: "application/pdf" }));
      download.href = downloadUrl;
      download.download = event.filename || "appunti.pdf";
      download.hidden = false;
      markdownUrl = URL.createObjectURL(new Blob([event.markdown], { type: "text/markdown;charset=utf-8" }));
      markdownDownload.href = markdownUrl;
      markdownDownload.download = event.markdown_filename || "appunti.md";
      markdownDownload.hidden = false;
      meter.value = 100;
      panel.dataset.state = "complete";
      status.textContent = "Appunti pronti. Scegli PDF o Markdown.";
      return;
    }
    if (event.type === "progress") {
      const message = describe(event);
      status.textContent = message;
      meter.value = Math.max(Number(meter.value), progressValue(event));
      if (!["starting", "part_done"].includes(event.phase)) {
        const item = document.createElement("li");
        item.textContent = message;
        steps.append(item);
        steps.scrollTop = steps.scrollHeight;
      }
    }
  }

  form.addEventListener("submit", async event => {
    if (event.submitter !== button) return;
    event.preventDefault();
    panel.hidden = false;
    panel.dataset.state = "running";
    status.setAttribute("role", "status");
    status.textContent = "Preparo le fonti selezionate…";
    meter.value = 0;
    steps.replaceChildren();
    download.hidden = true;
    markdownDownload.hidden = true;
    if (downloadUrl) URL.revokeObjectURL(downloadUrl);
    if (markdownUrl) URL.revokeObjectURL(markdownUrl);
    downloadUrl = null;
    markdownUrl = null;
    button.disabled = true;

    try {
      const body = new FormData(form);
      body.set("action", "notes");
      const response = await fetch(form.getAttribute("action"), {
        method: "POST",
        body,
        headers: { Accept: "application/x-ndjson" },
        credentials: "same-origin",
      });
      if (!response.ok) {
        const failure = await response.json();
        throw new Error(failure.error || "Impossibile generare gli appunti. Riprova.");
      }
      if (!response.body) throw new Error("Il browser non supporta il progresso in tempo reale.");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let pending = "";
      let finished = false;
      while (true) {
        const { value, done } = await reader.read();
        pending += decoder.decode(value || new Uint8Array(), { stream: !done });
        let newline;
        while ((newline = pending.indexOf("\n")) >= 0) {
          const line = pending.slice(0, newline).trim();
          pending = pending.slice(newline + 1);
          if (!line) continue;
          const update = JSON.parse(line);
          handleEvent(update);
          if (update.type === "complete" || update.type === "error") finished = true;
        }
        if (done) break;
      }
      if (!finished) throw new Error("La generazione si è interrotta. Riprova.");
    } catch (error) {
      showError(error instanceof Error ? error.message : "Impossibile generare gli appunti. Riprova.");
    } finally {
      button.disabled = false;
    }
  });
});
