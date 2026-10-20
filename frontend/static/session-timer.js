(() => {
  const clock = document.querySelector("[data-session-clock]");
  if (!clock) return;

  const start = clock.querySelector("[data-session-start]");
  const duration = clock.querySelector("[data-session-duration]");
  const startedAt = new Date(start.dateTime);
  if (!Number.isNaN(startedAt.getTime())) {
    start.textContent = new Intl.DateTimeFormat("it-IT", {
      dateStyle: "short", timeStyle: "short"
    }).format(startedAt);
  }

  const baseSeconds = Number(clock.dataset.elapsedSeconds) || 0;
  const running = clock.dataset.running === "true";
  const loadedAt = performance.now();
  const update = () => {
    const seconds = Math.floor(baseSeconds + (running ? (performance.now() - loadedAt) / 1000 : 0));
    const hours = String(Math.floor(seconds / 3600)).padStart(2, "0");
    const minutes = String(Math.floor(seconds / 60) % 60).padStart(2, "0");
    const remaining = String(seconds % 60).padStart(2, "0");
    duration.textContent = `${hours}:${minutes}:${remaining}`;
  };
  update();
  if (running) {
    window.setInterval(update, 1000);
    window.setInterval(() => {
      fetch(`/sessions/${clock.dataset.sessionId}/timer`, {
        method: "POST", credentials: "same-origin"
      }).catch(() => {});
    }, 15000);
  }
})();
