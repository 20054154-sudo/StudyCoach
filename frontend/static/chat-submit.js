document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector("[data-chat-composer]");
  if (!form) return;
  const button = form.querySelector("[data-chat-submit]");
  let submitted = false;

  form.addEventListener("submit", event => {
    if (submitted) {
      event.preventDefault();
      return;
    }
    submitted = true;
    button.disabled = true;
    button.textContent = "Il tutor sta preparando la risposta…";
  });
});
