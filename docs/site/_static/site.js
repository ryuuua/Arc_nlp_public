/* Local progressive enhancements. All documentation remains readable without JS. */
(() => {
  "use strict";
  const menu = document.querySelector(".menu-toggle");
  const nav = document.getElementById("site-nav");
  menu?.addEventListener("click", () => {
    const open = menu.getAttribute("aria-expanded") !== "true";
    menu.setAttribute("aria-expanded", String(open));
    nav.classList.toggle("is-open", open);
  });
  document.querySelectorAll("pre > code").forEach((code) => {
    if (!navigator.clipboard) return;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "copy-button";
    button.textContent = "Copy";
    button.setAttribute("aria-label", "Copy code to clipboard");
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(code.textContent);
        button.textContent = "Copied";
      } catch (_) {
        button.textContent = "Copy failed";
      }
      window.setTimeout(() => { button.textContent = "Copy"; }, 1600);
    });
    code.parentElement.append(button);
  });
  const dialog = document.getElementById("search-dialog");
  const input = document.getElementById("search-input");
  const status = document.getElementById("search-status");
  const results = document.getElementById("search-results");
  const root = new URL(document.body.dataset.siteRoot, window.location.href);
  let index = null;
  let request = null;
  const loadIndex = async () => {
    if (index) return index;
    if (!request) request = fetch(new URL("assets/search.json", root))
      .then(response => {
        if (!response.ok) throw new Error("Search index unavailable");
        return response.json();
      })
      .then(value => { index = value; return value; })
      .catch(error => { request = null; throw error; });
    return request;
  };
  let generation = 0;
  const search = async () => {
    const current = ++generation;
    const query = input.value.trim().toLocaleLowerCase();
    results.replaceChildren();
    if (!query) { status.textContent = "Type a term to search the documentation."; return; }
    status.textContent = "Searching…";
    try {
      const pages = await loadIndex();
      if (current !== generation) return;
      const terms = query.split(/\s+/);
      const matches = pages.map(page => {
        const title = page.title.toLocaleLowerCase();
        const text = page.text.toLocaleLowerCase();
        return {...page, score: terms.every(term => title.includes(term) || text.includes(term))
          ? terms.reduce((score, term) => score + (title.includes(term) ? 10 : 1), 0) : 0};
      }).filter(page => page.score > 0).sort((a, b) => b.score - a.score).slice(0, 15);
      status.textContent = matches.length ? `${matches.length} result${matches.length === 1 ? "" : "s"}` : "No matching pages. Try another term.";
      matches.forEach(page => {
        const item = document.createElement("li");
        const link = document.createElement("a");
        link.href = new URL(page.url, root).href;
        const title = document.createElement("strong");
        title.textContent = page.title;
        const summary = document.createElement("p");
        const position = page.text.toLocaleLowerCase().indexOf(terms[0]);
        const start = Math.max(0, position - 70);
        summary.textContent = `${start ? "…" : ""}${page.text.slice(start, start + 180)}…`;
        link.append(title, summary);
        item.append(link);
        results.append(item);
      });
    } catch (_) {
      if (current === generation) status.textContent = "Search is unavailable here. Serve the site over HTTP to use local search, or browse the navigation.";
    }
  };
  const openSearch = () => { dialog.showModal(); input.focus(); search(); };
  document.querySelector(".search-toggle")?.addEventListener("click", openSearch);
  document.querySelector(".search-close")?.addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", event => { if (event.target === dialog) dialog.close(); });
  input.addEventListener("input", search);
  document.addEventListener("keydown", event => {
    const editing = event.target instanceof HTMLElement && (event.target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(event.target.tagName));
    if (event.key === "/" && !editing && !dialog.open) { event.preventDefault(); openSearch(); }
    if (event.key === "Escape" && nav.classList.contains("is-open")) {
      nav.classList.remove("is-open"); menu.setAttribute("aria-expanded", "false"); menu.focus();
    }
  });
})();
