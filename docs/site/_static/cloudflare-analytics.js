(() => {
  "use strict";
  try {
    const paths = new Set([
      "/Arc_nlp_public/conference/",
      "/Arc_nlp_public/conference/index.html",
      "/Arc_nlp_public/conference/profile.html",
    ]);
    if (window.location.origin !== "https://ryuuua.github.io" || !paths.has(window.location.pathname)) return;

    const token = document.currentScript?.getAttribute("data-cloudflare-web-analytics-token") || "";
    if (!/^[0-9a-f]{32}$/.test(token) || document.querySelector("script[data-cf-beacon]")) return;

    const beacon = document.createElement("script");
    beacon.type = "module";
    beacon.src = "https://static.cloudflareinsights.com/beacon.min.js";
    beacon.setAttribute("data-cf-beacon", JSON.stringify({ token }));
    document.head.appendChild(beacon);
  } catch {
    // Optional analytics must not interrupt the page when unavailable.
  }
})();
