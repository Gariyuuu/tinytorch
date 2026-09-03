/* Docs reader.
 *
 * Fetches the markdown that site/build.sh copied out of the repository and
 * renders it in the browser. The repo's README.md and docs/*.md stay the single
 * source of truth — the site holds no copy of the prose, so it cannot drift.
 *
 * Three rendering concerns beyond plain markdown:
 *   · LaTeX  — registered as real marked *extensions*, not a regex pre-pass, so
 *              a `$` inside a shell code block is never mistaken for math.
 *   · mermaid— fenced ```mermaid blocks become <div class="mermaid"> for mermaid.run().
 *   · code   — highlight.js, themed through the Cobalt tokens in docs.css.
 */
(function () {
  "use strict";

  var DOCS = {
    autograd:     { file: "autograd.md",     title: "Reverse-mode autodiff" },
    architecture: { file: "architecture.md", title: "Architecture" },
    readme:       { file: "readme.md",       title: "Overview" }
  };
  var ORDER = ["readme", "autograd", "architecture"];

  var slug = (function () {
    var m = window.location.pathname.replace(/\/+$/, "").split("/").pop();
    return Object.prototype.hasOwnProperty.call(DOCS, m) ? m : "readme";
  })();

  var proseEl = document.getElementById("prose");
  var tocEl = document.getElementById("toc-list");
  var switchEl = document.getElementById("toc-switch");
  var titleEl = document.getElementById("doc-title");

  // ---- Doc switcher ------------------------------------------------------
  ORDER.forEach(function (key) {
    var a = document.createElement("a");
    a.href = "/docs/" + key;
    a.textContent = DOCS[key].title;
    if (key === slug) a.setAttribute("aria-current", "page");
    switchEl.appendChild(a);
  });
  document.title = DOCS[slug].title + " — TinyTorch";
  if (titleEl) titleEl.textContent = DOCS[slug].title;

  // ---- marked configuration ---------------------------------------------
  function slugify(text) {
    return String(text).toLowerCase().trim()
      .replace(/[^\w\s-]/g, "")
      .replace(/\s+/g, "-")
      .replace(/-+/g, "-");
  }

  var headings = [];

  function configureMarked() {
    var mathBlock = {
      name: "mathBlock",
      level: "block",
      start: function (src) { var i = src.indexOf("$$"); return i < 0 ? undefined : i; },
      tokenizer: function (src) {
        var m = /^\$\$([\s\S]+?)\$\$/.exec(src);
        if (m) return { type: "mathBlock", raw: m[0], text: m[1].trim() };
      },
      renderer: function (token) { return renderMath(token.text, true); }
    };

    var mathInline = {
      name: "mathInline",
      level: "inline",
      start: function (src) { var i = src.indexOf("$"); return i < 0 ? undefined : i; },
      tokenizer: function (src) {
        var m = /^\$([^\$\n]+?)\$/.exec(src);
        if (m) return { type: "mathInline", raw: m[0], text: m[1].trim() };
      },
      renderer: function (token) { return renderMath(token.text, false); }
    };

    marked.use({
      gfm: true,
      breaks: false,
      extensions: [mathBlock, mathInline],
      renderer: {
        // Signature differs across marked majors — accept both shapes.
        code: function (a, b) {
          var code = typeof a === "string" ? a : a.text;
          var lang = (typeof a === "string" ? b : a.lang) || "";
          lang = String(lang).split(/\s+/)[0];

          if (lang === "mermaid") {
            return '<div class="mermaid">' + escapeHtml(code) + "</div>";
          }
          var out = escapeHtml(code);
          if (window.hljs) {
            try {
              out = lang && hljs.getLanguage(lang)
                ? hljs.highlight(code, { language: lang }).value
                : hljs.highlightAuto(code).value;
            } catch (e) { /* fall back to the escaped source */ }
          }
          return '<pre><code class="hljs">' + out + "</code></pre>";
        },
        heading: function (a, b) {
          var text, depth;
          if (typeof a === "string") { text = a; depth = b; }
          else { depth = a.depth; text = this.parser.parseInline(a.tokens); }
          var plain = String(text).replace(/<[^>]*>/g, "");
          var id = slugify(plain);
          if (depth === 2 || depth === 3) headings.push({ id: id, text: plain, depth: depth });
          var anchor = (depth === 2 || depth === 3)
            ? ' <a class="anchor" href="#' + id + '" aria-label="Link to this section">#</a>'
            : "";
          return "<h" + depth + ' id="' + id + '">' + text + anchor + "</h" + depth + ">";
        }
        // Tables are left to marked's own renderer and wrapped in post(), which
        // works the same across marked majors.
      }
    });
  }

  function renderMath(tex, display) {
    if (!window.katex) return escapeHtml(display ? "$$" + tex + "$$" : "$" + tex + "$");
    try {
      return katex.renderToString(tex, { displayMode: display, throwOnError: false, strict: false });
    } catch (e) {
      return escapeHtml(tex);
    }
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  // ---- Post-render fixes -------------------------------------------------
  function post(root) {
    // Wrap any table that isn't already wrapped (covers marked versions whose
    // table renderer we could not override cleanly).
    Array.prototype.forEach.call(root.querySelectorAll("table"), function (t) {
      if (t.parentNode && t.parentNode.classList.contains("table-wrap")) return;
      var wrap = document.createElement("div");
      wrap.className = "table-wrap";
      t.parentNode.insertBefore(wrap, t);
      wrap.appendChild(t);
    });

    // Rewrite in-repo doc links to site routes.
    Array.prototype.forEach.call(root.querySelectorAll("a[href]"), function (a) {
      var href = a.getAttribute("href");
      if (!href || /^(https?:|#|mailto:)/.test(href)) {
        if (/^https?:/.test(href)) { a.target = "_blank"; a.rel = "noopener noreferrer"; }
        return;
      }
      var m = /(?:^|\/)(autograd|architecture)\.md(#.*)?$/.exec(href);
      if (m) { a.setAttribute("href", "/docs/" + m[1] + (m[2] || "")); return; }
      if (/README\.md$/i.test(href)) { a.setAttribute("href", "/docs/readme"); return; }
      // Anything else points at a repository file.
      a.setAttribute("href", "https://github.com/Gariyuuu/tinytorch/blob/main/" + href.replace(/^\.\//, ""));
      a.target = "_blank"; a.rel = "noopener noreferrer";
    });
  }

  function buildToc() {
    if (!headings.length) { tocEl.parentNode.style.display = "none"; return; }
    headings.forEach(function (h) {
      var li = document.createElement("li");
      var a = document.createElement("a");
      a.href = "#" + h.id;
      a.textContent = h.text;
      a.setAttribute("data-depth", String(h.depth));
      li.appendChild(a);
      tocEl.appendChild(li);
    });

    if (!("IntersectionObserver" in window)) return;
    var links = {};
    Array.prototype.forEach.call(tocEl.querySelectorAll("a"), function (a) {
      links[a.getAttribute("href").slice(1)] = a;
    });
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        var a = links[e.target.id];
        if (!a) return;
        if (e.isIntersecting) {
          Array.prototype.forEach.call(tocEl.querySelectorAll("a.is-active"), function (x) { x.classList.remove("is-active"); });
          a.classList.add("is-active");
        }
      });
    }, { rootMargin: "-70px 0px -72% 0px" });
    headings.forEach(function (h) {
      var el = document.getElementById(h.id);
      if (el) io.observe(el);
    });
  }

  function feedPalette() {
    if (!window.TT_PALETTE) return;
    var entries = [];
    ORDER.forEach(function (k) {
      entries.push({ t: DOCS[k].title, c: "Doc", href: "/docs/" + k });
    });
    headings.forEach(function (h) {
      entries.push({ t: h.text, c: DOCS[slug].title, href: "#" + h.id });
    });
    entries.push({ t: "GitHub repository", c: "External", href: "https://github.com/Gariyuuu/tinytorch" });
    window.TT_PALETTE.add(entries);
  }

  /* mermaid's colour parser (khroma) predates OKLCH and throws on it, so the
   * theme tokens have to be handed over as sRGB hex. Rather than duplicating
   * every colour as a hex literal — which would put tokens.css out of sync the
   * first time a colour changed — resolve each token through a 1x1 canvas.
   * The browser does the OKLCH -> sRGB conversion; tokens.css stays canonical. */
  var _swatch = null;
  function toHex(cssColor) {
    if (!cssColor) return null;
    if (/^#/.test(cssColor)) return cssColor;
    try {
      if (!_swatch) {
        var c = document.createElement("canvas");
        c.width = c.height = 1;
        _swatch = c.getContext("2d", { willReadFrequently: true });
      }
      _swatch.clearRect(0, 0, 1, 1);
      _swatch.fillStyle = "#000";
      _swatch.fillStyle = cssColor;
      _swatch.fillRect(0, 0, 1, 1);
      var d = _swatch.getImageData(0, 0, 1, 1).data;
      return "#" + [d[0], d[1], d[2]].map(function (n) {
        return ("0" + n.toString(16)).slice(-2);
      }).join("");
    } catch (e) {
      return null;
    }
  }

  function initMermaid() {
    if (!window.mermaid) return;
    var css = getComputedStyle(document.documentElement);
    var raw = function (n) { return css.getPropertyValue(n).trim(); };
    var hex = function (n) { return toHex(raw(n)); };

    try {
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: "loose",      // the diagrams use <br/> inside node labels
        theme: "base",
        fontFamily: raw("--font-body") || "Inter, sans-serif",
        themeVariables: {
          background:        hex("--color-paper-2"),
          primaryColor:      hex("--color-paper"),
          primaryTextColor:  hex("--color-ink"),
          primaryBorderColor:hex("--color-rule-2"),
          secondaryColor:    hex("--color-paper-3"),
          tertiaryColor:     hex("--color-paper-3"),
          lineColor:         hex("--color-ink-4"),
          textColor:         hex("--color-ink-2"),
          mainBkg:           hex("--color-paper"),
          nodeBorder:        hex("--color-rule-2"),
          clusterBkg:        hex("--color-paper-3"),
          clusterBorder:     hex("--color-rule-2"),
          edgeLabelBackground: hex("--color-paper-2"),
          fontSize: "13px"
        },
        flowchart: { curve: "basis", useMaxWidth: true, padding: 14 }
      });
    } catch (e) {
      // A theming failure must not cost the reader the diagram — fall back to
      // mermaid's own defaults rather than rendering nothing.
      try { mermaid.initialize({ startOnLoad: false, securityLevel: "loose" }); }
      catch (e2) { markDiagramFailure(); return; }
    }

    try {
      var out = mermaid.run({ querySelector: ".mermaid", suppressErrors: true });
      if (out && typeof out.catch === "function") out.catch(function () { markDiagramFailure(); });
    } catch (e) { markDiagramFailure(); }
  }

  function markDiagramFailure() {
    Array.prototype.forEach.call(document.querySelectorAll(".mermaid:not([data-processed])"), function (el) {
      el.classList.add("mermaid-error");
    });
  }

  // ---- Go ----------------------------------------------------------------
  function fail(message, detail) {
    proseEl.innerHTML = "";
    var h = document.createElement("div");
    h.className = "doc__state";
    h.innerHTML = "<p><strong>" + escapeHtml(message) + "</strong></p>" +
      (detail ? "<p><code>" + escapeHtml(detail) + "</code></p>" : "") +
      '<p>You can read this document on <a href="https://github.com/Gariyuuu/tinytorch/tree/main/docs">GitHub</a> instead.</p>';
    proseEl.appendChild(h);
  }

  fetch("/content/" + DOCS[slug].file, { cache: "no-cache" })
    .then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.text();
    })
    .then(function (md) {
      configureMarked();
      // The first H1 is replaced by the page's own title block.
      md = md.replace(/^#\s+.*\n+/, "");
      proseEl.innerHTML = marked.parse(md);
      post(proseEl);
      buildToc();
      feedPalette();
      initMermaid();
      if (window.location.hash) {
        var t = document.getElementById(window.location.hash.slice(1));
        if (t) t.scrollIntoView();
      }
    })
    .catch(function (e) { fail("Could not load this document.", String(e && e.message ? e.message : e)); });
})();
