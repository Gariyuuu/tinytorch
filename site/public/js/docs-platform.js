/* ============================================================================
 * Docs Platform — shared behaviour for the library documentation family.
 *
 * No dependencies, no build step. Everything degrades to a readable static
 * document if this file fails to load: highlighting, copy buttons, scroll-spy
 * and reveals are all progressive enhancements over working markup.
 *
 * Honours prefers-reduced-motion for every animation it owns.
 * ========================================================================== */
(function () {
  "use strict";

  var root = document.documentElement;
  var reduceMQ = window.matchMedia("(prefers-reduced-motion: reduce)");
  var reduced = function () { return reduceMQ.matches; };

  function $(sel, ctx) { return (ctx || document).querySelector(sel); }
  function $$(sel, ctx) { return Array.prototype.slice.call((ctx || document).querySelectorAll(sel)); }

  /* Build an <svg><use href="#i-x"> icon reference. */
  function icon(name, cls) {
    var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "dp-i" + (cls ? " " + cls : ""));
    svg.setAttribute("aria-hidden", "true");
    svg.setAttribute("viewBox", "0 0 24 24");
    var use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    use.setAttribute("href", "#i-" + name);
    svg.appendChild(use);
    return svg;
  }

  /* ==========================================================================
   * 1 · Theme — system / light / dark, persisted, announced
   * ======================================================================== */
  var THEME_KEY = "dp-theme";

  function storedTheme() {
    try { return localStorage.getItem(THEME_KEY); } catch (e) { return null; }
  }
  function resolvedTheme() {
    var t = storedTheme();
    if (t === "light" || t === "dark") return t;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  function applyTheme(t) {
    if (t === "light" || t === "dark") root.setAttribute("data-theme", t);
    else root.removeAttribute("data-theme");
    try { t ? localStorage.setItem(THEME_KEY, t) : localStorage.removeItem(THEME_KEY); } catch (e) {}
    syncThemeButton();
  }
  var themeBtn = null;
  function syncThemeButton() {
    if (!themeBtn) return;
    var stored = storedTheme();
    var res = resolvedTheme();
    themeBtn.setAttribute("data-resolved", res);
    var label = stored === "dark" ? "Theme: dark. Switch to system."
      : stored === "light" ? "Theme: light. Switch to dark."
      : "Theme: system (" + res + "). Switch to light.";
    themeBtn.setAttribute("aria-label", label);
    themeBtn.setAttribute("title", label);
  }

  function initTheme() {
    themeBtn = $(".dp-theme");
    if (!themeBtn) return;
    syncThemeButton();
    themeBtn.addEventListener("click", function () {
      // system -> light -> dark -> system
      var stored = storedTheme();
      applyTheme(stored === "light" ? "dark" : stored === "dark" ? null : "light");
    });
    var sysMQ = window.matchMedia("(prefers-color-scheme: dark)");
    var onSys = function () { if (!storedTheme()) syncThemeButton(); };
    sysMQ.addEventListener ? sysMQ.addEventListener("change", onSys) : sysMQ.addListener(onSys);
  }

  /* ==========================================================================
   * 2 · Syntax highlighting
   *
   * Deliberately small: comments, strings, numbers, keywords, call names and
   * shell prompts. That is the whole vocabulary these docs use, and it keeps
   * the highlighter auditable instead of shipping a parser.
   *
   * Tokens are emitted as real DOM nodes from textContent, never innerHTML,
   * so nothing in a code sample can become markup.
   * ======================================================================== */
  var PY_KW = "and|as|assert|async|await|break|class|continue|def|del|elif|else|except|finally|for|from|global|if|import|in|is|lambda|nonlocal|not|or|pass|raise|return|try|while|with|yield|True|False|None|self";
  var SH_KW = "if|then|else|fi|for|in|do|done|while|case|esac|function|export|source|local|return|exit";

  var GRAMMARS = {
    python: [
      ["com", /#[^\n]*/],
      ["str", /"""[\s\S]*?"""|'''[\s\S]*?'''|[rbfu]{0,2}"(?:\\.|[^"\\])*"|[rbfu]{0,2}'(?:\\.|[^'\\])*'/],
      ["key", new RegExp("\\b(?:" + PY_KW + ")\\b")],
      ["fun", /\b[A-Za-z_]\w*(?=\s*\()/],
      ["num", /\b\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d+)?\b|\b0[xXbBoO][0-9a-fA-F_]+\b/]
    ],
    shell: [
      ["pmt", /^\s*[$#>](?=\s)/m],
      ["com", /#[^\n]*/],
      ["str", /"(?:\\.|[^"\\])*"|'(?:[^'])*'/],
      ["key", new RegExp("(?:^|\\s)--?[A-Za-z][\\w-]*|\\b(?:" + SH_KW + ")\\b")],
      // Command names only at a true line start (or after a pipe). Indented
      // text is continuation or output, not a command — without this a
      // transcript's prose gets highlighted as if every line were a call.
      ["fun", /(?:^[A-Za-z_][\w.\/-]*|[|;&]\s*[A-Za-z_][\w.\/-]*)/m],
      ["num", /\b\d[\d_]*(?:\.\d+)?\b/]
    ],
    json: [
      ["com", /\/\/[^\n]*/],
      ["str", /"(?:\\.|[^"\\])*"(?=\s*:)/, "fun"],
      ["str", /"(?:\\.|[^"\\])*"/],
      ["key", /\b(?:true|false|null)\b/],
      ["num", /-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b/]
    ],
    toml: [
      ["com", /#[^\n]*/],
      ["str", /"""[\s\S]*?"""|"(?:\\.|[^"\\])*"|'(?:[^'])*'/],
      ["fun", /^\s*\[[^\]\n]*\]/m],
      ["key", /\b(?:true|false)\b/],
      ["num", /\b\d[\d_]*(?:\.\d+)?\b/]
    ],
    yaml: [
      ["com", /#[^\n]*/],
      ["str", /"(?:\\.|[^"\\])*"|'(?:[^'])*'/],
      ["fun", /^\s*-?\s*[A-Za-z_][\w.-]*(?=\s*:)/m],
      ["key", /\b(?:true|false|null|yes|no)\b/],
      ["num", /\b\d[\d_]*(?:\.\d+)?\b/]
    ],
    text: []
  };
  var ALIAS = {
    py: "python", python: "python", console: "shell", bash: "shell", sh: "shell",
    shell: "shell", zsh: "shell", terminal: "shell", json: "json", toml: "toml",
    yaml: "yaml", yml: "yaml", ini: "toml", cfg: "toml"
  };

  /* One left-to-right pass. Whichever rule matches earliest wins, which gives
   * the right answer for `# "not a string"` and `"# not a comment"` alike. */
  function tokenize(src, rules) {
    var out = [], pos = 0;
    if (!rules.length) return [{ t: null, s: src }];
    while (pos < src.length) {
      var best = null, bestIdx = Infinity, bestRule = null;
      for (var i = 0; i < rules.length; i++) {
        var re = new RegExp(rules[i][1].source, rules[i][1].flags.replace("g", "") + "g");
        re.lastIndex = pos;
        var m = re.exec(src);
        if (m && m.index < bestIdx) { bestIdx = m.index; best = m; bestRule = rules[i]; }
      }
      if (!best) break;
      if (bestIdx > pos) out.push({ t: null, s: src.slice(pos, bestIdx) });
      var cls = bestRule[2] || bestRule[0];
      // Shell command/flag rules capture leading whitespace; keep it unstyled.
      var raw = best[0];
      var lead = raw.match(/^[\s|;&]*/)[0];
      if (lead && (cls === "fun" || cls === "key")) {
        out.push({ t: null, s: lead });
        raw = raw.slice(lead.length);
      }
      if (raw) out.push({ t: cls, s: raw });
      pos = bestIdx + best[0].length;
      if (best[0].length === 0) pos++;
    }
    if (pos < src.length) out.push({ t: null, s: src.slice(pos) });
    return out;
  }

  function parseRanges(spec) {
    var set = {};
    (spec || "").split(",").forEach(function (part) {
      part = part.trim();
      if (!part) return;
      var d = part.split("-");
      var a = parseInt(d[0], 10), b = parseInt(d[1] !== undefined ? d[1] : d[0], 10);
      if (isNaN(a) || isNaN(b)) return;
      for (var i = a; i <= b; i++) set[i] = true;
    });
    return set;
  }

  /* `until` marks the boundary between source and the transcript it produced.
   * Past it, lines are emitted plain: printed output is prose, and grammar
   * rules applied to prose colour ordinary words as if they were keywords. */
  function highlight(codeEl, lang, hlSpec, until) {
    var src = codeEl.textContent.replace(/\n+$/, "");
    var rules = GRAMMARS[ALIAS[lang] || "text"] || [];
    var hl = parseRanges(hlSpec);
    var cut = until ? parseInt(until, 10) : Infinity;
    var frag = document.createDocumentFragment();
    var lines = src.split("\n");

    lines.forEach(function (line, i) {
      var lineEl = document.createElement("span");
      lineEl.className = "dp-ln" + (hl[i + 1] ? " dp-ln--hl" : "")
        + (i + 1 > cut ? " dp-ln--out" : "");
      tokenize(line, i + 1 > cut ? [] : rules).forEach(function (tok) {
        if (!tok.s) return;
        if (!tok.t) { lineEl.appendChild(document.createTextNode(tok.s)); return; }
        var s = document.createElement("span");
        s.className = "dp-t-" + tok.t;
        s.appendChild(document.createTextNode(tok.s));
        lineEl.appendChild(s);
      });
      // Preserve blank lines as real rows so the gutter keeps counting.
      if (!line.length) lineEl.appendChild(document.createTextNode("\u200b"));
      // No trailing "\n" text node: .dp-ln is block-level, and inside <pre> a
      // newline *and* a block would render every line twice.
      frag.appendChild(lineEl);
    });

    codeEl.textContent = "";
    codeEl.appendChild(frag);
  }

  /* ==========================================================================
   * 3 · Code blocks — highlight + copy affordance
   * ======================================================================== */
  function makeCopyButton(getText, label) {
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "dp-copy";
    btn.setAttribute("aria-label", label || "Copy code to clipboard");

    var idle = document.createElement("span");
    idle.className = "dp-copy__idle";
    idle.appendChild(icon("copy", "dp-i--sm"));

    var done = document.createElement("span");
    done.className = "dp-copy__done";
    done.appendChild(icon("check", "dp-i--sm"));
    var doneTxt = document.createElement("span");
    doneTxt.textContent = "Copied";
    done.appendChild(doneTxt);

    btn.appendChild(idle);
    btn.appendChild(done);

    var timer = null;
    btn.addEventListener("click", function () {
      var text = getText();
      var settle = function (state, msg) {
        btn.setAttribute("data-copied", state);
        announce(msg);
        clearTimeout(timer);
        timer = setTimeout(function () { btn.removeAttribute("data-copied"); }, 1800);
      };
      var ok = function () { settle("true", "Copied to clipboard"); };
      var fail = function () { settle("error", "Copy failed — select the code and copy manually"); };

      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(ok, function () { legacyCopy(text) ? ok() : fail(); });
      } else {
        legacyCopy(text) ? ok() : fail();
      }
    });
    return btn;
  }

  function legacyCopy(text) {
    try {
      var ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.style.cssText = "position:fixed;top:0;left:-9999px;opacity:0";
      document.body.appendChild(ta);
      ta.select();
      var ok = document.execCommand("copy");
      document.body.removeChild(ta);
      return ok;
    } catch (e) { return false; }
  }

  var liveRegion = null;
  function announce(msg) {
    if (!liveRegion) {
      liveRegion = document.createElement("div");
      liveRegion.className = "dp-sr";
      liveRegion.setAttribute("role", "status");
      liveRegion.setAttribute("aria-live", "polite");
      document.body.appendChild(liveRegion);
    }
    liveRegion.textContent = "";
    setTimeout(function () { liveRegion.textContent = msg; }, 30);
  }

  function initCode() {
    $$(".dp-code").forEach(function (block) {
      var codeEl = $("code", block);
      if (!codeEl) return;
      var lang = block.getAttribute("data-lang") || "text";
      var plain = codeEl.textContent.replace(/\n+$/, "");

      // data-nohl: the markup already carries hand-authored highlighting that
      // says more than a grammar could (a green PASS, a red failure). Leave it
      // alone and give it the chrome only.
      if (!block.hasAttribute("data-nohl")) {
        highlight(codeEl, lang, block.getAttribute("data-hl"), block.getAttribute("data-src-until"));
      }

      var bar = $(".dp-code__bar", block);
      if (!bar) {
        bar = document.createElement("figcaption");
        bar.className = "dp-code__bar";
        var tag = document.createElement("span");
        tag.className = "dp-code__lang";
        tag.appendChild(icon(block.getAttribute("data-icon") ||
          (ALIAS[lang] === "shell" ? "terminal" : "code"), "dp-i--sm"));
        var t = document.createElement("span");
        t.textContent = block.getAttribute("data-label") || lang;
        tag.appendChild(t);
        bar.appendChild(tag);
        var file = block.getAttribute("data-file");
        if (file) {
          var f = document.createElement("span");
          f.className = "dp-code__file";
          f.textContent = file;
          bar.appendChild(f);
        }
        block.insertBefore(bar, block.firstChild);
      }
      if (!$(".dp-copy", bar)) {
        bar.appendChild(makeCopyButton(function () { return plain; },
          "Copy the " + lang + " snippet to the clipboard"));
      }
    });

    // Bare <pre> blocks produced by someone else's renderer (a markdown pass,
    // highlight.js) get a floating copy control rather than being rewritten.
    $$("[data-dp-copy] pre").forEach(function (pre) {
      if (pre.parentElement && pre.parentElement.classList.contains("dp-prewrap")) return;
      var code = $("code", pre) || pre;
      var plain = code.textContent.replace(/\n+$/, "");
      var wrap = document.createElement("div");
      wrap.className = "dp-prewrap";
      pre.parentNode.insertBefore(wrap, pre);
      wrap.appendChild(pre);
      wrap.appendChild(makeCopyButton(function () { return plain; }, "Copy this code block"));
    });

    $$(".dp-cmd").forEach(function (cmd) {
      var codeEl = $("code", cmd);
      if (!codeEl) return;
      var plain = codeEl.textContent.trim();
      if (!$(".dp-i--prompt", cmd)) cmd.insertBefore(icon("terminal", "dp-i--prompt dp-i--sm"), cmd.firstChild);
      highlight(codeEl, cmd.getAttribute("data-lang") || "shell", null);
      if (!$(".dp-copy", cmd)) {
        cmd.appendChild(makeCopyButton(function () { return plain; }, "Copy command: " + plain));
      }
    });
  }

  /* ==========================================================================
   * 4 · Sidebar scroll-spy + reading progress
   * ======================================================================== */
  function initSpy() {
    var links = $$(".dp-toc a[href^='#']");
    if (!links.length) return;

    var byId = {};
    var sections = [];
    links.forEach(function (a) {
      var id = decodeURIComponent(a.getAttribute("href").slice(1));
      var el = document.getElementById(id);
      if (!el) return;
      byId[id] = a;
      sections.push(el);
    });
    if (!sections.length) return;

    var current = null;
    function setCurrent(id) {
      if (id === current) return;
      current = id;
      links.forEach(function (a) { a.removeAttribute("aria-current"); });
      if (byId[id]) byId[id].setAttribute("aria-current", "true");
    }

    // Pick the last section whose top has passed the reading line. Simple,
    // stable, and correct for short trailing sections that never fill the
    // viewport — which an IntersectionObserver ratio test gets wrong.
    var ticking = false;
    function update() {
      ticking = false;
      var line = window.scrollY + window.innerHeight * 0.28;
      var pick = sections[0];
      for (var i = 0; i < sections.length; i++) {
        if (sections[i].getBoundingClientRect().top + window.scrollY <= line) pick = sections[i];
      }
      // At the very bottom, the final section is what you are reading.
      if (window.innerHeight + window.scrollY >= document.body.scrollHeight - 4) {
        pick = sections[sections.length - 1];
      }
      setCurrent(pick.id);

      var max = document.documentElement.scrollHeight - window.innerHeight;
      var p = max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0;
      root.style.setProperty("--dp-progress", p.toFixed(4));

      var top = $(".dp-top");
      if (top) top.setAttribute("data-show", window.scrollY > window.innerHeight * 0.6 ? "true" : "false");
    }
    function onScroll() {
      if (!ticking) { ticking = true; requestAnimationFrame(update); }
    }
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll, { passive: true });
    update();

    // Keep the active item in view inside a long, independently scrolling rail.
    var side = $(".dp-side");
    if (side) {
      var mo = new MutationObserver(function () {
        var a = $(".dp-toc a[aria-current='true']");
        if (!a || side.scrollHeight <= side.clientHeight) return;
        var r = a.getBoundingClientRect(), sr = side.getBoundingClientRect();
        if (r.top < sr.top + 8 || r.bottom > sr.bottom - 8) {
          side.scrollTo({ top: side.scrollTop + (r.top - sr.top) - side.clientHeight / 2 + r.height,
                          behavior: reduced() ? "auto" : "smooth" });
        }
      });
      $$(".dp-toc a").forEach(function (a) { mo.observe(a, { attributes: true, attributeFilter: ["aria-current"] }); });
    }
  }

  /* ==========================================================================
   * 5 · Mobile contents disclosure
   * ======================================================================== */
  function initSideToggle() {
    var btn = $(".dp-side__toggle");
    var panel = $(".dp-side__panel");
    if (!btn || !panel) return;

    var mq = window.matchMedia("(max-width: 64rem)");
    function sync() {
      if (mq.matches) {
        btn.setAttribute("aria-expanded", "false");
        panel.hidden = true;
      } else {
        btn.setAttribute("aria-expanded", "true");
        panel.hidden = false;
      }
    }
    sync();
    mq.addEventListener ? mq.addEventListener("change", sync) : mq.addListener(sync);

    btn.addEventListener("click", function () {
      var open = btn.getAttribute("aria-expanded") === "true";
      btn.setAttribute("aria-expanded", String(!open));
      panel.hidden = open;
    });
    // Choosing a destination on mobile should put the document back in view.
    panel.addEventListener("click", function (e) {
      if (e.target.closest("a") && mq.matches) {
        btn.setAttribute("aria-expanded", "false");
        panel.hidden = true;
      }
    });
  }

  /* ==========================================================================
   * 6 · Back to top
   * ======================================================================== */
  function initTop() {
    var btn = $(".dp-top");
    if (!btn) return;
    btn.addEventListener("click", function () {
      window.scrollTo({ top: 0, behavior: reduced() ? "auto" : "smooth" });
      var first = $(".dp-h1") || $("main");
      if (first) { first.setAttribute("tabindex", "-1"); first.focus({ preventScroll: true }); }
    });
  }

  /* ==========================================================================
   * 7 · Table overflow affordance
   * ======================================================================== */
  function initTables() {
    $$(".dp-tablewrap").forEach(function (wrap) {
      var scroller = $(".dp-tablewrap__scroll", wrap) || wrap;
      function sync() {
        var state = [];
        if (scroller.scrollLeft > 2) state.push("start");
        if (scroller.scrollLeft + scroller.clientWidth < scroller.scrollWidth - 2) state.push("end");
        wrap.setAttribute("data-scroll", state.join(" "));
      }
      scroller.addEventListener("scroll", sync, { passive: true });
      window.addEventListener("resize", sync, { passive: true });
      sync();
      // A horizontally scrollable region must be reachable by keyboard.
      if (scroller.scrollWidth > scroller.clientWidth && !scroller.hasAttribute("tabindex")) {
        scroller.setAttribute("tabindex", "0");
        scroller.setAttribute("role", "region");
        var cap = $("caption", scroller);
        scroller.setAttribute("aria-label", (cap ? cap.textContent.trim() : "Table") + " (scrollable)");
      }
    });
  }

  /* ==========================================================================
   * 8 · Heading anchors
   * ======================================================================== */
  function initAnchors() {
    $$(".dp-sec[id]").forEach(function (sec) {
      var h = $(".dp-h2", sec);
      if (!h || $(".dp-anchor", h)) return;
      var a = document.createElement("a");
      a.className = "dp-anchor";
      a.href = "#" + sec.id;
      a.setAttribute("aria-label", "Link to this section: " + h.textContent.trim());
      a.appendChild(icon("link", "dp-i--sm"));
      h.appendChild(a);
    });
  }

  /* ==========================================================================
   * 9 · Entrance reveals — staggered, cheap, and skipped under reduced motion
   * ======================================================================== */
  function initReveal() {
    if (reduced() || !("IntersectionObserver" in window)) return;
    var targets = $$(".dp-reveal");
    if (!targets.length) return;
    root.setAttribute("data-dp-motion", "on");

    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        var el = entry.target;
        var group = el.parentElement ? Array.prototype.filter.call(el.parentElement.children, function (c) {
          return c.classList && c.classList.contains("dp-reveal");
        }) : [el];
        var i = Math.min(group.indexOf(el), 5);
        el.style.setProperty("--dp-delay", (i * 45) + "ms");
        el.setAttribute("data-shown", "true");
        io.unobserve(el);
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.05 });

    targets.forEach(function (t) { io.observe(t); });

    // If the user turns reduced-motion on mid-session, stop hiding anything.
    var onChange = function () {
      if (!reduced()) return;
      root.removeAttribute("data-dp-motion");
      targets.forEach(function (t) { t.setAttribute("data-shown", "true"); });
    };
    reduceMQ.addEventListener ? reduceMQ.addEventListener("change", onChange) : reduceMQ.addListener(onChange);
  }

  /* ==========================================================================
   * Boot
   * ======================================================================== */
  function boot() {
    try { initTheme(); } catch (e) {}
    try { initCode(); } catch (e) {}
    try { initAnchors(); } catch (e) {}
    try { initSpy(); } catch (e) {}
    try { initSideToggle(); } catch (e) {}
    try { initTop(); } catch (e) {}
    try { initTables(); } catch (e) {}
    try { initReveal(); } catch (e) {}
  }

  // Documents that render their prose at runtime (a markdown reader) call this
  // after each render so newly-inserted code blocks get the same affordances.
  window.DocsPlatform = {
    enhance: function () {
      try { initCode(); } catch (e) {}
      try { initTables(); } catch (e) {}
    }
  };

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
