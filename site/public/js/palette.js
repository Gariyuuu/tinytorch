/* ⌘K command palette — Cobalt's signature interactive move.
 *
 * Opens on click and on ⌘K / Ctrl+K. Esc or backdrop closes. Arrow keys move,
 * Enter follows. Focus is trapped to the input and restored on close. The index
 * is static for the landing page and extended by the docs reader (which appends
 * every heading it renders) via window.TT_PALETTE.add().
 */
(function () {
  "use strict";

  var BASE = [
    { t: "Overview", c: "Page", href: "/" },
    { t: "What you actually write", c: "Section", href: "/#tour" },
    { t: "How it is verified", c: "Section", href: "/#verification" },
    { t: "Overhead, measured", c: "Section", href: "/#benchmarks" },
    { t: "Scope is frozen", c: "Section", href: "/#scope" },
    { t: "Reverse-mode autodiff", c: "Doc", href: "/docs/autograd" },
    { t: "Architecture", c: "Doc", href: "/docs/architecture" },
    { t: "README", c: "Doc", href: "/docs/readme" },
    { t: "GitHub repository", c: "External", href: "https://github.com/Gariyuuu/tinytorch" }
  ];

  var items = BASE.slice();
  var pal = document.getElementById("pal");
  var input = document.getElementById("pal-input");
  var list = document.getElementById("pal-list");
  var openBtn = document.getElementById("pal-open");
  if (!pal || !input || !list) return;

  var active = 0;
  var shown = [];
  var lastFocus = null;

  function score(item, q) {
    if (!q) return 0;
    var t = item.t.toLowerCase();
    var i = t.indexOf(q);
    if (i === 0) return 3;
    if (i > 0) return 2;
    // subsequence match, so "grdchk" still finds "gradcheck"
    var qi = 0;
    for (var k = 0; k < t.length && qi < q.length; k++) if (t[k] === q[qi]) qi++;
    return qi === q.length ? 1 : -1;
  }

  function render() {
    var q = input.value.trim().toLowerCase();
    shown = items
      .map(function (it) { return { it: it, s: score(it, q) }; })
      .filter(function (r) { return r.s >= 0; })
      .sort(function (a, b) { return b.s - a.s; })
      .slice(0, 40)
      .map(function (r) { return r.it; });

    if (active >= shown.length) active = 0;
    list.innerHTML = "";

    if (!shown.length) {
      var empty = document.createElement("li");
      empty.className = "pal__empty";
      empty.textContent = "No matches.";
      list.appendChild(empty);
      return;
    }

    shown.forEach(function (it, i) {
      var li = document.createElement("li");
      li.className = "pal__item";
      li.setAttribute("role", "option");
      li.setAttribute("aria-selected", i === active ? "true" : "false");
      li.id = "pal-opt-" + i;

      var t = document.createElement("span");
      t.className = "pal__t";
      t.textContent = it.t;

      var c = document.createElement("span");
      c.className = "pal__c";
      c.textContent = it.c;

      li.appendChild(t);
      li.appendChild(c);
      li.addEventListener("click", function () { go(it); });
      li.addEventListener("mousemove", function () {
        if (active === i) return;
        active = i;
        syncSelection();
      });
      list.appendChild(li);
    });
    syncSelection();
  }

  function syncSelection() {
    var nodes = list.querySelectorAll(".pal__item");
    for (var i = 0; i < nodes.length; i++) {
      nodes[i].setAttribute("aria-selected", i === active ? "true" : "false");
    }
    input.setAttribute("aria-activedescendant", "pal-opt-" + active);
    var el = nodes[active];
    if (el && el.scrollIntoView) el.scrollIntoView({ block: "nearest" });
  }

  function go(item) {
    close();
    if (!item) return;
    if (/^https?:/.test(item.href)) { window.location.href = item.href; return; }
    window.location.href = item.href;
  }

  function open() {
    if (!pal.hidden) return;
    lastFocus = document.activeElement;
    pal.hidden = false;
    input.value = "";
    active = 0;
    render();
    input.focus();
  }

  function close() {
    if (pal.hidden) return;
    pal.hidden = true;
    // Opened by shortcut from nowhere in particular, focus would otherwise be
    // dropped on <body> and the next Tab would restart from the top of the page.
    var target = lastFocus;
    if (!target || target === document.body || !target.focus) target = openBtn;
    if (target && target.focus) target.focus();
  }

  if (openBtn) openBtn.addEventListener("click", open);

  Array.prototype.forEach.call(pal.querySelectorAll("[data-pal-close]"), function (el) {
    el.addEventListener("click", close);
  });

  input.addEventListener("input", function () { active = 0; render(); });

  document.addEventListener("keydown", function (e) {
    var k = e.key.toLowerCase();
    if (k === "k" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); pal.hidden ? open() : close(); return; }
    if (pal.hidden) return;
    if (e.key === "Escape") { e.preventDefault(); close(); return; }
    if (e.key === "ArrowDown") { e.preventDefault(); active = (active + 1) % Math.max(shown.length, 1); syncSelection(); return; }
    if (e.key === "ArrowUp") { e.preventDefault(); active = (active - 1 + shown.length) % Math.max(shown.length, 1); syncSelection(); return; }
    if (e.key === "Enter") { e.preventDefault(); go(shown[active]); return; }
    if (e.key === "Tab") { e.preventDefault(); input.focus(); }
  });

  window.TT_PALETTE = {
    add: function (entries) {
      items = BASE.concat(entries);
      if (!pal.hidden) render();
    }
  };
})();
