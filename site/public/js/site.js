/* Reveal-on-scroll and the one-shot hero caret.
 *
 * Motion is deliberately sparse: a fade + 10px rise per section, and a caret
 * that stops blinking once the reader has had a moment with the hero. Both are
 * gated behind prefers-reduced-motion.
 */
(function () {
  "use strict";

  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var nodes = document.querySelectorAll(".reveal");

  if (reduce || !("IntersectionObserver" in window)) {
    // Leave everything visible; never opt into the hidden state we can't undo.
    Array.prototype.forEach.call(nodes, function (n) { n.classList.add("is-in"); });
    return;
  }

  // Only now is it safe to hide: the observer below will bring each node back.
  document.documentElement.classList.add("has-reveal");

  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (!entry.isIntersecting) return;
      entry.target.classList.add("is-in");
      io.unobserve(entry.target);
    });
  }, { rootMargin: "0px 0px -8% 0px", threshold: 0.06 });

  Array.prototype.forEach.call(nodes, function (n) { io.observe(n); });

  // The caret blinks once for the arrival, then settles. It is decoration on a
  // finished run, not a fake "still working" signal.
  var caret = document.getElementById("caret");
  if (caret) window.setTimeout(function () { caret.style.animation = "none"; caret.style.opacity = "0.35"; }, 5200);
})();
