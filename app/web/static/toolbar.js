/*
 * Shared behaviour for the pagebar toolbar component (see
 * templates/components/toolbar.html). Everything is driven by data- attributes
 * so pages stay declarative and keep their own server/client filtering model.
 *
 *   Server model (GET form):
 *     <form data-toolbar-form>
 *       <input data-toolbar-search>          debounced auto-submit + focus restore
 *       <select data-submit-on-change>       submit on change
 *       <details class="combo">              multi-select; submit on close if changed
 *
 *   Client model (filter rows in place):
 *     <input  data-filter-input data-filter-target="#list">   (searches data-search)
 *     <select data-filter-input data-filter-target="#list" data-filter-key="country">
 *     <ul id="list"> <li data-filter-row data-search="…" data-country="…">
 */
(function () {
  'use strict';

  var SEARCH_DEBOUNCE_MS = 400;
  var FOCUS_FLAG = 'toolbarFocusSearch';

  function ready(fn) {
    if (document.readyState !== 'loading') fn();
    else document.addEventListener('DOMContentLoaded', fn);
  }

  // ---- Scrollbar width: keep the full-bleed sticky pagebar aligned ----
  // The band spans the viewport with vw units, which include the vertical
  // scrollbar, while the centered content column does not. Expose the live
  // scrollbar width as --sbw so the CSS can cancel it out; without this the
  // header shifts under the sidebar on pages long enough to scroll.
  function updateScrollbarWidth() {
    var sbw = window.innerWidth - document.documentElement.clientWidth;
    if (sbw < 0) sbw = 0;
    document.documentElement.style.setProperty('--sbw', sbw + 'px');
  }

  // ---- Server model: debounced search + submit-on-change + combo ----
  function initServerForms() {
    document.querySelectorAll('form[data-toolbar-form]').forEach(function (form) {
      if (form.__toolbarInit) return;
      form.__toolbarInit = true;

      // Debounced search box: submit a short beat after the user stops typing,
      // remembering to restore focus + caret after the reload.
      form.querySelectorAll('[data-toolbar-search]').forEach(function (input) {
        var timer = null;
        input.addEventListener('input', function () {
          clearTimeout(timer);
          timer = setTimeout(function () {
            try { sessionStorage.setItem(FOCUS_FLAG, input.name || '1'); } catch (e) {}
            form.requestSubmit();
          }, SEARCH_DEBOUNCE_MS);
        });
      });

      // Selects / date inputs that filter immediately.
      form.querySelectorAll('[data-submit-on-change]').forEach(function (el) {
        el.addEventListener('change', function () { form.requestSubmit(); });
      });

      // Multi-select combo dropdowns: apply once on close (so several options can
      // be toggled in one go), and close when clicking outside.
      form.querySelectorAll('details.combo').forEach(function (combo) {
        var dirty = false;
        combo.querySelectorAll('input[type="checkbox"]').forEach(function (cb) {
          cb.addEventListener('change', function () { dirty = true; });
        });
        combo.addEventListener('toggle', function () {
          if (!combo.open && dirty) { dirty = false; form.requestSubmit(); }
        });
        document.addEventListener('click', function (e) {
          if (combo.open && !combo.contains(e.target)) combo.open = false;
        });
      });
    });
  }

  function restoreSearchFocus() {
    var flag = null;
    try { flag = sessionStorage.getItem(FOCUS_FLAG); sessionStorage.removeItem(FOCUS_FLAG); } catch (e) {}
    if (!flag) return;
    var el = flag !== '1'
      ? document.querySelector('[data-toolbar-search][name="' + flag + '"]')
      : document.querySelector('[data-toolbar-search]');
    if (el) { el.focus(); var v = el.value; el.value = ''; el.value = v; }
  }

  // ---- Client model: filter rows in place ----
  function initClientFilters() {
    var groups = {};   // target selector -> [inputs]
    document.querySelectorAll('[data-filter-input]').forEach(function (input) {
      if (input.__toolbarFilter) return;
      input.__toolbarFilter = true;
      var target = input.getAttribute('data-filter-target');
      if (!target) return;
      (groups[target] = groups[target] || []).push(input);
      var ev = input.tagName === 'SELECT' ? 'change' : 'input';
      input.addEventListener(ev, function () { applyFilter(target, groups[target]); });
    });
    // Clear buttons reset every input pointed at their target, then re-filter.
    document.querySelectorAll('[data-filter-clear]').forEach(function (btn) {
      if (btn.__toolbarClear) return;
      btn.__toolbarClear = true;
      var target = btn.getAttribute('data-filter-clear');
      btn.addEventListener('click', function () {
        (groups[target] || []).forEach(function (input) { input.value = ''; });
        applyFilter(target, groups[target] || []);
      });
    });
    Object.keys(groups).forEach(function (t) { applyFilter(t, groups[t]); });
  }

  function applyFilter(targetSel, inputs) {
    var container = document.querySelector(targetSel);
    if (!container) return;
    var rows = container.querySelectorAll('[data-filter-row]');
    var shown = 0;
    rows.forEach(function (row) {
      var ok = inputs.every(function (input) {
        var val = (input.value || '').trim().toLowerCase();
        if (!val) return true;
        if (input.hasAttribute('data-filter-key')) {
          var key = input.getAttribute('data-filter-key');
          return (row.getAttribute('data-' + key) || '').toLowerCase() === val;
        }
        return (row.getAttribute('data-search') || '').toLowerCase().indexOf(val) !== -1;
      });
      row.hidden = !ok;
      if (ok) shown++;
    });
    var count = container.parentElement
      ? container.parentElement.querySelector('[data-filter-count]')
      : null;
    if (!count) count = document.querySelector('[data-filter-count][data-for="' + targetSel.replace('#', '') + '"]');
    if (count) count.textContent = shown;
  }

  // ---- Toast flash messages: auto-dismiss + manual close ----
  function initToasts() {
    document.querySelectorAll('.toast').forEach(function (toast) {
      if (toast.__toastInit) return;
      toast.__toastInit = true;
      var timer = null;
      function dismiss() {
        clearTimeout(timer);
        toast.classList.add('toast-hide');
        toast.addEventListener('animationend', function () {
          var wrap = toast.parentElement;
          toast.remove();
          if (wrap && wrap.classList.contains('toast-wrap') && !wrap.children.length) wrap.remove();
        }, { once: true });
      }
      var close = toast.querySelector('.toast-x');
      if (close) close.addEventListener('click', dismiss);
      timer = setTimeout(dismiss, 4500);
      // Pause the auto-dismiss while hovering so it can be read.
      toast.addEventListener('mouseenter', function () { clearTimeout(timer); });
      toast.addEventListener('mouseleave', function () { timer = setTimeout(dismiss, 2000); });
    });
  }

  ready(function () {
    updateScrollbarWidth();
    initServerForms();
    initClientFilters();
    restoreSearchFocus();
    initToasts();
    // Recompute when the layout reflows (resize, or content growing/shrinking
    // enough to add or remove the scrollbar — e.g. client-side filtering).
    window.addEventListener('resize', updateScrollbarWidth);
    if (window.ResizeObserver) {
      new ResizeObserver(updateScrollbarWidth).observe(document.body);
    }
  });
})();
