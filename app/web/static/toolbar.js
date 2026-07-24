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
  var TOAST_ICON_OK =
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>';
  var TOAST_ICON_ERR =
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 8v5M12 16h.01"/></svg>';

  // Auto-dismiss + manual close for a single toast node (server- or JS-created).
  function wireToast(toast) {
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
  }

  function initToasts() {
    document.querySelectorAll('.toast').forEach(wireToast);
  }

  // Programmatic toast so client-side actions (e.g. optimistic calendar moves)
  // can confirm success/failure without a full page reload. opts.variant:
  // 'error' for a red/failure toast; anything else is the default success look.
  function showToast(message, opts) {
    opts = opts || {};
    if (message == null || message === '') return null;
    var wrap = document.querySelector('.toast-wrap');
    if (!wrap) {
      wrap = document.createElement('div');
      wrap.className = 'toast-wrap';
      wrap.setAttribute('aria-live', 'polite');
      (document.body || document.documentElement).appendChild(wrap);
    }
    var isErr = opts.variant === 'error';
    var toast = document.createElement('div');
    toast.className = 'toast' + (isErr ? ' toast-error' : '');
    toast.setAttribute('role', isErr ? 'alert' : 'status');
    toast.innerHTML =
      '<span class="toast-ic">' + (isErr ? TOAST_ICON_ERR : TOAST_ICON_OK) + '</span>' +
      '<span class="toast-msg"></span>' +
      '<button type="button" class="toast-x" aria-label="Dismiss">\u2715</button>';
    toast.querySelector('.toast-msg').textContent = String(message);
    wrap.appendChild(toast);
    wireToast(toast);
    return toast;
  }
  window.toast = showToast;

  // ---- Instant feedback: mark the clicked action button busy on submit ----
  // POST actions round-trip to the server (often a remote DB), so the page can
  // sit for a beat. Flip the submitter into a spinner state the moment it's
  // used so the click always feels acknowledged. Disabling on a 0ms timeout
  // keeps the button's name/value in the submitted payload.
  var lastSubmitter = null;

  function setButtonBusy(btn, busy) {
    if (!btn) return;
    if (busy) {
      if (btn.classList.contains('is-loading')) return;
      // Lock the width so swapping the label for a spinner doesn't jump.
      if (btn.offsetWidth) btn.style.minWidth = btn.offsetWidth + 'px';
      btn.classList.add('is-loading');
      btn.setAttribute('aria-busy', 'true');
      setTimeout(function () { btn.disabled = true; }, 0);
    } else {
      btn.classList.remove('is-loading');
      btn.removeAttribute('aria-busy');
      btn.disabled = false;
      btn.style.minWidth = '';
    }
  }

  function initSubmitPending() {
    document.addEventListener('click', function (e) {
      var b = e.target.closest('button, input[type="submit"], input[type="image"]');
      if (b && (b.form || b.closest('form'))) lastSubmitter = b;
    }, true);

    document.addEventListener('submit', function (e) {
      var form = e.target;
      if (!form || (form.method && form.method.toLowerCase() !== 'post')) return;
      // These manage their own UX (search auto-submit / bespoke async handlers).
      if (form.hasAttribute('data-toolbar-form') || form.hasAttribute('data-async')) return;
      if (form.hasAttribute('data-no-pending')) return;
      var btn = e.submitter || lastSubmitter;
      if (btn && (btn.form || btn.closest('form')) !== form) btn = null;
      if (!btn) btn = form.querySelector('button:not([type="button"]), input[type="submit"]');
      if (btn && btn.type !== 'button' && !btn.hasAttribute('data-no-pending')) setButtonBusy(btn, true);
    });

    // Restore buttons if the page is served from the bfcache (back/forward),
    // otherwise a navigated-away action would come back stuck spinning.
    window.addEventListener('pageshow', function () {
      document.querySelectorAll('.is-loading').forEach(function (b) { setButtonBusy(b, false); });
    });
  }

  ready(function () {
    updateScrollbarWidth();
    initServerForms();
    initClientFilters();
    restoreSearchFocus();
    initToasts();
    initSubmitPending();
    // Recompute when the layout reflows (resize, or content growing/shrinking
    // enough to add or remove the scrollbar — e.g. client-side filtering).
    window.addEventListener('resize', updateScrollbarWidth);
    if (window.ResizeObserver) {
      new ResizeObserver(updateScrollbarWidth).observe(document.body);
    }
  });
})();
