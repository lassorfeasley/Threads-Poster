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
 *   Client model (filter + sort rows in place — see initClientFilters below):
 *     <input   data-filter-input data-filter-target="#list">   (searches data-search)
 *     <select  data-filter-input data-filter-target="#list" data-filter-key="country">
 *     <details class="combo" data-filter-input …>              (any-of match)
 *     <select  data-sort-target="#list">                       (reorder rows)
 *     <ul id="list"> <li data-filter-row data-search="…" data-country="…">
 *
 *   View tabs (peer views of one page — see initViewTabs below):
 *     <div role="tablist" data-view-tabs="storeKey" data-view-param="view">
 *       <button role="tab" data-tab="a" id="a-tab" aria-controls="a-panel">
 *     <div id="a-panel" role="tabpanel" aria-labelledby="a-tab">
 *     <span data-view-slot="a b">          controls only view a and b offer
 */
(function () {
  'use strict';

  var SEARCH_DEBOUNCE_MS = 400;
  var FOCUS_FLAG = 'toolbarFocusSearch';
  // Rows are hidden with a class, not the `hidden` attribute: card/table/flex
  // display rules outrank the attribute's UA style and would leave "hidden"
  // rows on screen.
  var HIDE = 'tb-hidden';

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

      // Multi-select combo dropdowns: apply once on close, so several options
      // can be toggled in one go before the page reloads.
      form.querySelectorAll('details.combo').forEach(function (combo) {
        var dirty = false;
        combo.addEventListener('change', function () { dirty = true; });
        combo.addEventListener('toggle', function () {
          if (!combo.open && dirty) { dirty = false; form.requestSubmit(); }
        });
      });
    });
  }

  // ---- Multi-select combos: live summary label + close on outside click ----
  function comboChecked(combo) {
    var checked = [];
    combo.querySelectorAll('input[type="checkbox"]').forEach(function (cb) {
      if (cb.checked) checked.push(cb);
    });
    return checked;
  }

  function updateComboLabel(combo) {
    var checked = comboChecked(combo);
    combo.classList.toggle('is-set', checked.length > 0);
    var out = combo.querySelector('[data-combo-label]');
    if (!out) return;
    if (!checked.length) out.textContent = out.getAttribute('data-combo-empty') || 'All';
    else if (checked.length === 1) out.textContent = checked[0].getAttribute('data-combo-text') || checked[0].value;
    else out.textContent = checked.length + ' selected';
  }

  // The panel hangs off the left edge of its summary by default; flip it to the
  // right edge when that would push it past the viewport.
  function positionComboPanel(combo) {
    var panel = combo.querySelector('.combo-panel');
    if (!panel) return;
    combo.classList.remove('combo-flip');
    if (panel.getBoundingClientRect().right > document.documentElement.clientWidth - 8) {
      combo.classList.add('combo-flip');
    }
  }

  function initCombos() {
    document.querySelectorAll('details.combo').forEach(function (combo) {
      if (combo.__comboInit) return;
      combo.__comboInit = true;
      combo.addEventListener('change', function () { updateComboLabel(combo); });
      combo.addEventListener('toggle', function () {
        if (combo.open) positionComboPanel(combo);
      });
      updateComboLabel(combo);
    });
  }

  function closeCombos(except) {
    document.querySelectorAll('details.combo[open]').forEach(function (combo) {
      if (!except || !combo.contains(except)) combo.open = false;
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

  // ---- Client model: filter + sort rows in place ----
  //
  // Controls opt in with data-filter-input + data-filter-target="#container";
  // the things they filter are [data-filter-row] elements inside it.
  //
  //   text input                          substring match on the row's data-search
  //   select[data-filter-key]             exact match on data-<key> ('' = any)
  //   details.combo[data-filter-key]      any-of match over the checked boxes
  //   input[type=date][data-filter-key]   ISO compare, data-filter-op="gte"|"lte"
  //
  // A control inside a .tb-hidden wrapper sits out, so a page with tabbed
  // sections can offer only the filters that apply to the open tab.
  //
  // Optional companions, all keyed off the same target selector:
  //   [data-filter-clear="#c"]     reset every control (hides itself when idle)
  //   [data-filter-active]         inside a clear button: count of live filters
  //   [data-sort-target="#c"]      reorder rows; values are "<key>-asc|desc"
  //                                and read data-sort-<key> off each row. A
  //                                tabbed page may render one per tab — the
  //                                first live one sorts.
  //   [data-filter-section="name"] scope for counts + a filtered-empty notice
  //   [data-filter-count="name"]   live count of matching rows in that section
  //   [data-filter-empty]          shown when a non-empty section filters to 0
  var filterGroups = {};   // target selector -> controls
  var sortControls = {};   // target selector -> selects

  function initClientFilters() {
    document.querySelectorAll('[data-filter-input]').forEach(function (control) {
      if (control.__toolbarFilter) return;
      control.__toolbarFilter = true;
      var target = control.getAttribute('data-filter-target');
      if (!target) return;
      (filterGroups[target] = filterGroups[target] || []).push(control);
      // Checkbox `change` bubbles up from inside a combo, so one listener on
      // the <details> covers every option in it.
      var ev = (control.tagName === 'INPUT' && control.type === 'text') ? 'input' : 'change';
      control.addEventListener(ev, function () { applyFilter(target); });
    });

    document.querySelectorAll('[data-filter-clear]').forEach(function (btn) {
      if (btn.__toolbarClear) return;
      btn.__toolbarClear = true;
      var target = btn.getAttribute('data-filter-clear');
      btn.addEventListener('click', function () {
        (filterGroups[target] || []).forEach(resetControl);
        applyFilter(target);
        var search = document.querySelector('[data-filter-target="' + target + '"].tb-search-input');
        if (search) search.focus();
      });
    });

    document.querySelectorAll('[data-sort-target]').forEach(function (select) {
      if (select.__toolbarSort) return;
      select.__toolbarSort = true;
      var target = select.getAttribute('data-sort-target');
      (sortControls[target] = sortControls[target] || []).push(select);
      select.addEventListener('change', function () { applyFilter(target); });
    });

    // Remember each row's server-rendered position so "default" sort can undo.
    Object.keys(filterGroups).concat(Object.keys(sortControls)).forEach(function (target) {
      var container = document.querySelector(target);
      if (!container) return;
      container.querySelectorAll('[data-filter-row]').forEach(function (row, i) {
        if (row.__order === undefined) row.__order = i;
      });
    });

    Object.keys(filterGroups).forEach(applyFilter);
  }

  function resetControl(control) {
    if (control.tagName === 'DETAILS') {
      control.querySelectorAll('input[type="checkbox"]').forEach(function (cb) { cb.checked = false; });
      updateComboLabel(control);
    } else {
      control.value = '';
    }
  }

  // Controls in a hidden wrapper (a tab that isn't showing) don't constrain.
  function isLive(control) {
    return !control.closest('.' + HIDE) && !control.closest('[hidden]');
  }

  function isSet(control) {
    if (control.tagName === 'DETAILS') return comboChecked(control).length > 0;
    return (control.value || '').trim() !== '';
  }

  function matches(control, row) {
    var key = control.getAttribute('data-filter-key');
    if (control.tagName === 'DETAILS') {
      var checked = comboChecked(control);
      if (!checked.length) return true;
      var rowVal = (row.getAttribute('data-' + key) || '').toLowerCase();
      return checked.some(function (cb) { return cb.value.toLowerCase() === rowVal; });
    }
    var val = (control.value || '').trim().toLowerCase();
    if (!val) return true;
    if (control.type === 'date') {
      var when = (row.getAttribute('data-' + key) || '').slice(0, 10);
      if (!when) return false;
      return control.getAttribute('data-filter-op') === 'lte' ? when <= val : when >= val;
    }
    if (key) return (row.getAttribute('data-' + key) || '').toLowerCase() === val;
    // Free-text search: every whitespace-separated term has to appear, so
    // "kxyz flood" narrows rather than finding nothing.
    var haystack = (row.getAttribute('data-search') || '').toLowerCase();
    return val.split(/\s+/).every(function (term) { return haystack.indexOf(term) !== -1; });
  }

  function applyFilter(targetSel) {
    var container = document.querySelector(targetSel);
    if (!container) return;
    var controls = (filterGroups[targetSel] || []).filter(isLive);

    container.querySelectorAll('[data-filter-row]').forEach(function (row) {
      var ok = controls.every(function (control) { return matches(control, row); });
      row.classList.toggle(HIDE, !ok);
    });

    applySort(container, (sortControls[targetSel] || []).filter(isLive)[0]);

    var sections = [].slice.call(container.querySelectorAll('[data-filter-section]'));
    if (container.matches('[data-filter-section]')) sections.unshift(container);
    sections.forEach(function (section) {
      var rows = section.querySelectorAll('[data-filter-row]');
      var shown = 0;
      rows.forEach(function (row) { if (!row.classList.contains(HIDE)) shown++; });
      var name = section.getAttribute('data-filter-section');
      document.querySelectorAll('[data-filter-count="' + name + '"]').forEach(function (el) {
        el.textContent = shown;
      });
      var empty = section.querySelector('[data-filter-empty]');
      if (empty) empty.classList.toggle(HIDE, !(rows.length && !shown));
    });

    // Clear button: show a live count of what's applied, and stay out of the
    // way entirely while nothing is.
    var active = controls.filter(isSet).length;
    document.querySelectorAll('[data-filter-clear="' + targetSel + '"]').forEach(function (btn) {
      btn.classList.toggle(HIDE, active === 0);
      var badge = btn.querySelector('[data-filter-active]');
      if (badge) badge.textContent = active;
    });
  }

  function applySort(container, select) {
    if (!select) return;
    var parts = (select.value || '').split('-');
    var key = parts[0];
    var dir = parts[1] === 'asc' ? 1 : -1;
    // Sort inside each row's own parent so separate grids/tables keep their rows.
    var groups = [];
    container.querySelectorAll('[data-filter-row]').forEach(function (row) {
      var parent = row.parentElement;
      var group = groups.find(function (g) { return g.parent === parent; });
      if (!group) groups.push(group = { parent: parent, rows: [] });
      group.rows.push(row);
    });
    groups.forEach(function (group) {
      group.rows.sort(function (a, b) {
        if (!key) return a.__order - b.__order;
        var av = a.getAttribute('data-sort-' + key) || '';
        var bv = b.getAttribute('data-sort-' + key) || '';
        // Rows missing the value sink to the bottom either way round.
        if (!av || !bv) return (!av && !bv) ? a.__order - b.__order : (av ? -1 : 1);
        var cmp = av.localeCompare(bv, undefined, { numeric: true, sensitivity: 'base' });
        return cmp ? cmp * dir : a.__order - b.__order;
      });
      group.rows.forEach(function (row) { group.parent.appendChild(row); });
    });
  }

  // Pages with tabbed sections call this after showing/hiding tab-scoped
  // filters, so the newly relevant controls take effect right away.
  window.refreshToolbarFilters = function () {
    Object.keys(filterGroups).forEach(applyFilter);
  };

  // ---- View tabs: peer views of one page, swapped in place ----
  //
  //   <div role="tablist" data-view-tabs="dashTab" data-view-param="view">
  //     <button role="tab" data-tab="candidates" id="candidates-tab"
  //             aria-controls="candidates-panel">
  //   <div id="candidates-panel" role="tabpanel" aria-labelledby="candidates-tab">
  //
  // A tab owns its panel through aria-controls, so there is no separate map to
  // keep in sync. Elements marked data-view-slot="videos cuts" show only for
  // the views they name, which is how a page offers filters that apply to the
  // open view alone.
  //
  // The open view is addressable: it resolves from the query param first (so a
  // link or a reload wins), then the last choice in localStorage, then whatever
  // the server rendered as selected. Switching pushes a history entry, so back
  // and forward step between views.
  function viewTabs(strip) {
    return [].slice.call(strip.querySelectorAll('[role="tab"][data-tab]'));
  }

  function viewKey(tab) {
    return tab.getAttribute('data-tab');
  }

  function showView(strip, key, opts) {
    opts = opts || {};
    var tabs = viewTabs(strip);
    var keys = tabs.map(viewKey);
    if (keys.indexOf(key) === -1) key = keys[0];
    if (!key) return;

    tabs.forEach(function (tab) {
      var on = viewKey(tab) === key;
      tab.classList.toggle('is-active', on);
      tab.setAttribute('aria-selected', on ? 'true' : 'false');
      // Roving tabindex: the strip is one stop, arrows move inside it.
      tab.tabIndex = on ? 0 : -1;
      var panel = document.getElementById(tab.getAttribute('aria-controls') || '');
      if (panel) panel.hidden = !on;
      if (on && opts.focus) tab.focus();
    });

    document.querySelectorAll('[data-view-slot]').forEach(function (slot) {
      var views = slot.getAttribute('data-view-slot').split(/\s+/);
      slot.classList.toggle(HIDE, views.indexOf(key) === -1);
    });

    var store = strip.getAttribute('data-view-tabs');
    if (store) { try { localStorage.setItem(store, key); } catch (e) {} }

    var param = strip.getAttribute('data-view-param');
    if (param) {
      var url = new URL(location.href);
      if (url.searchParams.get(param) !== key) {
        url.searchParams.set(param, key);
        history[opts.push ? 'pushState' : 'replaceState']({}, '', url);
      }
    }

    // Filters belonging to the view that just closed must stop constraining.
    window.refreshToolbarFilters();
  }

  function initialView(strip) {
    var keys = viewTabs(strip).map(viewKey);
    var param = strip.getAttribute('data-view-param');
    if (param) {
      var linked = new URLSearchParams(location.search).get(param);
      if (linked && keys.indexOf(linked) !== -1) return linked;
    }
    var store = strip.getAttribute('data-view-tabs');
    if (store) {
      var saved = null;
      try { saved = localStorage.getItem(store); } catch (e) {}
      if (saved && keys.indexOf(saved) !== -1) return saved;
    }
    var marked = strip.querySelector('[role="tab"][data-tab][aria-selected="true"]');
    return marked ? viewKey(marked) : keys[0];
  }

  function initViewTabs() {
    document.querySelectorAll('[data-view-tabs]').forEach(function (strip) {
      if (strip.__viewTabsInit) return;
      strip.__viewTabsInit = true;

      strip.addEventListener('click', function (e) {
        var tab = e.target.closest('[role="tab"][data-tab]');
        if (tab) showView(strip, viewKey(tab), { push: true });
      });

      strip.addEventListener('keydown', function (e) {
        var tabs = viewTabs(strip);
        var i = tabs.indexOf(document.activeElement);
        if (i === -1) return;
        var to = null;
        if (e.key === 'ArrowRight' || e.key === 'ArrowDown') to = tabs[(i + 1) % tabs.length];
        else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') to = tabs[(i - 1 + tabs.length) % tabs.length];
        else if (e.key === 'Home') to = tabs[0];
        else if (e.key === 'End') to = tabs[tabs.length - 1];
        if (!to) return;
        e.preventDefault();
        showView(strip, viewKey(to), { push: true, focus: true });
      });

      showView(strip, initialView(strip));
    });

    window.addEventListener('popstate', function () {
      document.querySelectorAll('[data-view-tabs]').forEach(function (strip) {
        var param = strip.getAttribute('data-view-param');
        var key = param && new URLSearchParams(location.search).get(param);
        if (key) showView(strip, key);
      });
    });
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

  // ---- Actions that don't need a page load ---------------------------------
  // A form marked data-async posts in the background instead of navigating.
  // The page its redirect would land on is usually the one already on screen,
  // and re-rendering it costs a full round of database reads for a change the
  // operator can already see. Composes with data-confirm (dialogs.js re-submits
  // through the form, so this handler still sees it).
  //
  //   data-async="Queued"           what to confirm with, as a toast
  //   data-async-remove="selector"  ancestor of the form to drop on success
  //   data-async-then="fnName"      window function called with (form, data)
  //
  // The endpoint must answer JSON to an Accept: application/json request —
  // otherwise fetch follows the redirect and re-renders the page anyway.
  function initAsyncForms() {
    document.addEventListener('submit', function (e) {
      var form = e.target;
      if (!form.hasAttribute || !form.hasAttribute('data-async')) return;
      e.preventDefault();
      var btn = e.submitter || form.querySelector('button:not([type="button"])');
      setButtonBusy(btn, true);
      fetch(form.action, {
        method: 'POST',
        headers: { 'Accept': 'application/json' },
        body: new FormData(form),
      }).then(function (r) {
        return r.json().catch(function () { return {}; }).then(function (d) {
          if (!r.ok) throw new Error(d.error || ('HTTP ' + r.status));
          return d;
        });
      }).then(function (d) {
        setButtonBusy(btn, false);
        var sel = form.getAttribute('data-async-remove');
        var gone = sel && form.closest(sel);
        if (gone) gone.remove();
        var then = form.getAttribute('data-async-then');
        if (then && typeof window[then] === 'function') window[then](form, d);
        var msg = form.getAttribute('data-async') || d.msg;
        if (msg) showToast(msg);
      }).catch(function (err) {
        setButtonBusy(btn, false);
        showToast(String((err && err.message) || err), { variant: 'error' });
      });
    });
  }

  ready(function () {
    updateScrollbarWidth();
    initCombos();
    initServerForms();
    // Before the filters: opening a view hides the slots belonging to the
    // others, and hidden controls have to be settled before the first pass.
    initViewTabs();
    initClientFilters();
    restoreSearchFocus();
    initToasts();
    initSubmitPending();
    initAsyncForms();
    document.addEventListener('click', function (e) { closeCombos(e.target); });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') closeCombos(null);
    });
    // Recompute when the layout reflows (resize, or content growing/shrinking
    // enough to add or remove the scrollbar — e.g. client-side filtering).
    window.addEventListener('resize', updateScrollbarWidth);
    if (window.ResizeObserver) {
      new ResizeObserver(updateScrollbarWidth).observe(document.body);
    }
  });
})();
