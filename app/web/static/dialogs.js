/* Styled, in-page dialogs used everywhere instead of the browser's native
 * alert()/confirm(). Native dialogs are inconsistent (and often suppressed or
 * auto-accepted) inside embedded webviews, so we route everything through one
 * shared <dialog class="modal"> that matches the app's styling.
 *
 * Two ways to use it:
 *   1. Programmatic  — `await confirmDialog(msg, opts)` / `await alertDialog(msg)`.
 *   2. Declarative   — add `data-confirm="Are you sure?"` to a form, submit
 *                      button, or link. Optional: data-confirm-title,
 *                      data-confirm-ok, data-confirm-cancel, data-confirm-variant.
 *
 * window.alert is transparently replaced (callers ignore its return value, so
 * async display is fine). window.confirm() stays native for any legacy
 * synchronous caller, but the app itself uses confirmDialog / data-confirm.
 */
(function () {
  'use strict';

  let dlg, titleEl, msgEl, okBtn, cancelBtn, closeBtn;

  function ensureDialog() {
    if (dlg) return dlg;
    dlg = document.createElement('dialog');
    dlg.className = 'modal app-dialog';
    dlg.innerHTML =
      '<div class="modal-head">' +
        '<strong data-role="title">Please confirm</strong>' +
        '<button type="button" class="btn ghost sm" data-role="close" aria-label="Close">✕</button>' +
      '</div>' +
      '<p class="meta dialog-msg" data-role="msg"></p>' +
      '<div class="btn-group" style="justify-content: flex-end;">' +
        '<button type="button" class="btn ghost" data-role="cancel">Cancel</button>' +
        '<button type="button" class="primary" data-role="ok">OK</button>' +
      '</div>';
    (document.body || document.documentElement).appendChild(dlg);
    titleEl = dlg.querySelector('[data-role="title"]');
    msgEl = dlg.querySelector('[data-role="msg"]');
    okBtn = dlg.querySelector('[data-role="ok"]');
    cancelBtn = dlg.querySelector('[data-role="cancel"]');
    closeBtn = dlg.querySelector('[data-role="close"]');
    return dlg;
  }

  function open(opts) {
    // No <dialog> support: degrade to the native equivalents.
    if (typeof document.createElement('dialog').showModal !== 'function') {
      if (opts.showCancel === false) { window.__nativeAlert(opts.message); return Promise.resolve(true); }
      return Promise.resolve(window.confirm(opts.message));
    }
    ensureDialog();
    titleEl.textContent = opts.title || 'Please confirm';
    msgEl.textContent = opts.message || '';
    okBtn.textContent = opts.okText || 'OK';
    okBtn.className = opts.okVariant || 'primary';
    cancelBtn.textContent = opts.cancelText || 'Cancel';
    const hideCancel = opts.showCancel === false;
    cancelBtn.style.display = hideCancel ? 'none' : '';
    closeBtn.style.display = hideCancel ? 'none' : '';

    return new Promise(function (resolve) {
      let done = false;
      function finish(val) {
        if (done) return;
        done = true;
        okBtn.removeEventListener('click', onOk);
        cancelBtn.removeEventListener('click', onCancel);
        closeBtn.removeEventListener('click', onCancel);
        dlg.removeEventListener('cancel', onEsc);
        try { dlg.close(); } catch (e) {}
        resolve(val);
      }
      function onOk() { finish(true); }
      function onCancel() { finish(false); }
      function onEsc(e) { e.preventDefault(); finish(false); }
      okBtn.addEventListener('click', onOk);
      cancelBtn.addEventListener('click', onCancel);
      closeBtn.addEventListener('click', onCancel);
      dlg.addEventListener('cancel', onEsc);
      dlg.showModal();
      okBtn.focus();
    });
  }

  window.confirmDialog = function (message, opts) {
    opts = opts || {};
    return open({
      title: opts.title || 'Please confirm',
      message: message,
      okText: opts.okText || 'OK',
      okVariant: opts.okVariant || 'primary',
      cancelText: opts.cancelText || 'Cancel',
      showCancel: true,
    });
  };

  window.alertDialog = function (message, opts) {
    opts = opts || {};
    return open({
      title: opts.title || 'Heads up',
      message: message,
      okText: opts.okText || 'OK',
      okVariant: opts.okVariant || 'primary',
      showCancel: false,
    });
  };

  // Keep a handle on the real alert, then route alert() through the styled one.
  window.__nativeAlert = window.alert.bind(window);
  window.alert = function (message) { window.alertDialog(message == null ? '' : String(message)); };

  // ---- Declarative data-confirm interception --------------------------------
  function optsFrom(el) {
    return {
      title: el.getAttribute('data-confirm-title') || 'Please confirm',
      okText: el.getAttribute('data-confirm-ok') || 'OK',
      okVariant: el.getAttribute('data-confirm-variant') || 'primary',
      cancelText: el.getAttribute('data-confirm-cancel') || 'Cancel',
    };
  }

  function isSubmitter(el) {
    return (el.tagName === 'BUTTON' && (el.type === 'submit' || el.type === '' || !el.type)) ||
           (el.tagName === 'INPUT' && el.type === 'submit');
  }

  // Submit buttons and links: intercept the click so we can honor formaction /
  // formnovalidate via requestSubmit(button) once the user confirms. This goes
  // through the form's submit path (the button, not the form, carries
  // data-confirm), so it won't re-trigger this handler.
  document.addEventListener('click', function (e) {
    const el = e.target.closest('[data-confirm]');
    if (!el) return;
    if (el.tagName !== 'A' && !isSubmitter(el)) return;
    e.preventDefault();
    e.stopPropagation();
    window.confirmDialog(el.getAttribute('data-confirm'), optsFrom(el)).then(function (ok) {
      if (!ok) return;
      if (el.tagName === 'A') { window.location.href = el.href; return; }
      const form = el.form || el.closest('form');
      if (!form) return;
      if (form.requestSubmit) form.requestSubmit(isSubmitter(el) ? el : undefined);
      else form.submit();
    });
  }, true);

  // Forms carrying data-confirm on the <form> itself (no specific submitter).
  document.addEventListener('submit', function (e) {
    const form = e.target;
    if (!form.hasAttribute || !form.hasAttribute('data-confirm')) return;
    if (form.dataset._confirmed === '1') { form.dataset._confirmed = ''; return; }
    e.preventDefault();
    e.stopPropagation();
    window.confirmDialog(form.getAttribute('data-confirm'), optsFrom(form)).then(function (ok) {
      if (!ok) return;
      form.dataset._confirmed = '1';
      if (form.requestSubmit) form.requestSubmit();
      else form.submit();
    });
  }, true);
})();
