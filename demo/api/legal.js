// /privacy and /terms (rewritten here by vercel.json). Reads published versions
// of the document from the platform Supabase project's legal_documents table,
// which the admin site (admin.continuity.marketing/legal) writes. The
// publishable key can only see published versions that are already in effect
// (a row-level security policy in app/admin/legal_schema.sql).
//
// Shows the version in force, or ?v=N for an earlier one, with the history
// underneath.
//
// Environment (Vercel project settings, never in the repo):
//   SUPABASE_URL              the platform project's URL
//   SUPABASE_PUBLISHABLE_KEY  its publishable key (sb_publishable_...). Supabase
//                             retires the legacy anon key by the end of 2026.

const KINDS = { privacy: "Privacy policy", terms: "Terms of use" };
const PATHS = { privacy: "/privacy", terms: "/terms" };

function esc(text) {
  return String(text == null ? "" : text).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function longDate(iso) {
  const [y, m, d] = String(iso).split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, d)).toLocaleDateString("en-US",
    { year: "numeric", month: "long", day: "numeric", timeZone: "UTC" });
}

async function published(kind) {
  const base = (process.env.SUPABASE_URL || "").replace(/\/+$/, "");
  const key = process.env.SUPABASE_PUBLISHABLE_KEY;
  if (!base || !key) throw new Error("SUPABASE_URL / SUPABASE_PUBLISHABLE_KEY not set");
  const url = `${base}/rest/v1/legal_documents?kind=eq.${kind}&status=eq.published`
    + "&select=version,title,summary,body_html,effective_date"
    + "&order=effective_date.desc,version.desc";
  // New-style keys go in the apikey header only; they aren't JWTs.
  const r = await fetch(url, { headers: { apikey: key } });
  if (!r.ok) throw new Error(`Supabase ${r.status}`);
  return r.json();
}

function page({ kind, doc, rows, isCurrent }) {
  const label = KINDS[kind];
  const other = kind === "privacy" ? "terms" : "privacy";
  const history = rows.length > 1 ? `
    <section class="history">
      <h2>Versions</h2>
      <ol>${rows.map((r, i) => `
        <li${doc && r.version === doc.version ? ' class="here"' : ""}>
          <a href="${PATHS[kind]}${i === 0 ? "" : `?v=${r.version}`}">Effective ${esc(longDate(r.effective_date))}</a>${i === 0 ? " <span>current</span>" : ""}
          ${r.summary ? `<p>${esc(r.summary)}</p>` : ""}
        </li>`).join("")}
      </ol>
    </section>` : "";
  const body = doc ? `
      <h1>${esc(doc.title || label)}</h1>
      <p class="meta">${isCurrent ? "Effective" : "This is an earlier version, effective"} ${esc(longDate(doc.effective_date))}.${isCurrent ? "" : ` <a href="${PATHS[kind]}">See the current version.</a>`}</p>
      <div class="doc">${doc.body_html}</div>` : `
      <h1>${label}</h1>
      <p class="meta">This page is being prepared. Questions: <a href="mailto:hello@continuity.marketing">hello@continuity.marketing</a>.</p>`;
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${label} · Continuity Marketing</title>
<meta name="theme-color" content="#f5f2ea">
${isCurrent ? "" : '<meta name="robots" content="noindex">'}
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,300..800;1,400&display=swap">
<style>
:root { --paper: #f5f2ea; --ink: #161512; --ink-2: #3d3a33; --muted: #77715f; --line: #ddd6c5; --accent: #1d5c43;
  --serif: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace; --sans: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace; }
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: var(--sans); font-size: 15px; line-height: 1.7; color: var(--ink); background: var(--paper); -webkit-text-size-adjust: 100%; }
a { color: inherit; }
.wrap { width: min(1120px, calc(100% - 40px)); margin: 0 auto; }
.col { max-width: 680px; }
nav .wrap { display: flex; align-items: center; justify-content: space-between; height: 72px; }
.wordmark { font-family: var(--serif); font-size: 16px; font-weight: 800; letter-spacing: -.03em; text-decoration: none; }
.wordmark em { font-style: normal; font-weight: 400; }
nav a.signin { font-size: 15px; color: var(--ink-2); text-decoration: none; }
main { padding: 56px 0 96px; }
h1, h2 { font-family: var(--serif); font-weight: 800; line-height: 1.1; letter-spacing: -.04em; }
h1 { font-size: clamp(28px, 4.6vw, 40px); }
.meta { margin-top: 12px; color: var(--muted); font-size: 15px; }
.doc { margin-top: 36px; color: var(--ink-2); }
.doc h2 { font-size: 19px; color: var(--ink); margin: 40px 0 10px; }
.doc h3 { font-size: 16px; color: var(--ink); margin: 28px 0 8px; }
.doc p, .doc ul, .doc ol { margin-bottom: 14px; }
.doc ul, .doc ol { padding-left: 22px; }
.doc li { margin-bottom: 6px; }
.doc strong { color: var(--ink); font-weight: 600; }
.history { margin-top: 64px; padding-top: 28px; border-top: 1px solid var(--line); }
.history h2 { font-size: 19px; margin-bottom: 14px; }
.history ol { list-style: none; display: grid; gap: 14px; }
.history li { font-size: 15px; }
.history li.here > a { font-weight: 600; text-decoration: none; }
.history span { font-size: 12px; color: var(--accent); font-weight: 600; margin-left: 6px; }
.history p { color: var(--muted); font-size: 14px; }
footer { padding: 32px 0 48px; border-top: 1px solid var(--line); font-size: 14px; color: var(--muted); }
footer .wrap { display: flex; flex-wrap: wrap; gap: 12px 28px; justify-content: space-between; }
</style>
</head>
<body>
<nav><div class="wrap"><a class="wordmark" href="/">Continuity <em>Marketing</em></a><a class="signin" href="/signin">Sign in</a></div></nav>
<main><div class="wrap"><div class="col">${body}${history}</div></div></main>
<footer><div class="wrap">
  <span><a href="${PATHS[other]}">${KINDS[other]}</a></span>
  <span>© ${new Date().getUTCFullYear()} Continuity Marketing</span>
</div></footer>
</body>
</html>`;
}

module.exports = async function handler(req, res) {
  const kind = String((req.query && req.query.kind) || "");
  if (!KINDS[kind]) return res.status(404).send("Not found");
  let rows;
  try {
    // Before the database is wired up, say the page is being prepared rather than fail.
    rows = (process.env.SUPABASE_URL && process.env.SUPABASE_PUBLISHABLE_KEY) ? await published(kind) : [];
  } catch (err) {
    console.error("legal:", err.message);
    res.setHeader("Cache-Control", "no-store");
    return res.status(503).send("This page is temporarily unavailable. Please try again shortly.");
  }
  const want = Number(req.query.v);
  const doc = want ? rows.find((r) => r.version === want) : rows[0];
  if (want && !doc) {
    res.setHeader("Location", PATHS[kind]);
    return res.status(302).end();
  }
  res.setHeader("Content-Type", "text/html; charset=utf-8");
  res.setHeader("Cache-Control", "public, s-maxage=300, stale-while-revalidate=86400");
  return res.status(200).send(page({ kind, doc, rows, isCurrent: !!doc && doc === rows[0] }));
};
