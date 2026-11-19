// "Start with Throughline": the landing page's form posts here, and the inquiry
// is emailed to us through Resend. Vercel runs this as a serverless function.
//
// Environment (Vercel project settings, never in the repo):
//   RESEND_API_KEY  a Resend key with sending access
//   LEAD_TO         where inquiries go (comma-separated for several)
//   LEAD_FROM       optional; defaults to Continuity Marketing <hello@continuity.marketing>

const LIMITS = { name: 120, email: 200, organization: 160, website: 300, message: 4000 };

function clean(value, max) {
  return String(value == null ? "" : value).replace(/\s+$/g, "").slice(0, max);
}

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

module.exports = async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ error: "POST only" });
  }
  let body = req.body;
  if (typeof body === "string") {
    try { body = JSON.parse(body); } catch { body = {}; }
  }
  body = body || {};

  // Bots fill every field; people never see this one.
  if (clean(body.company_url, 300)) return res.status(200).json({ ok: true });

  const lead = {};
  for (const [field, max] of Object.entries(LIMITS)) lead[field] = clean(body[field], max).trim();
  if (!lead.name || !lead.organization || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(lead.email)) {
    return res.status(400).json({ error: "name, organization and a valid email are required" });
  }

  const key = process.env.RESEND_API_KEY;
  const to = (process.env.LEAD_TO || "").split(",").map((s) => s.trim()).filter(Boolean);
  if (!key || !to.length) {
    console.error("start: RESEND_API_KEY or LEAD_TO is not set");
    return res.status(500).json({ error: "not configured" });
  }

  const rows = [
    ["Name", lead.name], ["Email", lead.email], ["Organization", lead.organization],
    ["Website", lead.website || "—"], ["What they want people to hear", lead.message || "—"],
  ];
  const text = rows.map(([k, v]) => `${k}: ${v}`).join("\n\n");
  const html = `<h2 style="font-family:Georgia,serif;font-weight:400">New Throughline inquiry</h2>` +
    rows.map(([k, v]) => `<p><b>${k}</b><br>${escapeHtml(v).replace(/\n/g, "<br>")}</p>`).join("");

  const r = await fetch("https://api.resend.com/emails", {
    method: "POST",
    headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      from: process.env.LEAD_FROM || "Continuity Marketing <hello@continuity.marketing>",
      to,
      reply_to: lead.email,
      subject: `Throughline inquiry: ${lead.organization}`,
      text,
      html,
    }),
  });
  if (!r.ok) {
    console.error("start: Resend refused", r.status, await r.text());
    return res.status(502).json({ error: "could not send" });
  }
  return res.status(200).json({ ok: true });
};
