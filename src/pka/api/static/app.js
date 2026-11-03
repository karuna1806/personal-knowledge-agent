"use strict";

const $ = (sel) => document.querySelector(sel);
const store = {
  get(k, d = "") { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};

let conversationId = store.get("pka.conversation") || null;
let currentSources = [];

// ── helpers ──────────────────────────────────────────────────────
const apiKeyInput = $("#api-key");
apiKeyInput.value = store.get("pka.apikey", "dev-key");
apiKeyInput.addEventListener("change", () => store.set("pka.apikey", apiKeyInput.value));

function headers(extra = {}) {
  return { "X-API-Key": apiKeyInput.value, ...extra };
}

async function api(path, opts = {}) {
  const res = await fetch(path, { ...opts, headers: headers(opts.headers || {}) });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* not json */ }
    throw new Error(`${res.status}: ${typeof detail === "string" ? detail : JSON.stringify(detail)}`);
  }
  return res.status === 204 ? null : res.json();
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children) node.append(c instanceof Node ? c : document.createTextNode(String(c ?? "")));
  return node;
}

function fmtDate(s) { return s ? new Date(s).toLocaleString() : ""; }

// ── tabs ─────────────────────────────────────────────────────────
document.querySelectorAll("nav button").forEach((btn) => btn.addEventListener("click", () => {
  document.querySelectorAll("nav button").forEach((b) => b.classList.toggle("active", b === btn));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${btn.dataset.tab}`));
  if (btn.dataset.tab === "corpus") { loadJobs(); loadDocuments(); }
  if (btn.dataset.tab === "memory") loadMemories();
}));

// ── chat ─────────────────────────────────────────────────────────
function showConversation() {
  $("#conv-id").textContent = conversationId ? `conversation ${conversationId.slice(0, 8)}…` : "new conversation";
}
showConversation();

$("#new-chat").addEventListener("click", () => {
  conversationId = null; store.set("pka.conversation", "");
  $("#messages").replaceChildren(); $("#sources").replaceChildren(); $("#memories-used").replaceChildren();
  showConversation();
});

$("#query").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#ask").requestSubmit(); }
});

function renderSources(sources) {
  currentSources = sources;
  $("#sources").replaceChildren(...sources.map((s) => el("li", { id: `src-${s.n}` },
    el("div", { class: "label" }, s.label, s.score != null ? `  ·  ${s.score}` : ""),
    el("pre", {}, s.snippet),
  )));
}

function highlightSource(n, claimEl) {
  document.querySelectorAll("#sources li.hl, .claim.hl").forEach((x) => x.classList.remove("hl"));
  const li = $(`#src-${n}`);
  if (li) { li.classList.add("hl"); li.scrollIntoView({ block: "nearest", behavior: "smooth" }); }
  if (claimEl) claimEl.classList.add("hl");
}

// Render the final answer with each claim span wrapped and its [n] markers clickable.
function renderAnswer(container, answer, citations, invalid) {
  const bySpan = new Map();
  for (const c of citations) {
    const key = `${c.span_start}:${c.span_end}`;
    if (!bySpan.has(key)) bySpan.set(key, c);
  }
  const frag = document.createDocumentFragment();
  const markerRe = /\[(\d{1,3}(?:\s*[,;]\s*\d{1,3})*)\]/g;
  const spans = [...bySpan.values()].sort((a, b) => a.span_start - b.span_start);
  let pos = 0;
  const pushText = (text) => {
    let last = 0;
    for (const m of text.matchAll(markerRe)) {
      frag.append(text.slice(last, m.index));
      for (const num of m[1].split(/\s*[,;]\s*/).map(Number)) {
        const ok = num >= 1 && num <= currentSources.length;
        frag.append(el("button", {
          class: ok ? "cite" : "cite bad", type: "button", title: ok ? currentSources[num - 1].label : "cited a source that does not exist",
          onclick: (e) => highlightSource(num, e.target.previousClaim),
        }, num));
      }
      last = m.index + m[0].length;
    }
    frag.append(text.slice(last));
  };
  for (const s of spans) {
    if (s.span_start < pos) continue;
    pushText(answer.slice(pos, s.span_start));
    const claim = el("span", { class: "claim" }, answer.slice(s.span_start, s.span_end));
    frag.append(claim);
    pos = s.span_end;
    // attach claim to the markers that follow it
    const before = frag.childNodes.length;
    const tail = answer.slice(pos).match(/^(\s*\[(?:\d{1,3}(?:\s*[,;]\s*\d{1,3})*)\])+/);
    if (tail) { pushText(tail[0]); pos += tail[0].length; }
    for (let i = before; i < frag.childNodes.length; i++) {
      const n = frag.childNodes[i];
      if (n.classList && n.classList.contains("cite")) n.previousClaim = claim;
    }
  }
  pushText(answer.slice(pos));
  container.replaceChildren(frag);
  if (invalid && invalid.length) container.append(el("span", { class: "meta" }, `⚠ cited non-existent source(s): ${invalid.join(", ")}`));
}

async function* sseEvents(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const raw = buffer.slice(0, idx); buffer = buffer.slice(idx + 2);
      let type = "message"; const data = [];
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) type = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
      }
      if (data.length) yield { type, data: JSON.parse(data.join("\n")) };
    }
  }
}

$("#ask").addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = $("#query").value.trim();
  if (!query) return;
  $("#query").value = "";
  $("#send").disabled = true;
  const messages = $("#messages");
  messages.append(el("div", { class: "msg user" }, query));
  const bubble = el("div", { class: "msg assistant" });
  const body = el("span", {}, "…");
  const meta = el("span", { class: "meta" });
  bubble.append(body, meta);
  messages.append(bubble);
  messages.scrollTop = messages.scrollHeight;

  let answer = "";
  try {
    const res = await fetch("/query", {
      method: "POST",
      headers: headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({ query, conversation_id: conversationId, mode: $("#mode").value, stream: true }),
    });
    if (!res.ok) {
      let detail = res.statusText; try { detail = (await res.json()).detail; } catch { /* */ }
      throw new Error(`${res.status}: ${typeof detail === "string" ? detail : JSON.stringify(detail)}`);
    }
    for await (const ev of sseEvents(res)) {
      if (ev.type === "meta") {
        conversationId = ev.data.conversation_id; store.set("pka.conversation", conversationId); showConversation();
        meta.textContent = `route: ${ev.data.route}${ev.data.cache_hit ? " · cached" : ""}` +
          (ev.data.standalone_query && ev.data.standalone_query !== query ? ` · searched: “${ev.data.standalone_query}”` : "");
      } else if (ev.type === "sources") {
        renderSources(ev.data.sources);
        const mem = ev.data.memories || [];
        $("#memories-used").textContent = mem.length ? `Memory used: ${mem.map((m) => m.text).join(" · ")}` : "";
      } else if (ev.type === "token") {
        answer += ev.data.text; body.textContent = answer;
        messages.scrollTop = messages.scrollHeight;
      } else if (ev.type === "done") {
        renderAnswer(body, answer.trim(), ev.data.citations, ev.data.invalid_citations);
        const t = ev.data.timings || {};
        meta.textContent += `  ·  ttft ${t.ttft_ms ?? "–"} ms · total ${t.total_ms ?? "–"} ms`;
      } else if (ev.type === "error") {
        throw new Error(ev.data.message);
      }
    }
  } catch (err) {
    bubble.classList.add("error");
    body.textContent = answer || "";
    meta.textContent = `Error: ${err.message}`;
  } finally {
    $("#send").disabled = false;
  }
});

// ── corpus ───────────────────────────────────────────────────────
async function queue(fn) {
  try { alertJobs(await fn()); } catch (err) { showInlineError(err); }
}
function showInlineError(err) {
  const tbody = $("#jobs tbody");
  tbody.prepend(el("tr", {}, el("td", { class: "status-failed" }, "error"), el("td", { colspan: 4 }, err.message)));
}
function alertJobs(r) {
  loadJobs();
  if (r && r.jobs) {
    const tbody = $("#jobs tbody");
    tbody.prepend(el("tr", {}, el("td", { class: "status-queued" }, "queued"), el("td", { colspan: 4 }, `${r.jobs.length} job(s) queued`)));
  }
}

$("#ingest-path").addEventListener("submit", (e) => { e.preventDefault(); queue(() => api("/ingest", {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ path: $("#path").value }),
})); });
$("#ingest-url").addEventListener("submit", (e) => { e.preventDefault(); queue(() => api("/ingest", {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url: $("#url").value }),
})); });
$("#ingest-upload").addEventListener("submit", (e) => {
  e.preventDefault();
  const fd = new FormData(); fd.append("file", $("#file").files[0]);
  queue(() => api("/ingest/upload", { method: "POST", body: fd }));
});
$("#refresh-jobs").addEventListener("click", () => { loadJobs(); loadDocuments(); });

async function loadJobs() {
  try {
    const jobs = await api("/ingest/jobs?limit=30");
    $("#jobs tbody").replaceChildren(...jobs.map((j) => el("tr", {},
      el("td", { class: `status-${j.status}` }, j.status),
      el("td", { class: "mono" }, j.target),
      el("td", {}, j.chunks ?? ""),
      el("td", {}, j.attempts),
      el("td", { class: "mono" }, j.error ?? ""),
    )));
  } catch (err) { showInlineError(err); }
}

async function loadDocuments() {
  try {
    const docs = await api("/documents?limit=200");
    $("#documents tbody").replaceChildren(...docs.map((d) => el("tr", {},
      el("td", {}, d.source_type),
      el("td", { class: "mono" }, d.source_uri),
      el("td", {}, d.title ?? ""),
      el("td", {}, d.chunk_count),
      el("td", {}, el("button", { class: "link", type: "button", onclick: async () => {
        await api(`/documents/${d.id}`, { method: "DELETE" }); loadDocuments();
      } }, "delete")),
    )));
  } catch (err) { showInlineError(err); }
}

// ── memory ───────────────────────────────────────────────────────
$("#show-history").addEventListener("change", loadMemories);
async function loadMemories() {
  const tbody = $("#memories tbody");
  try {
    const rows = await api(`/memories?include_history=${$("#show-history").checked}`);
    tbody.replaceChildren(...rows.map((m) => {
      const status = m.active ? "active" : (m.superseded_by ? `superseded by #${m.superseded_by}` : "retracted");
      return el("tr", {},
        el("td", {}, m.kind), el("td", {}, m.subject), el("td", { class: "mono" }, m.predicate),
        el("td", {}, m.object), el("td", {}, m.confidence.toFixed(2)), el("td", {}, fmtDate(m.valid_from)),
        el("td", { class: m.active ? "status-active" : "status-superseded" }, status),
        el("td", {}, m.active ? el("button", { class: "link", type: "button", onclick: async () => {
          await api(`/memories/${m.id}/retract`, { method: "POST" }); loadMemories();
        } }, "retract") : ""),
      );
    }));
    if (!rows.length) tbody.replaceChildren(el("tr", {}, el("td", { colspan: 8, class: "muted" }, "No memories yet. They are extracted in the background after each conversation turn.")));
  } catch (err) {
    tbody.replaceChildren(el("tr", {}, el("td", { colspan: 8, class: "status-failed" }, err.message)));
  }
}
