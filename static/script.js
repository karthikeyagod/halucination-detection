const $ = (id) => document.getElementById(id);
const input = $("input-text");
const button = $("analyze-btn");
const charCount = $("char-count");
const message = $("message");
const resultsContainer = $("results");
const historyContainer = $("history-list");
let lastResults = [];

// Character counter
if (input && charCount) {
  input.addEventListener("input", () => {
    const len = input.value.length;
    charCount.textContent = `${len.toLocaleString()} / 10,000 characters`;
    if (len > 10000) {
      charCount.style.color = "var(--red)";
    } else {
      charCount.style.color = "var(--muted)";
    }
  });
}

// HTML escaping to prevent XSS
function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  }[ch]));
}

// URL sanitization (only allow http/https)
function sanitizeUrl(url) {
  if (!url) return "";
  try {
    const parsed = new URL(url);
    if (parsed.protocol === "http:" || parsed.protocol === "https:") {
      return parsed.href;
    }
  } catch {
    // Relative or invalid URL
  }
  return "";
}

// Navigation active state handler
document.querySelectorAll("#sidebar-nav .nav-item").forEach((link) => {
  link.addEventListener("click", () => {
    document.querySelectorAll("#sidebar-nav .nav-item").forEach((l) => l.classList.remove("active"));
    link.classList.add("active");
  });
});

function normalizeVerdict(val) {
  const s = String(val || "").toUpperCase().trim();
  if (s.includes("SUPPORT")) return "SUPPORTED";
  if (s.includes("CONTRAD")) return "CONTRADICTED";
  return "INSUFFICIENT_EVIDENCE";
}

function renderResult(item) {
  const verdict = normalizeVerdict(item.verdict || item.status);
  const rawClaimText = item.claim || item.sentence || "";
  // Strip leading numbering from claimText to prevent duplicate numbering like "16. 16."
  const cleanClaimText = rawClaimText.replace(/^\d+[\.\)]\s*/, "");
  const claimId = item.id ? `${item.id}. ` : "";

  // Calibrated / estimated confidence badge
  const confType = (item.confidence_type === "calibrated") ? "Calibrated" : "Estimated";
  const confidenceBadge = (verdict !== "INSUFFICIENT_EVIDENCE" && typeof item.confidence === "number" && item.confidence > 0)
    ? `<span class="confidence" title="${confType} confidence calibrated from evidence relevance and model score">${confType} Confidence: ${(item.confidence * 100).toFixed(0)}%</span>`
    : `<span class="confidence neutral-conf" title="Evidence is inconclusive, missing, or unindexed">Inconclusive Evidence</span>`;

  // Evidence list formatting
  let evidenceHtml = "";
  if (Array.isArray(item.evidence) && item.evidence.length > 0) {
    const evList = item.evidence.map((ev) => {
      const safeUrl = sanitizeUrl(ev.source_url);
      const srcLink = safeUrl
        ? `<a class="source" href="${escapeHtml(safeUrl)}" target="_blank" rel="noopener noreferrer">${escapeHtml(ev.source_title || "Verified Source")} ↗</a>`
        : `<span class="source-plain">${escapeHtml(ev.source_title || "Reference Source")}</span>`;
      const relClass = ["ENTAILMENT", "CONTRADICTION", "NEUTRAL"].includes(ev.evidence_relationship || ev.relationship)
        ? (ev.evidence_relationship || ev.relationship)
        : "NEUTRAL";
      const relBadge = `<span class="rel-badge ${relClass}">${relClass}</span>`;
      const relScore = typeof ev.relevance_score === "number"
        ? `<span class="rel-score">Relevance: ${(ev.relevance_score * 100).toFixed(0)}%</span>`
        : "";

      const passageContent = ev.supporting_text || ev.text || "";

      return `
        <div class="evidence-item">
          <div class="evidence-meta">
            ${srcLink}
            <div class="evidence-tags">
              ${relBadge}
              ${relScore}
            </div>
          </div>
          <div class="evidence-text">${escapeHtml(passageContent)}</div>
        </div>
      `;
    }).join("");

    evidenceHtml = `
      <div class="evidence-container">
        <b>Retrieved Evidence & Proposition Alignment:</b>
        ${evList}
      </div>
    `;
  } else if (item.passage || item.evidence_text) {
    const pText = item.passage || item.evidence_text;
    const safeUrl = sanitizeUrl(item.source_url);
    const srcLink = safeUrl
      ? `<a class="source" href="${escapeHtml(safeUrl)}" target="_blank" rel="noopener noreferrer">${escapeHtml(item.source_title || item.source || "Source")} ↗</a>`
      : `<span class="source-plain">${escapeHtml(item.source_title || item.source || "Source")}</span>`;

    evidenceHtml = `
      <div class="evidence-container">
        <b>Retrieved Evidence:</b>
        <div class="evidence-item">
          <div class="evidence-meta">${srcLink}</div>
          <div class="evidence-text">${escapeHtml(pText)}</div>
        </div>
      </div>
    `;
  }

  // Insufficient evidence message when inconclusive
  let inconclusiveNotice = "";
  if (verdict === "INSUFFICIENT_EVIDENCE") {
    inconclusiveNotice = `
      <div class="empty-evidence">
        <b>Status Note:</b> The available public encyclopedic evidence does not establish whether this claim is true or false. Uncorroborated, private, or ambiguous statements remain <em>INSUFFICIENT_EVIDENCE</em>.
      </div>
    `;
  }

  // Factual correction (only displayed when verdict is CONTRADICTED and validated)
  const correctionHtml = (verdict === "CONTRADICTED" && item.correction)
    ? `<div class="correction"><b>Validated Factual Correction:</b> ${escapeHtml(item.correction)}</div>`
    : "";

  return `
    <article class="result">
      <div class="result-head">
        <div class="claim">${claimId}${escapeHtml(cleanClaimText)}</div>
        <div class="badges">
          ${confidenceBadge}
          <span class="badge ${verdict}">${verdict}</span>
        </div>
      </div>
      <div class="detail"><b>Explanation:</b> ${escapeHtml(item.explanation || "")}</div>
      ${inconclusiveNotice}
      ${evidenceHtml}
      ${correctionHtml}
    </article>
  `;
}

// Analyze button event listener
button.addEventListener("click", async () => {
  const text = input.value.trim();
  if (!text) {
    message.textContent = "Please paste or type text to analyze.";
    input.focus();
    return;
  }
  if (text.length > 10000) {
    message.textContent = "Input text exceeds the 10,000 character limit.";
    return;
  }

  button.disabled = true;
  button.textContent = "Analyzing Claims in Parallel…";
  message.textContent = "Retrieving evidence, evaluating proposition alignment, and verifying claims…";

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    let data;
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      data = await response.json();
    } else {
      const errText = await response.text();
      throw new Error(`Server returned HTTP ${response.status}: ${errText.slice(0, 120)}`);
    }

    if (!response.ok) {
      throw new Error(data.error || "Analysis failed.");
    }

    lastResults = data.results || [];
    resultsContainer.className = "results";
    resultsContainer.innerHTML = lastResults.length
      ? lastResults.map(renderResult).join("")
      : '<div class="empty">No sentences were detected in the input.</div>';

    const counts = data.counts || {};
    const sup = counts.SUPPORTED ?? counts.Supported ?? 0;
    const con = counts.CONTRADICTED ?? counts.Contradicted ?? 0;
    const ins = counts.INSUFFICIENT_EVIDENCE ?? counts.Unverified ?? 0;

    $("supported-count").textContent = sup;
    $("contradicted-count").textContent = con;
    $("unverified-count").textContent = ins;
    $("total-count").textContent = data.total_sentences || lastResults.length;
    $("corrected-paragraph").value = data.corrected_paragraph || "";

    message.textContent = "Analysis complete. Review the proposition alignment and predictions above.";
    loadHistory();
  } catch (err) {
    message.textContent = `Error: ${err.message || "Could not connect to the backend server."}`;
  } finally {
    button.disabled = false;
    button.textContent = "Analyze Text →";
  }
});

// Copy button with in-place feedback
const copyBtn = $("copy-btn");
if (copyBtn) {
  copyBtn.addEventListener("click", async () => {
    const text = $("corrected-paragraph").value;
    if (!text) {
      copyBtn.textContent = "No text to copy";
      setTimeout(() => { copyBtn.textContent = "Copy Paragraph"; }, 1500);
      return;
    }
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      $("corrected-paragraph").select();
      document.execCommand("copy");
    }
    copyBtn.textContent = "✓ Copied!";
    setTimeout(() => { copyBtn.textContent = "Copy Paragraph"; }, 1800);
  });
}

// Quick claim / chat verification
const chatInput = $("chat-input");
const chatBtn = $("chat-btn");
const chatResult = $("chat-result");

async function handleQuickVerify() {
  const msg = (chatInput?.value || "").trim();
  if (!msg) {
    if (chatInput) chatInput.focus();
    return;
  }
  chatBtn.disabled = true;
  chatBtn.textContent = "Verifying…";
  chatResult.style.display = "block";
  chatResult.innerHTML = '<div class="detail">Querying authoritative sources and evaluating claim proposition…</div>';

  try {
    const response = await fetch("/api/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ claim: msg })
    });
    const data = await response.json();
    chatResult.innerHTML = renderResult(data);
  } catch (err) {
    chatResult.innerHTML = `<div class="detail" style="color:var(--red);">Verification request failed: ${escapeHtml(err.message)}</div>`;
  } finally {
    chatBtn.disabled = false;
    chatBtn.textContent = "Verify Claim →";
  }
}

if (chatBtn && chatInput) {
  chatBtn.addEventListener("click", handleQuickVerify);
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleQuickVerify();
    }
  });
}

// History loader
async function loadHistory() {
  try {
    const response = await fetch("/api/history");
    const data = await response.json();
    const history = data.history || [];

    if (!history.length) {
      historyContainer.className = "history-list empty";
      historyContainer.innerHTML = "No previous analyses recorded yet.";
      return;
    }

    historyContainer.className = "history-list";
    historyContainer.innerHTML = history.map((item) => {
      const counts = item.counts || {};
      const sup = counts.SUPPORTED ?? counts.Supported ?? 0;
      const con = counts.CONTRADICTED ?? counts.Contradicted ?? 0;
      const ins = counts.INSUFFICIENT_EVIDENCE ?? counts.Unverified ?? 0;

      return `
        <div class="history-item">
          <div class="history-meta">
            <small>${new Date(item.timestamp).toLocaleTimeString()} · ${item.total_sentences} ${item.total_sentences === 1 ? "sentence" : "sentences"}</small>
            <span class="counts-badge">
              <span style="color:var(--green)">✓ ${sup}</span>
              <span style="color:var(--red)">✕ ${con}</span>
              <span style="color:var(--amber)">? ${ins}</span>
            </span>
          </div>
          <div class="history-preview">${escapeHtml(item.preview)}</div>
        </div>
      `;
    }).join("");
  } catch {
    historyContainer.innerHTML = '<div class="empty">History unavailable.</div>';
  }
}

// Clear history button with backend sync
const clearHistBtn = $("clear-history-btn");
if (clearHistBtn) {
  clearHistBtn.addEventListener("click", async () => {
    try {
      await fetch("/api/history/clear", { method: "POST" });
    } catch {
      // Continue even if network error
    }
    historyContainer.className = "history-list empty";
    historyContainer.innerHTML = "History cleared.";
  });
}

// Health check loader
async function loadHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    const model = data.verification_model || {};
    let statusText = "will load on first analysis";
    if (model.available === true) statusText = "ready & loaded";
    if (model.available === false) statusText = "unavailable (see README)";

    $("health").textContent = `Backend: online · NLI Model: ${model.name || "not configured"} (${statusText}).`;
  } catch {
    $("health").textContent = "Backend health check failed. Ensure app.py is running.";
  }
}

loadHealth();
loadHistory();
