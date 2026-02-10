const API_BASE = "";
const chatContainer = document.getElementById("chatContainer");
const chatForm = document.getElementById("chatForm");
const queryInput = document.getElementById("queryInput");
const sendBtn = document.getElementById("sendBtn");
const statusEl = document.getElementById("status");

// Health check on load
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/api/health`);
        const data = await res.json();

        statusEl.textContent =
            data.status === "healthy"
                ? "Ollama & Qdrant terhubung"
                : `Status: ${data.status} (Ollama: ${data.ollama ? "OK" : "OFF"}, Qdrant: ${data.qdrant ? "OK" : "OFF"})`;
        statusEl.className = `status ${data.status}`;
    } catch {
        statusEl.textContent = "Backend tidak terhubung";
        statusEl.className = "status error";
    }
}

function addMessage(content, type, sources = [], debug = null) {
    const msg = document.createElement("div");
    msg.className = `message ${type}-message`;

    let html = `<div class="message-content">${escapeHtml(content)}</div>`;

    if (debug) {
        const mode = debug.mode === "rag" ? "RAG" : "Chitchat";
        const time = `${debug.total_time_s}s`;
        const details = debug.mode === "rag"
            ? ` · Top-K: ${debug.similarity_top_k} → Re-rank: ${debug.reranker_top_n} · Sources: ${debug.sources_returned} · Model: ${debug.model}`
            : ` · Model: ${debug.model || "ollama"}`;
        html += `<div class="debug-bar">${mode} · ${time}${details}</div>`;
    }

    if (sources.length > 0) {
        html += `<details class="sources"><summary>Sumber (${sources.length})</summary>`;
        for (let i = 0; i < sources.length; i++) {
            const src = sources[i];
            const page = src.page ? `Hal. ${src.page}` : "";
            const chunk = src.chunk_index != null ? `Chunk #${src.chunk_index}` : "";
            const type = src.element_type ? `[${src.element_type}]` : "";
            const meta = [page, chunk, type].filter(Boolean).join(" · ");
            html += `
                <div class="source-item">
                    <div class="source-header">
                        <strong>${i + 1}. ${escapeHtml(src.file_name)}</strong>
                        <span class="source-score">Score: ${src.score}</span>
                    </div>
                    ${meta ? `<div class="source-meta">${meta}</div>` : ""}
                    <div class="source-preview">${escapeHtml(src.text_preview)}</div>
                </div>`;
        }
        html += `</details>`;
    }

    msg.innerHTML = html;
    chatContainer.appendChild(msg);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return msg;
}

function addLoading() {
    const msg = document.createElement("div");
    msg.className = "message bot-message loading";
    msg.id = "loadingMsg";
    msg.innerHTML = `
        <div class="message-content">
            <div class="dot-pulse">
                <span></span><span></span><span></span>
            </div>
            Sedang mencari jawaban...
        </div>`;
    chatContainer.appendChild(msg);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return msg;
}

function removeLoading() {
    const el = document.getElementById("loadingMsg");
    if (el) el.remove();
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

async function sendMessage(query) {
    addMessage(query, "user");
    queryInput.value = "";
    sendBtn.disabled = true;

    addLoading();

    try {
        const res = await fetch(`${API_BASE}/api/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query }),
        });

        removeLoading();

        if (!res.ok) {
            const err = await res.json();
            addMessage(`Error: ${err.detail || "Terjadi kesalahan"}`, "bot");
            return;
        }

        const data = await res.json();
        addMessage(data.answer, "bot", data.sources, data.debug);
    } catch (err) {
        removeLoading();
        addMessage("Tidak dapat terhubung ke server. Pastikan backend berjalan.", "bot");
    } finally {
        sendBtn.disabled = false;
        queryInput.focus();
    }
}

chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (query) sendMessage(query);
});

checkHealth();
