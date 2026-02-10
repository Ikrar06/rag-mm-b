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

function addMessage(content, type, sources = []) {
    const msg = document.createElement("div");
    msg.className = `message ${type}-message`;

    let html = `<div class="message-content">${escapeHtml(content)}</div>`;

    if (sources.length > 0) {
        html += `<details class="sources"><summary>Sumber (${sources.length})</summary>`;
        for (const src of sources) {
            const page = src.page ? ` — Hal. ${src.page}` : "";
            html += `
                <div class="source-item">
                    <strong>${escapeHtml(src.file_name)}</strong>${page}
                    <span class="source-score">Score: ${src.score}</span>
                    <br><small>${escapeHtml(src.text_preview)}</small>
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
        addMessage(data.answer, "bot", data.sources);
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
