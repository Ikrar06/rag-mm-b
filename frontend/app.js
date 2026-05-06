const API_BASE = "";

// ─── State ────────────────────────────────────────────────────────────────────
let currentSessionId = null;
let currentUser = null;

// ─── DOM refs ─────────────────────────────────────────────────────────────────
const chatContainer = document.getElementById("chatContainer");
const chatForm = document.getElementById("chatForm");
const queryInput = document.getElementById("queryInput");
const sendBtn = document.getElementById("sendBtn");
const statusEl = document.getElementById("status");
const loginOverlay = document.getElementById("loginOverlay");
const loginForm = document.getElementById("loginForm");
const loginError = document.getElementById("loginError");
const loginBtn = document.getElementById("loginBtn");
const userLabel = document.getElementById("userLabel");
const mainApp = document.getElementById("mainApp");

// ─── Auth ─────────────────────────────────────────────────────────────────────

async function checkAuth() {
    try {
        const res = await fetch(`${API_BASE}/api/auth/me`, { credentials: "include" });
        if (res.ok) {
            currentUser = await res.json();
            showApp();
        } else {
            showLogin();
        }
    } catch {
        showLogin();
    }
}

function showLogin() {
    loginOverlay.classList.remove("hidden");
    mainApp.classList.add("hidden");
}

function showApp() {
    loginOverlay.classList.add("hidden");
    mainApp.classList.remove("hidden");

    // Tampilkan info user di header
    const roleLabel = {
        "mahasiswa": "Mahasiswa",
        "staf_akademik": "Staf Akademik",
        "calon_mahasiswa": "Calon Mahasiswa",
        "public": "Tamu",
    }[currentUser.role] || currentUser.role;

    userLabel.textContent = `${currentUser.name} · ${roleLabel}`;

    // Session baru untuk sesi ini
    currentSessionId = localStorage.getItem("session_id_" + currentUser.username) || null;

    checkHealth();
}

loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    loginError.classList.add("hidden");
    loginBtn.disabled = true;
    loginBtn.textContent = "Masuk...";

    const username = document.getElementById("usernameInput").value.trim();
    const password = document.getElementById("passwordInput").value;

    try {
        const res = await fetch(`${API_BASE}/api/auth/login`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            credentials: "include",
            body: JSON.stringify({ username, password }),
        });

        if (res.ok) {
            currentUser = await res.json();
            showApp();
        } else {
            loginError.classList.remove("hidden");
        }
    } catch {
        loginError.textContent = "Tidak dapat terhubung ke server.";
        loginError.classList.remove("hidden");
    } finally {
        loginBtn.disabled = false;
        loginBtn.textContent = "Masuk";
    }
});

async function handleLogout() {
    await fetch(`${API_BASE}/api/auth/logout`, { method: "POST", credentials: "include" });
    currentUser = null;
    currentSessionId = null;
    chatContainer.innerHTML = `
        <div class="message bot-message">
            <div class="message-content">Halo! Saya asisten akademik UNHAS. Silakan ajukan pertanyaan tentang informasi akademik.</div>
        </div>`;
    showLogin();
}

function handleNewSession() {
    if (!currentUser) return;
    currentSessionId = null;
    localStorage.removeItem("session_id_" + currentUser.username);
    chatContainer.innerHTML = `
        <div class="message bot-message">
            <div class="message-content">Sesi baru dimulai. Ada yang ingin Anda tanyakan?</div>
        </div>`;
    queryInput.focus();
}

// ─── Health check ─────────────────────────────────────────────────────────────

async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/api/health`);
        const data = await res.json();

        const parts = [
            data.ollama ? "LLM OK" : "LLM OFF",
            data.qdrant ? "Qdrant OK" : "Qdrant OFF",
            data.postgres ? "DB OK" : "DB OFF",
            data.redis ? "Cache OK" : "Cache OFF",
        ];
        statusEl.textContent = parts.join(" · ");
        statusEl.className = `status ${data.status}`;
    } catch {
        statusEl.textContent = "Backend tidak terhubung";
        statusEl.className = "status error";
    }
}

// ─── Chat UI ──────────────────────────────────────────────────────────────────

function addMessage(content, type, sources = [], debug = null) {
    const msg = document.createElement("div");
    msg.className = `message ${type}-message`;

    let html = `<div class="message-content">${marked.parse(content)}</div>`;

    if (debug) {
        const modeMap = {
            "rag": "RAG",
            "rag_low_relevance": "RAG (low)",
            "cache_hit": "Cache",
            "chitchat": "Chitchat",
            "identity": "Chitchat",
            "blocked": "Blocked",
        };
        const modeLabel = modeMap[debug.mode] || debug.mode;
        const time = `${debug.total_time_s}s`;
        const score = debug.top_score != null ? ` · Score: ${debug.top_score}` : "";
        const sources_info = debug.sources_returned != null
            ? ` · Sources: ${debug.sources_returned}`
            : "";
        const model = debug.model ? ` · ${debug.model}` : "";
        html += `<div class="debug-bar">${modeLabel} · ${time}${score}${sources_info}${model}</div>`;
    }

    if (sources && sources.length > 0) {
        html += `<details class="sources"><summary>Sumber (${sources.length})</summary>`;
        for (let i = 0; i < sources.length; i++) {
            const src = sources[i];
            const page = src.page ? `Hal. ${src.page}` : "";
            const chunk = src.chunk_index != null ? `Chunk #${src.chunk_index}` : "";
            const elemType = src.element_type ? `[${src.element_type}]` : "";
            const meta = [page, chunk, elemType].filter(Boolean).join(" · ");
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
}

function removeLoading() {
    const el = document.getElementById("loadingMsg");
    if (el) el.remove();
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text || "";
    return div.innerHTML;
}

// ─── Send message (streaming) ─────────────────────────────────────────────────

async function sendMessage(userQuery) {
    addMessage(userQuery, "user");
    queryInput.value = "";
    sendBtn.disabled = true;

    // Create empty bot bubble immediately
    const botMsg = document.createElement("div");
    botMsg.className = "message bot-message";
    const contentEl = document.createElement("div");
    contentEl.className = "message-content streaming";
    botMsg.appendChild(contentEl);
    chatContainer.appendChild(botMsg);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    const body = { query: userQuery };
    if (currentSessionId) body.session_id = currentSessionId;

    let rawText = "";

    try {
        const res = await fetch(`${API_BASE}/api/chat/stream`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            credentials: "include",
            body: JSON.stringify(body),
        });

        if (res.status === 401) { showLogin(); return; }
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            contentEl.classList.remove("streaming");
            contentEl.textContent = `Error: ${err.detail || "Terjadi kesalahan"}`;
            return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n\n");
            buffer = lines.pop(); // keep incomplete chunk

            for (const line of lines) {
                if (!line.startsWith("data: ")) continue;
                let event;
                try { event = JSON.parse(line.slice(6)); } catch { continue; }

                if (event.type === "session") {
                    currentSessionId = event.session_id;
                    if (currentUser) {
                        localStorage.setItem("session_id_" + currentUser.username, event.session_id);
                    }
                } else if (event.type === "token") {
                    rawText += event.delta || "";
                    contentEl.textContent = rawText;
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                } else if (event.type === "meta") {
                    // Stream done — render with markdown + debug + sources
                    const finalAnswer = event.answer || rawText;

                    // Replace streaming content with markdown-rendered text
                    contentEl.classList.remove("streaming");
                    contentEl.innerHTML = marked.parse(finalAnswer);

                    // Append debug bar + sources to botMsg (outside contentEl, same as addMessage)
                    if (event.debug) {
                        const modeMap = {
                            "rag": "RAG", "rag_low_relevance": "RAG (low)",
                            "cache_hit": "Cache", "chitchat": "Chitchat",
                            "identity": "Chitchat", "blocked": "Blocked",
                        };
                        const d = event.debug;
                        const modeLabel = modeMap[d.mode] || d.mode;
                        const score = d.top_score != null ? ` · Score: ${d.top_score}` : "";
                        const srcs = d.sources_returned != null ? ` · Sources: ${d.sources_returned}` : "";
                        const mdl = d.model ? ` · ${d.model}` : "";
                        const debugEl = document.createElement("div");
                        debugEl.className = "debug-bar";
                        debugEl.textContent = `${modeLabel} · ${d.total_time_s}s${score}${srcs}${mdl}`;
                        botMsg.appendChild(debugEl);
                    }

                    const sources = event.sources || [];
                    if (sources.length > 0) {
                        let srcHtml = `<details class="sources"><summary>Sumber (${sources.length})</summary>`;
                        sources.forEach((src, i) => {
                            const page = src.page ? `Hal. ${src.page}` : "";
                            const chunk = src.chunk_index != null ? `Chunk #${src.chunk_index}` : "";
                            const elemType = src.element_type ? `[${src.element_type}]` : "";
                            const meta = [page, chunk, elemType].filter(Boolean).join(" · ");
                            srcHtml += `<div class="source-item">
                                <div class="source-header">
                                    <strong>${i + 1}. ${escapeHtml(src.file_name)}</strong>
                                    <span class="source-score">Score: ${src.score}</span>
                                </div>
                                ${meta ? `<div class="source-meta">${meta}</div>` : ""}
                                <div class="source-preview">${escapeHtml(src.text_preview)}</div>
                            </div>`;
                        });
                        srcHtml += `</details>`;
                        const srcEl = document.createElement("div");
                        srcEl.innerHTML = srcHtml;
                        botMsg.appendChild(srcEl.firstElementChild);
                    }

                    chatContainer.scrollTop = chatContainer.scrollHeight;
                } else if (event.type === "error") {
                    contentEl.classList.remove("streaming");
                    contentEl.textContent = `Error: ${event.message}`;
                }
            }
        }
    } catch (err) {
        contentEl.classList.remove("streaming");
        contentEl.textContent = "Tidak dapat terhubung ke server. Pastikan backend berjalan.";
    } finally {
        sendBtn.disabled = false;
        queryInput.focus();
    }
}

chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = queryInput.value.trim();
    if (q) sendMessage(q);
});

// ─── Init ─────────────────────────────────────────────────────────────────────
checkAuth();
