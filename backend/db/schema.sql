-- =============================================================================
-- Database Schema — RAG Chatbot UNHAS POC
-- Engine: PostgreSQL 16
-- Run: psql -U ragchat -d ragchat -f schema.sql
--      atau: docker compose exec postgres psql -U ragchat -d ragchat -f /schema.sql
-- =============================================================================

-- ── USERS ─────────────────────────────────────────────────────────────────────
-- Minimal fields yang dibutuhkan — asumsi login sudah dikelola Tim BE.
-- Table ini diisi dari sisi Tim BE setelah user login berhasil.

-- CREATE TABLE users (
--     id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
--     nim VARCHAR(20) UNIQUE,              -- nullable untuk staf
--     full_name VARCHAR(200),
--     email VARCHAR(200),
--     role VARCHAR(30) NOT NULL,           -- 'mahasiswa', 'calon_mahasiswa', 'staf_akademik'
--     created_at TIMESTAMPTZ DEFAULT NOW()
-- );

-- ── SESSIONS ──────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,                  -- REFERENCES users(id) ON DELETE CASCADE
    title VARCHAR(200),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_message_at TIMESTAMPTZ DEFAULT NOW(),
    is_archived BOOLEAN DEFAULT FALSE,
    message_count INT DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_active
    ON sessions(user_id, last_message_at DESC)
    WHERE is_archived = FALSE;

-- ── MESSAGES ──────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    sources JSONB,                          -- [{file_name, page, chunk_index, score, ...}]
    debug JSONB,                            -- {mode, total_time_s, top_score, model, ...}
    mode VARCHAR(30),                       -- chitchat|identity|blocked|rag|rag_low_relevance|cache_hit|vision_rag
    condensed_question TEXT,                -- hasil query condensation (untuk audit)
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);

-- ── MESSAGE_ATTACHMENTS (vision input) ────────────────────────────────────────

CREATE TABLE IF NOT EXISTS message_attachments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id UUID REFERENCES messages(id) ON DELETE CASCADE,  -- nullable: pre-attached sebelum send
    user_id UUID NOT NULL,                  -- REFERENCES users(id) ON DELETE CASCADE
    storage_key VARCHAR(500) NOT NULL,
    storage_url TEXT,
    original_filename VARCHAR(255),
    mime_type VARCHAR(100) NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    width INT,
    height INT,
    extracted_text TEXT,                    -- OCR cache (opsional)
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_attachments_message ON message_attachments(message_id);
CREATE INDEX IF NOT EXISTS idx_attachments_user_recent ON message_attachments(user_id, created_at DESC);

-- ── AUDIT LOG ─────────────────────────────────────────────────────────────────
-- Kriteria POC: audit logging tersedia (RBAC ≥ 95% accuracy, data non-public tidak bocor)

CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,                           -- nullable: untuk failed auth attempts
    action VARCHAR(50) NOT NULL,            -- login|logout|query|upload|access_doc|retrieve|rbac_denied|auth_failed
    resource_type VARCHAR(50),              -- session|message|attachment|document
    resource_id UUID,
    metadata JSONB,                         -- IP address, user agent, query content, dll
    success BOOLEAN NOT NULL,
    occurred_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_logs(user_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs(action, occurred_at DESC);

-- ── RATE LIMIT EVENTS (opsional — untuk analytics) ───────────────────────────

CREATE TABLE IF NOT EXISTS rate_limit_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    endpoint VARCHAR(100),
    occurred_at TIMESTAMPTZ DEFAULT NOW()
);
