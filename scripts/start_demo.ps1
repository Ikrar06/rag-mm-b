# =============================================================================
# Demo startup script — RAG Chatbot UNHAS
# Jalankan dari root project: .\scripts\start_demo.ps1
#
# PRASYARAT:
#   1. Ollama sudah jalan (ollama serve atau via system tray)
#   2. Docker Desktop sudah jalan
#   3. ngrok sudah diinstall: winget install ngrok
#   4. Akun ngrok (gratis) + static domain sudah di-claim:
#      - Daftar di https://dashboard.ngrok.com
#      - Buat static domain (1 gratis): Dashboard → Domains → New Domain
#      - Copy authtoken: Dashboard → Your Authtoken
#      - Simpan: ngrok config add-authtoken <TOKEN>
#      - Set NGROK_DOMAIN di bawah
# =============================================================================

# ── KONFIGURASI — Edit sesuai setup Anda ────────────────────────────────────

$BACKEND_PORT = 8000
$NGROK_DOMAIN = "remover-repressed-backboned.ngrok-free.dev"   # dari ngrok dashboard
$VENV_PATH     = ".\venv\Scripts\Activate.ps1"

# ── 1. Mulai Qdrant via Docker ───────────────────────────────────────────────

Write-Host "`n[1/4] Starting Qdrant + Postgres + Redis..." -ForegroundColor Cyan
docker compose -f docker-compose.dev.yml up -d
Start-Sleep -Seconds 5

# Verifikasi Qdrant
$qdrantOk = $false
for ($i = 0; $i -lt 5; $i++) {
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:6333/healthz" -UseBasicParsing -ErrorAction Stop
        if ($r.StatusCode -eq 200) { $qdrantOk = $true; break }
    } catch {}
    Start-Sleep -Seconds 2
}
if (-not $qdrantOk) {
    Write-Host "[ERROR] Qdrant tidak bisa diakses. Cek: docker ps" -ForegroundColor Red
    exit 1
}
Write-Host "  Qdrant + Postgres + Redis OK" -ForegroundColor Green

# ── 2. Cek Ollama ────────────────────────────────────────────────────────────

Write-Host "`n[2/4] Checking Ollama..." -ForegroundColor Cyan
try {
    $r = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -ErrorAction Stop
    Write-Host "  Ollama OK" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Ollama tidak bisa diakses di port 11434." -ForegroundColor Red
    Write-Host "  Pastikan Ollama sudah jalan: Start-Process ollama -ArgumentList 'serve'" -ForegroundColor Yellow
    exit 1
}

# ── 3. Mulai Backend FastAPI ─────────────────────────────────────────────────

Write-Host "`n[3/4] Starting FastAPI backend on port $BACKEND_PORT..." -ForegroundColor Cyan

# Aktifkan venv dan jalankan backend di background
$backendProcess = Start-Process -FilePath "powershell" -ArgumentList @(
    "-NoExit", "-Command",
    "& '$VENV_PATH'; uvicorn backend.main:app --host 0.0.0.0 --port $BACKEND_PORT --reload --reload-dir backend --reload-dir frontend"
) -PassThru

Start-Sleep -Seconds 5

# Verifikasi backend
$backendOk = $false
for ($i = 0; $i -lt 8; $i++) {
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:$BACKEND_PORT/api/health" -UseBasicParsing -ErrorAction Stop
        if ($r.StatusCode -eq 200) { $backendOk = $true; break }
    } catch {}
    Start-Sleep -Seconds 2
}
if (-not $backendOk) {
    Write-Host "[ERROR] Backend tidak bisa diakses. Cek window PowerShell yang baru dibuka." -ForegroundColor Red
    exit 1
}
Write-Host "  Backend OK — http://localhost:$BACKEND_PORT" -ForegroundColor Green

# ── 4. Mulai ngrok tunnel ────────────────────────────────────────────────────

Write-Host "`n[4/4] Starting ngrok tunnel..." -ForegroundColor Cyan
Write-Host "  Domain: https://$NGROK_DOMAIN" -ForegroundColor White

$ngrokExe = "C:\Users\ikrar\AppData\Local\Microsoft\WinGet\Packages\Ngrok.Ngrok_Microsoft.Winget.Source_8wekyb3d8bbwe\ngrok.exe"
$ngrokProcess = Start-Process -FilePath $ngrokExe -ArgumentList @(
    "http", "--domain=$NGROK_DOMAIN", "$BACKEND_PORT"
) -PassThru

Start-Sleep -Seconds 3

# ── QR Code untuk demo ───────────────────────────────────────────────────────

$url = "https://$NGROK_DOMAIN"
Write-Host "`n============================================================" -ForegroundColor Yellow
Write-Host "  CHATBOT UNHAS SIAP UNTUK DEMO" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "  URL Lokal : http://localhost:$BACKEND_PORT" -ForegroundColor Cyan
Write-Host "  URL Publik: $url" -ForegroundColor Cyan
Write-Host ""
Write-Host "  QR Code (scan untuk akses dari HP):" -ForegroundColor White
Write-Host "  https://qr-code-generator.vercel.app/?data=$url" -ForegroundColor White
Write-Host ""
Write-Host "  Atau buka ngrok dashboard: http://localhost:4040" -ForegroundColor Gray
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "  JANGAN tutup window ini selama demo berlangsung!" -ForegroundColor Red
Write-Host "  Semua service akan berhenti saat window ditutup." -ForegroundColor Red
Write-Host ""
Write-Host "  Tekan Ctrl+C untuk stop semua service." -ForegroundColor Gray

# ── Cleanup saat Ctrl+C ──────────────────────────────────────────────────────

try {
    $ngrokProcess | Wait-Process
} finally {
    Write-Host "`nShutting down..." -ForegroundColor Yellow
    if ($backendProcess -and !$backendProcess.HasExited) {
        Stop-Process -Id $backendProcess.Id -Force -ErrorAction SilentlyContinue
    }
    if ($ngrokProcess -and !$ngrokProcess.HasExited) {
        Stop-Process -Id $ngrokProcess.Id -Force -ErrorAction SilentlyContinue
    }
    docker compose -f docker-compose.dev.yml stop qdrant
    Write-Host "Done." -ForegroundColor Green
}
