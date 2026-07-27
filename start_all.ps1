# start_all.ps1
# Starts app.py, worker.py, and ingest_worker.py each in their own terminal window.
# Run this AFTER Kafka, ChromaDB, Redis, and PostgreSQL are already up.
#
# Usage:
#   .\start_all.ps1

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot

function Start-InNewTerminal($title, $command) {
    Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-Command",
        "cd '$projectRoot'; .\venv\Scripts\Activate.ps1; `$host.ui.RawUI.WindowTitle = '$title'; $command"
    )
}

Write-Host "Checking ChromaDB is reachable..." -ForegroundColor Cyan
try {
    $null = Invoke-WebRequest -Uri "http://localhost:8000/api/v2/heartbeat" -UseBasicParsing -TimeoutSec 3
    Write-Host "  ChromaDB is up." -ForegroundColor Green
} catch {
    Write-Host "  WARNING: ChromaDB does not appear to be running on localhost:8000." -ForegroundColor Yellow
    Write-Host "  Start it first (chroma run --path .\chroma_data --port 8000) or app startup may fail." -ForegroundColor Yellow
}

Write-Host "`nStarting app.py, worker.py, ingest_worker.py in separate terminals..." -ForegroundColor Cyan

Start-InNewTerminal "app.py"           "python app.py"
Start-Sleep -Seconds 2
Start-InNewTerminal "worker.py"        "python worker.py"
Start-Sleep -Seconds 2
Start-InNewTerminal "ingest_worker.py" "python ingest_worker.py"

Write-Host "`nAll three started. Check each window for startup errors." -ForegroundColor Green
Write-Host "Reminder: Kafka, ChromaDB, Redis, and PostgreSQL must already be running." -ForegroundColor Yellow