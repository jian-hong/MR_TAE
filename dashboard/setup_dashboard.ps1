# Build React frontend and optionally start Flask API
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $PSScriptRoot
if (-not (Test-Path ".\frontend\package.json")) {
    Write-Error "frontend/package.json not found"
    exit 1
}
Push-Location frontend
npm ci
if ($LASTEXITCODE -ne 0) { npm install }
npm run build
Pop-Location
Write-Host "Frontend built to dashboard/frontend/build. Start API from repo root:"
Write-Host "  .\.venv\Scripts\python.exe dashboard\app.py"
