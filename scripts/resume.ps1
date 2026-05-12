# resume.ps1 — Resume training after a crash/interrupt (Windows)
# Usage: .\scripts\resume.ps1

$ErrorActionPreference = "Stop"

# Activate venv
$venv = Join-Path $PSScriptRoot "..\.venv\Scripts\Activate.ps1"
if (Test-Path $venv) {
    & $venv
} else {
    $myenv = Join-Path $PSScriptRoot "..\myenv\Scripts\Activate.ps1"
    if (Test-Path $myenv) {
        & $myenv
    } else {
        Write-Host "[ERROR] No virtual environment found. Create with: python -m venv .venv" -ForegroundColor Red
        exit 1
    }
}

# Verify CUDA
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] CUDA not available" -ForegroundColor Yellow
}

# Check session state
$session = Join-Path $PSScriptRoot "..\session_state.json"
if (Test-Path $session) {
    Write-Host "[RESUME] Found session_state.json" -ForegroundColor Green
    python pipeline/mlflow_orchestrator.py --resume
} else {
    Write-Host "[FRESH] No session found — starting from scratch" -ForegroundColor Yellow
    python pipeline/mlflow_orchestrator.py --run-all
}
