# Train all registry models with resilience + MLflow
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    & ".\.venv\Scripts\Activate.ps1"
} elseif (Test-Path ".\myenv\Scripts\Activate.ps1") {
    & ".\myenv\Scripts\Activate.ps1"
}
New-Item -ItemType Directory -Force -Path logs | Out-Null
python pipeline\mlflow_orchestrator.py --run-all
