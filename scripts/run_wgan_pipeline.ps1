# Extract inter-pulse noise segments from Q.Lin pool, train WGAN-GP noise generator.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    & .\.venv\Scripts\Activate.ps1
} elseif (Test-Path ".\myenv\Scripts\Activate.ps1") {
    & .\myenv\Scripts\Activate.ps1
}
$py = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }
Write-Host "Using $py"
& $py data\wgan\extract_noise_segments.py --out data\real_noise_segments.pt
& $py data\wgan\wgan_noise.py --noise-segments data\real_noise_segments.pt --epochs 300 --out data\wgan\noise_generator.pth
Write-Host "Generator weights: data/wgan/noise_generator.pth"
Write-Host "Optional: mix generated noise into synthetic training batches (see docs/MODEL_COMPARISON.md)."
