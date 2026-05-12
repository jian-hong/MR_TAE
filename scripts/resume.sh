#!/usr/bin/env bash
# resume.sh — One command to pick up exactly where training stopped.
#
# Usage (from project root):
#   bash resume.sh
#
# What it does:
#   1. Activates .venv
#   2. Checks GPU + CUDA
#   3. Reads session_state.json and prints current status
#   4. Launches the orchestrator with --resume flag
#   5. Starts the dashboard in background so you can monitor in browser
#
# WiFi-proof: if the process dies mid-training, just run this again.
# It will pick up from the last saved checkpoint.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 1. Activate venv ─────────────────────────────────────────────────────────
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "[ENV] .venv activated: $(which python)"
else
    echo "[ERROR] .venv not found. Run: python3 -m venv .venv && source .venv/bin/activate"
    exit 1
fi

# ── 2. GPU check ─────────────────────────────────────────────────────────────
echo "[GPU] Checking CUDA..."
python -c "import torch; print(f'  Device: {torch.cuda.get_device_name(0)}'); print(f'  CUDA: {torch.version.cuda}')" || {
    echo "[ERROR] CUDA not available. Check GPU drivers."
    exit 1
}

# ── 3. Print current session state ───────────────────────────────────────────
echo ""
echo "[SESSION] Current training state:"
if [ -f "session_state.json" ]; then
    python - <<'EOF'
import json
with open("session_state.json") as f:
    s = json.load(f)
models = s.get("models", {})
for mid, d in models.items():
    status = d.get("status", "?")
    ep = d.get("current_epoch", 0)
    ncc = d.get("best_ncc", 0)
    skip = d.get("skip_reason", "")
    marker = {"done":"✓","running":"▶","skipped":"✗","queued":"○"}.get(status,"?")
    print(f"  {marker} {mid:<30} {status:<10} ep={ep:<5} best_ncc={ncc:.4f}", end="")
    if skip:
        print(f"  [{skip[:60]}]", end="")
    print()
last = s.get("last_updated","never")
print(f"\n  Last update: {last}")
EOF
else
    echo "  No session found — will start fresh"
fi

# ── 4. Show resume prompt if it exists ───────────────────────────────────────
if [ -f "RESUME_PROMPT.md" ]; then
    echo ""
    echo "[PROMPT] RESUME_PROMPT.md exists — copy it into a new Claude/Cursor session if needed"
fi

# ── 5. Start dashboard in background ─────────────────────────────────────────
echo ""
echo "[DASH] Starting dashboard on http://localhost:8080 ..."
nohup python dashboard/app.py > logs/dashboard.log 2>&1 &
DASH_PID=$!
echo "  Dashboard PID: $DASH_PID (logs/dashboard.log)"

# Give dashboard a moment to start
sleep 1

# ── 6. Start MLflow UI in background ─────────────────────────────────────────
echo "[MLFLOW] Starting MLflow UI on http://localhost:5000 ..."
nohup mlflow ui --host 0.0.0.0 --port 5000 > logs/mlflow.log 2>&1 &
MLFLOW_PID=$!
echo "  MLflow PID: $MLFLOW_PID (logs/mlflow.log)"

sleep 1

# ── 7. Launch training ────────────────────────────────────────────────────────
echo ""
echo "[TRAIN] Launching orchestrator with --resume ..."
echo "  Monitor:   http://localhost:8080"
echo "  MLflow:    http://localhost:5000"
echo "  Live log:  tail -f logs/training_live.log"
echo "  To stop:   Ctrl+C (checkpoint will be saved automatically)"
echo ""

python pipeline/mlflow_orchestrator.py --resume

# ── 8. On exit, remind user ───────────────────────────────────────────────────
echo ""
echo "[DONE] Training session ended."
echo "  Run 'bash resume.sh' again to continue if there are remaining models."
echo "  Run 'python evaluation/benchmark_runner.py --compare-all' when all done."
