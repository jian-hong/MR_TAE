"""
dashboard/app.py — Research Dashboard Backend

Flask API serving experiment data, live training stream, and file access.
Pairs with the React frontend in dashboard/frontend/.

Run:
    python dashboard/app.py
    # Open http://localhost:8080
"""

import os
import json
import time
import subprocess
import threading
import glob
from pathlib import Path
from datetime import datetime

from flask import Flask, jsonify, send_file, request, Response, send_from_directory
from flask_cors import CORS
import mlflow
from mlflow.tracking import MlflowClient

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
MLFLOW_TRACKING_URI = str(PROJECT_ROOT / "mlruns")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

app = Flask(__name__, static_folder="frontend/build", static_url_path="/")
CORS(app)

# ---------------------------------------------------------------------------
# Serve React app
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

# ---------------------------------------------------------------------------
# Experiments API
# ---------------------------------------------------------------------------
@app.route("/api/experiments")
def list_experiments():
    """List all MLflow experiments with summary stats."""
    experiments = client.search_experiments()
    result = []
    for exp in experiments:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.val/ncc DESC"],
            max_results=1
        )
        best_ncc = None
        if runs:
            best_ncc = runs[0].data.metrics.get("test_real/ncc")

        result.append({
            "id": exp.experiment_id,
            "name": exp.name,
            "artifact_location": exp.artifact_location,
            "lifecycle_stage": exp.lifecycle_stage,
            "best_real_ncc": best_ncc,
            "run_count": len(client.search_runs([exp.experiment_id])),
        })
    return jsonify(result)


@app.route("/api/experiments/<experiment_name>")
def get_experiment(experiment_name):
    """Full details for one experiment: all runs, metrics, params."""
    exp = client.get_experiment_by_name(experiment_name)
    if not exp:
        return jsonify({"error": "Experiment not found"}), 404

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"]
    )

    runs_data = []
    for run in runs:
        # Build metric history for charts
        metric_history = {}
        for key in ["train/loss", "train/ncc", "val/ncc", "train/rmse", "val/rmse",
                    "gpu/temp_c", "gpu/vram_used_gb"]:
            history = client.get_metric_history(run.info.run_id, key)
            metric_history[key] = [{"step": h.step, "value": h.value} for h in history]

        runs_data.append({
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName", run.info.run_id[:8]),
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "params": run.data.params,
            "metrics": run.data.metrics,
            "metric_history": metric_history,
            "artifacts": _list_artifacts(run.info.run_id),
        })

    return jsonify({
        "experiment": {
            "id": exp.experiment_id,
            "name": exp.name,
        },
        "runs": runs_data,
    })


def _list_artifacts(run_id: str):
    """Return list of artifact paths for a run."""
    try:
        artifacts = client.list_artifacts(run_id)
        return [a.path for a in artifacts if not a.is_dir]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Live Training Stream (SSE)
# ---------------------------------------------------------------------------
_live_metrics = {}  # model_id -> latest metrics dict

def update_live_metrics():
    """Background thread: tail MLflow for the most recent active run."""
    while True:
        try:
            active_runs = client.search_runs(
                experiment_ids=["*"],
                filter_string="attributes.status = 'RUNNING'",
                max_results=5,
            )
            for run in active_runs:
                model_id = run.data.tags.get("model_id", "unknown")
                _live_metrics[model_id] = {
                    "run_id": run.info.run_id,
                    "model_id": model_id,
                    "phase": run.data.metrics.get("train/phase", 0),
                    "epoch": run.data.metrics.get("epoch", 0),
                    "train_loss": run.data.metrics.get("train/loss"),
                    "val_ncc": run.data.metrics.get("val/ncc"),
                    "val_rmse": run.data.metrics.get("val/rmse"),
                    "gpu_temp": run.data.metrics.get("gpu/temp_c"),
                    "vram_gb": run.data.metrics.get("gpu/vram_used_gb"),
                    "timestamp": time.time(),
                }
        except Exception:
            pass
        time.sleep(5)

threading.Thread(target=update_live_metrics, daemon=True).start()


@app.route("/api/training/live")
def live_training_stream():
    """Server-Sent Events stream of live training metrics."""
    def generate():
        last_sent = {}
        while True:
            if _live_metrics != last_sent:
                data = json.dumps(list(_live_metrics.values()))
                yield f"data: {data}\n\n"
                last_sent = dict(_live_metrics)
            time.sleep(2)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# Benchmark API
# ---------------------------------------------------------------------------
@app.route("/api/benchmark")
def get_benchmark():
    """Return the latest benchmark comparison table."""
    benchmark_file = RESULTS_DIR / "benchmark" / "comparison_table.json"
    if not benchmark_file.exists():
        # Return mock data if not yet generated
        return jsonify({"status": "not_ready", "message": "Run benchmark_runner.py first"})

    with open(benchmark_file) as f:
        return jsonify(json.load(f))


# ---------------------------------------------------------------------------
# Files API
# ---------------------------------------------------------------------------
@app.route("/api/files")
def list_files():
    """Recursive listing of results/ directory."""
    def scan(path: Path):
        items = []
        for p in sorted(path.iterdir()):
            item = {
                "name": p.name,
                "path": str(p.relative_to(PROJECT_ROOT)),
                "type": "dir" if p.is_dir() else "file",
                "size": p.stat().st_size if p.is_file() else None,
                "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat() if p.exists() else None,
            }
            if p.is_dir():
                item["children"] = scan(p)
            items.append(item)
        return items

    return jsonify(scan(RESULTS_DIR))


@app.route("/api/files/<path:filepath>")
def serve_file(filepath):
    """Serve a file from the results directory."""
    full_path = PROJECT_ROOT / filepath
    if not full_path.exists() or not full_path.is_file():
        return jsonify({"error": "File not found"}), 404
    return send_file(full_path)


# ---------------------------------------------------------------------------
# Training Control API
# ---------------------------------------------------------------------------
@app.route("/api/training/start", methods=["POST"])
def start_training():
    """Launch a training run via subprocess."""
    body = request.get_json() or {}
    model_id = body.get("model_id", "MR-TAE-FULL")
    hparams = body.get("hparams", "config/training_config.yaml")

    cmd = [
        "python", "pipeline/mlflow_orchestrator.py",
        "--model", model_id,
        "--hparams", hparams,
    ]
    proc = subprocess.Popen(
        cmd, cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    return jsonify({"status": "started", "pid": proc.pid, "model_id": model_id})


@app.route("/api/training/stop", methods=["POST"])
def stop_training():
    """Send SIGINT to running training process (triggers graceful checkpoint save)."""
    body = request.get_json() or {}
    pid = body.get("pid")
    if not pid:
        return jsonify({"error": "pid required"}), 400
    try:
        os.kill(int(pid), 2)  # SIGINT
        return jsonify({"status": "stopped", "pid": pid})
    except ProcessLookupError:
        return jsonify({"error": "Process not found"}), 404


# ---------------------------------------------------------------------------
# GPU Status API
# ---------------------------------------------------------------------------
@app.route("/api/gpu")
def gpu_status():
    """Return current GPU stats via nvidia-smi."""
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits"
        ]).decode().strip()
        fields = [f.strip() for f in out.split(",")]
        return jsonify({
            "name": fields[0],
            "temp_c": int(fields[1]),
            "utilization_pct": int(fields[2]),
            "memory_used_mb": int(fields[3]),
            "memory_total_mb": int(fields[4]),
            "power_draw_w": float(fields[5]) if fields[5] != "[N/A]" else None,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Session & Resilience APIs
# ---------------------------------------------------------------------------
@app.route("/api/session")
def get_session():
    """Return full session_state.json for the dashboard status panel."""
    session_file = PROJECT_ROOT / "session_state.json"
    if not session_file.exists():
        return jsonify({"status": "no_session", "message": "No training session found yet."})
    with open(session_file) as f:
        return jsonify(json.load(f))


@app.route("/api/session/note", methods=["POST"])
def add_session_note():
    """Add a human note to the session (e.g. 'WiFi dropped at 14:32')."""
    body = request.get_json() or {}
    note = body.get("note", "").strip()
    if not note:
        return jsonify({"error": "note required"}), 400
    session_file = PROJECT_ROOT / "session_state.json"
    if session_file.exists():
        with open(session_file) as f:
            s = json.load(f)
        s.setdefault("notes", []).append({
            "time": datetime.now().isoformat(), "note": note
        })
        tmp = session_file.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(s, f, indent=2)
        tmp.replace(session_file)
    return jsonify({"status": "ok", "note": note})


@app.route("/api/resume-prompt")
def get_resume_prompt():
    """Return the current RESUME_PROMPT.md content."""
    prompt_file = PROJECT_ROOT / "RESUME_PROMPT.md"
    if not prompt_file.exists():
        return jsonify({"content": "No resume prompt yet — start training first."})
    with open(prompt_file) as f:
        return jsonify({"content": f.read()})


@app.route("/api/live-log")
def live_log_stream():
    """SSE stream of logs/training_live.log — tails the file in real time."""
    log_file = PROJECT_ROOT / "logs" / "training_live.log"

    def generate():
        # Send last 50 lines on connect
        if log_file.exists():
            with open(log_file) as f:
                lines = f.readlines()
            for line in lines[-50:]:
                yield f"data: {json.dumps({'line': line.rstrip()})}\n\n"

        # Then tail for new lines
        with open(log_file, "a+") as f:
            f.seek(0, 2)  # seek to end
            while True:
                line = f.readline()
                if line:
                    yield f"data: {json.dumps({'line': line.rstrip()})}\n\n"
                else:
                    time.sleep(0.5)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.touch(exist_ok=True)
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    print("🔬 AE-PD Research Dashboard")
    print("   Backend: http://localhost:8080")
    print("   MLflow:  http://localhost:5000")
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
