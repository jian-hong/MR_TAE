# Full training launch (resilience + MLflow)

1. **Environment (Windows / PowerShell)**  
   Use the project venv at repo root (`.venv` recommended; junction to `Data/FYP_MASTER/.venv` if configured):

   ```powershell
   & .\.venv\Scripts\Activate.ps1
   ```

2. **Start all registered models** (uses `pipeline/mlflow_orchestrator.py`, session state, checkpoints under `results/<MODEL_ID>/`):

   ```powershell
   .\scripts\start_training.ps1
   ```

   Or directly:

   ```powershell
   python pipeline\mlflow_orchestrator.py --run-all
   ```

3. **Resume after interrupt / reboot**

   ```powershell
   .\scripts\resume.ps1
   ```

4. **Single model**

   ```powershell
   python pipeline\mlflow_orchestrator.py --model MR-TAE-FULL
   ```

5. **Dashboard** (after `dashboard\setup_dashboard.ps1`):

   ```powershell
   python dashboard\app.py
   ```

   Open `http://127.0.0.1:8080`.

MLflow runs are stored under `mlruns/`. Live log: `logs/training.log`.
