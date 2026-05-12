# RESUME PROMPT
*Auto-generated: 2026-04-10 22:53:23*
*Last update: 2026-04-10T22:53:23.544219*

---

## Paste this into a new Claude / Cursor session to continue:

```
I am resuming the AE-PD Denoising research pipeline.
Project: Intelligent Partial Discharge Denoising — Oo Jian Hong, University of Malaya 2026

## Current state (from session_state.json):

COMPLETED: MR-TAE-noBiGRU, MR-TAE-noSwin, MR-TAE-noAttn, MR-TAE-noWavelet, MR-TAE-FULL, MR-TAE-noMTL
SKIPPED:   none
REMAINING: none


## Instructions for the agent:

1. Activate venv: .venv\Scripts\activate (Windows) or source .venv/bin/activate (Linux)
2. Verify: python -c "import torch; print(torch.cuda.get_device_name(0))"
3. Read session_state.json — it has the full state of every model
4. If any model has status='running': resume it from its checkpoint with
   python pipeline/mlflow_orchestrator.py --resume
5. If no model is running: continue with the next 'queued' model
6. Do NOT restart already-completed or skipped models
7. After all models done: run evaluation/benchmark_runner.py --compare-all

The session_state.json and logs/training_live.log have full details.
```

---

## Quick status table

| Model | Status | Best NCC | Epoch | Skip reason |
|---|---|---|---|---|
| MR-TAE-noBiGRU | done | 0.9758 | 99 | — |
| MR-TAE-noSwin | done | 0.9764 | 99 | — |
| MR-TAE-noAttn | done | 0.9701 | 24 | — |
| MR-TAE-noWavelet | done | — | 0 | — |
| MR-TAE-FULL | done | 0.9764 | 99 | — |
| MR-TAE-noMTL | done | — | 0 | — |

---
*Log file: logs/training_live.log*
*Session file: session_state.json*
*To add a note: resilience.add_note('your message')*