# Frontend — Local Setup Guide

The frontend is a React (Vite) app that talks to a FastAPI backend.
It lets you run A/B tests between two detection configurations side-by-side.

---

## Architecture

```
Browser (port 3000)
  └─ Vite dev server  (proxies API calls to →)
       └─ FastAPI backend  (port 8000)
            ├─ Baseline detector   (models/model_enhanced.joblib)
            ├─ RoBERTa detector    (models/roberta/roberta-large or roberta-base)
            └─ Ollama LLM backend  (port 11434)
```

---

## Quick Start (one command)

Run this from the project root. It handles everything automatically.

```bash
bash scripts/run_local.sh
```

> Make sure your Python environment is active before running (venv, conda, or global Python — any works as long as `pip install -r requirements.txt` has been run in it).

Then open **http://localhost:3000** in your browser.

---

## What the script does

1. Installs Python requirements if not already installed
2. Runs `npm install` inside `frontend/` if `node_modules` is missing
3. Checks that trained model files exist and warns if any are missing
4. Checks if Ollama is installed and running
5. Pulls the default lightweight LLM (`llama3.2:3b`) if no Ollama model is present
6. Starts the FastAPI backend on port 8000 (background)
7. Starts the Vite dev server on port 3000 (foreground)

---

## Manual step-by-step (if you prefer)

### Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.10+ | python.org |
| Node.js | 18+ | nodejs.org |
| Ollama | latest | ollama.com |

### 1 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2 — Make sure trained models exist

You need to have run these first (you only do this once):

```bash
# Data pipeline (downloads datasets)
python src/prepare_data.py

# Baseline model (fast, ~2 min)
python src/baseline.py --mode enhanced
python src/baseline.py --mode simple

# RoBERTa fine-tuning (needs GPU, ~30–90 min)
python src/train_roberta.py --model roberta-base
```

The API will load whichever models it finds. Baseline works without a GPU.
RoBERTa inference at serving time is fast even on CPU for short prompts.

### 3 — Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 4 — Start Ollama (optional but needed for the chat feature)

```bash
# In a separate terminal — start the Ollama daemon
ollama serve

# Pull at least one model (pick based on your VRAM)
ollama pull llama3.2:3b      # ~2 GB  — lowest VRAM, good for testing
ollama pull mistral:7b       # ~4 GB  — best for injection demos
```

You can skip Ollama entirely — the `/detect` endpoint and A/B detection still work,
you just won't see LLM responses in the chat panels.

### 5 — Start the backend

```bash
uvicorn src.api:app --reload --port 8000
```

Check it's up: http://localhost:8000/health

### 6 — Start the frontend

```bash
cd frontend
npm run dev
```

Open: **http://localhost:3000**

---

## Do I need to run compare_results.py?

No. `compare_results.py` generates offline comparison charts and tables
(saved to `results/comparison/`). It has nothing to do with the live frontend.

Run it after your training runs if you want the academic comparison plots:

```bash
python src/compare_results.py
```

---

## Frontend usage

| Panel | Default config | What it shows |
|-------|---------------|---------------|
| Left  | No detector (unprotected) | Raw LLM response |
| Right | RoBERTa detector (protected) | Blocked or passed response |

- **Detector dropdown** — swap between `none`, `baseline`, `roberta`
- **LLM dropdown** — populated from your locally installed Ollama models
- **Threshold slider** — tune the detection confidence cutoff (0.0–1.0)
- Blocked messages are highlighted in red with the confidence score

---

## Environment variables (optional overrides)

Create a `.env` file in the project root to override defaults:

```bash
DETECTOR_MODEL=roberta         # roberta | baseline
DETECTOR_THRESHOLD=0.5
LLM_API_URL=http://localhost:11434/v1
LLM_MODEL=mistral:7b
BLOCK_ON_ERROR=false           # true = fail-closed
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `http://localhost:3000` shows blank page | Check the Vite terminal for errors; make sure `npm install` ran |
| `/health` returns 502 or no detectors loaded | Check uvicorn terminal; verify model files exist in `models/` |
| Chat sends but no LLM reply | Start Ollama (`ollama serve`) and pull a model |
| RoBERTa detector missing | Run `python src/train_roberta.py --model roberta-base` first |
| Port 8000 already in use | Kill the existing process or change the port with `--port 8001` and update `vite.config.js` |
