# Team Contributions

## Project: Prompt Injection Detection in LLM-Based Software Services

---

## Team Members & Responsibilities

---

### Member 1 - Sindhu — Data & Baseline Modeling

**Responsibilities:**
- Sourced and merged five HuggingFace datasets into a unified binary-classification corpus (~450k samples)
- Implemented the full data pipeline (`src/prepare_data.py`): downloading, schema standardization, deduplication, and 80/10/10 train/val/test splitting
- Maintained two held-out out-of-distribution test sets (`test_deepset.parquet`, `test_wildcard.parquet`) for generalization evaluation
- Built the TF-IDF + Logistic Regression baseline (`src/baseline.py`) in two feature modes:
  - **Simple:** word unigrams/bigrams, 10k features
  - **Enhanced:** word n-grams + character n-grams + handcrafted security keywords (e.g., "ignore", "system prompt"), special character ratio, URL/email presence, bracket balance
- Ran 5-fold cross-validation grid search over regularization strength (C)
- Evaluated the baseline models across all four data splits and produced confusion matrices, ROC curves, and precision-recall curves saved to `results/baseline/`

**Key files:**
- `src/prepare_data.py`
- `src/baseline.py`
- `src/evaluate.py`
- `.env.example`

---

### Member 2 - Maulik — Deep Learning Model (RoBERTa Fine-Tuning)

**Responsibilities:**
- Designed and implemented the RoBERTa fine-tuning pipeline (`src/train_roberta.py`)
- Supported two model scales: `roberta-base` (125M params) and `roberta-large` (355M params, recommended)
- Implemented a custom PyTorch Dataset + DataLoader for tokenization and batch generation
- Applied weighted cross-entropy loss to handle class imbalance
- Enabled mixed-precision (fp16) training for GPU efficiency
- Added early stopping based on validation loss to prevent overfitting
- Evaluated fine-tuned models on all four data splits (val, test, test_deepset, test_wildcard)
- Ran results aggregation and comparison (`src/compare_results.py`), generating:
  - `summary_table.csv` and `summary_table.tex` (LaTeX-ready for academic paper)
  - Grouped bar charts, heatmaps, radar plots, and generalization gap scatter plots in `results/comparison/`
- Ran experiments on GPU (RTX 4090): ~30 min for roberta-base, ~90 min for roberta-large

**Key files:**
- `src/train_roberta.py`
- `src/compare_results.py`
- `results/` (generated outputs)
- `scripts/check_cuda.py`
- `notebooks/01_EDA.ipynb`

---

### Member 3 - Anish Paul Singareddy — Backend API & Database

**Responsibilities:**
- Designed and implemented the full FastAPI backend (`src/api.py`, 950 lines)
- Built the detector registry: loads all available models (baseline, roberta-base, roberta-large) at startup
- Implemented the core API endpoints:
  - `/detect` — single prompt classification without LLM forwarding
  - `/ab/chat` — A/B testing endpoint running two detector+LLM configurations concurrently via `asyncio.gather`
  - `/v1/chat/completions` — OpenAI SDK-compatible proxy with injection screening
  - `/health`, `/available-detectors`, `/available-llms` — status and discovery endpoints
  - `/sessions/*` — session CRUD
  - `/results/*` and `/app` — static file serving for frontend and result plots
- Implemented fail-closed mode (`BLOCK_ON_ERROR`), per-request threshold overrides, and response headers (`X-Injection-Detected`, `X-Injection-Probability`, etc.)
- Designed and implemented the async SQLite database layer (`src/database.py`) using `aiosqlite`:
  - `sessions` table — stores A/B session configuration and metadata
  - `exchanges` table — stores per-turn user prompts, detector verdicts, LLM responses, and latencies
  - WAL mode, foreign key constraints, and cascade deletes

**Key files:**
- `src/api.py`
- `src/database.py`
- `requirements.txt`
- `requirements.docker.txt`

---

### Member 4 - Isaac — Frontend, Docker & Documentation

**Responsibilities:**
- Built the full React + Vite frontend (`frontend/`) for interactive A/B testing:
  - Dual chat panels with per-side detector and LLM dropdowns
  - Threshold slider (0.0–1.0) for real-time sensitivity tuning
  - Session sidebar with create, load, rename, and delete
  - Dark/light theme toggle persisted to `localStorage`
  - Tab system: Chat | Results | Models
- Developed all UI components:
  - `ChatPanel.jsx` — main chat panel with detector badges and message history
  - `MessageBubble.jsx` — message UI with blocking indicators and detection probability
  - `ResultsPanel.jsx` — metrics table, ROC/PR curves, confusion matrices
  - `ModelsPanel.jsx` — Ollama model download manager with VRAM requirements
  - `SessionsSidebar.jsx` — persistent session history
  - `RocCurveChart.jsx`, `PrCurveChart.jsx`, `ConfusionMatrixChart.jsx` — interactive chart components using Recharts
  - `Icons.jsx` — SVG icon library
- Wrote the HTTP client (`frontend/src/api.js`) for all FastAPI calls
- Configured Vite with API proxy for local development (`vite.config.js`)
- Authored the multi-stage `Dockerfile`:
  - Stage 1: Node 20 — builds React app into `frontend/dist/`
  - Stage 2: Python 3.11 slim — installs PyTorch (CUDA 12.1 on amd64, CPU on arm64), copies app and frontend
- Wrote `docker-compose.yml` orchestrating Ollama + app containers with GPU passthrough, health checks, and persistent volumes
- Wrote `scripts/run_local.sh` for one-command local development startup
- Authored all project documentation:
  - `README.md` — overview, quick start, architecture diagram
  - `docs/COMMANDS.md` — step-by-step experiment reproduction guide (including SLURM/HPC instructions)
  - `docs/DOCKER.md` — Docker build, push, deployment, and troubleshooting guide
  - `docs/FRONTEND.md` — frontend architecture and component reference
  - `docs/PROJECT_JOURNAL.md` — project development notes and design decisions

**Key files:**
- `frontend/` (all files)
- `Dockerfile`
- `docker-compose.yml`
- `scripts/run_local.sh`
- `docs/`
- `README.md`
