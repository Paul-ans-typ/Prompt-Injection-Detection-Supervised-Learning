#!/usr/bin/env bash
# run_local.sh — Start the prompt injection detection frontend locally.
#
# Usage (from project root):
#   bash scripts/run_local.sh
#
# What it does:
#   1. Installs Python requirements if needed
#   2. Runs npm install in frontend/ if node_modules is missing
#   3. Checks that trained model files exist
#   4. Checks Ollama; pulls a default LLM on first run if none are present
#   5. Starts FastAPI backend on :8000 (background)
#   6. Starts Vite dev server on :3000 (foreground — Ctrl+C to stop both)

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths — resolve relative to this script regardless of where you call it from
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
MODELS_DIR="$PROJECT_ROOT/models"

# Default Ollama model pulled on first run (smallest footprint, ~2 GB)
DEFAULT_OLLAMA_MODEL="llama3.2:3b"

# Colour helpers (no-op if terminal doesn't support them)
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ---------------------------------------------------------------------------
# 1. Python environment note
# ---------------------------------------------------------------------------
# The script uses whatever is on PATH. If uvicorn fails to start later,
# make sure your venv/conda env is activated and run:
#   pip install -r requirements.txt

# ---------------------------------------------------------------------------
# 2. Frontend npm dependencies
# ---------------------------------------------------------------------------
info "Checking frontend npm dependencies..."

if ! command -v npm &>/dev/null; then
    error "Node.js / npm not found. Install from https://nodejs.org"
    exit 1
fi

if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    info "Running npm install in frontend/..."
    npm install --prefix "$FRONTEND_DIR"
else
    info "node_modules already present — skipping npm install."
fi

# ---------------------------------------------------------------------------
# 3. Verify trained model files
# ---------------------------------------------------------------------------
info "Checking for trained model files..."
MISSING_MODELS=0

# Baseline
if [ ! -f "$MODELS_DIR/model_enhanced.joblib" ] && [ ! -f "$MODELS_DIR/model_simple.joblib" ]; then
    warn "No baseline model found in models/. Run:"
    warn "  python src/baseline.py --mode enhanced"
    MISSING_MODELS=1
else
    info "Baseline model found."
fi

# RoBERTa
if [ ! -d "$MODELS_DIR/roberta/roberta-large" ] && [ ! -d "$MODELS_DIR/roberta/roberta-base" ]; then
    warn "No RoBERTa model found in models/roberta/. Run:"
    warn "  python src/train_roberta.py --model roberta-base"
    MISSING_MODELS=1
else
    info "RoBERTa model found."
fi

if [ "$MISSING_MODELS" -eq 1 ]; then
    warn "Some models are missing — the API will still start but those detectors won't be available."
fi

# ---------------------------------------------------------------------------
# 4. Ollama — check, start hint, pull default model on first run
# ---------------------------------------------------------------------------
info "Checking Ollama..."

if ! command -v ollama &>/dev/null; then
    warn "Ollama not found. Chat features will be unavailable."
    warn "Install from: https://ollama.com"
else
    # Try to reach the Ollama API
    OLLAMA_RUNNING=false
    if curl -sf http://localhost:11434/api/tags -o /dev/null 2>/dev/null; then
        OLLAMA_RUNNING=true
    fi

    if [ "$OLLAMA_RUNNING" = false ]; then
        warn "Ollama is installed but not running."
        warn "Start it in another terminal with:  ollama serve"
        warn "Then re-run this script, or pull models manually."
    else
        info "Ollama is running."

        # Get the list of installed model names
        INSTALLED=$(curl -sf http://localhost:11434/api/tags \
            | python -c "
import sys, json
data = json.load(sys.stdin)
models = [m['name'] for m in data.get('models', [])]
print('\n'.join(models))
" 2>/dev/null || true)

        if [ -z "$INSTALLED" ]; then
            info "No Ollama models found — pulling default model: $DEFAULT_OLLAMA_MODEL"
            ollama pull "$DEFAULT_OLLAMA_MODEL"
        else
            info "Ollama models already installed:"
            echo "$INSTALLED" | sed 's/^/         /'
        fi
    fi
fi

# ---------------------------------------------------------------------------
# 5. Kill any stale backend on port 8000
# ---------------------------------------------------------------------------
BACKEND_PID=""
if command -v lsof &>/dev/null; then
    STALE=$(lsof -ti :8000 2>/dev/null || true)
    if [ -n "$STALE" ]; then
        warn "Port 8000 already in use (PID $STALE) — killing stale process."
        kill "$STALE" 2>/dev/null || true
        sleep 1
    fi
fi

# ---------------------------------------------------------------------------
# 6. Start FastAPI backend in background
# ---------------------------------------------------------------------------
info "Starting FastAPI backend on http://localhost:8000 ..."
cd "$PROJECT_ROOT"
uvicorn src.api:app --port 8000 --log-level warning &
BACKEND_PID=$!

# Wait up to 10 s for the backend to be ready
for i in $(seq 1 10); do
    if curl -sf http://localhost:8000/health -o /dev/null 2>/dev/null; then
        info "Backend is up."
        break
    fi
    sleep 1
done

if ! curl -sf http://localhost:8000/health -o /dev/null 2>/dev/null; then
    warn "Backend did not respond after 10 s — check for errors above."
fi

# ---------------------------------------------------------------------------
# 7. Trap Ctrl+C to kill the background backend when the frontend exits
# ---------------------------------------------------------------------------
cleanup() {
    echo ""
    info "Shutting down..."
    [ -n "$BACKEND_PID" ] && kill "$BACKEND_PID" 2>/dev/null || true
    info "Done."
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# 8. Start Vite dev server (foreground — stays until Ctrl+C)
# ---------------------------------------------------------------------------
info "Starting Vite frontend on http://localhost:3000 ..."
info "Press Ctrl+C to stop both servers."
echo ""
npm run dev --prefix "$FRONTEND_DIR"
