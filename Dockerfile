# ── Stage 1: Build React frontend ─────────────────────────────────────────────
FROM node:20-slim AS frontend-build
WORKDIR /build
COPY frontend/package*.json ./
RUN npm ci --silent
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python application ───────────────────────────────────────────────
FROM python:3.11-slim AS app
WORKDIR /app

# Build arg set automatically by Docker BuildKit (amd64 or arm64)
ARG TARGETARCH

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.1 support on amd64; CPU-only on arm64.
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121; \
    else \
        pip install --no-cache-dir torch; \
    fi

# Install remaining inference-only dependencies
COPY requirements.docker.txt ./
RUN pip install --no-cache-dir -r requirements.docker.txt

# Source code
COPY src/ ./src/

# Trained model weights.
# Training checkpoints and QLoRA artifacts are excluded via .dockerignore —
# only the final weights needed for inference are copied.
COPY models/ ./models/

# Pre-generated result plots served by the /results endpoints
COPY results/ ./results/

# Built React frontend (served by FastAPI at /app)
COPY --from=frontend-build /build/dist ./frontend/dist

# SQLite chat history lives here; mount a named volume for persistence
RUN mkdir -p /app/data

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
