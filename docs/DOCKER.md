# Docker — Build, Publish & Run Guide

Everything you need to build the images, push them to Docker Hub, and run the stack locally or on any machine.

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- A [Docker Hub](https://hub.docker.com/) account
- The trained model weights present locally (`models/` directory must exist with at least `model_enhanced.joblib` and `models/roberta/roberta-large/`)

---

## 1. One-time setup

### Log in to Docker Hub

```bash
docker login
# enter your Docker Hub username and password when prompted
```

### Enable BuildKit (required for multi-arch builds)

BuildKit is enabled by default in Docker Desktop. If you are on a headless Linux machine, export this variable first:

```bash
export DOCKER_BUILDKIT=1
```

---

## 2. Build the app image

Replace `youruser` with your Docker Hub username in every command below.

There is only **one** image to build — the app (FastAPI + React frontend).
Ollama uses the official `ollama/ollama` image pulled automatically by Docker Compose.

```bash
docker compose build
```

### Tag for Docker Hub

```bash
docker build -t youruser/prompt-injection-app:latest .
```

---

## 3. Push to Docker Hub

Only the app image needs pushing. Ollama is the official public image.

```bash
docker push youruser/prompt-injection-app:latest
```

---

## 4. (Optional) Multi-arch build for Intel + Apple Silicon

If you want a single image that works natively on both `amd64` (Intel/AMD) and `arm64` (Apple Silicon M1/M2/M3) without emulation, use `docker buildx`.

### Create a buildx builder (one-time)

```bash
docker buildx create --name multiarch --use
docker buildx inspect --bootstrap
```

### Build and push multi-arch in one command

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t youruser/prompt-injection-app:latest \
  --push \
  .
```

---

## 5. Using the published images (for end users)

Once the images are on Docker Hub, anyone can run the full stack without cloning the repo or installing Python / Node / Ollama.

### Option A — docker compose (recommended)

Create a `docker-compose.yml` anywhere on your machine with the following content (replace `youruser`):

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    healthcheck:
      test: ["CMD-SHELL", "ollama list || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 12
      start_period: 10s
    restart: unless-stopped

  app:
    image: youruser/prompt-injection-app:latest
    ports:
      - "8000:8000"
    depends_on:
      ollama:
        condition: service_healthy
    environment:
      LLM_API_URL: http://ollama:11434/v1
      LLM_API_KEY: ollama
      LLM_MODEL: llama3.2:3b
      DETECTOR_MODEL: roberta
      DETECTOR_THRESHOLD: "0.5"
      BLOCK_ON_ERROR: "false"
    volumes:
      - chat_data:/app/data
    restart: unless-stopped

volumes:
  ollama_data:
  chat_data:
```

Then run:

```bash
docker compose pull   # pulls youruser/prompt-injection-app + official ollama/ollama
docker compose up     # starts both containers
```

Open **http://localhost:8000/app** in your browser.

Go to the **Models** tab and pull whichever model you want to use. Models are stored in the `ollama_data` volume and survive container restarts — you only download once.

### Option B — docker run (no compose file needed)

```bash
# 1. Create a volume for model persistence
docker volume create ollama_data

# 2. Start Ollama (official image, no custom build needed)
docker run -d --name ollama \
  -p 11434:11434 \
  -v ollama_data:/root/.ollama \
  ollama/ollama:latest

# 3. Wait ~5 seconds for Ollama to be ready, then start the app
docker run -d --name app \
  -p 8000:8000 \
  --link ollama:ollama \
  -e LLM_API_URL=http://ollama:11434/v1 \
  -e LLM_MODEL=llama3.2:3b \
  -e DETECTOR_MODEL=roberta \
  youruser/prompt-injection-app:latest
```

---

## 6. Downloading models

The Ollama container starts empty. Open the **Models** tab in the app to pull models. It shows:

- All recommended models with parameter count, VRAM requirement, and safety rating
- Real-time download progress: percentage, bytes transferred, speed (MB/s), and ETA
- A cancel button if you change your mind mid-download

Models are stored in the `ollama_data` named volume and persist across restarts. Once downloaded you never need to download them again unless you remove the volume.

You can also pull from the terminal if you prefer:

```bash
docker exec -it ollama ollama pull mistral:7b
```

---

## 7. Stopping and cleaning up

```bash
# Stop containers (keeps volumes/data — models and chat history are preserved)
docker compose down

# Stop AND delete everything including downloaded models and chat history
docker compose down -v

# Remove the built app image from disk (Ollama uses the official upstream image, no need to remove)
docker rmi youruser/prompt-injection-app:latest

# Full nuclear cleanup (removes all stopped containers, unused images, build cache)
docker system prune -af
```

---

## 8. Updating the app image

After making code changes:

```bash
docker compose build          # rebuild the app image (cached, usually fast)
docker compose build --no-cache  # force full rebuild (e.g. after dep changes)

# Re-tag and push
docker build -t youruser/prompt-injection-app:latest . \
  && docker push youruser/prompt-injection-app:latest
```

Ollama never needs rebuilding — it's always the official upstream image.

---

## 9. Environment variable reference

All variables can be overridden in `docker-compose.yml` under `environment:` or passed with `-e` to `docker run`.

| Variable | Default | Description |
|----------|---------|-------------|
| `DETECTOR_MODEL` | `roberta` | Active detector: `roberta` or `baseline` |
| `DETECTOR_THRESHOLD` | `0.5` | Injection confidence threshold (0.0–1.0) |
| `DETECTOR_MAX_LENGTH` | `256` | Tokenizer max token length |
| `LLM_API_URL` | `http://ollama:11434/v1` | Ollama base URL (internal Docker network) |
| `LLM_API_KEY` | `ollama` | API key sent to Ollama (can be any string) |
| `LLM_MODEL` | `llama3.2:3b` | Default LLM used for chat |
| `BLOCK_ON_ERROR` | `false` | `true` = block requests when the detector errors |

---

## 10. Troubleshooting

**App container starts but shows "No detectors loaded"**
The `models/` directory was missing or empty when the image was built. Rebuild after confirming `models/roberta/roberta-large/model.safetensors` exists.

**Chat returns "model not found" errors**
No model has been pulled yet. Open the **Models** tab and pull one.

**Very slow inference on Mac**
Expected — Docker on macOS runs Linux containers without Metal GPU access. RoBERTa inference on CPU takes ~200–500 ms per request instead of ~30 ms on GPU. The `baseline` detector is much faster if needed:
```yaml
DETECTOR_MODEL: baseline
```

**Port 8000 or 11434 already in use**
Change the host-side port in `docker-compose.yml`:
```yaml
ports:
  - "8080:8000"   # access the app at localhost:8080 instead
```
