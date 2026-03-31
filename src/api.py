"""
Multi-detector FastAPI middleware for prompt injection detection.

Loads every trained detector it can find at startup and exposes:
  /available-detectors  — list of loaded detector names
  /available-llms       — Ollama models available locally + recommended pull list
  /detect               — classify a single prompt (uses primary detector)
  /ab/chat              — A/B test endpoint: runs two configs concurrently
  /v1/chat/completions  — OpenAI-compatible proxy (uses primary detector)
  /health               — server health
  /models               — alias for /health

A/B test flow:
  Client sends one message + two side configs (detector + llm).
  Both sides run concurrently (asyncio.gather).
  Side without detector → forwards straight to LLM.
  Side with detector → screens first, blocks or forwards.

Usage:
  uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

Environment variables (all optional):
  DETECTOR_MODEL        Primary detector for /v1/chat/completions: baseline|roberta
  DETECTOR_MODEL_PATH   Override auto-detection path for primary detector
  DETECTOR_THRESHOLD    Default threshold (default: 0.5)
  DETECTOR_MAX_LENGTH   Tokenizer max length (default: 256)
  LLM_API_URL           Ollama / OpenAI-compatible base URL (default: http://localhost:11434/v1)
  LLM_API_KEY           API key (default: ollama)
  LLM_MODEL             Default model forwarded to LLM (default: mistral:7b)
  BLOCK_ON_ERROR        true = fail-closed, false = fail-open (default: false)
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src import database

import httpx
import torch
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field as PydanticField

log = logging.getLogger("api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PRIMARY_DETECTOR   = os.getenv("DETECTOR_MODEL", "roberta").lower()
DETECTOR_PATH      = os.getenv("DETECTOR_MODEL_PATH", "")
DETECTOR_THRESHOLD = float(os.getenv("DETECTOR_THRESHOLD", "0.5"))
DETECTOR_MAX_LEN   = int(os.getenv("DETECTOR_MAX_LENGTH", "256"))

LLM_API_URL  = os.getenv("LLM_API_URL",  "http://localhost:11434/v1")
LLM_API_KEY  = os.getenv("LLM_API_KEY",  "ollama")
LLM_MODEL    = os.getenv("LLM_MODEL",    "mistral:7b")

BLOCK_ON_ERROR = os.getenv("BLOCK_ON_ERROR", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Detector registry
# ---------------------------------------------------------------------------

@dataclass
class DetectorState:
    model_type: str
    model:      Any
    tokenizer:  Any               # None for baseline
    device:     torch.device
    loaded:     bool = True


# Global registry: name → DetectorState
# "none" is a virtual entry meaning "no detector"
_detectors: dict[str, DetectorState] = {}


# ---------------------------------------------------------------------------
# VRAM / model catalogue (shown in /available-llms for models not yet pulled)
# ---------------------------------------------------------------------------

RECOMMENDED_MODELS = [
    {"name": "mistral:7b",       "vram_gb": 4.1, "params": "7B",  "safety": "low",      "note": "Best for injection demos — minimal safety RLHF"},
    {"name": "openhermes:latest","vram_gb": 4.1, "params": "7B",  "safety": "low",      "note": "Fine-tuned Mistral, minimal safety training"},
    {"name": "llama3.2:3b",      "vram_gb": 2.0, "params": "3B",  "safety": "moderate", "note": "Small and fast"},
    {"name": "llama3.1:8b",      "vram_gb": 4.7, "params": "8B",  "safety": "moderate", "note": "Well-rounded, decent for demos"},
    {"name": "gemma2:9b",        "vram_gb": 5.5, "params": "9B",  "safety": "moderate", "note": "Google Gemma 2"},
    {"name": "deepseek-r1:7b",   "vram_gb": 4.7, "params": "7B",  "safety": "moderate", "note": "Reasoning model — interesting injection behaviour"},
    {"name": "phi4:14b",         "vram_gb": 8.7, "params": "14B", "safety": "good",     "note": "Microsoft Phi-4 — strong safety training"},
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _auto_path(model_type: str) -> Optional[Path]:
    candidates: dict[str, list[Path]] = {
        "baseline": [
            PROJECT_ROOT / "models" / "model_enhanced.joblib",
            PROJECT_ROOT / "models" / "model_simple.joblib",
        ],
        "roberta": [
            PROJECT_ROOT / "models" / "roberta" / "roberta-large",
            PROJECT_ROOT / "models" / "roberta" / "roberta-base",
        ],
    }
    for path in candidates.get(model_type, []):
        if path.exists():
            return path
    return None


def _try_load_baseline(path: Path) -> Optional[DetectorState]:
    try:
        import joblib
        log.info(f"Loading baseline from {path}")
        model = joblib.load(path)
        return DetectorState(
            model_type="baseline",
            model=model,
            tokenizer=None,
            device=torch.device("cpu"),
        )
    except Exception as exc:
        log.warning(f"Baseline load failed: {exc}")
        return None


def _try_load_roberta(path: Path) -> Optional[DetectorState]:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        log.info(f"Loading RoBERTa from {path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(str(path))
        model = AutoModelForSequenceClassification.from_pretrained(str(path))
        model.eval().to(device)
        return DetectorState(model_type="roberta", model=model, tokenizer=tokenizer, device=device)
    except Exception as exc:
        log.warning(f"RoBERTa load failed: {exc}")
        return None


def load_all_detectors() -> None:
    """Try to load every available detector at startup and register them."""
    loaders = {
        "baseline": _try_load_baseline,
        "roberta":  _try_load_roberta,
    }

    for name, loader in loaders.items():
        # If this is the primary detector and a custom path was set, use that
        if name == PRIMARY_DETECTOR and DETECTOR_PATH:
            path = Path(DETECTOR_PATH)
        else:
            path = _auto_path(name)

        if path is None:
            log.info(f"  {name}: not found (train it first to enable)")
            continue

        state = loader(path)
        if state is not None:
            _detectors[name] = state
            log.info(f"  {name}: loaded from {path}  (device={state.device})")

    if not _detectors:
        msg = "No detectors loaded. Run training scripts first."
        if BLOCK_ON_ERROR:
            raise RuntimeError(msg)
        log.warning(msg + " API will pass all requests through unscreened.")
    else:
        log.info(f"Loaded detectors: {list(_detectors.keys())}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def _infer(ds: DetectorState, text: str, threshold: float) -> tuple[float, int]:
    if ds.model_type == "baseline":
        prob  = float(ds.model.predict_proba([text])[0][1])
        return prob, int(prob >= threshold)

    enc = ds.tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=DETECTOR_MAX_LEN,
        return_tensors="pt",
    )
    enc = {k: v.to(ds.device) for k, v in enc.items()}
    out  = ds.model(**enc)
    prob = float(torch.softmax(out.logits.float(), dim=-1)[0, 1].cpu())
    return prob, int(prob >= threshold)


def _screen(detector_name: str, text: str, threshold: Optional[float] = None) -> dict:
    """
    Run the named detector and return a result dict.
    detector_name='none' → always safe (no screening).
    Never raises — errors are handled based on BLOCK_ON_ERROR.
    """
    eff_threshold = threshold if threshold is not None else DETECTOR_THRESHOLD

    if detector_name == "none" or detector_name not in _detectors:
        return {
            "verdict": "safe", "probability": 0.0, "label": 0,
            "threshold": eff_threshold, "detector": "none", "latency_ms": 0.0,
        }

    ds = _detectors[detector_name]
    t0 = time.perf_counter()
    try:
        prob, label = _infer(ds, text, eff_threshold)
        verdict = "blocked" if label == 1 else "safe"
    except Exception as exc:
        log.exception(f"Detector {detector_name!r} inference failed: {exc}")
        prob, label, verdict = (1.0, 1, "blocked") if BLOCK_ON_ERROR else (0.0, 0, "safe")

    return {
        "verdict":     verdict,
        "probability": round(prob, 6),
        "label":       label,
        "threshold":   eff_threshold,
        "detector":    detector_name,
        "latency_ms":  round((time.perf_counter() - t0) * 1000, 2),
    }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=" * 60)
    log.info("Prompt Injection Detection API — startup")
    log.info(f"Primary detector : {PRIMARY_DETECTOR}")
    log.info(f"Default LLM      : {LLM_MODEL}  at  {LLM_API_URL}")
    log.info(f"Threshold        : {DETECTOR_THRESHOLD}")
    log.info(f"Block on error   : {BLOCK_ON_ERROR}")
    log.info("=" * 60)
    load_all_detectors()
    await database.init_db()
    log.info(f"Chat history DB  : {database.DB_PATH}")
    yield
    log.info("API shutting down.")


app = FastAPI(
    title="Prompt Injection Detection API",
    description=(
        "Multi-detector middleware with A/B testing support. "
        "OpenAI-compatible /v1/chat/completions endpoint for transparent proxy use."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve built React frontend from /app if it exists
_FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
if _FRONTEND_DIST.exists():
    app.mount("/app", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")
    log.info(f"Serving frontend from {_FRONTEND_DIST}")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class DetectRequest(BaseModel):
    text:      str                    = PydanticField(..., description="Prompt text to screen")
    detector:  Optional[str]          = PydanticField(None, description="Detector to use (overrides primary)")
    threshold: Optional[float]        = PydanticField(None, description="Override threshold for this request")


class DetectResponse(BaseModel):
    verdict:     str
    probability: float
    label:       int
    threshold:   float
    detector:    str
    latency_ms:  float


class Message(BaseModel):
    role:    str
    content: str


class SideConfig(BaseModel):
    detector: str  = PydanticField("none",        description="Detector name or 'none'")
    llm:      str  = PydanticField("mistral:7b",  description="Ollama model tag")
    label:    Optional[str] = PydanticField(None, description="Display label for this side")


class AbChatRequest(BaseModel):
    messages:   list[Message]
    messages_a: Optional[list[Message]] = None  # per-side history; falls back to messages
    messages_b: Optional[list[Message]] = None
    side_a:     SideConfig
    side_b:     SideConfig
    threshold:  Optional[float] = None
    session_id: Optional[str]   = None   # persist to this session; None = auto-create


class SideResult(BaseModel):
    verdict:     str
    probability: float
    detector:    str
    llm:         str
    label:       str
    response:    str
    blocked:     bool
    detect_ms:   float
    total_ms:    float


class AbChatResponse(BaseModel):
    side_a:     SideResult
    side_b:     SideResult
    session_id: str


# ── Session schemas ──────────────────────────────────────────────────────────

class SessionCreate(BaseModel):
    threshold: float             = 0.5
    a_config:  dict              = PydanticField(default_factory=dict)
    b_config:  dict              = PydanticField(default_factory=dict)
    title:     Optional[str]     = None


class SessionPatch(BaseModel):
    title: str


class ChatCompletionRequest(BaseModel):
    model:       Optional[str]  = None
    messages:    list[Message]
    temperature: Optional[float] = 0.7
    max_tokens:  Optional[int]   = None
    stream:      Optional[bool]  = False
    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health check")
async def health():
    return {
        "status":       "ok" if _detectors else "degraded",
        "loaded_detectors": list(_detectors.keys()),
        "primary_detector": PRIMARY_DETECTOR,
        "threshold":    DETECTOR_THRESHOLD,
        "llm_backend":  LLM_API_URL,
        "default_llm":  LLM_MODEL,
        "block_on_error": BLOCK_ON_ERROR,
    }


@app.get("/models", summary="Alias for /health")
async def models_alias():
    return await health()


@app.get("/available-detectors", summary="List all loaded detector names")
async def available_detectors():
    """
    Returns detector names that can be used in /ab/chat and /detect.
    Always includes 'none' (no screening).
    """
    entries = [
        {
            "name":       "none",
            "loaded":     True,
            "model_type": "none",
            "device":     "n/a",
            "description": "No detector — all prompts pass through",
        }
    ]
    for name, ds in _detectors.items():
        entries.append({
            "name":       name,
            "loaded":     True,
            "model_type": ds.model_type,
            "device":     str(ds.device),
            "description": {
                "baseline": "TF-IDF + Logistic Regression",
                "roberta":  "Fine-tuned RoBERTa",
            }.get(name, name),
        })
    return {"detectors": entries}


@app.get("/available-llms", summary="LLMs available in Ollama + recommended pull list")
async def available_llms():
    """
    Queries the local Ollama instance for installed models.
    Also returns a recommended pull list with VRAM estimates.
    """
    installed: list[dict] = []
    try:
        ollama_base = LLM_API_URL.replace("/v1", "")
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{ollama_base}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            installed = [
                {
                    "name":       m["name"],
                    "size_gb":    round(m.get("size", 0) / 1e9, 1),
                    "installed":  True,
                }
                for m in data.get("models", [])
            ]
    except Exception as exc:
        log.warning(f"Could not query Ollama: {exc}")

    installed_names = {m["name"] for m in installed}
    recommended = [
        {**m, "installed": m["name"] in installed_names}
        for m in RECOMMENDED_MODELS
    ]

    return {
        "installed": installed,
        "recommended": recommended,
        "ollama_url": LLM_API_URL,
    }


class PullRequest(BaseModel):
    name: str = PydanticField(..., description="Model tag to pull, e.g. 'llama3.2:3b'")


@app.post("/ollama/pull", summary="Pull an Ollama model — streams progress as SSE")
async def ollama_pull(req: PullRequest):
    """
    Proxies the Ollama /api/pull stream back to the caller as Server-Sent Events.
    Each SSE event is a JSON object:
      { status, digest, total, completed, percent, speed_bps, eta_s }
    The last event will have status='success' or status='error'.
    """
    ollama_base = LLM_API_URL.replace("/v1", "").rstrip("/")

    async def event_stream():
        prev_digest    = ""
        prev_completed = 0
        prev_time      = time.perf_counter()
        smooth_speed   = 0.0   # bytes/sec — exponential moving average
        alpha          = 0.3   # EMA factor

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{ollama_base}/api/pull",
                    json={"name": req.name},
                ) as resp:
                    resp.raise_for_status()
                    async for raw_line in resp.aiter_lines():
                        if not raw_line.strip():
                            continue
                        try:
                            data = json.loads(raw_line)
                        except json.JSONDecodeError:
                            continue

                        status    = data.get("status", "")
                        digest    = data.get("digest", "")
                        total     = data.get("total", 0)
                        completed = data.get("completed", 0)
                        now       = time.perf_counter()

                        # Each blob starts fresh — reset speed tracker on digest change
                        if digest and digest != prev_digest:
                            prev_digest    = digest
                            prev_completed = 0
                            prev_time      = now
                            smooth_speed   = 0.0

                        speed_bps = 0.0
                        eta_s     = None
                        percent   = 0.0

                        if total > 0:
                            percent = round(completed / total * 100, 1)

                        dt = now - prev_time
                        if dt > 0.2 and completed > prev_completed:
                            instant      = (completed - prev_completed) / dt
                            smooth_speed = (alpha * instant + (1 - alpha) * smooth_speed) if smooth_speed > 0 else instant
                            speed_bps    = smooth_speed
                            prev_completed = completed
                            prev_time      = now
                        elif smooth_speed > 0:
                            speed_bps = smooth_speed

                        if speed_bps > 0 and total > completed:
                            eta_s = round((total - completed) / speed_bps)

                        event = {
                            "status":    status,
                            "digest":    digest,
                            "total":     total,
                            "completed": completed,
                            "percent":   percent,
                            "speed_bps": round(speed_bps),
                            "eta_s":     eta_s,
                        }
                        yield f"data: {json.dumps(event)}\n\n"

            yield f"data: {json.dumps({'status': 'success'})}\n\n"

        except Exception as exc:
            log.warning(f"Ollama pull error for {req.name!r}: {exc}")
            yield f"data: {json.dumps({'status': 'error', 'error': str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/ollama/models/{model_name:path}", summary="Delete an Ollama model")
async def ollama_delete(model_name: str):
    """Delete a model from the local Ollama instance."""
    ollama_base = LLM_API_URL.replace("/v1", "").rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.request(
                "DELETE",
                f"{ollama_base}/api/delete",
                json={"name": model_name},
            )
            resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, f"Ollama delete failed: {exc.response.text}")
    except Exception as exc:
        raise HTTPException(500, f"Could not reach Ollama: {exc}")
    return {"deleted": model_name}


@app.post("/detect", response_model=DetectResponse, summary="Classify a single prompt")
async def detect(req: DetectRequest):
    """Classify a prompt without forwarding to any LLM."""
    detector_name = req.detector or PRIMARY_DETECTOR
    if detector_name not in _detectors and detector_name != "none":
        available = ["none"] + list(_detectors.keys())
        raise HTTPException(400, f"Unknown detector {detector_name!r}. Available: {available}")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _screen, detector_name, req.text, req.threshold)

    log.info(
        f"/detect  detector={detector_name}  verdict={result['verdict']}  "
        f"prob={result['probability']:.4f}  latency={result['latency_ms']:.1f}ms"
    )
    return DetectResponse(**result)


@app.post("/ab/chat", response_model=AbChatResponse, summary="A/B test: two configs, one message")
async def ab_chat(req: AbChatRequest):
    """
    Send the same conversation to two different configurations simultaneously.
    Both sides run concurrently via asyncio.gather.
    Supports per-side message histories via messages_a / messages_b so that
    blocked prompts are never included in a side's future LLM context.
    """
    # Per-side histories (fall back to shared messages if not provided)
    msgs_a = req.messages_a or req.messages
    msgs_b = req.messages_b or req.messages

    user_msgs_a = [m for m in msgs_a if m.role == "user"]
    user_msgs_b = [m for m in msgs_b if m.role == "user"]
    if not user_msgs_a and not user_msgs_b:
        raise HTTPException(400, "No user message found.")

    # Screen text is always the latest user message on each side
    screen_text_a = user_msgs_a[-1].content if user_msgs_a else ""
    screen_text_b = user_msgs_b[-1].content if user_msgs_b else ""

    # Use side B's screen text for the session title (or side A's as fallback)
    screen_text = screen_text_b or screen_text_a

    async def run_side(cfg: SideConfig, messages: list[Message], screen_text: str) -> SideResult:
        t_start = time.perf_counter()
        label   = cfg.label or cfg.detector

        # 1. Screen
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _screen, cfg.detector, screen_text, req.threshold)
        detect_ms = result["latency_ms"]

        if result["verdict"] == "blocked":
            total_ms = round((time.perf_counter() - t_start) * 1000, 1)
            prob     = result["probability"]
            return SideResult(
                verdict="blocked",
                probability=prob,
                detector=cfg.detector,
                llm=cfg.llm,
                label=label,
                response=(
                    f"[BLOCKED] Prompt injection detected "
                    f"(confidence: {prob:.1%}). "
                    "Request was not forwarded to the LLM."
                ),
                blocked=True,
                detect_ms=detect_ms,
                total_ms=total_ms,
            )

        # 2. Forward to LLM using this side's history
        payload = {
            "model":    cfg.llm,
            "messages": [m.model_dump() for m in messages],
            "stream":   False,
        }
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                llm_resp = await client.post(
                    f"{LLM_API_URL.rstrip('/')}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {LLM_API_KEY}"},
                )
                llm_resp.raise_for_status()
                content = llm_resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            content = f"[ERROR] Could not reach LLM backend: {exc}"

        total_ms = round((time.perf_counter() - t_start) * 1000, 1)
        return SideResult(
            verdict="safe",
            probability=result["probability"],
            detector=cfg.detector,
            llm=cfg.llm,
            label=label,
            response=content,
            blocked=False,
            detect_ms=detect_ms,
            total_ms=total_ms,
        )

    side_a, side_b = await asyncio.gather(
        run_side(req.side_a, msgs_a, screen_text_a),
        run_side(req.side_b, msgs_b, screen_text_b),
    )

    log.info(
        f"/ab/chat  A({req.side_a.detector}/{req.side_a.llm})={side_a.verdict}  "
        f"B({req.side_b.detector}/{req.side_b.llm})={side_b.verdict}"
    )

    # ── Persist to DB ────────────────────────────────────────────────────────
    session_id = req.session_id

    # Auto-generate title from the first user message (up to 60 chars)
    title = (screen_text[:60] + "…") if len(screen_text) > 60 else screen_text

    if not session_id:
        # First message in a new conversation — create a session
        session_id = await database.create_session(
            threshold=req.threshold if req.threshold is not None else DETECTOR_THRESHOLD,
            a_config=req.side_a.model_dump(),
            b_config=req.side_b.model_dump(),
            title=title,
        )
    else:
        # Verify the session exists; create a new one if it was deleted
        existing = await database.get_session(session_id)
        if existing is None:
            session_id = await database.create_session(
                threshold=req.threshold if req.threshold is not None else DETECTOR_THRESHOLD,
                a_config=req.side_a.model_dump(),
                b_config=req.side_b.model_dump(),
                title=title,
            )

    await database.save_exchange(
        session_id=session_id,
        user_text=screen_text,
        side_a=side_a.model_dump(),
        side_b=side_b.model_dump(),
    )

    return AbChatResponse(side_a=side_a, side_b=side_b, session_id=session_id)


# ---------------------------------------------------------------------------
# Session / chat-history endpoints
# ---------------------------------------------------------------------------

@app.get("/sessions", summary="List all saved chat sessions")
async def list_sessions():
    return await database.list_sessions()


@app.post("/sessions", summary="Create a new chat session", status_code=201)
async def create_session(body: SessionCreate):
    title = body.title or "New conversation"
    sid = await database.create_session(
        threshold=body.threshold,
        a_config=body.a_config,
        b_config=body.b_config,
        title=title,
    )
    return {"session_id": sid}


@app.get("/sessions/{session_id}", summary="Get a session with all its exchanges")
async def get_session(session_id: str):
    s = await database.get_session(session_id)
    if s is None:
        raise HTTPException(404, f"Session {session_id!r} not found.")
    return s


@app.patch("/sessions/{session_id}", summary="Rename a session")
async def patch_session(session_id: str, body: SessionPatch):
    s = await database.get_session(session_id)
    if s is None:
        raise HTTPException(404, f"Session {session_id!r} not found.")
    await database.update_session_title(session_id, body.title.strip() or "Untitled")
    return {"ok": True}


@app.delete("/sessions/{session_id}", summary="Delete a session and all its exchanges")
async def delete_session(session_id: str):
    await database.delete_session(session_id)
    return {"ok": True}


@app.post("/v1/chat/completions", summary="OpenAI-compatible proxy with injection screening")
async def chat_completions(req: ChatCompletionRequest, response: Response):
    """
    OpenAI-SDK compatible endpoint. Uses the primary detector configured at startup.
    Set X-Injection-* response headers on every request.
    """
    user_msgs = [m for m in req.messages if m.role == "user"]
    if not user_msgs:
        raise HTTPException(400, "No user message found.")

    screen_text = user_msgs[-1].content
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _screen, PRIMARY_DETECTOR, screen_text, None)

    response.headers["X-Injection-Detected"]    = str(result["label"] == 1).lower()
    response.headers["X-Injection-Probability"] = str(result["probability"])
    response.headers["X-Injection-Verdict"]     = result["verdict"]
    response.headers["X-Detector-Model"]        = result["detector"]

    log.info(
        f"/v1/chat/completions  verdict={result['verdict']}  "
        f"prob={result['probability']:.4f}  latency={result['latency_ms']:.1f}ms"
    )

    if result["verdict"] == "blocked":
        return _make_blocked_response(req.model or LLM_MODEL, result["probability"])

    payload = req.model_dump(exclude_none=True)
    payload.setdefault("model", LLM_MODEL)

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            if req.stream:
                return await _proxy_stream(client, payload)
            llm_resp = await client.post(
                f"{LLM_API_URL.rstrip('/')}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            )
            llm_resp.raise_for_status()
            return JSONResponse(content=llm_resp.json())
    except httpx.HTTPStatusError as exc:
        raise HTTPException(502, f"LLM backend returned {exc.response.status_code}.")
    except httpx.RequestError as exc:
        raise HTTPException(502, f"Cannot reach LLM backend at {LLM_API_URL}: {exc}")


# ---------------------------------------------------------------------------
# Results endpoints
# ---------------------------------------------------------------------------

def _scan_results() -> list[dict]:
    """Walk the results/ directory and collect all available metrics + image paths."""
    results_dir = PROJECT_ROOT / "results"
    models = []

    # ── Baseline ────────────────────────────────────────────────────────────
    for mode in ("enhanced", "simple"):
        metrics_file = results_dir / "baseline" / f"metrics_{mode}.json"
        if not metrics_file.exists():
            continue
        metrics = json.loads(metrics_file.read_text())
        confusions = []
        for split in ("val", "test", "test_deepset"):
            img = results_dir / "baseline" / f"confusion_{mode}_{split}.png"
            if img.exists():
                confusions.append({"split": split, "url": f"/results/image/baseline/confusion_{mode}_{split}.png"})
        images = {}
        for plot in ("roc", "pr"):
            img = results_dir / "baseline" / f"{plot}_{mode}.png"
            if img.exists():
                images[plot] = f"/results/image/baseline/{plot}_{mode}.png"
        if confusions:
            images["confusions"] = confusions
        curves = None
        curves_file = results_dir / "baseline" / f"curves_{mode}.json"
        if curves_file.exists():
            curves = json.loads(curves_file.read_text())
        models.append({
            "id":      f"baseline_{mode}",
            "label":   f"Baseline ({'Enhanced TF-IDF' if mode == 'enhanced' else 'Simple TF-IDF'})",
            "metrics": metrics,
            "images":  images,
            "curves":  curves,
        })

    # ── RoBERTa ─────────────────────────────────────────────────────────────
    roberta_dir = results_dir / "roberta"
    if roberta_dir.exists():
        for variant_dir in sorted(roberta_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            metrics_file = variant_dir / "metrics.json"
            if not metrics_file.exists():
                continue
            metrics = json.loads(metrics_file.read_text())
            confusions = []
            for split in ("val", "test", "test_deepset"):
                img = variant_dir / f"confusion_{split}.png"
                if img.exists():
                    confusions.append({"split": split, "url": f"/results/image/roberta/{variant_dir.name}/confusion_{split}.png"})
            images = {}
            for plot in ("roc", "pr"):
                img = variant_dir / f"{plot}.png"
                if img.exists():
                    images[plot] = f"/results/image/roberta/{variant_dir.name}/{plot}.png"
            if confusions:
                images["confusions"] = confusions
            curves = None
            curves_file = variant_dir / "curves.json"
            if curves_file.exists():
                curves = json.loads(curves_file.read_text())
            models.append({
                "id":      f"roberta_{variant_dir.name}",
                "label":   f"RoBERTa ({variant_dir.name})",
                "metrics": metrics,
                "images":  images,
                "curves":  curves,
            })

    return models


@app.get("/results", summary="All available training metrics and plot image paths")
async def get_results():
    """
    Scans results/ for metrics JSON files and returns structured data.
    Only includes models for which results actually exist.
    """
    return {"models": _scan_results()}


@app.get("/results/image/{path:path}", summary="Serve a results plot image")
async def get_results_image(path: str):
    """Serve PNG files from the results/ directory."""
    # Prevent path traversal
    results_dir = PROJECT_ROOT / "results"
    target = (results_dir / path).resolve()
    if not str(target).startswith(str(results_dir.resolve())):
        raise HTTPException(403, "Access denied.")
    if not target.exists() or target.suffix.lower() != ".png":
        raise HTTPException(404, f"Image not found: {path}")
    return FileResponse(str(target), media_type="image/png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_blocked_response(model: str, probability: float) -> JSONResponse:
    created = int(time.time())
    return JSONResponse(
        status_code=200,
        content={
            "id":      f"blocked-{created}",
            "object":  "chat.completion",
            "created": created,
            "model":   model,
            "choices": [{
                "index":   0,
                "message": {
                    "role":    "assistant",
                    "content": (
                        f"[BLOCKED] Prompt injection detected "
                        f"(confidence: {probability:.1%}). "
                        "Request not forwarded to the language model."
                    ),
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        },
    )


async def _proxy_stream(client: httpx.AsyncClient, payload: dict) -> StreamingResponse:
    async def _gen():
        async with client.stream(
            "POST",
            f"{LLM_API_URL.rstrip('/')}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            timeout=120.0,
        ) as r:
            async for chunk in r.aiter_bytes():
                yield chunk
    return StreamingResponse(_gen(), media_type="text/event-stream")
