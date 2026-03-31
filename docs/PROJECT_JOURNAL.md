# Project Journal — Prompt Injection Detection

Development log covering the full project lifecycle: timeline, decisions made, and meeting notes.

---

## Project Timeline

### Week 1 — Scoping and Data

- Surveyed the landscape of publicly available prompt-injection datasets on HuggingFace
- Selected five complementary datasets covering direct injections, jailbreaks, and role-play attacks
- Wrote `prepare_data.py` to download, merge, deduplicate, and stratified-split the corpus (~450k prompts)
- Set up a held-out `test_deepset` benchmark (~662 curated examples) for cross-dataset generalisation evaluation
- Established the binary label convention: `0` benign / `1` malicious

### Week 2 — Baseline and RoBERTa Training

- Implemented the TF-IDF + Logistic Regression baseline (`baseline.py`) — trains in ~5 minutes on CPU and provides a strong competitive reference point
- Fine-tuned `roberta-base` then `roberta-large` on the full training split (`train_roberta.py`)
- Wrote `evaluate.py` to generate confusion matrices, ROC curves, and PR curves for every model × split combination
- Wrote `compare_results.py` to aggregate all metrics into a side-by-side comparison table
- RoBERTa-large reached **F1 > 0.97** on the held-out test set; DAN-style jailbreaks that fooled the baseline were caught reliably

### Week 3 — API and A/B Testing Frontend

- Built `src/api.py`: a FastAPI app that loads all available detectors at startup, exposes an OpenAI-compatible `/v1/chat/completions` proxy with injection screening, and adds `X-Injection-*` response headers
- Implemented `/ab/chat` — a concurrent A/B testing endpoint that runs two independently configured sides (detector + LLM) against the same message using `asyncio.gather`
- Built the React + Vite frontend with two synchronised chat panels, per-panel detector/LLM selectors, a threshold slider, and a Results tab rendering training metrics and plots
- Integrated Ollama as the LLM backend; implemented `/ollama/pull` with SSE progress streaming and `/ollama/models` for in-UI model management
- Added `src/database.py` (SQLite via aiosqlite) with full session persistence and a collapsible sidebar

### Week 4 — Containerisation, Bug Fixes, and Polish

- Created a multi-stage `Dockerfile`: Stage 1 builds the React frontend with Node; Stage 2 installs the Python inference stack and serves everything through a single FastAPI process on port 8000
- Added NVIDIA GPU passthrough to `docker-compose.yml` and switched to the CUDA 12.1 PyTorch wheel
- Fixed a Vite `base` path bug causing a blank page when the frontend was served from `/app/` (assets were referenced as `/assets/...` instead of `/app/assets/...`; fixed by setting `base: '/app/'` in `vite.config.js`)
- Fixed a conversation-history isolation bug: the A/B chat was using a single shared message history, so a blocked DAN-mode injection would still reach the LLM on future turns. Implemented **per-side independent histories** — blocked prompts are never appended to that side's LLM context
- Added sync-scroll, light/dark theme toggle, and example injection prompts to the frontend

---

## Meeting Notes

### Meeting 1 — Project Kickoff
**Date:** Week 1, Monday
**Attendees:** Full team

**Agenda:** Define scope, agree on datasets, divide initial work.

**Discussion:**
- Reviewed the OWASP LLM Top 10 list; prompt injection (LLM01) is ranked the highest-severity risk for LLM-integrated applications
- Debated between unsupervised anomaly detection and a supervised binary classifier. Decided on supervised: labelled datasets exist, interpretability is higher, and we can produce concrete precision/recall numbers
- Identified five HuggingFace datasets with complementary coverage: direct injections, indirect injections, DAN-style jailbreaks, and role-play attacks
- Agreed to hold out the deepset dataset entirely from training and use it only as a cross-dataset generalisation benchmark
- Assigned: data pipeline (Ryan), baseline model scaffold (Paul), project repo setup (team)

**Action items:**
- [ ] Finish `prepare_data.py` with deduplication and stratified split by end of week
- [ ] Confirm Kaggle API credentials are shared for dataset download
- [ ] Create `.env.example` template so credentials are never committed

---

### Meeting 2 — Baseline Results Review
**Date:** Week 1, Friday
**Attendees:** Full team

**Agenda:** Review baseline numbers, decide whether to proceed with fine-tuning.

**Discussion:**
- Baseline (TF-IDF + Logistic Regression) achieved ~0.93 F1 on the primary test set — better than expected for a bag-of-words approach
- Precision dropped sharply on the deepset held-out benchmark (~0.81 F1), suggesting the model is memorising surface patterns rather than understanding semantic intent
- Tested adversarial examples manually: the baseline missed almost every DAN-style jailbreak and indirect injection. These are the cases that matter most in production
- Agreed to proceed with RoBERTa fine-tuning — transformer semantic understanding should handle paraphrase-based evasion far better
- Discussed model size trade-offs: `roberta-base` trains ~3× faster but `roberta-large` may recover meaningful accuracy on harder examples. Plan: train both and compare

**Action items:**
- [ ] Set up GPU training environment (SLURM or local RTX 4090)
- [ ] Write `train_roberta.py` with configurable model size, learning rate, and checkpoint saving
- [ ] Define the final evaluation suite: val, test, test_deepset splits + confusion matrix, ROC, PR curve per model

---

### Meeting 3 — Model Results and API Design
**Date:** Week 2, Wednesday
**Attendees:** Full team

**Agenda:** Finalise model selection, design the serving layer.

**Discussion:**
- RoBERTa-large results: **F1 0.974** on primary test, **F1 0.961** on deepset benchmark — significant improvement over baseline, especially on adversarial examples. DAN-mode prompts that fooled the baseline are now caught with >95% confidence
- RoBERTa-base is close but not there on the deepset benchmark; agreed to ship large as the default and keep base as a lighter alternative
- Discussed serving architecture. Three options considered:
  1. Standalone gRPC inference server — complex, overkill for a demo
  2. FastAPI middleware that wraps the model — straightforward, integrates easily
  3. OpenAI-compatible proxy — best option: existing applications can be pointed at it with zero code changes
- Decided on option 3 as the primary interface, with `/detect` for programmatic use and `/ab/chat` for the demo UI
- Agreed on the A/B testing concept: two chat panels, same message sent to both, one with detection and one without — makes the detector's value immediately visible to a non-technical audience
- Agreed to use Ollama as the local LLM backend (no API key required, easy to demo offline)

**Action items:**
- [ ] Scaffold `src/api.py` with lifespan model loading, `/detect`, `/v1/chat/completions`, and `/ab/chat`
- [ ] Design Pydantic schemas for all request/response types
- [ ] Start React frontend with two-panel layout and shared input bar

---

### Meeting 4 — Frontend Demo and Docker Planning
**Date:** Week 3, Thursday
**Attendees:** Full team

**Agenda:** Frontend demo review, plan containerisation.

**Discussion:**
- Live demo of the A/B testing frontend. Sent a DAN-mode jailbreak — the protected panel blocked it instantly at 98% confidence while the unprotected panel generated the jailbreak response. Reaction was positive
- Identified a bug: after a blocked message, the next normal question on the protected side sometimes replied in character as if the jailbreak had succeeded
- Root cause: both sides shared a single message history. The blocked user message was still appended to shared history and sent to the LLM on the next turn. The detector only screens the most recent message, so subsequent benign prompts pass — but the LLM has the full DAN context in its window
- Discussed two fixes: (a) strip blocked messages from shared history, (b) maintain fully independent per-side histories. Agreed on (b): each side builds its own LLM context; blocked turns are never added to that side's history. This is a correctness requirement, not just a UX nicety
- Reviewed Docker requirements: multi-stage build (Node compiles frontend, Python runs API), NVIDIA Container Toolkit for GPU passthrough, CPU-only PyTorch wheel must be swapped for CUDA 12.1

**Action items:**
- [ ] Implement per-side independent history in `App.jsx`; update `/ab/chat` API to accept `messages_a` / `messages_b`
- [ ] Write `Dockerfile` (multi-stage) and `docker-compose.yml` with GPU passthrough
- [ ] Add SQLite session persistence with collapsible sidebar UI

---

### Meeting 5 — Final Review and Wrap-Up
**Date:** Week 4, Friday
**Attendees:** Full team

**Agenda:** Final bug review, documentation, project retrospective.

**Discussion:**
- Walked through the Docker build end-to-end. Caught a blank-page bug: Vite builds with `base: '/'` by default, but FastAPI mounts the frontend at `/app/`, so `index.html` referenced `/assets/index.js` instead of `/app/assets/index.js`. Fixed by setting `base: '/app/'` in `vite.config.js`
- Confirmed per-side history isolation is working: sent a DAN jailbreak (blocked on protected side), then a normal question — protected side has no knowledge of the jailbreak, responds normally. Unprotected side retains full context and responds in DAN mode, clearly demonstrating the detector's value
- Reviewed the Results tab: confusion matrices, ROC curves, and PR curves for all model/split combinations are served live and rendered interactively in the browser
- Retrospective takeaways:
  - Transformer-based detectors are meaningfully better than bag-of-words for semantic injection variants; the gap narrows on simple keyword-based attacks where TF-IDF is sufficient
  - Conversation history isolation is a correctness requirement — without it the detector provides a false sense of security on multi-turn conversations
  - The OpenAI-compatible proxy design was the right call: the demo integrated with off-the-shelf tools instantly
- Remaining work: finalise `COMMANDS.md`, `DOCKER.md`, `FRONTEND.md`; tag `v1.0.0` release

**Action items:**
- [ ] Finalise and push all documentation under `docs/`
- [ ] Tag `v1.0.0` release on GitHub once docs are complete
- [ ] Write up findings in the project report
