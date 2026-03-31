const BASE = ''  // same origin — Vite proxies in dev, FastAPI serves in prod

export async function fetchDetectors() {
  const r = await fetch(`${BASE}/available-detectors`)
  if (!r.ok) throw new Error(`Detectors fetch failed: ${r.status}`)
  return r.json()
}

export async function fetchLlms() {
  const r = await fetch(`${BASE}/available-llms`)
  if (!r.ok) throw new Error(`LLMs fetch failed: ${r.status}`)
  return r.json()
}

export async function fetchHealth() {
  const r = await fetch(`${BASE}/health`)
  if (!r.ok) throw new Error(`Health check failed: ${r.status}`)
  return r.json()
}

/**
 * Send a message to both sides concurrently.
 * @param {Array<{role:string, content:string}>} messages
 * @param {{ detector: string, llm: string, label?: string }} sideA
 * @param {{ detector: string, llm: string, label?: string }} sideB
 * @param {number|null} threshold
 * @param {string|null} sessionId  — pass null on first turn; backend auto-creates
 * @returns {Promise<{side_a, side_b, session_id}>}
 */
export async function sendAbChat(messagesA, messagesB, sideA, sideB, threshold = null, sessionId = null, signal = null) {
  const body = {
    messages:   messagesA,   // fallback / used by older clients
    messages_a: messagesA,
    messages_b: messagesB,
    side_a: sideA,
    side_b: sideB,
    ...(threshold  !== null ? { threshold }            : {}),
    ...(sessionId  !== null ? { session_id: sessionId } : {}),
  }
  const r = await fetch(`${BASE}/ab/chat`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(body),
    ...(signal ? { signal } : {}),
  })
  if (!r.ok) {
    const err = await r.text()
    throw new Error(`AB chat failed (${r.status}): ${err}`)
  }
  return r.json()
}

export async function fetchResults() {
  const r = await fetch(`${BASE}/results`)
  if (!r.ok) throw new Error(`Results fetch failed: ${r.status}`)
  return r.json()
}

export async function detectOnly(text, detector = null, threshold = null) {
  const body = {
    text,
    ...(detector  !== null ? { detector }  : {}),
    ...(threshold !== null ? { threshold } : {}),
  }
  const r = await fetch(`${BASE}/detect`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(body),
  })
  if (!r.ok) throw new Error(`Detect failed: ${r.status}`)
  return r.json()
}

// ── Session API ──────────────────────────────────────────────────────────────

export async function fetchSessions() {
  const r = await fetch(`${BASE}/sessions`)
  if (!r.ok) throw new Error(`Sessions fetch failed: ${r.status}`)
  return r.json()  // array of session summaries
}

export async function fetchSession(sessionId) {
  const r = await fetch(`${BASE}/sessions/${sessionId}`)
  if (!r.ok) throw new Error(`Session fetch failed: ${r.status}`)
  return r.json()  // { id, title, exchanges: [...], ... }
}

export async function renameSession(sessionId, title) {
  const r = await fetch(`${BASE}/sessions/${sessionId}`, {
    method:  'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ title }),
  })
  if (!r.ok) throw new Error(`Rename failed: ${r.status}`)
  return r.json()
}

export async function deleteSession(sessionId) {
  const r = await fetch(`${BASE}/sessions/${sessionId}`, { method: 'DELETE' })
  if (!r.ok) throw new Error(`Delete failed: ${r.status}`)
  return r.json()
}

// ── Ollama model management ──────────────────────────────────────────────────

/**
 * Pull an Ollama model and stream progress events back via SSE.
 * @param {string} name  Model tag, e.g. 'llama3.1:8b'
 * @param {(event: object) => void} onEvent  Called for each SSE progress event
 * @param {AbortSignal} signal  Pass an AbortController.signal to cancel
 */
export async function pullModel(name, onEvent, signal) {
  const r = await fetch(`${BASE}/ollama/pull`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ name }),
    signal,
  })
  if (!r.ok) throw new Error(`Pull request failed: ${r.status}`)

  const reader  = r.body.getReader()
  const decoder = new TextDecoder()
  let   buffer  = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      // SSE frames are separated by \n\n
      const frames = buffer.split('\n\n')
      buffer = frames.pop() ?? ''
      for (const frame of frames) {
        for (const line of frame.split('\n')) {
          if (line.startsWith('data: ')) {
            try { onEvent(JSON.parse(line.slice(6))) } catch { /* ignore malformed */ }
          }
        }
      }
    }
  } finally {
    reader.cancel()
  }
}

/**
 * Delete a model from the local Ollama instance.
 * @param {string} name  Model tag, e.g. 'llama3.2:3b'
 */
export async function deleteOllamaModel(name) {
  const r = await fetch(`${BASE}/ollama/models/${encodeURIComponent(name)}`, {
    method: 'DELETE',
  })
  if (!r.ok) throw new Error(`Delete failed (${r.status})`)
  return r.json()
}
