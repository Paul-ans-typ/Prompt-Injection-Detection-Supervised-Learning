import { useState, useEffect, useRef, useCallback } from 'react'
import { fetchLlms, pullModel, deleteOllamaModel } from '../api'
import {
  DownloadIcon, RefreshCwIcon, HardDriveIcon, PackageIcon,
  TrashIcon, CheckIcon, XIcon, AlertTriangleIcon, ServerIcon,
} from './Icons'

// ── Formatters ───────────────────────────────────────────────────────────────

function fmtBytes(bytes) {
  if (!bytes || bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${(bytes / Math.pow(k, i)).toFixed(i >= 2 ? 1 : 0)} ${sizes[i]}`
}

function fmtSpeed(bps) {
  if (!bps || bps === 0) return null
  return `${fmtBytes(bps)}/s`
}

function fmtEta(seconds) {
  if (seconds === null || seconds === undefined || seconds < 0) return null
  if (seconds < 60) return `${seconds}s`
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  if (m < 60) return `${m}m ${s > 0 ? ` ${s}s` : ''}`
  return `${Math.floor(m / 60)}h ${m % 60}m`
}

// ── Safety badge ─────────────────────────────────────────────────────────────

function SafetyBadge({ level }) {
  const cls = {
    low:      'safety-low',
    moderate: 'safety-moderate',
    good:     'safety-good',
  }[level] ?? 'safety-moderate'
  return <span className={`safety-badge ${cls}`}>{level} safety</span>
}

// ── Pull progress row ─────────────────────────────────────────────────────────

function PullProgress({ pull }) {
  const shortDigest = pull.digest ? pull.digest.replace('sha256:', '').slice(0, 12) : null

  return (
    <div className="pull-progress-area">
      <div className="pull-status-line">
        {shortDigest
          ? <span className="pull-layer-tag">{pull.status.replace(pull.digest ?? '', shortDigest + '…')}</span>
          : <span className="pull-layer-tag">{pull.status || 'connecting…'}</span>
        }
      </div>

      {pull.total > 0 && (
        <>
          <div className="progress-track">
            <div
              className="progress-fill"
              style={{ width: `${Math.min(pull.percent, 100)}%` }}
            />
          </div>

          <div className="pull-stats-row">
            <span className="pull-pct">{pull.percent.toFixed(1)}%</span>
            <span className="pull-bytes">{fmtBytes(pull.completed)} / {fmtBytes(pull.total)}</span>
            {fmtSpeed(pull.speed_bps) && (
              <span className="pull-speed">{fmtSpeed(pull.speed_bps)}</span>
            )}
            {fmtEta(pull.eta_s) && (
              <span className="pull-eta">ETA {fmtEta(pull.eta_s)}</span>
            )}
          </div>
        </>
      )}
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export default function ModelsPanel({ onPullSuccess }) {
  const [data,      setData]     = useState(null)   // { installed, recommended, ollama_url }
  const [loading,   setLoading]  = useState(true)
  const [fetchErr,  setFetchErr] = useState(null)
  const [pulls,     setPulls]    = useState({})     // { modelName: pullState }
  const [deleting,  setDeleting] = useState(new Set())
  const abortRefs = useRef({})

  // ── Load / refresh ──────────────────────────────────────────────────────────
  const load = useCallback(async () => {
    setLoading(true)
    setFetchErr(null)
    try {
      setData(await fetchLlms())
    } catch (e) {
      setFetchErr(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  // ── Pull a model ────────────────────────────────────────────────────────────
  const handlePull = async (name) => {
    abortRefs.current[name]?.abort()
    const ctrl = new AbortController()
    abortRefs.current[name] = ctrl

    setPulls(prev => ({
      ...prev,
      [name]: {
        status: 'connecting…', digest: '', percent: 0,
        speed_bps: 0, eta_s: null, total: 0, completed: 0,
        done: false, error: null,
      },
    }))

    try {
      await pullModel(name, (ev) => {
        if (ev.status === 'success') {
          setPulls(prev => ({
            ...prev,
            [name]: { ...prev[name], status: 'success', done: true, percent: 100 },
          }))
          load()
          onPullSuccess?.()
          setTimeout(() => {
            setPulls(prev => { const n = { ...prev }; delete n[name]; return n })
          }, 2500)
        } else if (ev.status === 'error') {
          setPulls(prev => ({
            ...prev,
            [name]: { ...prev[name], error: ev.error ?? 'Unknown error', done: true },
          }))
        } else {
          setPulls(prev => ({
            ...prev,
            [name]: {
              ...prev[name],
              status:    ev.status    ?? prev[name].status,
              digest:    ev.digest    ?? prev[name].digest,
              percent:   ev.percent   ?? prev[name].percent,
              speed_bps: ev.speed_bps ?? prev[name].speed_bps,
              eta_s:     ev.eta_s,
              total:     ev.total     ?? prev[name].total,
              completed: ev.completed ?? prev[name].completed,
            },
          }))
        }
      }, ctrl.signal)
    } catch (e) {
      if (e.name === 'AbortError') {
        // User cancelled — remove entry
        setPulls(prev => { const n = { ...prev }; delete n[name]; return n })
      } else {
        setPulls(prev => ({
          ...prev,
          [name]: { ...prev[name], error: e.message, done: true },
        }))
      }
    }
  }

  const handleCancel = (name) => {
    abortRefs.current[name]?.abort()
    setPulls(prev => { const n = { ...prev }; delete n[name]; return n })
  }

  // ── Delete a model ──────────────────────────────────────────────────────────
  const handleDelete = async (name) => {
    if (!window.confirm(`Delete "${name}" from Ollama? This cannot be undone.`)) return
    setDeleting(prev => new Set([...prev, name]))
    try {
      await deleteOllamaModel(name)
      await load()
    } catch (e) {
      setFetchErr(`Could not delete "${name}": ${e.message}`)
    } finally {
      setDeleting(prev => { const n = new Set(prev); n.delete(name); return n })
    }
  }

  // ── Derived state ───────────────────────────────────────────────────────────
  const isPulling = Object.values(pulls).some(p => !p.done)
  const installed  = data?.installed    ?? []
  const recommended = data?.recommended ?? []

  const installedNames  = new Set(installed.map(m => m.name))
  const availableToPull = recommended.filter(r => !installedNames.has(r.name))
  const installedRec    = recommended.filter(r =>  installedNames.has(r.name))

  const totalInstalledGb = installed
    .reduce((sum, m) => sum + (m.size_gb ?? 0), 0)
    .toFixed(1)

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div className="models-panel">

      {/* ── Toolbar ── */}
      <div className="models-toolbar">
        <div className="models-toolbar-left">
          <PackageIcon size={16} />
          <span className="models-toolbar-title">Ollama Models</span>
          {data && (
            <span className="models-toolbar-meta">
              {data.ollama_url}
            </span>
          )}
        </div>
        <button
          className={`btn-ghost${loading ? ' btn-spinning' : ''}`}
          onClick={load}
          disabled={loading}
          title="Refresh model list"
        >
          <RefreshCwIcon size={13} />
          Refresh
        </button>
      </div>

      {/* ── Fetch error ── */}
      {fetchErr && (
        <div className="models-error-banner">
          <AlertTriangleIcon size={14} />
          {fetchErr}
          <button onClick={() => setFetchErr(null)} className="models-error-dismiss">
            <XIcon size={13} />
          </button>
        </div>
      )}

      {/* ── Loading skeleton ── */}
      {loading && !data && (
        <div className="models-loading">
          <span className="spinner" />
          Querying Ollama…
        </div>
      )}

      {data && (
        <div className="models-content">

          {/* ════════════════════════ INSTALLED ════════════════════════ */}
          <section className="models-section">
            <div className="models-section-header">
              <HardDriveIcon size={14} />
              <span>Installed</span>
              <span className="models-section-meta">
                {installed.length} model{installed.length !== 1 ? 's' : ''}
                {installed.length > 0 && ` · ${totalInstalledGb} GB on disk`}
              </span>
            </div>

            {installed.length === 0 ? (
              <div className="models-empty">
                No models installed yet. Pull one from the list below.
              </div>
            ) : (
              <div className="models-list">
                {installed.map(m => {
                  const recInfo = recommended.find(r => r.name === m.name)
                  return (
                    <div className="model-row" key={m.name}>
                      <div className="model-row-left">
                        <span className="model-row-name">{m.name}</span>
                        {recInfo && (
                          <>
                            <span className="model-row-params">{recInfo.params}</span>
                            <SafetyBadge level={recInfo.safety} />
                          </>
                        )}
                        {m.size_gb > 0 && (
                          <span className="model-row-size">{m.size_gb} GB</span>
                        )}
                      </div>
                      <button
                        className="btn-ghost btn-danger"
                        onClick={() => handleDelete(m.name)}
                        disabled={deleting.has(m.name)}
                        title={`Delete ${m.name}`}
                      >
                        {deleting.has(m.name)
                          ? <span className="spinner spinner-sm" />
                          : <TrashIcon size={13} />
                        }
                        Delete
                      </button>
                    </div>
                  )
                })}
              </div>
            )}
          </section>

          {/* ════════════════════════ AVAILABLE ════════════════════════ */}
          <section className="models-section">
            <div className="models-section-header">
              <DownloadIcon size={14} />
              <span>Available to Download</span>
              <span className="models-section-meta">
                {availableToPull.length} model{availableToPull.length !== 1 ? 's' : ''}
              </span>
            </div>

            {availableToPull.length === 0 ? (
              <div className="models-empty">
                <CheckIcon size={14} />
                All recommended models are installed.
              </div>
            ) : (
              <div className="models-list models-list-cards">
                {availableToPull.map(m => {
                  const pull = pulls[m.name]
                  const isActive = pull && !pull.done

                  return (
                    <div className={`model-card${isActive ? ' model-card-active' : ''}`} key={m.name}>
                      <div className="model-card-top">
                        <div className="model-card-info">
                          <span className="model-card-name">{m.name}</span>
                          <span className="model-card-params">{m.params}</span>
                          <span className="model-card-vram">{m.vram_gb} GB VRAM</span>
                          <SafetyBadge level={m.safety} />
                        </div>

                        <div className="model-card-actions">
                          {!pull ? (
                            <button
                              className="btn-pull"
                              onClick={() => handlePull(m.name)}
                              disabled={isPulling}
                              title={isPulling ? 'Another download is in progress' : `Pull ${m.name}`}
                            >
                              <DownloadIcon size={13} />
                              Pull
                            </button>
                          ) : pull.done ? (
                            pull.error ? (
                              <span className="pull-result-error">
                                <AlertTriangleIcon size={13} /> Error
                              </span>
                            ) : (
                              <span className="pull-result-ok">
                                <CheckIcon size={13} /> Done
                              </span>
                            )
                          ) : (
                            <button
                              className="btn-ghost btn-danger btn-sm"
                              onClick={() => handleCancel(m.name)}
                              title="Cancel download"
                            >
                              <XIcon size={13} />
                              Cancel
                            </button>
                          )}
                        </div>
                      </div>

                      <div className="model-card-note">{m.note}</div>

                      {/* In-progress pull ── */}
                      {isActive && <PullProgress pull={pull} />}

                      {/* Error state ── */}
                      {pull?.done && pull?.error && (
                        <div className="pull-error-msg">
                          <AlertTriangleIcon size={12} />
                          {pull.error}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            )}
          </section>

          {/* ════════════════════════ ALREADY INSTALLED FROM RECOMMENDED ════════════════════════ */}
          {installedRec.length > 0 && (
            <section className="models-section models-section-dimmed">
              <div className="models-section-header">
                <CheckIcon size={14} />
                <span>Also Recommended — Already Installed</span>
              </div>
              <div className="models-list models-list-compact">
                {installedRec.map(m => (
                  <div className="model-row model-row-installed" key={m.name}>
                    <div className="model-row-left">
                      <CheckIcon size={12} className="icon-check" />
                      <span className="model-row-name">{m.name}</span>
                      <span className="model-row-params">{m.params}</span>
                      <SafetyBadge level={m.safety} />
                    </div>
                    <span className="model-row-note">{m.note}</span>
                  </div>
                ))}
              </div>
            </section>
          )}

        </div>
      )}
    </div>
  )
}
