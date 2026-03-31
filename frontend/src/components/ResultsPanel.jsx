import { useState, useEffect } from 'react'
import { fetchResults } from '../api'
import { ActivityIcon, TrendingUpIcon, GridIcon } from './Icons'
import RocCurveChart      from './RocCurveChart'
import PrCurveChart       from './PrCurveChart'
import ConfusionMatrixChart from './ConfusionMatrixChart'

const METRIC_COLS = [
  { key: 'accuracy',      label: 'Acc'  },
  { key: 'precision',     label: 'Prec' },
  { key: 'recall',        label: 'Rec'  },
  { key: 'f1',            label: 'F1'   },
  { key: 'roc_auc',       label: 'AUC'  },
  { key: 'avg_precision', label: 'AP'   },
]

const SPLIT_LABELS = {
  val:           'Validation',
  test:          'Test',
  test_deepset:  'Test (Deepset)',
  test_wildcard: 'Test (Wildcard)',
}

function pct(v) {
  if (v == null) return '—'
  return (v * 100).toFixed(1) + '%'
}

function f1Color(f1) {
  if (f1 == null) return ''
  if (f1 >= 0.95) return 'metric-great'
  if (f1 >= 0.85) return 'metric-good'
  if (f1 >= 0.70) return 'metric-ok'
  return 'metric-poor'
}

// Build sidebar plot list from a model entry.
// Prefer curves data (interactive). Fall back to static image URLs.
function buildPlotList(model) {
  const plots = []
  const curves = model.curves

  // ROC
  if (curves?.roc?.length) {
    plots.push({ id: 'roc', label: 'ROC Curve', type: 'roc', mode: 'interactive' })
  } else if (model.images?.roc) {
    plots.push({ id: 'roc', label: 'ROC Curve', type: 'roc', mode: 'image', url: model.images.roc })
  }

  // PR
  if (curves?.pr?.length) {
    plots.push({ id: 'pr', label: 'PR Curve', type: 'pr', mode: 'interactive' })
  } else if (model.images?.pr) {
    plots.push({ id: 'pr', label: 'PR Curve', type: 'pr', mode: 'image', url: model.images.pr })
  }

  // Confusion matrices
  if (curves?.confusions?.length) {
    plots.push({ id: 'confusion', label: 'Confusion Matrices', type: 'confusion', mode: 'interactive' })
  } else {
    ;(model.images?.confusions || []).forEach(c => {
      plots.push({
        id:    `confusion_${c.split}`,
        label: `Confusion — ${SPLIT_LABELS[c.split] || c.split}`,
        type:  'confusion',
        mode:  'image',
        url:   c.url,
      })
    })
  }

  return plots
}

function PlotTypeIcon({ type }) {
  if (type === 'roc')      return <ActivityIcon   size={14} />
  if (type === 'pr')       return <TrendingUpIcon size={14} />
  return <GridIcon size={14} />
}

export default function ResultsPanel({ theme = 'dark' }) {
  const [models,      setModels]      = useState(null)
  const [loading,     setLoading]     = useState(true)
  const [error,       setError]       = useState(null)
  const [activeModel, setActiveModel] = useState(null)
  const [activePlot,  setActivePlot]  = useState(null)

  useEffect(() => {
    fetchResults()
      .then(data => {
        setModels(data.models || [])
        if (data.models?.length > 0) setActiveModel(data.models[0].id)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  // Reset active plot when active model changes
  useEffect(() => {
    if (!models || !activeModel) return
    const model = models.find(m => m.id === activeModel)
    if (!model) return
    const plots = buildPlotList(model)
    setActivePlot(plots.length > 0 ? plots[0].id : null)
  }, [activeModel, models])

  if (loading) return <div className="results-empty">Loading results…</div>
  if (error)   return <div className="results-empty results-error">Could not load results: {error}</div>
  if (!models || models.length === 0) {
    return (
      <div className="results-empty">
        No training results found yet. Run the training scripts first.
      </div>
    )
  }

  const model    = models.find(m => m.id === activeModel) || models[0]
  const plotList = buildPlotList(model)
  const currentPlot = plotList.find(p => p.id === activePlot)

  function renderPlot(plot) {
    if (!plot) return null
    if (plot.mode === 'image') {
      return (
        <div className="results-plot-view">
          <img key={plot.id} src={plot.url} alt={plot.label} className="results-plot-img" />
        </div>
      )
    }
    // interactive
    if (plot.type === 'roc') {
      return <RocCurveChart rocData={model.curves?.roc} theme={theme} />
    }
    if (plot.type === 'pr') {
      return <PrCurveChart prData={model.curves?.pr} theme={theme} />
    }
    if (plot.type === 'confusion') {
      return <ConfusionMatrixChart confusions={model.curves?.confusions} theme={theme} />
    }
    return null
  }

  return (
    <div className="results-panel">
      {/* ── Model selector (top bar) ─────────────────────────── */}
      <div className="results-header">
        <div className="results-tabs">
          {models.map(m => (
            <button
              key={m.id}
              className={`results-tab ${m.id === activeModel ? 'results-tab-active' : ''}`}
              onClick={() => setActiveModel(m.id)}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      <div className="results-body">
        {/* ── Left sidebar — plot navigation ───────────────────── */}
        {plotList.length > 0 && (
          <nav className="results-sidebar">
            <div className="sidebar-section-title">Plots</div>
            {plotList.map(plot => (
              <button
                key={plot.id}
                className={`sidebar-item ${plot.id === activePlot ? 'sidebar-item-active' : ''}`}
                onClick={() => setActivePlot(plot.id)}
              >
                <PlotTypeIcon type={plot.type} />
                <span>{plot.label}</span>
                {plot.mode === 'interactive' && (
                  <span className="plot-interactive-badge">live</span>
                )}
              </button>
            ))}
          </nav>
        )}

        {/* ── Main content area ────────────────────────────────── */}
        <div className="results-main">
          {/* Metrics table */}
          <div className="results-section">
            <div className="results-section-title">Metrics per split</div>
            <div className="results-table-wrap">
              <table className="results-table">
                <thead>
                  <tr>
                    <th>Split</th>
                    {METRIC_COLS.map(c => <th key={c.key}>{c.label}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {model.metrics.map(row => (
                    <tr key={row.split}>
                      <td className="split-cell">{SPLIT_LABELS[row.split] || row.split}</td>
                      {METRIC_COLS.map(c => (
                        <td key={c.key} className={c.key === 'f1' ? f1Color(row[c.key]) : ''}>
                          {pct(row[c.key])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Selected plot */}
          {currentPlot ? (
            <div className="results-section results-section-plot">
              {renderPlot(currentPlot)}
            </div>
          ) : plotList.length === 0 && (
            <div className="results-no-plots">No plots available for this model.</div>
          )}
        </div>
      </div>
    </div>
  )
}
