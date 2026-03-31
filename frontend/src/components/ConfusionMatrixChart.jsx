const SPLIT_LABELS = {
  val:          'Validation',
  test:         'Test',
  test_deepset: 'Test — Deepset',
}

function SingleMatrix({ entry, theme }) {
  const { matrix, labels = ['Benign', 'Malicious'], split } = entry
  if (!matrix?.length) return null

  const total = matrix.flat().reduce((a, b) => a + b, 0)
  const pct = (v) => total > 0 ? ((v / total) * 100).toFixed(1) : '0.0'

  const TP = matrix[1]?.[1] ?? 0
  const TN = matrix[0]?.[0] ?? 0
  const FP = matrix[0]?.[1] ?? 0
  const FN = matrix[1]?.[0] ?? 0

  const cellBg = (row, col) => {
    const isDiag = row === col
    if (theme === 'light') {
      return isDiag ? 'rgba(20,212,154,0.18)' : 'rgba(239,68,68,0.10)'
    }
    return isDiag ? 'rgba(20,212,154,0.14)' : 'rgba(239,68,68,0.08)'
  }

  return (
    <div className="cm-card">
      <div className="cm-split-label">{SPLIT_LABELS[split] ?? split}</div>

      {/* Axis hint row */}
      <div className="cm-axis-hint">
        <span className="cm-axis-x">Predicted →</span>
      </div>

      <div className="cm-table-wrap">
        {/* Y-axis label */}
        <div className="cm-axis-y">
          <span>↓ Actual</span>
        </div>

        {/* Grid */}
        <div
          className="cm-grid"
          style={{ gridTemplateColumns: `80px repeat(${labels.length}, 1fr)` }}
        >
          {/* top-left empty corner */}
          <div className="cm-corner" />

          {/* column headers */}
          {labels.map(l => (
            <div key={l} className="cm-col-header">{l}</div>
          ))}

          {/* data rows */}
          {matrix.map((row, ri) => (
            <>
              <div key={`rh-${ri}`} className="cm-row-header">{labels[ri] ?? ri}</div>
              {row.map((val, ci) => (
                <div
                  key={`${ri}-${ci}`}
                  className={`cm-cell ${ri === ci ? 'cm-correct' : 'cm-wrong'}`}
                  style={{ background: cellBg(ri, ci) }}
                >
                  <span className="cm-count">{val.toLocaleString()}</span>
                  <span className="cm-pct">{pct(val)}%</span>
                </div>
              ))}
            </>
          ))}
        </div>
      </div>

      {/* Summary stats */}
      <div className="cm-stats">
        <div className="cm-stat-group cm-stat-group-ok">
          <span className="cm-stat-item">
            <span className="cm-stat-key">TP</span>
            <span className="cm-stat-val">{TP.toLocaleString()}</span>
          </span>
          <span className="cm-stat-item">
            <span className="cm-stat-key">TN</span>
            <span className="cm-stat-val">{TN.toLocaleString()}</span>
          </span>
        </div>
        <div className="cm-stat-group cm-stat-group-bad">
          <span className="cm-stat-item">
            <span className="cm-stat-key">FP</span>
            <span className="cm-stat-val">{FP.toLocaleString()}</span>
          </span>
          <span className="cm-stat-item">
            <span className="cm-stat-key">FN</span>
            <span className="cm-stat-val">{FN.toLocaleString()}</span>
          </span>
        </div>
      </div>
    </div>
  )
}

export default function ConfusionMatrixChart({ confusions, theme }) {
  if (!confusions?.length) return <div className="chart-empty">No confusion matrix data available.</div>

  return (
    <div className="chart-wrap">
      <div className="chart-header">
        <span className="chart-label">Rows = actual class · Columns = predicted class</span>
      </div>
      <div className="cm-grid-outer">
        {confusions.map(entry => (
          <SingleMatrix key={entry.split} entry={entry} theme={theme} />
        ))}
      </div>
    </div>
  )
}
