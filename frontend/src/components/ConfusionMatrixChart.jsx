const SPLIT_LABELS = {
  val:          'Val',
  test:         'Test',
  test_deepset: 'Test (Deepset)',
}

function SingleMatrix({ entry, theme }) {
  const { matrix, labels = ['Benign', 'Malicious'], split } = entry
  if (!matrix?.length) return null

  // Flatten for percent display
  const total = matrix.flat().reduce((a, b) => a + b, 0)
  const pct = (v) => total > 0 ? ((v / total) * 100).toFixed(1) : '0.0'

  const TP = matrix[1]?.[1] ?? 0
  const TN = matrix[0]?.[0] ?? 0
  const FP = matrix[0]?.[1] ?? 0
  const FN = matrix[1]?.[0] ?? 0

  const cellColor = (row, col) => {
    const isDiag = row === col
    if (theme === 'light') {
      return isDiag
        ? 'rgba(20, 212, 154, 0.18)'
        : 'rgba(239, 68, 68, 0.12)'
    }
    return isDiag
      ? 'rgba(20, 212, 154, 0.15)'
      : 'rgba(239, 68, 68, 0.10)'
  }

  return (
    <div className="cm-card">
      <div className="cm-split-label">{SPLIT_LABELS[split] ?? split}</div>
      <div className="cm-grid" style={{ gridTemplateColumns: `auto repeat(${labels.length}, 1fr)` }}>
        {/* top-left empty corner */}
        <div className="cm-corner">
          <span className="cm-axis-label cm-axis-pred">Predicted →</span>
          <span className="cm-axis-label cm-axis-actual">↓ Actual</span>
        </div>
        {/* column headers */}
        {labels.map(l => (
          <div key={l} className="cm-header">{l}</div>
        ))}
        {/* rows */}
        {matrix.map((row, ri) => (
          <>
            <div key={`row-${ri}`} className="cm-row-header">{labels[ri] ?? ri}</div>
            {row.map((val, ci) => (
              <div
                key={`${ri}-${ci}`}
                className={`cm-cell ${ri === ci ? 'cm-correct' : 'cm-wrong'}`}
                style={{ background: cellColor(ri, ci) }}
              >
                <span className="cm-count">{val.toLocaleString()}</span>
                <span className="cm-pct">{pct(val)}%</span>
              </div>
            ))}
          </>
        ))}
      </div>
      <div className="cm-stats">
        <span className="cm-stat-item"><span className="cm-stat-key">TP</span>{TP}</span>
        <span className="cm-stat-item"><span className="cm-stat-key">TN</span>{TN}</span>
        <span className="cm-stat-item cm-stat-bad"><span className="cm-stat-key">FP</span>{FP}</span>
        <span className="cm-stat-item cm-stat-bad"><span className="cm-stat-key">FN</span>{FN}</span>
      </div>
    </div>
  )
}

export default function ConfusionMatrixChart({ confusions, theme }) {
  if (!confusions?.length) return <div className="chart-empty">No confusion matrix data available.</div>

  return (
    <div className="chart-wrap">
      <div className="chart-title">
        Confusion Matrices
        <span className="chart-subtitle">Rows = actual class · Columns = predicted class</span>
      </div>
      <div className="cm-row">
        {confusions.map(entry => (
          <SingleMatrix key={entry.split} entry={entry} theme={theme} />
        ))}
      </div>
    </div>
  )
}
