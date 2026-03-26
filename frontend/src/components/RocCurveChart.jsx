import { useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, ReferenceLine,
} from 'recharts'

const SPLIT_COLORS = {
  val:          '#5294ff',
  test:         '#14d49a',
  test_deepset: '#f59e0b',
}

const SPLIT_LABELS = {
  val:          'Val',
  test:         'Test',
  test_deepset: 'Test (Deepset)',
}

// Recharts needs one unified array of {x, val_y, test_y, ...} for multi-line.
// We downsample each curve to ≤200 pts for performance.
function buildChartData(rocEntries) {
  if (!rocEntries?.length) return []
  const N = 200
  return Array.from({ length: N + 1 }, (_, i) => {
    const x = i / N
    const pt = { x: +x.toFixed(4) }
    for (const entry of rocEntries) {
      const fpr = entry.fpr
      const tpr = entry.tpr
      // binary search closest fpr index
      let lo = 0, hi = fpr.length - 1
      while (lo < hi) {
        const mid = (lo + hi) >> 1
        if (fpr[mid] < x) lo = mid + 1
        else hi = mid
      }
      pt[entry.split] = +tpr[lo].toFixed(4)
    }
    return pt
  })
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip-title">FPR: {label}</div>
      {payload.map(p => (
        <div key={p.dataKey} className="chart-tooltip-row" style={{ color: p.color }}>
          <span>{SPLIT_LABELS[p.dataKey] ?? p.dataKey}</span>
          <span>TPR: {p.value}</span>
        </div>
      ))}
    </div>
  )
}

export default function RocCurveChart({ rocData, theme }) {
  const [hovered, setHovered] = useState(null)
  if (!rocData?.length) return <div className="chart-empty">No ROC data available.</div>

  const chartData = buildChartData(rocData)

  return (
    <div className="chart-wrap">
      <div className="chart-title">
        ROC Curve
        <span className="chart-subtitle">Receiver Operating Characteristic</span>
      </div>
      <ResponsiveContainer width="100%" height={340}>
        <LineChart data={chartData} margin={{ top: 8, right: 24, bottom: 24, left: 8 }}>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke={theme === 'light' ? '#e2e8f0' : '#2a2d3e'}
          />
          <XAxis
            dataKey="x"
            type="number"
            domain={[0, 1]}
            tickCount={6}
            tick={{ fill: theme === 'light' ? '#64748b' : '#8b92a9', fontSize: 11 }}
            label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -12, fill: theme === 'light' ? '#64748b' : '#8b92a9', fontSize: 12 }}
          />
          <YAxis
            domain={[0, 1]}
            tickCount={6}
            tick={{ fill: theme === 'light' ? '#64748b' : '#8b92a9', fontSize: 11 }}
            label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft', offset: 12, fill: theme === 'light' ? '#64748b' : '#8b92a9', fontSize: 12 }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            formatter={(val) => SPLIT_LABELS[val] ?? val}
            wrapperStyle={{ fontSize: 12, paddingTop: 8 }}
          />
          <ReferenceLine
            x={0} y={0}
            stroke={theme === 'light' ? '#94a3b8' : '#4a5068'}
            strokeDasharray="4 4"
          />
          {/* diagonal random-classifier line */}
          {[{ x: 0, diag: 0 }, { x: 1, diag: 1 }].length > 0 && (
            <ReferenceLine
              segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
              stroke={theme === 'light' ? '#94a3b8' : '#4a5068'}
              strokeDasharray="4 4"
              label={{ value: 'Random', position: 'insideTopLeft', fill: theme === 'light' ? '#94a3b8' : '#4a5068', fontSize: 10 }}
            />
          )}
          {rocData.map(entry => (
            <Line
              key={entry.split}
              type="monotone"
              dataKey={entry.split}
              stroke={SPLIT_COLORS[entry.split] ?? '#888'}
              strokeWidth={hovered === null || hovered === entry.split ? 2.5 : 1}
              opacity={hovered === null || hovered === entry.split ? 1 : 0.3}
              dot={false}
              activeDot={{ r: 4 }}
              name={`${SPLIT_LABELS[entry.split] ?? entry.split} (AUC ${entry.auc?.toFixed(3) ?? '?'})`}
              onMouseEnter={() => setHovered(entry.split)}
              onMouseLeave={() => setHovered(null)}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      <div className="chart-auc-badges">
        {rocData.map(entry => (
          <span key={entry.split} className="chart-badge" style={{ borderColor: SPLIT_COLORS[entry.split] ?? '#888' }}>
            {SPLIT_LABELS[entry.split] ?? entry.split}: AUC {entry.auc?.toFixed(3) ?? '?'}
          </span>
        ))}
      </div>
    </div>
  )
}
