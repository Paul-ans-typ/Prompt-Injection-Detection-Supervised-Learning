import { useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer,
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

function buildChartData(prEntries) {
  if (!prEntries?.length) return []
  const N = 200
  return Array.from({ length: N + 1 }, (_, i) => {
    const x = i / N  // recall axis
    const pt = { x: +x.toFixed(4) }
    for (const entry of prEntries) {
      const recall = entry.recall
      const prec   = entry.precision
      // find closest recall index (recall is typically descending in sklearn)
      let best = 0, bestDist = Infinity
      for (let j = 0; j < recall.length; j++) {
        const d = Math.abs(recall[j] - x)
        if (d < bestDist) { bestDist = d; best = j }
      }
      pt[entry.split] = +prec[best].toFixed(4)
    }
    return pt
  })
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip-title">Recall: {label}</div>
      {payload.map(p => (
        <div key={p.dataKey} className="chart-tooltip-row" style={{ color: p.color }}>
          <span>{SPLIT_LABELS[p.dataKey] ?? p.dataKey}</span>
          <span>Precision: {p.value}</span>
        </div>
      ))}
    </div>
  )
}

export default function PrCurveChart({ prData, theme }) {
  const [hovered, setHovered] = useState(null)
  if (!prData?.length) return <div className="chart-empty">No PR data available.</div>

  const chartData = buildChartData(prData)

  return (
    <div className="chart-wrap">
      <div className="chart-title">
        Precision–Recall Curve
        <span className="chart-subtitle">Higher area = better detector</span>
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
            label={{ value: 'Recall', position: 'insideBottom', offset: -12, fill: theme === 'light' ? '#64748b' : '#8b92a9', fontSize: 12 }}
          />
          <YAxis
            domain={[0, 1]}
            tickCount={6}
            tick={{ fill: theme === 'light' ? '#64748b' : '#8b92a9', fontSize: 11 }}
            label={{ value: 'Precision', angle: -90, position: 'insideLeft', offset: 12, fill: theme === 'light' ? '#64748b' : '#8b92a9', fontSize: 12 }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            formatter={(val) => SPLIT_LABELS[val] ?? val}
            wrapperStyle={{ fontSize: 12, paddingTop: 8 }}
          />
          {prData.map(entry => (
            <Line
              key={entry.split}
              type="monotone"
              dataKey={entry.split}
              stroke={SPLIT_COLORS[entry.split] ?? '#888'}
              strokeWidth={hovered === null || hovered === entry.split ? 2.5 : 1}
              opacity={hovered === null || hovered === entry.split ? 1 : 0.3}
              dot={false}
              activeDot={{ r: 4 }}
              name={`${SPLIT_LABELS[entry.split] ?? entry.split} (AP ${entry.ap?.toFixed(3) ?? '?'})`}
              onMouseEnter={() => setHovered(entry.split)}
              onMouseLeave={() => setHovered(null)}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      <div className="chart-auc-badges">
        {prData.map(entry => (
          <span key={entry.split} className="chart-badge" style={{ borderColor: SPLIT_COLORS[entry.split] ?? '#888' }}>
            {SPLIT_LABELS[entry.split] ?? entry.split}: AP {entry.ap?.toFixed(3) ?? '?'}
          </span>
        ))}
      </div>
    </div>
  )
}
