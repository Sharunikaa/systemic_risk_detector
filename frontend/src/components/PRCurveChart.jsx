import React from 'react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer, Legend
} from 'recharts'

/**
 * PRCurveChart — Precision-Recall curve for the crisis class.
 *
 * Props:
 *   precision  : array of precision values
 *   recall     : array of recall values
 *   thresholds : array of thresholds
 *   prAuc      : PR-AUC value (number)
 *   baseline   : random classifier baseline (crisis prevalence rate)
 *   optThresh  : optimal threshold to mark
 */
export default function PRCurveChart({
  precision = [],
  recall = [],
  thresholds = [],
  prAuc = 0,
  baseline = 0.06,
  optThresh = 0.5,
}) {
  // Build chart data from PR curve arrays
  const data = recall.map((r, i) => ({
    recall: parseFloat(r.toFixed(4)),
    precision: parseFloat((precision[i] ?? 0).toFixed(4)),
    threshold: thresholds[i] ?? null,
  })).reverse()

  // Find the closest point to optThresh for the dot
  let optPoint = null
  let minDiff = Infinity
  data.forEach(d => {
    if (d.threshold != null) {
      const diff = Math.abs(d.threshold - optThresh)
      if (diff < minDiff) { minDiff = diff; optPoint = d }
    }
  })

  const CustomDot = ({ cx, cy, payload }) => {
    if (optPoint && Math.abs(payload.recall - optPoint.recall) < 0.005) {
      return (
        <g>
          <circle cx={cx} cy={cy} r={6} fill="var(--quantum)" stroke="var(--bg-surface)" strokeWidth={2} />
          <text x={cx + 10} y={cy - 8} fontSize={10} fill="var(--quantum)" fontFamily="JetBrains Mono">
            t={optThresh.toFixed(2)}
          </text>
        </g>
      )
    }
    return null
  }

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
        <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Crisis Class PR Curve</span>
        <span className="font-mono" style={{
          fontSize: '0.8rem', color: 'var(--crisis)',
          background: 'rgba(239,68,68,0.1)',
          padding: '2px 8px', borderRadius: 4
        }}>
          PR-AUC = {prAuc.toFixed(4)}
        </span>
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" strokeOpacity={0.35} />
          <XAxis dataKey="recall" type="number" domain={[0,1]}
            tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
            label={{ value: 'Recall', position: 'insideBottom', offset: -2,
                     fill: 'var(--text-muted)', fontSize: 11 }}
            axisLine={{ stroke: 'var(--border)' }} tickLine={false} />
          <YAxis domain={[0,1]}
            tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
            axisLine={{ stroke: 'var(--border)' }} tickLine={false} width={35} />
          <Tooltip
            formatter={(v) => [v.toFixed(4), '']}
            contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)',
                            borderRadius: 8, color: 'var(--text-primary)', fontSize: '0.8rem' }}
          />
          {/* Baseline — random classifier */}
          <ReferenceLine y={baseline} stroke="rgba(148,163,184,0.4)"
            strokeDasharray="4 4"
            label={{ value: `Baseline (${(baseline*100).toFixed(1)}%)`, position: 'right',
                     fill: 'var(--text-muted)', fontSize: 10 }} />
          <Area
            type="monotone" dataKey="precision"
            stroke="var(--crisis)" strokeWidth={2}
            fill="var(--crisis)" fillOpacity={0.15}
            dot={<CustomDot />}
            name="Precision"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
