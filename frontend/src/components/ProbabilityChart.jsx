import React from 'react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend
} from 'recharts'

/**
 * ProbabilityChart — Stacked 100% area chart showing regime probabilities.
 *
 * Props:
 *   predictions : array of { date, prob_crisis, prob_normal, prob_highvol }
 *   height      : chart height in px (default 200)
 */
export default function ProbabilityChart({ predictions = [], height = 200 }) {
  const data = predictions.map(p => ({
    date:    p.date,
    crisis:  parseFloat((p.prob_crisis  * 100).toFixed(2)),
    normal:  parseFloat((p.prob_normal  * 100).toFixed(2)),
    highvol: parseFloat((p.prob_highvol * 100).toFixed(2)),
  }))

  // Sample if too dense
  const sampled = data.length > 200 ? data.filter((_, i) => i % 2 === 0) : data

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={sampled} stackOffset="expand" margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" strokeOpacity={0.3} />
        <XAxis
          dataKey="date"
          tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
          tickFormatter={v => v?.slice(0, 7)}
          interval={Math.floor(sampled.length / 8)}
          axisLine={{ stroke: 'var(--border)' }}
          tickLine={false}
        />
        <YAxis
          tickFormatter={v => `${(v * 100).toFixed(0)}%`}
          tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
          axisLine={{ stroke: 'var(--border)' }}
          tickLine={false}
          width={40}
        />
        <Tooltip
          formatter={(v, name) => [`${(v * 100).toFixed(1)}%`, name]}
          contentStyle={{
            background: 'var(--bg-elevated)',
            border: '1px solid var(--border)',
            borderRadius: 8,
            color: 'var(--text-primary)',
            fontSize: '0.8rem'
          }}
        />
        <Area type="monotone" dataKey="crisis"  stackId="p"
          fill="var(--crisis)"  stroke="none" fillOpacity={0.8} name="Crisis" />
        <Area type="monotone" dataKey="highvol" stackId="p"
          fill="var(--highvol)" stroke="none" fillOpacity={0.6} name="High-Vol" />
        <Area type="monotone" dataKey="normal"  stackId="p"
          fill="var(--normal)"  stroke="none" fillOpacity={0.5} name="Normal" />
        <Legend
          wrapperStyle={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}
          iconType="square"
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}
