import React, { useMemo } from 'react'
import {
  ComposedChart, Area, Line, ReferenceLine, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts'

const CRISIS_EVENTS = [
  { date: '2020-03-16', label: 'COVID-19' },
  { date: '2022-05-09', label: 'LUNA' },
  { date: '2023-03-10', label: 'SVB' },
]

const REGIME_COLORS = {
  CRISIS: 'var(--crisis)',
  NORMAL: 'var(--normal)',
  'HIGH-VOL': 'var(--highvol)',
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="card" style={{ padding: '10px 14px', minWidth: 160, fontSize: '0.8rem' }}>
      <div style={{ color: 'var(--text-muted)', marginBottom: 6 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ display: 'flex', justifyContent: 'space-between', gap: 12, marginBottom: 2 }}>
          <span style={{ color: p.color }}>{p.name}</span>
          <span className="font-mono" style={{ color: 'var(--text-primary)' }}>
            {typeof p.value === 'number' ? p.value.toFixed(2) : p.value}
          </span>
        </div>
      ))}
    </div>
  )
}

/**
 * RegimeTimeline — Full-width composite chart showing:
 *   - Area bands for CRISIS / NORMAL / HIGH-VOL
 *   - BTC price normalized to base 100 (secondary axis, dashed)
 *   - Vertical reference lines for key crisis events
 *
 * Props:
 *   regimes : array of { date, regime_label, crisis_flag }
 *   prices  : array of { date, spx, gold, btc } (normalized to 100)
 */
export default function RegimeTimeline({ regimes = [], prices = [] }) {
  const data = useMemo(() => {
    const pricesByDate = Object.fromEntries(prices.map(p => [p.date, p]))

    return regimes.map(r => {
      const p = pricesByDate[r.date] || {}
      return {
        date:     r.date,
        CRISIS:   r.regime_label === 'CRISIS'   ? 1 : 0,
        NORMAL:   r.regime_label === 'NORMAL'   ? 1 : 0,
        HIGHVOL:  r.regime_label === 'HIGH-VOL' ? 1 : 0,
        btc:      p.btc ?? null,
        regime:   r.regime_label,
      }
    })
  }, [regimes, prices])

  // Sample every Nth point to reduce chart density
  const sampled = useMemo(() => data.filter((_, i) => i % 5 === 0), [data])

  return (
    <ResponsiveContainer width="100%" height={220}>
      <ComposedChart data={sampled} margin={{ top: 5, right: 40, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" strokeOpacity={0.35} />
        <XAxis
          dataKey="date"
          tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
          tickFormatter={v => v?.slice(0, 7)}
          interval={Math.floor(sampled.length / 12)}
          axisLine={{ stroke: 'var(--border)' }}
          tickLine={false}
        />
        <YAxis
          yAxisId="left"
          domain={[0, 1]}
          tick={false}
          axisLine={false}
          tickLine={false}
          width={0}
        />
        <YAxis
          yAxisId="right"
          orientation="right"
          tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
          axisLine={{ stroke: 'var(--border)' }}
          tickLine={false}
          width={40}
        />
        <Tooltip content={<CustomTooltip />} />

        <Area yAxisId="left" type="stepAfter" dataKey="CRISIS"
          fill="var(--crisis)" stroke="none" fillOpacity={0.35} stackId="r" name="Crisis" />
        <Area yAxisId="left" type="stepAfter" dataKey="NORMAL"
          fill="var(--normal)" stroke="none" fillOpacity={0.25} stackId="r" name="Normal" />
        <Area yAxisId="left" type="stepAfter" dataKey="HIGHVOL"
          fill="var(--highvol)" stroke="none" fillOpacity={0.25} stackId="r" name="High-Vol" />

        <Line yAxisId="right" type="monotone" dataKey="btc"
          stroke="var(--quantum)" strokeWidth={1.5} dot={false}
          strokeDasharray="4 2" name="BTC (base 100)" connectNulls />

        {CRISIS_EVENTS.map(ev => (
          <ReferenceLine
            key={ev.date}
            yAxisId="left"
            x={ev.date}
            stroke="rgba(255,255,255,0.25)"
            strokeDasharray="3 3"
            label={{ value: ev.label, position: 'insideTopLeft',
                     fill: 'var(--text-muted)', fontSize: 10 }}
          />
        ))}

        <Legend
          wrapperStyle={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}
          iconType="square"
        />
      </ComposedChart>
    </ResponsiveContainer>
  )
}
