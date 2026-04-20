import React from 'react'

/**
 * ArcGauge — SVG arc gauge showing crisis probability from 0-100%.
 *
 * The arc fills from the left (0%) to the right (100%) with a crisis-red color.
 * A label in the center shows the percentage with monospace font.
 *
 * Props:
 *   value  : number in [0, 1] (crisis probability)
 *   size   : number (diameter in px, default 160)
 *   label  : string shown below the value
 */
export default function ArcGauge({ value = 0, size = 160, label = 'VQH Crisis Prob' }) {
  const clampedValue = Math.max(0, Math.min(1, value))
  const pct   = clampedValue * 100
  const r     = size * 0.38
  const cx    = size / 2
  const cy    = size / 2
  const sw    = size * 0.075     // stroke width

  // Arc goes from 135° to 405° (270° sweep = max)
  const startAngle = 135
  const sweepAngle = 270
  const endAngle = startAngle + sweepAngle * clampedValue

  const toRad = (deg) => (deg * Math.PI) / 180

  const arcPath = (start, end) => {
    const x1 = cx + r * Math.cos(toRad(start))
    const y1 = cy + r * Math.sin(toRad(start))
    const x2 = cx + r * Math.cos(toRad(end))
    const y2 = cy + r * Math.sin(toRad(end))
    const large = end - start > 180 ? 1 : 0
    return `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`
  }

  const bgEnd   = startAngle + sweepAngle
  const fillEnd = endAngle

  // Color for gauge fill
  const fillColor = pct > 40 ? 'var(--crisis)' : pct > 20 ? 'var(--highvol)' : 'var(--normal)'
  const labelColor = fillColor

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {/* Background track */}
        <path
          d={arcPath(startAngle, bgEnd)}
          fill="none"
          stroke="var(--bg-elevated)"
          strokeWidth={sw}
          strokeLinecap="round"
        />
        {/* Fill arc */}
        {clampedValue > 0 && (
          <path
            d={arcPath(startAngle, fillEnd < startAngle + 1 ? startAngle + 1 : fillEnd)}
            fill="none"
            stroke={fillColor}
            strokeWidth={sw}
            strokeLinecap="round"
            style={{ filter: pct > 40 ? `drop-shadow(0 0 6px ${fillColor}60)` : 'none' }}
          />
        )}
        {/* Center value */}
        <text
          x={cx} y={cy - 4}
          textAnchor="middle"
          fontSize={size * 0.18}
          fontWeight={700}
          fill={labelColor}
          fontFamily="'JetBrains Mono', monospace"
        >
          {pct.toFixed(1)}%
        </text>
        {/* Min / Max labels */}
        <text x={cx - r * 0.7} y={cy + r * 0.82} textAnchor="middle"
          fontSize={size * 0.07} fill="var(--text-muted)" fontFamily="Inter,sans-serif">0%</text>
        <text x={cx + r * 0.7} y={cy + r * 0.82} textAnchor="middle"
          fontSize={size * 0.07} fill="var(--text-muted)" fontFamily="Inter,sans-serif">100%</text>
      </svg>
      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textAlign: 'center' }}>
        {label}
      </div>
    </div>
  )
}
