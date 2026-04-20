import React from 'react'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

/**
 * MetricCard — single KPI tile with accent border, value, subtitle, delta.
 *
 * Props:
 *   label      : string
 *   value      : string | number
 *   subtitle   : string
 *   accent     : CSS color
 *   delta      : number
 *   deltaLabel : string
 *   size       : 'sm' | 'md' | 'lg'
 */
export default function MetricCard({
  label,
  value,
  subtitle,
  accent = 'var(--quantum)',
  delta,
  deltaLabel,
  size = 'md',
}) {
  const valueSizes = { sm: '1.25rem', md: '1.75rem', lg: '2.25rem' }
  const labelSizes = { sm: '0.6875rem', md: '0.6875rem', lg: '0.75rem' }
  const valueSize  = valueSizes[size] ?? valueSizes.md
  const labelSize  = labelSizes[size] ?? labelSizes.md

  const deltaColor = delta == null
    ? 'var(--text-muted)'
    : delta > 0 ? 'var(--normal)' : delta < 0 ? 'var(--crisis)' : 'var(--text-muted)'

  const DeltaIcon = delta > 0 ? TrendingUp : delta < 0 ? TrendingDown : Minus

  return (
    <div className="card" style={{
      borderLeft: `2px solid ${accent}`,
      display: 'flex',
      flexDirection: 'column',
      gap: '0.375rem',
      padding: '1rem 1.125rem',
    }}>
      {/* Label */}
      <div style={{
        fontSize: labelSize,
        color: 'var(--text-muted)',
        fontWeight: 500,
        textTransform: 'uppercase',
        letterSpacing: '0.07em',
        lineHeight: 1,
      }}>
        {label}
      </div>

      {/* Value */}
      <div className="font-mono" style={{
        fontSize: valueSize,
        fontWeight: 700,
        color: 'var(--text-primary)',
        lineHeight: 1.1,
        letterSpacing: '-0.02em',
      }}>
        {value ?? '—'}
      </div>

      {/* Subtitle */}
      {subtitle && (
        <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', lineHeight: 1 }}>
          {subtitle}
        </div>
      )}

      {/* Delta */}
      {delta != null && (
        <div style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: 4,
          fontSize: '0.75rem',
          color: deltaColor,
          fontWeight: 600,
          marginTop: 2,
          padding: '2px 7px',
          borderRadius: 4,
          background: delta > 0
            ? 'rgba(16,185,129,0.1)'
            : delta < 0 ? 'rgba(244,63,94,0.1)' : 'rgba(107,107,133,0.1)',
          width: 'fit-content',
        }}>
          <DeltaIcon size={11} strokeWidth={2.5} />
          <span className="font-mono">
            {delta > 0 ? '+' : ''}{typeof delta === 'number' ? delta.toFixed(2) : delta}
          </span>
          {deltaLabel && (
            <span style={{ color: 'var(--text-muted)', fontWeight: 400, fontSize: '0.6875rem' }}>
              {deltaLabel}
            </span>
          )}
        </div>
      )}
    </div>
  )
}
