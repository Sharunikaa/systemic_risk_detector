import React from 'react'

const CLASS_NAMES = ['CRISIS', 'NORMAL', 'HIGH-VOL']

const CELL_COLORS = {
  CRISIS:    { bg: 'rgba(239,68,68,', text: 'var(--crisis)' },
  NORMAL:    { bg: 'rgba(34,197,94,',  text: 'var(--normal)' },
  'HIGH-VOL':{ bg: 'rgba(245,158,11,', text: 'var(--highvol)' },
}

/**
 * ConfusionMatrix — 3×3 heat-map with count + percentage in each cell.
 *
 * Props:
 *   matrix    : 3×3 nested array [[...],[...],[...]]
 *               Rows = true labels, Cols = predicted labels
 *               Order: [CRISIS, NORMAL, HIGH-VOL]
 *   modelName : string
 */
export default function ConfusionMatrix({ matrix = [], modelName = '' }) {
  if (!matrix?.length) return (
    <div style={{ color: 'var(--text-muted)', padding: '2rem', textAlign: 'center' }}>
      No confusion matrix data
    </div>
  )

  const totalPerRow = matrix.map(row => row.reduce((a, b) => a + b, 0))

  return (
    <div>
      <div style={{
        fontSize: '0.75rem',
        color: 'var(--text-muted)',
        marginBottom: '0.75rem',
        textAlign: 'center',
        textTransform: 'uppercase',
        letterSpacing: '0.06em',
      }}>
        {modelName} — Test Set 2023–2024
      </div>

      {/* Header row */}
      <div style={{ display: 'grid', gridTemplateColumns: '80px repeat(3, 1fr)', gap: 2 }}>
        <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', padding: '4px 0', textAlign: 'center' }}>
          True ↓ Pred →
        </div>
        {CLASS_NAMES.map(cn => (
          <div key={cn} style={{
            fontSize: '0.7rem',
            color: CELL_COLORS[cn].text,
            fontWeight: 600,
            padding: '6px 4px',
            textAlign: 'center',
            letterSpacing: '0.04em',
          }}>
            {cn}
          </div>
        ))}

        {/* Data rows */}
        {CLASS_NAMES.map((trueCls, ri) => {
          const rowTotal = totalPerRow[ri] || 1
          return (
            <React.Fragment key={trueCls}>
              <div style={{
                fontSize: '0.7rem',
                color: CELL_COLORS[trueCls].text,
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'flex-end',
                paddingRight: 8,
                letterSpacing: '0.04em',
              }}>
                {trueCls}
              </div>
              {CLASS_NAMES.map((predCls, ci) => {
                const val  = matrix[ri]?.[ci] ?? 0
                const pct  = ((val / rowTotal) * 100).toFixed(1)
                const isCorrect = ri === ci
                const intensity = isCorrect
                  ? Math.min(0.7, val / rowTotal * 0.9)
                  : Math.min(0.35, val / rowTotal * 0.5)

                const cellColors = CELL_COLORS[isCorrect ? trueCls : predCls]

                return (
                  <div
                    key={predCls}
                    style={{
                      background: `${cellColors.bg}${intensity})`,
                      border: isCorrect
                        ? `1px solid ${cellColors.text}40`
                        : '1px solid rgba(71,85,105,0.3)',
                      borderRadius: 6,
                      padding: '10px 6px',
                      textAlign: 'center',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 2,
                    }}
                  >
                    <div className="font-mono" style={{
                      fontSize: '1.1rem',
                      fontWeight: 700,
                      color: isCorrect ? cellColors.text : 'var(--text-primary)',
                    }}>
                      {val}
                    </div>
                    <div className="font-mono" style={{
                      fontSize: '0.65rem',
                      color: 'var(--text-muted)',
                    }}>
                      {pct}%
                    </div>
                  </div>
                )
              })}
            </React.Fragment>
          )
        })}
      </div>
    </div>
  )
}
