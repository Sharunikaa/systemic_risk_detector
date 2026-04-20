import React from 'react'
import { NavLink } from 'react-router-dom'
import { LayoutGrid, BarChart2, Cpu, Activity, Zap } from 'lucide-react'

const NAV_ITEMS = [
  { path: '/',            label: 'Overview',         icon: LayoutGrid, accent: 'var(--normal)'  },
  { path: '/comparison',  label: 'Model Comparison',  icon: BarChart2,  accent: 'var(--highvol)' },
  { path: '/quantum',     label: 'Quantum Results',   icon: Cpu,        accent: 'var(--quantum)' },
  { path: '/predictions', label: 'Predictions',       icon: Activity,   accent: 'var(--crisis)'  },
]

export default function Sidebar() {
  return (
    <aside style={{
      width: 'var(--sidebar-w)',
      minWidth: 'var(--sidebar-w)',
      background: 'var(--bg-surface)',
      borderRight: '1px solid var(--border)',
      display: 'flex',
      flexDirection: 'column',
      position: 'sticky',
      top: 0,
      height: '100vh',
      overflow: 'hidden',
    }}>

      {/* ── Brand ── */}
      <div style={{
        padding: '1.25rem 1rem 1rem',
        borderBottom: '1px solid var(--border)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 34, height: 34,
            borderRadius: 9,
            background: 'linear-gradient(135deg, #818cf8 0%, #6d28d9 100%)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            flexShrink: 0,
            boxShadow: '0 0 14px rgba(129,140,248,0.35)',
          }}>
            <Zap size={16} color="white" strokeWidth={2.5} />
          </div>
          <div>
            <div style={{
              fontSize: '0.8125rem',
              fontWeight: 700,
              color: 'var(--text-primary)',
              letterSpacing: '0.04em',
              lineHeight: 1.2,
            }}>
              QML System
            </div>
            <div style={{
              fontSize: '0.6875rem',
              color: 'var(--text-muted)',
              letterSpacing: '0.02em',
              marginTop: 1,
            }}>
              Contagion Detection
            </div>
          </div>
        </div>
      </div>

      {/* ── Nav label ── */}
      <div style={{
        padding: '1rem 1rem 0.375rem',
        fontSize: '0.6875rem',
        fontWeight: 600,
        color: 'var(--text-disabled)',
        textTransform: 'uppercase',
        letterSpacing: '0.08em',
      }}>
        Navigation
      </div>

      {/* ── Nav items ── */}
      <nav style={{ padding: '0 0.5rem', flex: 1 }}>
        {NAV_ITEMS.map(({ path, label, icon: Icon, accent }) => (
          <NavLink
            key={path}
            to={path}
            end={path === '/'}
            style={({ isActive }) => ({
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              padding: '9px 10px',
              marginBottom: 2,
              borderRadius: 8,
              textDecoration: 'none',
              fontSize: '0.8125rem',
              fontWeight: isActive ? 600 : 400,
              color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
              background: isActive ? 'var(--bg-elevated)' : 'transparent',
              borderLeft: isActive ? `2px solid ${accent}` : '2px solid transparent',
              transition: 'all 0.15s ease',
              position: 'relative',
            })}
          >
            {({ isActive }) => (
              <>
                <div style={{
                  width: 28, height: 28,
                  borderRadius: 7,
                  background: isActive ? `${accent}18` : 'transparent',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  flexShrink: 0,
                  transition: 'background 0.15s ease',
                }}>
                  <Icon
                    size={15}
                    color={isActive ? accent : 'var(--text-muted)'}
                    strokeWidth={isActive ? 2.5 : 2}
                  />
                </div>
                <span style={{ lineHeight: 1 }}>{label}</span>
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* ── System status ── */}
      <div style={{
        margin: '0 0.5rem 0.5rem',
        padding: '0.75rem',
        background: 'var(--bg-elevated)',
        borderRadius: 8,
        border: '1px solid var(--border)',
      }}>
        <div style={{
          fontSize: '0.6875rem',
          fontWeight: 600,
          color: 'var(--text-muted)',
          textTransform: 'uppercase',
          letterSpacing: '0.07em',
          marginBottom: '0.5rem',
        }}>
          System
        </div>
        {[
          { label: 'VQH Circuit',    color: 'var(--quantum)', detail: '3 qubits · 2L' },
          { label: 'Models Active',  color: 'var(--normal)',  detail: '5 / 5' },
          { label: 'Test Window',    color: 'var(--highvol)', detail: '2023–2024' },
        ].map(({ label, color, detail }) => (
          <div key={label} style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            marginBottom: 5,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <div style={{
                width: 5, height: 5, borderRadius: '50%',
                background: color,
                boxShadow: `0 0 5px ${color}`,
                flexShrink: 0,
              }} />
              <span style={{ fontSize: '0.6875rem', color: 'var(--text-muted)' }}>{label}</span>
            </div>
            <span style={{ fontSize: '0.6875rem', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>
              {detail}
            </span>
          </div>
        ))}
      </div>

      {/* ── Footer ── */}
      <div style={{
        padding: '0.625rem 1rem',
        borderTop: '1px solid var(--border)',
        fontSize: '0.6875rem',
        color: 'var(--text-disabled)',
        textAlign: 'center',
      }}>
        QML Contagion v1.0
      </div>
    </aside>
  )
}
