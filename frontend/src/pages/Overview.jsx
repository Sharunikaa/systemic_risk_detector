import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer
} from 'recharts'
import RegimeTimeline from '../components/RegimeTimeline'
import MetricCard from '../components/MetricCard'
import { AlertTriangle, Clock, TrendingUp, Zap } from 'lucide-react'

const REGIME_COLORS = {
  CRISIS: 'var(--crisis)',
  NORMAL: 'var(--normal)',
  'HIGH-VOL': 'var(--highvol)',
}

function useFetch(url) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  useEffect(() => {
    setLoading(true)
    setError(null)
    axios.get(url)
      .then(r => setData(r.data))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [url])   // re-fetches whenever url changes (date params included)
  return { data, loading, error }
}

function LoadingBar() {
  return (
    <div className="skeleton" style={{ height: 24, borderRadius: 6, marginBottom: 8 }} />
  )
}

function ErrorMsg({ message, hint }) {
  return (
    <div className="card" style={{ borderLeft: '3px solid var(--highvol)', padding: '1rem' }}>
      <div style={{ color: 'var(--highvol)', fontSize: '0.85rem', marginBottom: 4 }}>
        <AlertTriangle size={14} style={{ marginRight: 6, verticalAlign: 'middle' }} />
        {message}
      </div>
      {hint && <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>{hint}</div>}
    </div>
  )
}

export default function Overview() {
  const [startDate, setStartDate] = useState('2014-09-18')
  const [endDate, setEndDate] = useState('2024-12-31')
  
  const { data: regimes, loading: rl, error: re }   = useFetch(`/api/data/regimes?start_date=${startDate}&end_date=${endDate}`)
  const { data: prices,  loading: pl, error: pe }   = useFetch(`/api/data/prices?start_date=${startDate}&end_date=${endDate}`)
  const { data: benchmark, loading: bl, error: be } = useFetch('/api/models/benchmark')
  const { data: vqhPreds, loading: vl }              = useFetch(`/api/models/vqh/predictions?start_date=${startDate}&end_date=${endDate}`)
  const { data: health }                             = useFetch('/api/health')

  // Current regime (last entry in regimes)
  const currentRegime = regimes?.at(-1)
  const currentLabel  = currentRegime?.regime_label ?? 'LOADING'
  const regimeColor   = REGIME_COLORS[currentLabel] ?? 'var(--text-muted)'

  // VQH latest crisis prob
  const latestVqhProb = vqhPreds?.at(-1)?.prob_crisis ?? null

  // Days since last crisis
  const daysSinceCrisis = (() => {
    if (!regimes) return null
    const lastCrisis = [...regimes].reverse().findIndex(r => r.crisis_flag === 1)
    return lastCrisis === -1 ? '—' : lastCrisis
  })()

  // Timing lead from VQH metrics
  const vqhMetricsPromise = useFetch('/api/models/metrics')
  const vqhMetrics = vqhMetricsPromise.data?.vqh
  const timingLead = vqhMetrics?.quantum_metrics?.crisis_timing_lead_days ?? null

  // Benchmark bar chart data
  const benchmarkData = (() => {
    if (!benchmark) return []
    const models = benchmark.models ?? []
    return models.map(m => ({
      model:         m,
      'Macro F1':    benchmark.comparison?.macro_f1?.[m] ?? 0,
      'Crisis Recall': benchmark.comparison?.crisis_recall?.[m] ?? 0,
      'Crisis PR-AUC': benchmark.comparison?.crisis_pr_auc?.[m] ?? 0,
    }))
  })()

  return (
    <div className="fade-in">
      {/* Page header */}
      <div className="page-header">
        <h1>Systemic Risk Overview</h1>
        <div className="page-subtitle">
          <span>S&amp;P 500</span><span className="dot" />
          <span>Bitcoin</span><span className="dot" />
          <span>Gold</span><span className="dot" />
          <span>VQH Quantum Prediction</span>
          {health && (
            <>
              <span className="dot" />
              <span style={{
                color: 'var(--normal)',
                display: 'flex', alignItems: 'center', gap: 4,
              }}>
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--normal)', display: 'inline-block' }} />
                {health.models_loaded}/5 models loaded
              </span>
            </>
          )}
        </div>
      </div>

      {/* Date Range Filter */}
      <div className="card" style={{ marginBottom: '1.5rem', padding: '1rem 1.25rem' }}>
        <div style={{
          display: 'flex', gap: '1.25rem', alignItems: 'flex-end', flexWrap: 'wrap',
        }}>
          <div className="input-group" style={{ minWidth: 160 }}>
            <label>Start Date</label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              min="2014-09-18"
              max="2024-12-31"
            />
          </div>
          <div className="input-group" style={{ minWidth: 160 }}>
            <label>End Date</label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              min="2014-09-18"
              max="2024-12-31"
            />
          </div>
          <div style={{
            fontSize: '0.75rem', color: 'var(--text-muted)',
            paddingBottom: '0.5rem',
          }}>
            Filters regime timeline &amp; price data
          </div>
        </div>
      </div>

      {/* ── Section 1: Status cards ── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
        {/* Current Regime */}
        <div className="card" style={{ borderLeft: `3px solid ${regimeColor}` }}>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>
            Current Regime
          </div>
          {rl ? <LoadingBar /> : (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <div className={currentLabel === 'CRISIS' ? 'pulse-crisis' : ''}
                style={{
                  padding: '4px 14px', borderRadius: 8, fontSize: '1.1rem',
                  fontWeight: 700, color: regimeColor,
                  background: `${regimeColor}18`,
                  border: `1.5px solid ${regimeColor}40`,
                }}>
                {currentLabel}
              </div>
            </div>
          )}
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 8 }}>
            {currentRegime?.date ?? '—'}
          </div>
        </div>

        {/* VQH Crisis Probability */}
        <MetricCard
          label="VQH Crisis Probability"
          value={latestVqhProb != null ? `${(latestVqhProb * 100).toFixed(1)}%` : '—'}
          accent="var(--quantum)"
          subtitle="VQH quantum prediction"
        />

        {/* Days Since Last Crisis */}
        <MetricCard
          label="Days Since Last Crisis"
          value={daysSinceCrisis}
          accent="var(--normal)"
          subtitle="trading days"
        />

        {/* VQH Lead */}
        <MetricCard
          label="VQH Lead vs Classical"
          value={timingLead != null ? `${timingLead > 0 ? '+' : ''}${timingLead.toFixed(1)} days` : '—'}
          accent="var(--quantum)"
          subtitle="advance warning"
          delta={timingLead}
        />
      </div>

      {/* ── Section 2: Regime Timeline ── */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '1rem' }}>
          Regime Timeline — 2014 to 2024
        </div>
        {rl || pl ? (
          <div className="skeleton" style={{ height: 220 }} />
        ) : re ? (
          <ErrorMsg message={re} hint="Run scripts/run_phase1_preprocessing.py first" />
        ) : (
          <RegimeTimeline regimes={regimes} prices={prices} />
        )}
      </div>

      {/* ── Section 3: Asset Prices ── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
        {[
          { key: 'spx',  label: 'S&P 500',  color: 'var(--normal)' },
          { key: 'btc',  label: 'Bitcoin',   color: 'var(--quantum)' },
          { key: 'gold', label: 'Gold (GLD)', color: 'var(--highvol)' },
        ].map(({ key, label, color }) => {
          const latest = prices?.at(-1)?.[key]
          const first  = prices?.[0]?.[key]
          const change = first ? ((latest - first) / first * 100).toFixed(1) : null
          return (
            <div key={key} className="card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
                <div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</div>
                  <div className="font-mono" style={{ fontSize: '1.5rem', fontWeight: 700, color }}>
                    {latest != null ? latest.toFixed(1) : '—'}
                  </div>
                </div>
                {change != null && (
                  <div className="font-mono" style={{
                    fontSize: '0.85rem', fontWeight: 600,
                    color: change >= 0 ? 'var(--normal)' : 'var(--crisis)',
                    background: change >= 0 ? 'rgba(34,197,94,0.1)' : 'rgba(239,68,68,0.1)',
                    padding: '2px 8px', borderRadius: 6
                  }}>
                    {change >= 0 ? '+' : ''}{change}%
                  </div>
                )}
              </div>
              <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>
                Normalized to 100 at 2014-09-17
              </div>
            </div>
          )
        })}
      </div>

      {/* ── Section 4: Model Benchmark ── */}
      <div className="card">
        <div style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '1rem' }}>
          Model Benchmark — Crisis Detection Performance
        </div>
        {bl ? (
          <div className="skeleton" style={{ height: 220 }} />
        ) : be ? (
          <ErrorMsg message={be} hint="Run scripts/run_phase2_classical.py first" />
        ) : (
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={benchmarkData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" strokeOpacity={0.35} />
              <XAxis dataKey="model" tick={{ fill: 'var(--text-muted)', fontSize: 12 }}
                axisLine={{ stroke: 'var(--border)' }} tickLine={false} />
              <YAxis domain={[0, 1]} tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                axisLine={{ stroke: 'var(--border)' }} tickLine={false} width={35} />
              <Tooltip
                contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)',
                  borderRadius: 8, color: 'var(--text-primary)', fontSize: '0.8rem' }}
              />
              <Legend wrapperStyle={{ fontSize: '0.75rem', color: 'var(--text-muted)' }} />
              <Bar dataKey="Macro F1"      fill="var(--highvol)" radius={[3,3,0,0]} />
              <Bar dataKey="Crisis Recall" fill="var(--crisis)"  radius={[3,3,0,0]} />
              <Bar dataKey="Crisis PR-AUC" fill="var(--quantum)" radius={[3,3,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        )}
        {benchmark && (
          <div style={{ marginTop: '0.75rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Best model: <span style={{ color: 'var(--quantum)', fontWeight: 600 }}>
              {benchmark.best_model_by_crisis_pr_auc}
            </span> &nbsp;·&nbsp; Primary metric: Crisis PR-AUC
          </div>
        )}
      </div>
    </div>
  )
}
