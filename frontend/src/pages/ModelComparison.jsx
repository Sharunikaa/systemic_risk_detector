import React, { useState, useEffect, useCallback } from 'react'
import axios from 'axios'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, Legend, Tooltip
} from 'recharts'
import MetricCard from '../components/MetricCard'
import ConfusionMatrix from '../components/ConfusionMatrix'
import PRCurveChart from '../components/PRCurveChart'
import { AlertTriangle, CheckCircle } from 'lucide-react'

const MODELS = ['HMM', 'XGBoost', 'LSTM', 'TFT', 'VQH']
const MODEL_KEYS = { HMM: 'hmm', XGBoost: 'xgboost', LSTM: 'lstm', TFT: 'tft', VQH: 'vqh' }
const MODEL_COLORS = {
  HMM: '#06b6d4', XGBoost: '#f59e0b', LSTM: '#22c55e', TFT: '#3b82f6', VQH: '#8b5cf6'
}
const CLASS_NAMES = ['CRISIS', 'NORMAL', 'HIGH-VOL']

function useFetch(url) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  useEffect(() => {
    setLoading(true); setError(null)
    axios.get(url)
      .then(r => setData(r.data))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [url])
  return { data, loading, error }
}

export default function ModelComparison() {
  const [activeModel, setActiveModel] = useState('LSTM')
  const [prCurve, setPrCurve] = useState(null)

  const { data: allMetrics, loading: ml, error: me } = useFetch('/api/models/metrics')
  const { data: benchmark } = useFetch('/api/models/benchmark')
  const { data: cmData }    = useFetch(`/api/models/${MODEL_KEYS[activeModel]}/confusion-matrix`)

  // Load PR curve for active model
  useEffect(() => {
    axios.get(`/api/models/${MODEL_KEYS[activeModel]}/pr-curve`)
      .then(r => setPrCurve(r.data))
      .catch(() => setPrCurve(null))
  }, [activeModel])

  const metrics = allMetrics?.[MODEL_KEYS[activeModel]]
  const best    = benchmark?.best_model_by_crisis_pr_auc

  // Radar data
  const radarData = [
    { axis: 'Macro F1',    ...Object.fromEntries(MODELS.map(m => [m, allMetrics?.[MODEL_KEYS[m]]?.overall?.macro_f1 ?? 0])) },
    { axis: 'Crisis Recall', ...Object.fromEntries(MODELS.map(m => [m, allMetrics?.[MODEL_KEYS[m]]?.per_class?.CRISIS?.recall ?? 0])) },
    { axis: 'PR-AUC',      ...Object.fromEntries(MODELS.map(m => [m, allMetrics?.[MODEL_KEYS[m]]?.crisis_specific?.pr_auc ?? 0])) },
    { axis: 'Accuracy',    ...Object.fromEntries(MODELS.map(m => [m, allMetrics?.[MODEL_KEYS[m]]?.overall?.accuracy ?? 0])) },
  ]

  // Best value per column (for bold highlighting in table)
  const bestPerMetric = {
    accuracy:    Math.max(...MODELS.map(m => allMetrics?.[MODEL_KEYS[m]]?.overall?.accuracy ?? 0)),
    macro_f1:    Math.max(...MODELS.map(m => allMetrics?.[MODEL_KEYS[m]]?.overall?.macro_f1 ?? 0)),
    crisis_recall: Math.max(...MODELS.map(m => allMetrics?.[MODEL_KEYS[m]]?.per_class?.CRISIS?.recall ?? 0)),
    pr_auc:      Math.max(...MODELS.map(m => allMetrics?.[MODEL_KEYS[m]]?.crisis_specific?.pr_auc ?? 0)),
    roc_auc:     Math.max(...MODELS.map(m => allMetrics?.[MODEL_KEYS[m]]?.crisis_specific?.roc_auc ?? 0)),
  }

  const fmt = (v, d=4) => v != null ? v.toFixed(d) : '—'
  const fmtPct = v => v != null ? `${(v*100).toFixed(1)}%` : '—'

  return (
    <div className="fade-in">
      <div className="page-header">
        <h1>Model Comparison</h1>
        <div className="page-subtitle">
          <span>Test window: 2023-01-01 — 2024-12-31</span>
          <span className="dot" />
          <span>Primary metric: Crisis PR-AUC</span>
        </div>
      </div>

      {/* Tab bar */}
      <div className="tab-bar" style={{ marginBottom: '1.5rem', width: 'fit-content' }}>
        {MODELS.map(m => (
          <button
            key={m}
            id={`tab-${m}`}
            onClick={() => setActiveModel(m)}
            className={`tab ${activeModel === m ? 'active' : ''}`}
            style={{
              borderColor: activeModel === m ? MODEL_COLORS[m] : 'transparent',
              color: activeModel === m ? MODEL_COLORS[m] : undefined,
            }}
          >
            {m}
            {m === best && (
              <span style={{ marginLeft: 4, fontSize: '0.65rem', color: 'var(--quantum)' }}>★</span>
            )}
          </button>
        ))}
      </div>

      {me && (
        <div className="card" style={{ borderLeft: '3px solid var(--highvol)', marginBottom: '1rem' }}>
          <AlertTriangle size={14} style={{ color: 'var(--highvol)', marginRight: 6 }} />
          <span style={{ color: 'var(--highvol)', fontSize: '0.85rem' }}>{me}</span>
          <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem', marginTop: 4 }}>
            Run scripts/run_phase2_classical.py to generate model results.
          </div>
        </div>
      )}

      {/* Row 1: Metric cards + Confusion matrix */}
      <div style={{ display: 'grid', gridTemplateColumns: '40% 60%', gap: '1rem', marginBottom: '1.5rem' }}>
        {/* Left: metric cards */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
            <MetricCard
              label="Accuracy"
              value={fmtPct(metrics?.overall?.accuracy)}
              accent={MODEL_COLORS[activeModel]}
            />
            <MetricCard
              label="Macro F1"
              value={fmt(metrics?.overall?.macro_f1)}
              accent={MODEL_COLORS[activeModel]}
            />
            <MetricCard
              label="Crisis Recall"
              value={fmtPct(metrics?.per_class?.CRISIS?.recall)}
              accent="var(--crisis)"
            />
            <MetricCard
              label="Crisis PR-AUC"
              value={fmt(metrics?.crisis_specific?.pr_auc)}
              accent="var(--crisis)"
            />
          </div>

          {/* Per-class table */}
          <div className="card">
            <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Classification Report
            </div>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Class</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1</th>
                  <th>N</th>
                </tr>
              </thead>
              <tbody>
                {CLASS_NAMES.map(cls => {
                  const c = metrics?.per_class?.[cls]
                  const color = cls === 'CRISIS' ? 'var(--crisis)' : cls === 'NORMAL' ? 'var(--normal)' : 'var(--highvol)'
                  return (
                    <tr key={cls}>
                      <td><span style={{ color, fontWeight: 600, fontSize: '0.8rem' }}>{cls}</span></td>
                      <td className="font-mono">{fmt(c?.precision, 3)}</td>
                      <td className="font-mono">{fmt(c?.recall, 3)}</td>
                      <td className="font-mono">{fmt(c?.f1, 3)}</td>
                      <td className="font-mono">{c?.support ?? '—'}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Right: confusion matrix */}
        <div className="card">
          <ConfusionMatrix matrix={cmData?.matrix} modelName={activeModel} />
        </div>
      </div>

      {/* Row 2: PR Curve */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        {prCurve ? (
          <PRCurveChart
            precision={prCurve.precision}
            recall={prCurve.recall}
            thresholds={prCurve.thresholds}
            prAuc={prCurve.pr_auc}
            baseline={prCurve.baseline}
            optThresh={metrics?.crisis_specific?.optimal_threshold ?? 0.5}
          />
        ) : (
          <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', padding: '1rem', textAlign: 'center' }}>
            PR curve data not yet available for {activeModel}
          </div>
        )}
      </div>

      {/* Row 3: All Models Comparison Table */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: '1rem' }}>
          All Models — Test Window 2023–2024
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>Macro F1</th>
                <th>Crisis Recall</th>
                <th>Crisis PR-AUC</th>
                <th>ROC-AUC</th>
              </tr>
            </thead>
            <tbody>
              {MODELS.map(m => {
                const mk = MODEL_KEYS[m]
                const mm = allMetrics?.[mk]
                const isBest = m === best
                const acc   = mm?.overall?.accuracy
                const mf1   = mm?.overall?.macro_f1
                const cr    = mm?.per_class?.CRISIS?.recall
                const prauc = mm?.crisis_specific?.pr_auc
                const roc   = mm?.crisis_specific?.roc_auc

                const bold = (val, best) => Math.abs((val ?? 0) - (best ?? 0)) < 0.0001

                return (
                  <tr key={m} style={{
                    borderLeft: isBest ? '3px solid var(--quantum)' : '3px solid transparent',
                    background: isBest ? 'rgba(139,92,246,0.04)' : undefined,
                  }}>
                    <td style={{ color: MODEL_COLORS[m], fontWeight: 700, fontSize: '0.85rem' }}>
                      {m}{isBest ? ' ★' : ''}
                    </td>
                    <td className="font-mono" style={{ fontWeight: bold(acc, bestPerMetric.accuracy) ? 700 : 400, color: bold(acc, bestPerMetric.accuracy) ? 'var(--normal)' : undefined }}>
                      {fmtPct(acc)}
                    </td>
                    <td className="font-mono" style={{ fontWeight: bold(mf1, bestPerMetric.macro_f1) ? 700 : 400, color: bold(mf1, bestPerMetric.macro_f1) ? 'var(--normal)' : undefined }}>
                      {fmt(mf1)}
                    </td>
                    <td className="font-mono" style={{ fontWeight: bold(cr, bestPerMetric.crisis_recall) ? 700 : 400, color: bold(cr, bestPerMetric.crisis_recall) ? 'var(--normal)' : undefined }}>
                      {fmtPct(cr)}
                    </td>
                    <td className="font-mono" style={{ fontWeight: bold(prauc, bestPerMetric.pr_auc) ? 700 : 400, color: bold(prauc, bestPerMetric.pr_auc) ? 'var(--quantum)' : undefined }}>
                      {fmt(prauc)}
                    </td>
                    <td className="font-mono" style={{ fontWeight: bold(roc, bestPerMetric.roc_auc) ? 700 : 400, color: bold(roc, bestPerMetric.roc_auc) ? 'var(--normal)' : undefined }}>
                      {fmt(roc)}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        <div style={{ marginTop: '0.75rem', fontSize: '0.72rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
          ★ Best model by Crisis PR-AUC &nbsp;|&nbsp;
          Primary metric: Crisis PR-AUC. Accuracy is misleading for imbalanced datasets (crisis ≈ 6% of days).
        </div>
      </div>

      {/* Row 4: Radar chart */}
      <div className="card">
        <div style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: '1rem' }}>
          Performance Radar — All 5 Models
        </div>
        <ResponsiveContainer width="100%" height={320}>
          <RadarChart data={radarData} margin={{ top: 10, bottom: 10 }}>
            <PolarGrid stroke="var(--border)" strokeOpacity={0.4} />
            <PolarAngleAxis dataKey="axis" tick={{ fill: 'var(--text-muted)', fontSize: 12 }} />
            <PolarRadiusAxis domain={[0, 1]} tick={false} axisLine={false} />
            <Tooltip
              contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)',
                borderRadius: 8, fontSize: '0.8rem', color: 'var(--text-primary)' }}
            />
            {MODELS.map(m => (
              <Radar key={m} name={m} dataKey={m}
                stroke={MODEL_COLORS[m]} fill={MODEL_COLORS[m]} fillOpacity={0.08}
                strokeWidth={activeModel === m ? 2.5 : 1.2}
              />
            ))}
            <Legend wrapperStyle={{ fontSize: '0.75rem' }} />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
