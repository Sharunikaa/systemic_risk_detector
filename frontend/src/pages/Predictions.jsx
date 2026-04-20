import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { CheckCircle, XCircle, AlertTriangle } from 'lucide-react'
import ArcGauge from '../components/ArcGauge'
import ProbabilityChart from '../components/ProbabilityChart'

const REGIME_LABELS = { 0: 'CRISIS', 1: 'NORMAL', 2: 'HIGH-VOL' }
const REGIME_COLORS = { 0: 'var(--crisis)', 1: 'var(--normal)', 2: 'var(--highvol)' }
const BADGE_CLASS  = { 0: 'badge badge-crisis', 1: 'badge badge-normal', 2: 'badge badge-highvol' }

const BLACK_SWAN_EVENTS = [
  { name: 'COVID-19 2020',    period: 'Feb–May 2020',  key: 'COVID_2020',  inTest: false },
  { name: 'Crypto Winter 2022', period: 'May–Jul 2022', key: 'Crypto_2022', inTest: true },
  { name: 'SVB Collapse 2023',  period: 'Mar–May 2023', key: 'SVB_2023',    inTest: true },
]

function useFetch(url) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  useEffect(() => {
    axios.get(url)
      .then(r => setData(r.data))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [url])
  return { data, loading, error }
}

function RecallCell({ recall }) {
  if (recall == null) return <td style={{ color: 'var(--text-muted)' }}>N/A</td>
  const pct = recall * 100
  const color = pct >= 70 ? 'var(--normal)' : pct >= 40 ? 'var(--highvol)' : 'var(--crisis)'
  const bg    = pct >= 70 ? 'rgba(34,197,94,0.1)' : pct >= 40 ? 'rgba(245,158,11,0.1)' : 'rgba(239,68,68,0.1)'
  return (
    <td>
      <span className="font-mono" style={{ color, background: bg, padding: '2px 8px', borderRadius: 4, fontSize: '0.85rem', fontWeight: 600 }}>
        {pct.toFixed(1)}%
      </span>
    </td>
  )
}

export default function Predictions() {
  const { data: allMetrics }                          = useFetch('/api/models/metrics')
  const { data: vqhPreds,  loading: vl, error: ve }  = useFetch('/api/models/vqh/predictions')
  const { data: latestLog, loading: ll }              = useFetch('/api/predictions/latest?days=30')

  // Custom prediction state
  const [customInput, setCustomInput] = useState({
    btc_return: 0.0,
    spx_return: 0.0,
    gold_return: 0.0,
    btc_volatility: 0.2,
    spx_volatility: 0.15,
    gold_volatility: 0.12,
    btc_spx_corr: 0.3,
    btc_gold_corr: 0.2,
    spx_gold_corr: 0.4,
    vix_level: 20.0,
  })
  const [customPredictions, setCustomPredictions] = useState(null)
  const [predicting, setPredicting] = useState(false)

  const vqhMetrics  = allMetrics?.vqh
  const lstmMetrics = allMetrics?.lstm

  // Handle custom prediction
  const handleCustomPrediction = async () => {
    setPredicting(true)
    try {
      const response = await axios.post('/api/models/predict', customInput)
      setCustomPredictions(response.data)
    } catch (error) {
      console.error('Custom prediction failed:', error)
      setCustomPredictions(null)
    } finally {
      setPredicting(false)
    }
  }

  // Use last VQH prediction for current state
  const latest = vqhPreds?.at(-1) ?? latestLog?.at(-1)
  const curRegime   = latest?.pred_regime ?? latest?.true_regime ?? 1
  const crisPct     = latest?.prob_crisis  ?? 0
  const normalPct   = latest?.prob_normal  ?? 1
  const highvolPct  = latest?.prob_highvol ?? 0
  const lastDate    = latest?.date ?? '—'

  // Black-swan data from VQH and LSTM metrics
  const buildBlackSwan = (metricsObj, modelLabel) => {
    const bs = metricsObj?.black_swan ?? {}
    return BLACK_SWAN_EVENTS.map(ev => {
      const w = bs[ev.key]
      if (!w || !w.in_test) return { ...ev, model: modelLabel, n_crisis: '—', n_detected: '—', recall: null, lead: null }
      return {
        ...ev,
        model:      modelLabel,
        n_crisis:   w.n_crisis_days,
        n_detected: w.n_detected,
        recall:     w.recall,
      }
    })
  }

  const vqhBs  = buildBlackSwan(vqhMetrics, 'VQH')
  const lstmBs = buildBlackSwan(lstmMetrics, 'LSTM')
  const leadDelta = vqhMetrics?.quantum_metrics?.crisis_timing_lead_days

  return (
    <div className="fade-in">
      <div className="page-header">
        <h1>Predictions</h1>
        <div className="page-subtitle">
          <span>VQH quantum next-step regime forecasting</span>
          <span className="dot" />
          <span>Test window 2023–2024</span>
        </div>
      </div>

      {/* Custom Prediction Form */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div style={{ marginBottom: '1rem' }}>
          <h3 style={{ marginBottom: '0.25rem' }}>Custom Market Data Prediction</h3>
          <p style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', margin: 0 }}>
            Enter current market conditions to get predictions from all 5 models.
          </p>
        </div>

        {/* Returns row */}
        <div style={{ marginBottom: '0.75rem' }}>
          <div style={{
            fontSize: '0.6875rem', fontWeight: 600, color: 'var(--text-muted)',
            textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: '0.625rem',
          }}>
            Returns
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem' }}>
            {[
              { key: 'btc_return',  label: 'BTC Return',  step: '0.01' },
              { key: 'spx_return',  label: 'SPX Return',  step: '0.01' },
              { key: 'gold_return', label: 'Gold Return', step: '0.01' },
            ].map(({ key, label, step }) => (
              <div className="input-group" key={key}>
                <label>{label}</label>
                <input
                  type="number" step={step}
                  value={customInput[key]}
                  onChange={(e) => setCustomInput({ ...customInput, [key]: parseFloat(e.target.value) || 0 })}
                />
              </div>
            ))}
          </div>
        </div>

        {/* Volatility row */}
        <div style={{ marginBottom: '0.75rem' }}>
          <div style={{
            fontSize: '0.6875rem', fontWeight: 600, color: 'var(--text-muted)',
            textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: '0.625rem',
          }}>
            Volatility
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem' }}>
            {[
              { key: 'btc_volatility',  label: 'BTC Volatility',  min: '0', max: '2' },
              { key: 'spx_volatility',  label: 'SPX Volatility',  min: '0', max: '2' },
              { key: 'gold_volatility', label: 'Gold Volatility', min: '0', max: '2' },
            ].map(({ key, label, min, max }) => (
              <div className="input-group" key={key}>
                <label>{label}</label>
                <input
                  type="number" step="0.01" min={min} max={max}
                  value={customInput[key]}
                  onChange={(e) => setCustomInput({ ...customInput, [key]: parseFloat(e.target.value) || 0 })}
                />
              </div>
            ))}
          </div>
        </div>

        {/* Correlations + VIX row */}
        <div style={{ marginBottom: '1.25rem' }}>
          <div style={{
            fontSize: '0.6875rem', fontWeight: 600, color: 'var(--text-muted)',
            textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: '0.625rem',
          }}>
            Correlations &amp; Fear Index
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '0.75rem' }}>
            {[
              { key: 'btc_spx_corr',  label: 'BTC–SPX Corr',  min: '-1', max: '1' },
              { key: 'btc_gold_corr', label: 'BTC–Gold Corr', min: '-1', max: '1' },
              { key: 'spx_gold_corr', label: 'SPX–Gold Corr', min: '-1', max: '1' },
              { key: 'vix_level',     label: 'VIX Level',     min: '0',  max: '100', step: '0.1' },
            ].map(({ key, label, min, max, step = '0.01' }) => (
              <div className="input-group" key={key}>
                <label>{label}</label>
                <input
                  type="number" step={step} min={min} max={max}
                  value={customInput[key]}
                  onChange={(e) => setCustomInput({ ...customInput, [key]: parseFloat(e.target.value) || 0 })}
                />
              </div>
            ))}
          </div>
        </div>

        <button
          className="btn btn-primary"
          onClick={handleCustomPrediction}
          disabled={predicting}
        >
          {predicting ? 'Running models…' : 'Get Predictions'}
        </button>
      </div>

      {/* Custom Prediction Results */}
      {customPredictions && (
        <div className="card" style={{ marginBottom: '1.5rem' }}>
          <h3 style={{ marginBottom: '1rem' }}>Model Comparison Results</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.75rem' }}>
            {customPredictions.predictions.map((pred) => {
              const isBest = pred.model_name === customPredictions.best_model
              const rColor = REGIME_COLORS[pred.pred_regime] ?? 'var(--text-muted)'
              return (
                <div key={pred.model_name} style={{
                  border: `1px solid ${isBest ? 'var(--quantum-border)' : 'var(--border)'}`,
                  borderRadius: 9,
                  padding: '1rem',
                  background: isBest ? 'var(--quantum-dim)' : 'var(--bg-elevated)',
                  position: 'relative',
                }}>
                  <div style={{
                    display: 'flex', justifyContent: 'space-between',
                    alignItems: 'center', marginBottom: '0.625rem',
                  }}>
                    <span style={{ fontWeight: 700, fontSize: '0.875rem', color: 'var(--text-primary)' }}>
                      {pred.model_name}
                    </span>
                    {isBest && (
                      <span className="badge badge-quantum" style={{ fontSize: '0.6rem' }}>BEST</span>
                    )}
                  </div>
                  <div style={{
                    fontSize: '1rem', fontWeight: 700, color: rColor,
                    marginBottom: '0.625rem',
                  }}>
                    {REGIME_LABELS[pred.pred_regime] ?? 'UNKNOWN'}
                  </div>
                  {[
                    ['Crisis',   pred.prob_crisis,   'var(--crisis)'],
                    ['Normal',   pred.prob_normal,   'var(--normal)'],
                    ['High-Vol', pred.prob_highvol,  'var(--highvol)'],
                  ].map(([lbl, val, col]) => (
                    <div key={lbl} style={{ marginBottom: 5 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                        <span style={{ fontSize: '0.6875rem', color: 'var(--text-muted)' }}>{lbl}</span>
                        <span className="font-mono" style={{ fontSize: '0.6875rem', color: col }}>
                          {(val * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="progress-bar">
                        <div className="progress-fill" style={{ width: `${val * 100}%`, background: col }} />
                      </div>
                    </div>
                  ))}
                  <div style={{
                    marginTop: '0.5rem', fontSize: '0.6875rem',
                    color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between',
                  }}>
                    <span>Confidence</span>
                    <span className="font-mono" style={{ color: 'var(--text-secondary)' }}>
                      {(pred.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {ve && (
        <div className="card" style={{ borderLeft: '3px solid var(--highvol)', marginBottom: '1rem', fontSize: '0.85rem', color: 'var(--highvol)' }}>
          <AlertTriangle size={14} style={{ marginRight: 6, verticalAlign: 'middle' }} />
          {ve} — Run scripts/run_phase3_qml.py to generate VQH predictions.
        </div>
      )}

      {/* ── Section 1: Two-column layout ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '35% 65%', gap: '1rem', marginBottom: '1.5rem', alignItems: 'start' }}>
        {/* Left: Current Market State */}
        <div className="card" style={{ borderLeft: '3px solid var(--quantum)', textAlign: 'center' }}>
          <div style={{ fontSize: '0.7rem', color: 'var(--quantum)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '1rem' }}>
            VQH Quantum Prediction
          </div>

          {/* Regime badge */}
          <div style={{ marginBottom: '1.25rem' }}>
            <span className={BADGE_CLASS[curRegime]} style={{ fontSize: '1rem', padding: '6px 18px' }}>
              {REGIME_LABELS[curRegime] ?? 'UNKNOWN'}
            </span>
          </div>

          {/* Arc gauge */}
          <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1.25rem' }}>
            <ArcGauge value={crisPct} size={140} label="VQH Crisis Prob" />
          </div>

          {/* Probability bars */}
          {[
            { label: 'Crisis',   val: crisPct,   color: 'var(--crisis)' },
            { label: 'Normal',   val: normalPct, color: 'var(--normal)' },
            { label: 'High-Vol', val: highvolPct, color: 'var(--highvol)' },
          ].map(({ label, val, color }) => (
            <div key={label} style={{ marginBottom: '0.6rem', textAlign: 'left' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{label}</span>
                <span className="font-mono" style={{ fontSize: '0.75rem', color }}>{(val*100).toFixed(1)}%</span>
              </div>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${val*100}%`, background: color }} />
              </div>
            </div>
          ))}

          <div style={{ marginTop: '0.75rem', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
            Last date: {lastDate}
          </div>
        </div>

        {/* Right: Probability history */}
        <div className="card">
          <div style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: '1rem' }}>
            Probability History — Last 252 Trading Days
          </div>
          {vl ? (
            <div className="skeleton" style={{ height: 200 }} />
          ) : vqhPreds ? (
            <ProbabilityChart predictions={vqhPreds.slice(-252)} height={240} />
          ) : (
            <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', textAlign: 'center', padding: '3rem' }}>
              Run scripts/run_phase3_qml.py to generate predictions.
            </div>
          )}
        </div>
      </div>

      {/* ── Section 2: Black Swan Event Table ── */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: '1rem' }}>
          Black Swan Event Detection
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Event</th>
                <th>Period</th>
                <th>Crisis Days</th>
                <th>LSTM Detected</th>
                <th>VQH Detected</th>
                <th>VQH Lead</th>
              </tr>
            </thead>
            <tbody>
              {BLACK_SWAN_EVENTS.map((ev, i) => {
                const vbs  = vqhBs[i]
                const lbs  = lstmBs[i]
                const notInTest = !ev.inTest

                const vqhDetected  = vbs?.recall
                const lstmDetected = lbs?.recall

                return (
                  <tr key={ev.name}>
                    <td style={{ fontWeight: 600, fontSize: '0.85rem' }}>{ev.name}</td>
                    <td style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>{ev.period}</td>
                    <td className="font-mono">{notInTest ? 'In training' : (vbs?.n_crisis ?? '—')}</td>
                    {notInTest ? (
                      <td colSpan={3} style={{ color: 'var(--text-muted)', fontSize: '0.8rem', fontStyle: 'italic' }}>
                        Falls in training window — not evaluated
                      </td>
                    ) : (
                      <>
                        <RecallCell recall={lstmDetected} />
                        <RecallCell recall={vqhDetected} />
                        <td className="font-mono" style={{ color: 'var(--quantum)' }}>
                          {leadDelta != null ? `${leadDelta > 0 ? '+' : ''}${leadDelta.toFixed(1)}d` : '—'}
                        </td>
                      </>
                    )}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Section 3: Last 30 Days Log ── */}
      <div className="card">
        <div style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: '1rem' }}>
          Last 30 Predictions Log
        </div>
        {ll ? (
          <div className="skeleton" style={{ height: 200 }} />
        ) : (
          <div style={{ overflowX: 'auto', maxHeight: '400px', overflowY: 'auto' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>True Regime</th>
                  <th>VQH Prediction</th>
                  <th>Match</th>
                  <th>Crisis Prob</th>
                </tr>
              </thead>
              <tbody>
                {(latestLog ?? []).slice().reverse().map(row => {
                  const isMatch     = row.match
                  const trueLbl     = REGIME_LABELS[row.true_regime]  ?? 'NORMAL'
                  const predLbl     = REGIME_LABELS[row.pred_regime]  ?? 'NORMAL'
                  const isMissedCrisis = !isMatch && row.true_regime === 0
                  const isFalseAlarm   = !isMatch && row.pred_regime === 0 && row.true_regime !== 0

                  const rowBg = isMatch ? undefined
                    : isMissedCrisis ? 'rgba(239,68,68,0.06)'
                    : isFalseAlarm ? 'rgba(245,158,11,0.04)' : undefined

                  return (
                    <tr key={row.date} style={{ background: rowBg }}>
                      <td className="font-mono" style={{ fontSize: '0.82rem' }}>{row.date}</td>
                      <td>
                        <span className={`badge badge-${trueLbl.toLowerCase().replace('-','')}`}
                          style={{ fontSize: '0.72rem' }}>
                          {trueLbl}
                        </span>
                      </td>
                      <td>
                        <span className={`badge badge-${predLbl.toLowerCase().replace('-','')}`}
                          style={{ fontSize: '0.72rem' }}>
                          {predLbl}
                        </span>
                      </td>
                      <td style={{ textAlign: 'center' }}>
                        {isMatch
                          ? <CheckCircle size={14} color="var(--normal)" />
                          : <XCircle    size={14} color={isMissedCrisis ? 'var(--crisis)' : 'var(--highvol)'} />
                        }
                      </td>
                      <td className="font-mono" style={{
                        fontSize: '0.82rem',
                        color: row.prob_crisis > 0.4 ? 'var(--crisis)' : 'var(--text-muted)',
                        fontWeight: row.prob_crisis > 0.4 ? 700 : 400,
                      }}>
                        {(row.prob_crisis * 100).toFixed(1)}%
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
        <div style={{ marginTop: '0.5rem', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
          <span style={{ color: 'var(--crisis)' }}>■ Red row</span> = missed crisis  &nbsp;|&nbsp;
          <span style={{ color: 'var(--highvol)' }}>■ Amber row</span> = false alarm &nbsp;|&nbsp;
          Green check = correct prediction
        </div>
      </div>
    </div>
  )
}
