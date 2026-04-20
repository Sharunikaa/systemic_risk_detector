import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  AreaChart, Area, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ReferenceArea, ResponsiveContainer, Legend
} from 'recharts'
import MetricCard from '../components/MetricCard'
import { Cpu, AlertTriangle, TrendingUp, TrendingDown } from 'lucide-react'

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

function DeltaIndicator({ vqhVal, lstmVal, decimals = 4, isPercent = false }) {
  if (vqhVal == null || lstmVal == null) return <span style={{ color: 'var(--text-muted)' }}>—</span>
  const diff    = vqhVal - lstmVal
  const color   = diff > 0 ? 'var(--normal)' : diff < 0 ? 'var(--crisis)' : 'var(--text-muted)'
  const Icon    = diff > 0 ? TrendingUp : TrendingDown
  const fmt     = v => isPercent ? `${(v*100).toFixed(1)}%` : v.toFixed(decimals)
  return (
    <span style={{ color, display: 'inline-flex', alignItems: 'center', gap: 4 }}>
      <Icon size={12} />
      {diff > 0 ? '+' : ''}{isPercent ? `${(diff*100).toFixed(1)}%` : diff.toFixed(decimals)}
    </span>
  )
}

export default function QuantumResults() {
  const { data: allMetrics } = useFetch('/api/models/metrics')
  const { data: entropy,  loading: el, error: ee } = useFetch('/api/quantum/entanglement')
  const { data: leadLag,  loading: ll, error: le } = useFetch('/api/quantum/lead-lag')

  const vqhM  = allMetrics?.vqh
  const lstmM = allMetrics?.lstm

  const vqhPrAuc    = vqhM?.crisis_specific?.pr_auc
  const lstmPrAuc   = lstmM?.crisis_specific?.pr_auc
  const vqhRecall   = vqhM?.per_class?.CRISIS?.recall
  const lstmRecall  = lstmM?.per_class?.CRISIS?.recall
  const vqhMf1      = vqhM?.overall?.macro_f1
  const lstmMf1     = lstmM?.overall?.macro_f1
  const timingLead  = vqhM?.quantum_metrics?.crisis_timing_lead_days
  const meanEntCris = vqhM?.quantum_metrics?.mean_entanglement_entropy_crisis?.toFixed(4)
  const meanEntNorm = vqhM?.quantum_metrics?.mean_entanglement_entropy_noncris?.toFixed(4)

  // Build entropy chart data
  const entropyData = entropy?.map(e => ({
    date:    e.date,
    entropy: e.entropy,
    crisis:  e.true_regime === 0 ? 1 : 0,
  })) ?? []

  // Lead-lag chart data
  const leadData = leadLag?.map(d => ({
    date:     d.date,
    vqh:      d.vqh_prob_crisis,
    lstm:     d.lstm_prob_crisis,
  })) ?? []

  const optThresh = vqhM?.crisis_specific?.optimal_threshold ?? 0.4

  return (
    <div className="fade-in">
      <div className="page-header">
        <h1 style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <Cpu size={20} color="var(--quantum)" strokeWidth={2} />
          Quantum Results
        </h1>
        <div className="page-subtitle">
          <span>Variational Quantum HMM</span>
          <span className="dot" />
          <span>3 qubits · 2 layers</span>
          <span className="dot" />
          <span>StronglyEntanglingLayers</span>
        </div>
      </div>

      {/* ── Section 1: Circuit Architecture Card ── */}
      <div className="card" style={{ marginBottom: '1.5rem', borderLeft: '3px solid var(--quantum)' }}>
        <div style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--quantum)', marginBottom: '1rem' }}>
          VQH Circuit Architecture
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table className="data-table" style={{ minWidth: 600 }}>
            <thead>
              <tr>
                <th>Component</th>
                <th>Implementation</th>
                <th>Financial Meaning</th>
              </tr>
            </thead>
            <tbody>
              {[
                ['AngleEmbedding', 'Ry(p × π) on each qubit', 'Encodes market regime uncertainty as qubit rotation on Bloch sphere'],
                ['CNOT(0→1)', 'BTC crisis qubit → SPX normal qubit', 'Models crypto-to-equity contagion channel (non-separable)'],
                ['CNOT(1→2)', 'SPX regime → Gold high-vol qubit', 'Models equity-to-gold flight-to-quality dynamics'],
                ['CNOT(2→0)', 'Gold qubit → BTC crisis qubit', 'Gold safe-haven rotation feeding back to BTC stress'],
                ['StronglyEntanglingLayers', '2 layers, all-to-all connectivity', 'Full cross-asset contagion modeling (not achievable classically)'],
                ['PauliZ measurement', '⟨Z_i⟩ ∈ [−1, +1] per qubit', 'Extracts next-step regime transition probabilities'],
              ].map(([comp, impl, fin]) => (
                <tr key={comp}>
                  <td style={{ color: 'var(--quantum)', fontWeight: 600, fontSize: '0.82rem', fontFamily: 'monospace' }}>{comp}</td>
                  <td className="font-mono" style={{ fontSize: '0.78rem' }}>{impl}</td>
                  <td style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>{fin}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Section 2: VQH vs Best Classical ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1.5rem' }}>
        {/* Best Classical (LSTM) */}
        <div className="card" style={{ borderLeft: '3px solid #3b82f6' }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.75rem' }}>
            LSTM (Best Classical)
          </div>
          {[
            ['Crisis PR-AUC', lstmPrAuc?.toFixed(4), null, null],
            ['Crisis Recall',  lstmRecall != null ? `${(lstmRecall*100).toFixed(1)}%` : '—', null, null],
            ['Macro F1',       lstmMf1?.toFixed(4), null, null],
            ['Timing Lead',    '0.0 days', null, null],
          ].map(([label, val]) => (
            <div key={label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 0', borderBottom: '1px solid rgba(71,85,105,0.2)' }}>
              <span style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>{label}</span>
              <span className="font-mono" style={{ color: '#3b82f6', fontWeight: 600, fontSize: '0.9rem' }}>{val ?? '—'}</span>
            </div>
          ))}
        </div>

        {/* VQH */}
        <div className="card" style={{ borderLeft: '3px solid var(--quantum)' }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--quantum)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: 6 }}>
            <Cpu size={12} /> VQH (Quantum)
          </div>
          {[
            ['Crisis PR-AUC', vqhPrAuc?.toFixed(4),  vqhPrAuc,  lstmPrAuc],
            ['Crisis Recall',  vqhRecall != null ? `${(vqhRecall*100).toFixed(1)}%` : '—', vqhRecall, lstmRecall],
            ['Macro F1',       vqhMf1?.toFixed(4),    vqhMf1,    lstmMf1],
            ['Timing Lead',    timingLead != null ? `${timingLead > 0 ? '+' : ''}${timingLead.toFixed(1)} days` : '—', timingLead, 0],
          ].map(([label, val, vqhN, lstmN]) => (
            <div key={label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 0', borderBottom: '1px solid rgba(71,85,105,0.2)' }}>
              <span style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>{label}</span>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 1 }}>
                <span className="font-mono" style={{ color: 'var(--quantum)', fontWeight: 600, fontSize: '0.9rem' }}>{val ?? '—'}</span>
                {vqhN != null && lstmN != null && (
                  <span style={{ fontSize: '0.7rem' }}>
                    <DeltaIndicator vqhVal={typeof vqhN === 'number' ? vqhN : 0} lstmVal={typeof lstmN === 'number' ? lstmN : 0} decimals={3} />
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Section 3: Entanglement Entropy ── */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <div style={{ fontSize: '0.875rem', fontWeight: 600 }}>
            Quantum Entanglement Entropy — Proxy for Systemic Risk
          </div>
          <div style={{ display: 'flex', gap: '1rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            {meanEntCris && <span>Crisis: <span className="font-mono" style={{ color: 'var(--crisis)' }}>{meanEntCris}</span></span>}
            {meanEntNorm && <span>Non-crisis: <span className="font-mono" style={{ color: 'var(--normal)' }}>{meanEntNorm}</span></span>}
          </div>
        </div>
        {el ? (
          <div className="skeleton" style={{ height: 200 }} />
        ) : ee ? (
          <div className="card" style={{ borderLeft: '3px solid var(--highvol)', color: 'var(--highvol)', fontSize: '0.85rem' }}>
            {ee} — Run scripts/run_phase3_qml.py first.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={entropyData} margin={{ top: 5, right: 40, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" strokeOpacity={0.3} />
              <XAxis dataKey="date" tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                tickFormatter={v => v?.slice(0,7)} interval={Math.floor(entropyData.length/8)}
                axisLine={{ stroke: 'var(--border)' }} tickLine={false} />
              <YAxis yAxisId="left" domain={[0, 1]}
                tick={{ fill: 'var(--text-muted)', fontSize: 11 }} axisLine={{ stroke: 'var(--border)' }}
                tickLine={false} width={35} />
              <YAxis yAxisId="right" orientation="right" domain={[0, 2]}
                tick={false} axisLine={false} tickLine={false} width={0} />
              <Tooltip
                formatter={(v, n) => [v.toFixed(4), n]}
                contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)',
                  borderRadius: 8, color: 'var(--text-primary)', fontSize: '0.8rem' }}
              />
              <Area yAxisId="left" type="monotone" dataKey="entropy"
                stroke="var(--quantum)" strokeWidth={2} fill="var(--quantum)" fillOpacity={0.12}
                name="Entropy" />
              <Area yAxisId="left" type="monotone" dataKey="entropy"
                fill="var(--quantum)" fillOpacity={0.25} stroke="none"
                activeDot={false}
                baseValue={0.5}
                name=""
              />
              <Area yAxisId="right" type="stepAfter" dataKey="crisis"
                fill="var(--crisis)" fillOpacity={0.12} stroke="none" name="Crisis" />
              <ReferenceLine yAxisId="left" y={0.5} stroke="rgba(139,92,246,0.4)"
                strokeDasharray="3 3" label={{ value: '0.5', position: 'right', fill: 'var(--quantum)', fontSize: 10 }} />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* ── Section 4: Lead/Lag Chart ── */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: '1rem' }}>
          Crisis Probability — VQH vs LSTM around SVB 2023
        </div>
        {ll ? (
          <div className="skeleton" style={{ height: 200 }} />
        ) : le ? (
          <div className="card" style={{ borderLeft: '3px solid var(--highvol)', color: 'var(--highvol)', fontSize: '0.85rem' }}>
            {le} — Run scripts/run_phase3_qml.py first.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={leadData} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" strokeOpacity={0.3} />
              <XAxis dataKey="date" tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                tickFormatter={v => v?.slice(5)} axisLine={{ stroke: 'var(--border)' }} tickLine={false} />
              <YAxis domain={[0, 1]} tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                axisLine={{ stroke: 'var(--border)' }} tickLine={false} width={35} />
              <Tooltip
                formatter={(v, n) => [v != null ? v.toFixed(4) : '—', n]}
                contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)',
                  borderRadius: 8, color: 'var(--text-primary)', fontSize: '0.8rem' }}
              />
              <ReferenceArea x1="2023-03-10" x2="2023-05-31" fill="rgba(239,68,68,0.08)" />
              <ReferenceLine x="2023-03-10" stroke="var(--crisis)" strokeDasharray="4 3"
                label={{ value: 'SVB closure', position: 'top', fill: 'var(--crisis)', fontSize: 10 }} />
              <ReferenceLine y={optThresh} stroke="rgba(148,163,184,0.4)" strokeDasharray="4 3"
                label={{ value: `t=${optThresh.toFixed(2)}`, position: 'right', fill: 'var(--text-muted)', fontSize: 10 }} />
              <Line type="monotone" dataKey="lstm" stroke="#3b82f6" strokeWidth={2}
                strokeDasharray="5 3" dot={false} name="LSTM" connectNulls />
              <Line type="monotone" dataKey="vqh" stroke="var(--quantum)" strokeWidth={2.5}
                dot={false} name="VQH" connectNulls />
              <Legend wrapperStyle={{ fontSize: '0.75rem', color: 'var(--text-muted)' }} />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* ── Section 5: Summary ── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
        <MetricCard
          label="Crisis PR-AUC Improvement"
          value={vqhPrAuc != null && lstmPrAuc != null
            ? `${((vqhPrAuc - lstmPrAuc) / lstmPrAuc * 100).toFixed(1)}%`
            : '—'}
          accent="var(--quantum)"
          subtitle="vs best classical (LSTM)"
        />
        <MetricCard
          label="Mean Entanglement (Crisis)"
          value={meanEntCris ?? '—'}
          accent="var(--crisis)"
          subtitle="Von Neumann entropy"
        />
        <MetricCard
          label="Crisis Timing Lead"
          value={timingLead != null ? `${timingLead > 0 ? '+' : ''}${timingLead.toFixed(1)} days` : '—'}
          accent={timingLead > 0 ? 'var(--quantum)' : 'var(--crisis)'}
          subtitle="vs LSTM advance warning"
          delta={timingLead}
        />
      </div>
    </div>
  )
}
