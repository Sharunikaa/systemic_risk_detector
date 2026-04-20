import React from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import Overview from './pages/Overview'
import ModelComparison from './pages/ModelComparison'
import QuantumResults from './pages/QuantumResults'
import Predictions from './pages/Predictions'

export default function App() {
  return (
    <BrowserRouter>
      <div style={{ display: 'flex', minHeight: '100vh', background: 'var(--bg-base)' }}>
        <Sidebar />
        <main style={{ flex: 1, overflow: 'auto', padding: '1.5rem' }}>
          <Routes>
            <Route path="/"           element={<Overview />} />
            <Route path="/comparison" element={<ModelComparison />} />
            <Route path="/quantum"    element={<QuantumResults />} />
            <Route path="/predictions" element={<Predictions />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
