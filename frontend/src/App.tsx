import { useState, useEffect, useCallback } from 'react'
import { Routes, Route, Link, useLocation } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import Trades from './pages/Trades'
import Positions from './pages/Positions'
import Strategies from './pages/Strategies'
import Settings from './pages/Settings'
import Login from './pages/Login'
import ConnectionIndicator from './components/common/ConnectionIndicator'
import Toast, { type ToastMessage } from './components/common/Toast'
import { useWebSocket } from './hooks/useWebSocket'
import type { Trade } from './api/types'

const navItems = [
  { path: '/', label: 'Dashboard' },
  { path: '/positions', label: 'Positions' },
  { path: '/trades', label: 'Trades' },
  { path: '/strategies', label: 'Strategies' },
  { path: '/settings', label: 'Settings' },
]

const WS_URL = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/ws`

function App() {
  const location = useLocation()
  const { connected, data: wsMessage } = useWebSocket(WS_URL)
  const [toasts, setToasts] = useState<ToastMessage[]>([])

  const dismissToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id))
  }, [])

  // Show toast on new trade events
  useEffect(() => {
    if (wsMessage?.type === 'trade') {
      const trade = wsMessage.payload as unknown as Trade
      const side = trade.side || 'TRADE'
      const symbol = trade.symbol || 'Unknown'
      const price = trade.price ? `$${trade.price.toLocaleString()}` : ''
      const toast: ToastMessage = {
        id: `trade-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        message: `${side} ${symbol} ${price}`.trim(),
        type: side === 'BUY' ? 'success' : 'info',
      }
      setToasts((prev) => [...prev.slice(-4), toast])
    }
  }, [wsMessage])

  if (location.pathname === '/login') {
    return (
      <Routes>
        <Route path="/login" element={<Login />} />
      </Routes>
    )
  }

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      <Toast toasts={toasts} onDismiss={dismissToast} />
      <nav className="bg-gray-800 border-b border-gray-700 px-6 py-3">
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-bold text-white">Trading Bot</h1>
            <ConnectionIndicator connected={connected} />
          </div>
          <div className="flex gap-4">
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`px-3 py-1.5 rounded text-sm transition-colors ${
                  location.pathname === item.path
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-300 hover:text-white hover:bg-gray-700'
                }`}
              >
                {item.label}
              </Link>
            ))}
          </div>
        </div>
      </nav>
      <main className="max-w-7xl mx-auto px-6 py-6">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/positions" element={<Positions />} />
          <Route path="/trades" element={<Trades />} />
          <Route path="/strategies" element={<Strategies />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </main>
    </div>
  )
}

export default App
