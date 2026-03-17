import { useEffect, useState, useCallback } from 'react'
import apiClient from '../api/client'
import type { CompositeSignal, SignalHistoryEntry, WsStatusPayload } from '../api/types'
import { useWebSocket } from '../hooks/useWebSocket'
import SignalGauge from '../components/signals/SignalGauge'
import SignalBreakdown from '../components/signals/SignalBreakdown'
import SignalHistoryChart from '../components/signals/SignalHistoryChart'

const WS_URL = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/ws`

function Signals() {
  const [signals, setSignals] = useState<Record<string, CompositeSignal>>({})
  const [history, setHistory] = useState<SignalHistoryEntry[]>([])
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const { data: wsMessage } = useWebSocket(WS_URL)

  // Apply WebSocket updates for signals
  useEffect(() => {
    if (wsMessage?.type === 'status_update') {
      const p = wsMessage.payload as unknown as WsStatusPayload
      if (p.onchain_signals && Object.keys(p.onchain_signals).length > 0) {
        setSignals(p.onchain_signals)
        setLoading(false)
      }
    }
  }, [wsMessage])

  // Fetch initial data
  const fetchData = useCallback(async () => {
    try {
      const [signalsRes, historyRes] = await Promise.allSettled([
        apiClient.get('/onchain-signals'),
        apiClient.get('/signals/history', { params: { hours: 24 } }),
      ])

      if (signalsRes.status === 'fulfilled' && signalsRes.value.data.signals) {
        setSignals(signalsRes.value.data.signals)
      }
      if (historyRes.status === 'fulfilled' && historyRes.value.data.history) {
        setHistory(historyRes.value.data.history)
      }
    } catch {
      // silently fail
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchData()
    // Refresh history every 5 minutes
    const interval = setInterval(fetchData, 300000)
    return () => clearInterval(interval)
  }, [fetchData])

  // Auto-select first symbol
  const symbolKeys = Object.keys(signals)
  useEffect(() => {
    if (!selectedSymbol && symbolKeys.length > 0) {
      setSelectedSymbol(symbolKeys[0])
    }
  }, [symbolKeys, selectedSymbol])

  if (loading) {
    return (
      <div>
        <h2 className="text-2xl font-bold mb-4">Signals</h2>
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700 animate-pulse">
          <div className="h-8 bg-gray-700 rounded mb-4 w-1/3" />
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-12 bg-gray-700 rounded" />
            ))}
          </div>
        </div>
      </div>
    )
  }

  if (symbolKeys.length === 0) {
    return (
      <div>
        <h2 className="text-2xl font-bold mb-4">Signals</h2>
        <div className="bg-gray-800 rounded-lg p-8 border border-gray-700 text-center text-gray-400">
          No signal data available. Signals appear after the onchain trader engine runs its first cycle.
        </div>
      </div>
    )
  }

  const activeSignal = selectedSymbol ? signals[selectedSymbol] : null

  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Signals</h2>

      {/* Symbol tabs */}
      <div className="flex gap-2 mb-6 overflow-x-auto">
        {symbolKeys.map((symbol) => {
          const sig = signals[symbol]
          const displaySymbol = symbol.replace('/USDT', '').replace(':USDT', '')
          const isActive = symbol === selectedSymbol
          return (
            <button
              key={symbol}
              onClick={() => setSelectedSymbol(symbol)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors whitespace-nowrap ${
                isActive
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-800 text-gray-300 hover:bg-gray-700 border border-gray-700'
              }`}
            >
              <span>{displaySymbol}</span>
              <span className={`text-xs font-bold ${
                sig.action === 'BUY' ? 'text-green-400' :
                sig.action === 'SELL' ? 'text-red-400' : 'text-gray-400'
              }`}>
                {sig.score > 0 ? '+' : ''}{sig.score.toFixed(0)}
              </span>
            </button>
          )
        })}
      </div>

      {activeSignal && selectedSymbol && (
        <>
          {/* Large signal gauge */}
          <div className="mb-6">
            <SignalGauge symbol={selectedSymbol} signal={activeSignal} />
          </div>

          {/* Signal breakdown table */}
          <div className="mb-6">
            <h3 className="text-sm font-semibold text-gray-400 mb-3">Signal Breakdown</h3>
            <SignalBreakdown signals={activeSignal.signals} />
          </div>

          {/* History chart */}
          <div>
            <h3 className="text-sm font-semibold text-gray-400 mb-3">Signal History (24h)</h3>
            <SignalHistoryChart history={history} symbol={selectedSymbol} />
          </div>
        </>
      )}
    </div>
  )
}

export default Signals
