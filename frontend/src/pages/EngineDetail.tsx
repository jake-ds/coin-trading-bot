import { useState, useEffect, useCallback, useRef } from 'react'
import { useParams, Link } from 'react-router-dom'
import apiClient from '../api/client'
import { useWebSocket } from '../hooks/useWebSocket'
import type { EngineCycleLogEntry, EngineDecisionStep, EngineInfo } from '../api/types'

const WS_URL = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/ws`

const categoryColors: Record<string, string> = {
  execute: 'border-green-500 bg-green-900/20',
  decide: 'border-blue-500 bg-blue-900/20',
  skip: 'border-yellow-500 bg-yellow-900/20',
  evaluate: 'border-gray-500 bg-gray-800/40',
}

const categoryBadge: Record<string, string> = {
  execute: 'bg-green-600 text-white',
  decide: 'bg-blue-600 text-white',
  skip: 'bg-yellow-600 text-black',
  evaluate: 'bg-gray-600 text-white',
}

function DecisionCard({ step }: { step: EngineDecisionStep }) {
  const colors = categoryColors[step.category] || categoryColors.evaluate
  const badge = categoryBadge[step.category] || categoryBadge.evaluate

  return (
    <div className={`border-l-4 rounded p-3 mb-2 ${colors}`}>
      <div className="flex items-center gap-2 mb-1">
        <span className={`text-xs px-1.5 py-0.5 rounded font-medium uppercase ${badge}`}>
          {step.category}
        </span>
        <span className="font-medium text-sm">{step.label}</span>
      </div>
      <div className="text-xs space-y-0.5 text-gray-300">
        <p>
          <span className="text-gray-500">Observation:</span> {step.observation}
        </p>
        <p>
          <span className="text-gray-500">Threshold:</span> {step.threshold}
        </p>
        <p className="font-medium text-gray-100">
          <span className="text-gray-500">Result:</span> {step.result}
        </p>
      </div>
    </div>
  )
}

function CyclePanel({ cycle }: { cycle: EngineCycleLogEntry }) {
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-3 gap-3 text-sm">
        <div className="bg-gray-800 rounded p-3">
          <span className="text-gray-500 text-xs">Cycle</span>
          <p className="font-mono text-lg">#{cycle.cycle_num}</p>
        </div>
        <div className="bg-gray-800 rounded p-3">
          <span className="text-gray-500 text-xs">Duration</span>
          <p className="font-mono text-lg">{cycle.duration_ms.toFixed(0)}ms</p>
        </div>
        <div className="bg-gray-800 rounded p-3">
          <span className="text-gray-500 text-xs">PnL</span>
          <p
            className={`font-mono text-lg ${cycle.pnl_update >= 0 ? 'text-green-400' : 'text-red-400'}`}
          >
            {cycle.pnl_update >= 0 ? '+' : ''}
            {cycle.pnl_update.toFixed(4)}
          </p>
        </div>
      </div>

      <div>
        <h4 className="text-sm font-semibold mb-2 text-gray-300">
          Decisions ({cycle.decisions?.length || 0})
        </h4>
        {cycle.decisions && cycle.decisions.length > 0 ? (
          cycle.decisions.map((d, i) => <DecisionCard key={i} step={d} />)
        ) : (
          <p className="text-gray-500 text-sm">No decisions recorded</p>
        )}
      </div>

      {cycle.actions_taken.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold mb-2 text-gray-300">
            Actions ({cycle.actions_taken.length})
          </h4>
          <div className="space-y-1">
            {cycle.actions_taken.map((action, i) => (
              <div key={i} className="bg-gray-800 rounded p-2 text-xs font-mono">
                {JSON.stringify(action)}
              </div>
            ))}
          </div>
        </div>
      )}

      {cycle.signals.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold mb-2 text-gray-300">
            Signals ({cycle.signals.length})
          </h4>
          <div className="space-y-1">
            {cycle.signals.map((sig, i) => (
              <div key={i} className="bg-gray-800 rounded p-2 text-xs font-mono">
                {JSON.stringify(sig)}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function EngineDetail() {
  const { name } = useParams<{ name: string }>()
  const [engine, setEngine] = useState<EngineInfo | null>(null)
  const [cycles, setCycles] = useState<EngineCycleLogEntry[]>([])
  const [selectedIndex, setSelectedIndex] = useState<number>(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const cycleListRef = useRef<HTMLDivElement>(null)

  const { data: wsMessage } = useWebSocket(WS_URL)

  const fetchData = useCallback(async () => {
    if (!name) return
    try {
      const [enginesRes, cycleRes] = await Promise.all([
        apiClient.get(`/engines`),
        apiClient.get(`/engines/${name}/cycle-log`),
      ])
      const info = enginesRes.data[name]
      if (info) setEngine(info)
      const log = cycleRes.data.cycle_log || []
      setCycles(log)
      if (log.length > 0) setSelectedIndex(log.length - 1)
      setError(null)
    } catch {
      setError('Failed to fetch engine data')
    } finally {
      setLoading(false)
    }
  }, [name])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  // Handle real-time WebSocket updates
  useEffect(() => {
    if (
      wsMessage?.type === 'engine_cycle' &&
      wsMessage.payload?.engine_name === name
    ) {
      const newCycle = wsMessage.payload as unknown as EngineCycleLogEntry
      setCycles((prev) => {
        const updated = [...prev, newCycle]
        // Keep last 50
        return updated.length > 50 ? updated.slice(-50) : updated
      })
      setSelectedIndex((prev) => prev + 1)
    }
  }, [wsMessage, name])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-400">Loading engine detail...</div>
      </div>
    )
  }

  if (error || !engine) {
    return (
      <div className="space-y-4">
        <Link to="/engines" className="text-blue-400 hover:text-blue-300 text-sm">
          &larr; Back to Engines
        </Link>
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <p className="text-yellow-400">{error || 'Engine not found'}</p>
        </div>
      </div>
    )
  }

  const selectedCycle = cycles[selectedIndex] || null

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Link to="/engines" className="text-blue-400 hover:text-blue-300 text-sm">
            &larr; Engines
          </Link>
          <h2 className="text-2xl font-bold">{engine.name}</h2>
          <span className="text-xs px-2 py-0.5 rounded bg-gray-700 text-gray-300 uppercase">
            {engine.status}
          </span>
        </div>
        <span className="text-sm text-gray-400">
          {cycles.length} cycles loaded
        </span>
      </div>

      <p className="text-sm text-gray-400">{engine.description}</p>

      <div className="flex gap-4" style={{ height: 'calc(100vh - 220px)' }}>
        {/* Left: Cycle list */}
        <div
          ref={cycleListRef}
          className="w-64 flex-shrink-0 overflow-y-auto bg-gray-800 rounded-lg border border-gray-700"
        >
          {cycles.length === 0 ? (
            <div className="p-4 text-gray-500 text-sm text-center">
              No cycles yet
            </div>
          ) : (
            cycles.map((c, i) => (
              <button
                key={`${c.cycle_num}-${c.timestamp}`}
                className={`w-full text-left px-3 py-2 border-b border-gray-700 text-sm transition-colors ${
                  i === selectedIndex
                    ? 'bg-blue-900/40 border-l-2 border-l-blue-500'
                    : 'hover:bg-gray-700/50'
                }`}
                onClick={() => setSelectedIndex(i)}
              >
                <div className="flex justify-between items-center">
                  <span className="font-mono">#{c.cycle_num}</span>
                  <span
                    className={`text-xs ${c.pnl_update >= 0 ? 'text-green-400' : 'text-red-400'}`}
                  >
                    {c.pnl_update >= 0 ? '+' : ''}
                    {c.pnl_update.toFixed(4)}
                  </span>
                </div>
                <div className="text-xs text-gray-500 mt-0.5">
                  {new Date(c.timestamp).toLocaleTimeString()}
                  {c.decisions && c.decisions.length > 0 && (
                    <span className="ml-2 text-gray-400">
                      {c.decisions.length} decisions
                    </span>
                  )}
                </div>
              </button>
            ))
          )}
        </div>

        {/* Right: Detail panel */}
        <div className="flex-1 overflow-y-auto bg-gray-800 rounded-lg border border-gray-700 p-4">
          {selectedCycle ? (
            <CyclePanel cycle={selectedCycle} />
          ) : (
            <div className="flex items-center justify-center h-full text-gray-500">
              Select a cycle to view details
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default EngineDetail
