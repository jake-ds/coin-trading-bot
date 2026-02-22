import { useState, useEffect, useCallback } from 'react'
import apiClient from '../api/client'
import { useWebSocket } from '../hooks/useWebSocket'
import type {
  CycleLogEntry,
  CycleLogResponse,
  CycleSymbolDetail,
  StrategySignalEntry,
  WsStatusPayload,
} from '../api/types'

const WS_URL = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/ws`

function CycleLog() {
  const [cycleLog, setCycleLog] = useState<CycleLogEntry[]>([])
  const [selectedCycle, setSelectedCycle] = useState<CycleLogEntry | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const { connected, data: wsMessage } = useWebSocket(WS_URL)

  const fetchCycleLog = useCallback(async () => {
    try {
      const res = await apiClient.get<CycleLogResponse>('/cycle-log')
      const log = res.data.cycle_log
      setCycleLog(log)
      if (log.length > 0) {
        setSelectedCycle(log[log.length - 1])
      }
      setError(null)
    } catch {
      setError('Failed to load cycle log')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchCycleLog()
  }, [fetchCycleLog])

  // WebSocket updates — push new latest cycle
  useEffect(() => {
    if (wsMessage?.type === 'status_update') {
      const payload = wsMessage.payload as unknown as WsStatusPayload
      if (payload.cycle_log_latest) {
        const latest = payload.cycle_log_latest
        setCycleLog((prev) => {
          const exists = prev.some(
            (c) => c.cycle_num === latest.cycle_num
          )
          if (exists) return prev
          const updated = [...prev, latest]
          if (updated.length > 50) updated.shift()
          return updated
        })
        // Auto-select latest if user is viewing the most recent
        setSelectedCycle((prev) => {
          if (!prev || prev.cycle_num === latest.cycle_num - 1) {
            return latest
          }
          return prev
        })
      }
    }
  }, [wsMessage])

  // Polling fallback
  useEffect(() => {
    if (connected) return
    const interval = setInterval(fetchCycleLog, 10000)
    return () => clearInterval(interval)
  }, [connected, fetchCycleLog])

  if (loading) {
    return (
      <div>
        <h2 className="text-2xl font-bold mb-6">Cycle Log</h2>
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-8 animate-pulse">
          <div className="h-6 bg-gray-700 rounded w-1/3 mb-4" />
          <div className="h-4 bg-gray-700 rounded w-2/3 mb-2" />
          <div className="h-4 bg-gray-700 rounded w-1/2" />
        </div>
      </div>
    )
  }

  if (error && cycleLog.length === 0) {
    return (
      <div>
        <h2 className="text-2xl font-bold mb-6">Cycle Log</h2>
        <div className="bg-red-900/30 border border-red-700 rounded-lg p-4 text-red-300">
          {error}
        </div>
      </div>
    )
  }

  return (
    <div>
      <h2 className="text-2xl font-bold mb-6">Cycle Log</h2>

      {error && (
        <div className="bg-red-900/30 border border-red-700 rounded-lg p-3 mb-4 text-red-300 text-sm">
          {error}
        </div>
      )}

      {cycleLog.length === 0 ? (
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-8 text-center text-gray-400">
          No cycle data yet. Waiting for bot to complete cycles...
        </div>
      ) : (
        <div className="grid grid-cols-12 gap-4">
          {/* Cycle history sidebar */}
          <div className="col-span-3">
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-3">
              <h3 className="text-sm font-semibold text-gray-400 mb-3">
                History ({cycleLog.length})
              </h3>
              <div className="space-y-1 max-h-[calc(100vh-240px)] overflow-y-auto">
                {[...cycleLog].reverse().map((entry) => (
                  <CycleHistoryItem
                    key={entry.cycle_num}
                    entry={entry}
                    isSelected={selectedCycle?.cycle_num === entry.cycle_num}
                    onClick={() => setSelectedCycle(entry)}
                  />
                ))}
              </div>
            </div>
          </div>

          {/* Cycle detail */}
          <div className="col-span-9">
            {selectedCycle ? (
              <CycleDetail entry={selectedCycle} />
            ) : (
              <div className="bg-gray-800 rounded-lg border border-gray-700 p-8 text-center text-gray-400">
                Select a cycle to view details
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function CycleHistoryItem({
  entry,
  isSelected,
  onClick,
}: {
  entry: CycleLogEntry
  isSelected: boolean
  onClick: () => void
}) {
  const hasAction = Object.values(entry.symbols).some(
    (s) => s.final_action !== 'HOLD'
  )
  const symbolCount = Object.keys(entry.symbols).length

  return (
    <button
      onClick={onClick}
      className={`w-full text-left px-3 py-2 rounded text-sm transition-colors ${
        isSelected
          ? 'bg-blue-600/30 border border-blue-500/50 text-white'
          : 'hover:bg-gray-700 text-gray-300'
      }`}
    >
      <div className="flex items-center justify-between">
        <span className="font-medium">#{entry.cycle_num}</span>
        {hasAction && (
          <span className="w-2 h-2 rounded-full bg-green-400" />
        )}
      </div>
      <div className="text-xs text-gray-500 mt-0.5">
        {new Date(entry.timestamp).toLocaleTimeString()} &middot;{' '}
        {entry.duration_ms != null ? `${entry.duration_ms}ms` : '—'} &middot;{' '}
        {symbolCount} symbol{symbolCount !== 1 ? 's' : ''}
      </div>
    </button>
  )
}

function CycleDetail({ entry }: { entry: CycleLogEntry }) {
  const symbols = Object.entries(entry.symbols)

  return (
    <div className="space-y-4">
      {/* Cycle header */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold">
            Cycle #{entry.cycle_num}
          </h3>
          <div className="flex items-center gap-4 text-sm text-gray-400">
            <span>{new Date(entry.timestamp).toLocaleString()}</span>
            {entry.duration_ms != null && (
              <span className="px-2 py-0.5 bg-gray-700 rounded">
                {entry.duration_ms}ms
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Per-symbol breakdown */}
      {symbols.map(([symbol, detail]) => (
        <SymbolBreakdown key={symbol} symbol={symbol} detail={detail} />
      ))}
    </div>
  )
}

function SymbolBreakdown({
  symbol,
  detail,
}: {
  symbol: string
  detail: CycleSymbolDetail
}) {
  const [expanded, setExpanded] = useState(true)

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700">
      {/* Symbol header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between p-4 hover:bg-gray-750 transition-colors"
      >
        <div className="flex items-center gap-3">
          <span className="font-semibold text-white">{symbol}</span>
          <span className="text-sm text-gray-400">
            ${detail.price.toLocaleString(undefined, { maximumFractionDigits: 2 })}
          </span>
          {detail.regime && <RegimeBadge regime={detail.regime} />}
          {detail.trend && (
            <span className="text-xs px-2 py-0.5 rounded bg-gray-700 text-gray-300">
              {detail.trend}
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          <ActionBadge action={detail.final_action} />
          <span className="text-gray-500 text-sm">{expanded ? '—' : '+'}</span>
        </div>
      </button>

      {expanded && (
        <div className="px-4 pb-4 space-y-4">
          {/* Strategy signals table */}
          <div>
            <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
              Strategy Signals
            </h4>
            {detail.strategies.length === 0 ? (
              <p className="text-sm text-gray-500">No signals collected</p>
            ) : (
              <div className="overflow-hidden rounded border border-gray-700">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-gray-750">
                      <th className="text-left px-3 py-1.5 text-gray-400 font-medium">
                        Strategy
                      </th>
                      <th className="text-left px-3 py-1.5 text-gray-400 font-medium">
                        Action
                      </th>
                      <th className="text-left px-3 py-1.5 text-gray-400 font-medium">
                        Confidence
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {detail.strategies.map((s, i) => (
                      <StrategyRow key={i} signal={s} />
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Ensemble result */}
          <div>
            <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
              Ensemble Vote
            </h4>
            <div className="bg-gray-900/50 rounded border border-gray-700 p-3">
              <div className="flex items-center gap-4">
                <ActionBadge action={detail.ensemble.action} />
                <div className="text-sm">
                  <span className="text-gray-400">Confidence: </span>
                  <span className="text-white">
                    {(detail.ensemble.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="text-sm">
                  <span className="text-gray-400">Agreement: </span>
                  <span className="text-white">
                    {detail.ensemble.agreement}/{detail.strategies.length}
                  </span>
                </div>
              </div>
              {detail.ensemble.agreeing_strategies.length > 0 && (
                <div className="mt-2 text-xs text-gray-400">
                  Agreeing:{' '}
                  {detail.ensemble.agreeing_strategies.join(', ')}
                </div>
              )}
              {detail.ensemble.reason && (
                <div className="mt-2 text-xs text-yellow-400">
                  Reason: {detail.ensemble.reason}
                </div>
              )}
            </div>
          </div>

          {/* Risk check */}
          {detail.risk_check && (
            <div>
              <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
                Risk Check
              </h4>
              <div
                className={`rounded border p-3 text-sm ${
                  detail.risk_check.passed
                    ? 'bg-green-900/20 border-green-700/50 text-green-300'
                    : 'bg-red-900/20 border-red-700/50 text-red-300'
                }`}
              >
                <div className="flex items-center gap-2">
                  <span>
                    {detail.risk_check.passed ? 'Passed' : 'Rejected'}
                  </span>
                  <span className="text-xs px-1.5 py-0.5 rounded bg-gray-700/50 text-gray-400">
                    {detail.risk_check.stage}
                  </span>
                </div>
                {detail.risk_check.reason && (
                  <p className="mt-1 text-xs opacity-80">
                    {detail.risk_check.reason}
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Order details */}
          {detail.order && (
            <div>
              <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
                Executed Order
              </h4>
              <div className="bg-blue-900/20 border border-blue-700/50 rounded p-3 text-sm">
                <div className="flex items-center gap-4">
                  <ActionBadge action={detail.order.side} />
                  <span className="text-gray-300">
                    {detail.order.quantity} @ $
                    {detail.order.price.toLocaleString(undefined, {
                      maximumFractionDigits: 2,
                    })}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function StrategyRow({ signal }: { signal: StrategySignalEntry }) {
  return (
    <tr className="border-t border-gray-700/50">
      <td className="px-3 py-1.5 text-gray-200">{signal.name}</td>
      <td className="px-3 py-1.5">
        <ActionBadge action={signal.action} small />
      </td>
      <td className="px-3 py-1.5">
        <div className="flex items-center gap-2">
          <div className="flex-1 h-1.5 bg-gray-700 rounded-full max-w-[100px]">
            <div
              className="h-full rounded-full bg-blue-500"
              style={{ width: `${Math.min(signal.confidence * 100, 100)}%` }}
            />
          </div>
          <span className="text-xs text-gray-400">
            {(signal.confidence * 100).toFixed(1)}%
          </span>
        </div>
      </td>
    </tr>
  )
}

function ActionBadge({
  action,
  small = false,
}: {
  action: string
  small?: boolean
}) {
  const colors: Record<string, string> = {
    BUY: 'bg-green-600/30 text-green-400 border-green-600/40',
    SELL: 'bg-red-600/30 text-red-400 border-red-600/40',
    HOLD: 'bg-gray-600/30 text-gray-400 border-gray-600/40',
  }
  const cls = colors[action] || colors.HOLD

  return (
    <span
      className={`inline-block border rounded font-medium ${cls} ${
        small ? 'text-xs px-1.5 py-0.5' : 'text-sm px-2 py-0.5'
      }`}
    >
      {action}
    </span>
  )
}

function RegimeBadge({ regime }: { regime: string }) {
  const colors: Record<string, string> = {
    TRENDING_UP: 'bg-green-600/20 text-green-400 border-green-600/40',
    TRENDING_DOWN: 'bg-red-600/20 text-red-400 border-red-600/40',
    RANGING: 'bg-yellow-600/20 text-yellow-400 border-yellow-600/40',
    HIGH_VOLATILITY: 'bg-purple-600/20 text-purple-400 border-purple-600/40',
  }
  const cls = colors[regime] || 'bg-gray-600/20 text-gray-400 border-gray-600/40'

  return (
    <span className={`text-xs px-2 py-0.5 rounded border ${cls}`}>
      {regime}
    </span>
  )
}

export default CycleLog
