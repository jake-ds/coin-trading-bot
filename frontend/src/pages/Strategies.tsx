import { useState, useEffect, useCallback } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  CartesianGrid,
} from 'recharts'
import apiClient from '../api/client'
import { useWebSocket } from '../hooks/useWebSocket'
import type {
  StrategiesResponse,
  StrategyStats,
  ToggleResponse,
  WsStatusPayload,
} from '../api/types'

const WS_URL = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/ws`

function Strategies() {
  const [strategies, setStrategies] = useState<Record<string, StrategyStats>>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [confirmDialog, setConfirmDialog] = useState<{
    name: string
    positionCount: number
  } | null>(null)
  const [togglingStrategy, setTogglingStrategy] = useState<string | null>(null)

  const { connected, data: wsMessage } = useWebSocket(WS_URL)

  // Initial fetch
  const fetchStrategies = useCallback(async () => {
    try {
      const res = await apiClient.get<StrategiesResponse>('/strategies')
      setStrategies(res.data.strategies)
      setError(null)
    } catch {
      setError('Failed to load strategies')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchStrategies()
  }, [fetchStrategies])

  // WebSocket updates
  useEffect(() => {
    if (wsMessage?.type === 'status_update') {
      const payload = wsMessage.payload as unknown as WsStatusPayload
      if (payload.strategy_stats) {
        setStrategies(payload.strategy_stats)
      }
    }
  }, [wsMessage])

  // Polling fallback when WS disconnected
  useEffect(() => {
    if (connected) return
    const interval = setInterval(fetchStrategies, 10000)
    return () => clearInterval(interval)
  }, [connected, fetchStrategies])

  const handleToggle = useCallback(
    async (name: string, force = false) => {
      setTogglingStrategy(name)
      try {
        const url = force
          ? `/strategies/${name}/toggle?force=true`
          : `/strategies/${name}/toggle`
        const res = await apiClient.post<ToggleResponse>(url)

        if (res.data.has_open_positions && !res.data.success) {
          setConfirmDialog({
            name,
            positionCount: res.data.open_position_count || 0,
          })
          return
        }

        if (res.data.success) {
          // Optimistic update
          setStrategies((prev) => ({
            ...prev,
            [name]: { ...prev[name], active: res.data.active },
          }))
        }
      } catch {
        setError(`Failed to toggle strategy: ${name}`)
      } finally {
        setTogglingStrategy(null)
      }
    },
    []
  )

  const handleForceDisable = useCallback(async () => {
    if (confirmDialog) {
      await handleToggle(confirmDialog.name, true)
      setConfirmDialog(null)
    }
  }, [confirmDialog, handleToggle])

  const strategyEntries = Object.entries(strategies)

  // Bar chart data
  const comparisonData = strategyEntries.map(([name, stats]) => ({
    name,
    pnl: stats.total_pnl,
    fill: stats.total_pnl >= 0 ? '#22c55e' : '#ef4444',
  }))

  if (loading) {
    return (
      <div>
        <h2 className="text-2xl font-bold mb-6">Strategies</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <div
              key={i}
              className="bg-gray-800 rounded-lg border border-gray-700 p-6 animate-pulse"
            >
              <div className="h-6 bg-gray-700 rounded w-1/3 mb-4" />
              <div className="h-4 bg-gray-700 rounded w-2/3 mb-2" />
              <div className="h-4 bg-gray-700 rounded w-1/2 mb-2" />
              <div className="h-32 bg-gray-700 rounded mt-4" />
            </div>
          ))}
        </div>
      </div>
    )
  }

  if (error && strategyEntries.length === 0) {
    return (
      <div>
        <h2 className="text-2xl font-bold mb-6">Strategies</h2>
        <div className="bg-red-900/30 border border-red-700 rounded-lg p-4 text-red-300">
          {error}
        </div>
      </div>
    )
  }

  return (
    <div>
      <h2 className="text-2xl font-bold mb-6">Strategies</h2>

      {error && (
        <div className="bg-red-900/30 border border-red-700 rounded-lg p-3 mb-4 text-red-300 text-sm">
          {error}
        </div>
      )}

      {/* Confirmation Dialog */}
      {confirmDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-gray-800 border border-gray-600 rounded-lg p-6 max-w-md mx-4">
            <h3 className="text-lg font-bold text-yellow-400 mb-3">
              Open Positions Warning
            </h3>
            <p className="text-gray-300 mb-4">
              Strategy <strong>{confirmDialog.name}</strong> has{' '}
              <strong>{confirmDialog.positionCount}</strong> open position
              {confirmDialog.positionCount > 1 ? 's' : ''}. Disabling it will
              not close existing positions, but no new trades will be opened.
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setConfirmDialog(null)}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleForceDisable}
                className="px-4 py-2 bg-red-600 hover:bg-red-500 rounded text-sm transition-colors"
              >
                Disable Anyway
              </button>
            </div>
          </div>
        </div>
      )}

      {strategyEntries.length === 0 ? (
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-8 text-center text-gray-400">
          No strategies registered yet.
        </div>
      ) : (
        <>
          {/* Strategy Comparison Chart */}
          {comparisonData.length > 0 && (
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-6 mb-6">
              <h3 className="text-lg font-semibold mb-4">
                Strategy Comparison — Total PnL
              </h3>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={comparisonData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis
                    dataKey="name"
                    tick={{ fill: '#9ca3af', fontSize: 12 }}
                    axisLine={{ stroke: '#4b5563' }}
                  />
                  <YAxis
                    tick={{ fill: '#9ca3af', fontSize: 12 }}
                    axisLine={{ stroke: '#4b5563' }}
                    tickFormatter={(v: number) => `$${v.toFixed(0)}`}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1f2937',
                      border: '1px solid #4b5563',
                      borderRadius: '8px',
                    }}
                    labelStyle={{ color: '#e5e7eb' }}
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    formatter={(value: any) => [`$${Number(value).toFixed(2)}`, 'PnL']}
                  />
                  <Bar dataKey="pnl" radius={[4, 4, 0, 0]}>
                    {comparisonData.map((entry, i) => (
                      <rect key={i} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Strategy Cards Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {strategyEntries.map(([name, stats]) => (
              <StrategyCard
                key={name}
                name={name}
                stats={stats}
                toggling={togglingStrategy === name}
                onToggle={handleToggle}
              />
            ))}
          </div>
        </>
      )}
    </div>
  )
}

function StrategyCard({
  name,
  stats,
  toggling,
  onToggle,
}: {
  name: string
  stats: StrategyStats
  toggling: boolean
  onToggle: (name: string) => void
}) {
  const isActive = stats.active
  const isAutoDisabled = stats.disabled

  // Mini chart data from pnl_history
  const pnlHistory = stats.pnl_history || []
  const chartData = pnlHistory.map((_val, i) => {
    const cumPnl = pnlHistory.slice(0, i + 1).reduce((sum, p) => sum + p, 0)
    return { trade: i + 1, pnl: cumPnl }
  })

  // Determine auto-disable reason display
  let disableReason = ''
  if (isAutoDisabled && stats.disabled_reason) {
    if (stats.disabled_reason.includes('consecutive_losses')) {
      disableReason = `${stats.consecutive_losses} consecutive losses`
    } else if (stats.disabled_reason.includes('low_win_rate')) {
      disableReason = `Win rate ${stats.win_rate.toFixed(1)}% below threshold`
    } else {
      disableReason = stats.disabled_reason
    }
  }

  return (
    <div
      className={`bg-gray-800 rounded-lg border p-5 ${
        isAutoDisabled
          ? 'border-yellow-600/50'
          : isActive
          ? 'border-gray-700'
          : 'border-gray-700 opacity-75'
      }`}
    >
      {/* Header: name + toggle */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <h3 className="text-lg font-semibold">{name}</h3>
          {isAutoDisabled && (
            <span className="text-xs px-2 py-0.5 bg-yellow-600/20 text-yellow-400 rounded-full border border-yellow-600/40">
              Auto-disabled
            </span>
          )}
        </div>
        <button
          onClick={() => onToggle(name)}
          disabled={toggling}
          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none ${
            toggling
              ? 'bg-gray-600 cursor-wait'
              : isActive
              ? 'bg-green-600 hover:bg-green-500'
              : 'bg-gray-600 hover:bg-gray-500'
          }`}
          title={isActive ? 'Click to disable' : 'Click to enable'}
        >
          <span
            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
              isActive ? 'translate-x-6' : 'translate-x-1'
            }`}
          />
        </button>
      </div>

      {/* Auto-disable warning */}
      {isAutoDisabled && disableReason && (
        <div className="text-xs text-yellow-400 bg-yellow-600/10 border border-yellow-600/30 rounded px-3 py-1.5 mb-3">
          {disableReason}
        </div>
      )}

      {/* Stats grid */}
      <div className="grid grid-cols-3 gap-3 mb-4">
        <StatValue
          label="Total PnL"
          value={`$${stats.total_pnl.toFixed(2)}`}
          color={stats.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}
        />
        <StatValue
          label="Win Rate"
          value={`${stats.win_rate.toFixed(1)}%`}
          color={stats.win_rate >= 50 ? 'text-green-400' : 'text-red-400'}
        />
        <StatValue label="Trades" value={String(stats.total_trades)} />
        <StatValue
          label="Consec. Losses"
          value={String(stats.consecutive_losses)}
          color={stats.consecutive_losses >= 3 ? 'text-yellow-400' : undefined}
        />
        <StatValue
          label="Sharpe"
          value={stats.sharpe_ratio.toFixed(2)}
          color={
            stats.sharpe_ratio > 0
              ? 'text-green-400'
              : stats.sharpe_ratio < 0
              ? 'text-red-400'
              : undefined
          }
        />
        <StatValue
          label="Profit Factor"
          value={stats.profit_factor.toFixed(2)}
          color={stats.profit_factor > 1 ? 'text-green-400' : undefined}
        />
      </div>

      {/* Mini PnL Chart */}
      {chartData.length > 1 && (
        <div className="mt-2">
          <p className="text-xs text-gray-500 mb-1">
            Cumulative PnL (last {chartData.length} trades)
          </p>
          <ResponsiveContainer width="100%" height={100}>
            <LineChart data={chartData}>
              <Line
                type="monotone"
                dataKey="pnl"
                stroke={
                  chartData[chartData.length - 1].pnl >= 0
                    ? '#22c55e'
                    : '#ef4444'
                }
                strokeWidth={1.5}
                dot={false}
              />
              <XAxis dataKey="trade" hide />
              <YAxis hide domain={['auto', 'auto']} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #4b5563',
                  borderRadius: '6px',
                  fontSize: '12px',
                }}
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                formatter={(value: any) => [
                  `$${Number(value).toFixed(2)}`,
                  'Cumulative PnL',
                ]}
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                labelFormatter={(label: any) => `Trade #${label}`}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}

function StatValue({
  label,
  value,
  color,
}: {
  label: string
  value: string
  color?: string
}) {
  return (
    <div>
      <p className="text-xs text-gray-500">{label}</p>
      <p className={`text-sm font-semibold ${color || 'text-gray-200'}`}>
        {value}
      </p>
    </div>
  )
}

export default Strategies
