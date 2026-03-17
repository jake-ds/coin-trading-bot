import { useEffect, useState, useCallback, useRef } from 'react'
import apiClient from '../api/client'
import type { WsStatusPayload, CompositeSignal, Position, Trade, SignalHistoryEntry } from '../api/types'
import { useWebSocket } from '../hooks/useWebSocket'
import StatusBadge from '../components/common/StatusBadge'
import MetricCard from '../components/common/MetricCard'
import PortfolioSummary from '../components/common/PortfolioSummary'
import { MetricCardSkeleton, PortfolioSummarySkeleton } from '../components/common/Skeleton'
import SignalGauge from '../components/signals/SignalGauge'
import FearGreedIndicator from '../components/signals/FearGreedIndicator'
import SignalOverviewChart from '../components/signals/SignalOverviewChart'
import SignalHistoryChart from '../components/signals/SignalHistoryChart'

interface DashboardData {
  status: string
  started_at: string | null
  portfolio: WsStatusPayload['portfolio'] | null
  metrics: Record<string, number>
  positions: Position[]
  trades: Trade[]
  emergency: { active: boolean }
  enginePerformance: Record<string, { pnl: number; win_rate: number; total_trades: number }>
  signals: Record<string, CompositeSignal>
  cycleMetrics: {
    cycle_count: number
    average_cycle_duration: number
    last_cycle_time: number | null
  } | null
}

const WS_URL = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/ws`

function Dashboard() {
  const [data, setData] = useState<DashboardData>({
    status: 'stopped',
    started_at: null,
    portfolio: null,
    metrics: {},
    positions: [],
    trades: [],
    emergency: { active: false },
    enginePerformance: {},
    signals: {},
    cycleMetrics: null,
  })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [signalHistory, setSignalHistory] = useState<SignalHistoryEntry[]>([])
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const { connected, data: wsMessage } = useWebSocket(WS_URL)

  // Apply WebSocket status_update messages
  useEffect(() => {
    if (wsMessage?.type === 'status_update') {
      const p = wsMessage.payload as unknown as WsStatusPayload
      setData({
        status: p.status,
        started_at: p.started_at,
        portfolio: p.portfolio,
        metrics: p.metrics,
        positions: p.open_positions || [],
        trades: p.trades || [],
        emergency: p.emergency || { active: false },
        enginePerformance: p.engine_performance || {},
        signals: p.onchain_signals || {},
        cycleMetrics: p.cycle_metrics,
      })
      setLoading(false)
      setError(null)
    }
  }, [wsMessage])

  const fetchData = useCallback(async () => {
    try {
      const [statusRes, portfolioRes, signalsRes, positionsRes, historyRes] = await Promise.allSettled([
        apiClient.get('/status'),
        apiClient.get('/portfolio'),
        apiClient.get('/onchain-signals'),
        apiClient.get('/positions'),
        apiClient.get('/signals/history', { params: { hours: 6 } }),
      ])
      setData((prev) => ({
        ...prev,
        status: statusRes.status === 'fulfilled' ? statusRes.value.data.status : prev.status,
        cycleMetrics: statusRes.status === 'fulfilled' ? statusRes.value.data.cycle_metrics : prev.cycleMetrics,
        portfolio: portfolioRes.status === 'fulfilled' ? portfolioRes.value.data.portfolio : prev.portfolio,
        signals: signalsRes.status === 'fulfilled' ? signalsRes.value.data.signals : prev.signals,
        positions: positionsRes.status === 'fulfilled' ? positionsRes.value.data.positions : prev.positions,
      }))
      if (historyRes.status === 'fulfilled' && historyRes.value.data.history) {
        setSignalHistory(historyRes.value.data.history)
      }
      setError(null)
    } catch {
      setError('Failed to fetch dashboard data')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  // Fallback polling when WebSocket is disconnected
  useEffect(() => {
    if (connected) {
      if (pollingRef.current) {
        clearInterval(pollingRef.current)
        pollingRef.current = null
      }
    } else {
      if (!pollingRef.current) {
        pollingRef.current = setInterval(fetchData, 10000)
      }
    }
    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current)
        pollingRef.current = null
      }
    }
  }, [connected, fetchData])

  if (loading) {
    return (
      <div>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold">Dashboard</h2>
        </div>
        <div className="mb-6">
          <PortfolioSummarySkeleton />
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
          {Array.from({ length: 6 }).map((_, i) => (
            <MetricCardSkeleton key={i} />
          ))}
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div>
        <h2 className="text-2xl font-bold mb-4">Dashboard</h2>
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-red-400">
          {error}
        </div>
      </div>
    )
  }

  const portfolio = data.portfolio
  const totalValue = portfolio?.total_value || 0
  const metrics = data.metrics
  const enginePerf = Object.values(data.enginePerformance)[0]
  const winRate = enginePerf?.win_rate ? (enginePerf.win_rate * 100) : Number(metrics.win_rate || 0)
  const totalTrades = enginePerf?.total_trades ?? Number(metrics.total_trades || 0)
  const totalPnl = enginePerf?.pnl ?? 0

  // Extract Fear & Greed from any signal's sentiment sub-signal
  let fearGreedValue: number | null = null
  let fearGreedClassification: string | undefined
  const firstSignal = Object.values(data.signals)[0]
  if (firstSignal?.signals) {
    const sentimentSignal = firstSignal.signals.find((s) => s.name === 'sentiment')
    if (sentimentSignal && sentimentSignal.reason) {
      const match = sentimentSignal.reason.match(/Fear\s*(?:&|and)?\s*Greed\s+(\d+)/i)
      if (match) {
        fearGreedValue = parseInt(match[1], 10)
      }
      const classMatch = sentimentSignal.reason.match(/"([^"]+)"/)
      if (classMatch) {
        fearGreedClassification = classMatch[1]
      }
    }
  }

  const signalEntries = Object.entries(data.signals)
  const positions = data.positions.slice(0, 5)
  const recentTrades = [...(data.trades || [])].reverse().slice(0, 5)

  return (
    <div>
      {/* Status bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-6">
        <h2 className="text-2xl font-bold">Dashboard</h2>
        <div className="flex items-center gap-3">
          {data.emergency.active && (
            <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-semibold bg-red-500/20 text-red-400 animate-pulse">
              EMERGENCY STOP
            </span>
          )}
          <StatusBadge status={data.status} />
        </div>
      </div>

      {/* Portfolio summary */}
      <div className="mb-6">
        <PortfolioSummary
          totalValue={totalValue}
          totalReturn={Number(metrics.total_return_pct || 0)}
          cashValue={portfolio?.cash_value as number | undefined}
          unrealizedPnl={portfolio?.unrealized_pnl as number | undefined}
        />
      </div>

      {/* Fear & Greed + Key Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 mb-6">
        <FearGreedIndicator value={fearGreedValue} classification={fearGreedClassification} />
        <MetricCard
          label="Total PnL"
          value={`${totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(2)}`}
          suffix="USDT"
          colorClass={totalPnl >= 0 ? 'text-green-400' : 'text-red-400'}
        />
        <MetricCard
          label="Win Rate"
          value={winRate.toFixed(1)}
          suffix="%"
          colorClass={winRate >= 50 ? 'text-green-400' : winRate > 0 ? 'text-amber-400' : 'text-gray-400'}
        />
        <MetricCard
          label="Total Trades"
          value={totalTrades}
        />
      </div>

      {/* Signal overview chart + top signal gauges */}
      {signalEntries.length > 0 && (
        <div className="mb-6">
          <h3 className="text-sm font-semibold text-gray-400 mb-3">Onchain Signal Scores</h3>
          <SignalOverviewChart signals={data.signals} />
        </div>
      )}

      {/* Signal history chart (6h) */}
      {signalHistory.length > 0 && signalEntries.length > 0 && (
        <div className="mb-6">
          <h3 className="text-sm font-semibold text-gray-400 mb-3">Signal Trend (6h)</h3>
          <SignalHistoryChart history={signalHistory} symbol={signalEntries[0][0]} />
        </div>
      )}

      {/* Top signal gauges (top 6 by absolute score) */}
      {signalEntries.length > 0 && (
        <div className="mb-6">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-gray-400">Top Signals</h3>
            <a href="/signals" className="text-xs text-blue-400 hover:text-blue-300">
              View all ({signalEntries.length})
            </a>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {signalEntries
              .sort(([, a], [, b]) => Math.abs(b.score) - Math.abs(a.score))
              .slice(0, 6)
              .map(([symbol, signal]) => (
                <SignalGauge key={symbol} symbol={symbol} signal={signal} />
              ))}
          </div>
        </div>
      )}

      {/* Open positions summary */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-gray-400">Open Positions</h3>
          {data.positions.length > 5 && (
            <a href="/positions" className="text-xs text-blue-400 hover:text-blue-300">
              View all ({data.positions.length})
            </a>
          )}
        </div>
        {positions.length === 0 ? (
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700 text-center text-gray-500 text-sm">
            No open positions
          </div>
        ) : (
          <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-700 text-gray-400">
                  <th className="text-left px-4 py-2 font-medium">Symbol</th>
                  <th className="text-right px-4 py-2 font-medium">Entry</th>
                  <th className="text-right px-4 py-2 font-medium">Current</th>
                  <th className="text-right px-4 py-2 font-medium">PnL</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((pos, idx) => {
                  const isProfit = pos.unrealized_pnl >= 0
                  return (
                    <tr key={`${pos.symbol}-${idx}`} className="border-b border-gray-700/50">
                      <td className="px-4 py-2 font-medium text-white">{pos.symbol}</td>
                      <td className="px-4 py-2 text-right text-gray-300">${pos.entry_price.toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
                      <td className="px-4 py-2 text-right text-gray-300">${pos.current_price.toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
                      <td className={`px-4 py-2 text-right font-medium ${isProfit ? 'text-green-400' : 'text-red-400'}`}>
                        {isProfit ? '+' : ''}{pos.unrealized_pnl.toFixed(2)}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Recent trades */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-gray-400">Recent Trades</h3>
          {data.trades.length > 5 && (
            <a href="/trades" className="text-xs text-blue-400 hover:text-blue-300">
              View all
            </a>
          )}
        </div>
        {recentTrades.length === 0 ? (
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700 text-center text-gray-500 text-sm">
            No trades yet
          </div>
        ) : (
          <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-700 text-gray-400">
                  <th className="text-left px-4 py-2 font-medium">Time</th>
                  <th className="text-left px-4 py-2 font-medium">Symbol</th>
                  <th className="text-center px-4 py-2 font-medium">Side</th>
                  <th className="text-right px-4 py-2 font-medium">Price</th>
                  <th className="text-right px-4 py-2 font-medium">PnL</th>
                </tr>
              </thead>
              <tbody>
                {recentTrades.map((trade, idx) => {
                  const isBuy = trade.side === 'BUY'
                  const hasPnl = trade.pnl != null
                  const isProfit = (trade.pnl ?? 0) >= 0
                  return (
                    <tr key={`${trade.timestamp}-${idx}`} className="border-b border-gray-700/50">
                      <td className="px-4 py-2 text-gray-300 text-xs">{new Date(trade.timestamp).toLocaleString()}</td>
                      <td className="px-4 py-2 font-medium text-white">{trade.symbol}</td>
                      <td className="px-4 py-2 text-center">
                        <span className={`inline-block px-2 py-0.5 rounded text-xs font-bold ${
                          isBuy ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                        }`}>
                          {trade.side}
                        </span>
                      </td>
                      <td className="px-4 py-2 text-right text-gray-300">${trade.price.toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
                      <td className={`px-4 py-2 text-right font-medium ${
                        hasPnl ? (isProfit ? 'text-green-400' : 'text-red-400') : 'text-gray-500'
                      }`}>
                        {hasPnl ? `${isProfit ? '+' : ''}${trade.pnl!.toFixed(2)}` : '--'}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

export default Dashboard
