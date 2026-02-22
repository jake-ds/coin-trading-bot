import { useState, useEffect, useCallback } from 'react'
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Brush,
  ReferenceDot,
} from 'recharts'
import apiClient from '../api/client'
import type {
  AnalyticsResponse,
  EquityPoint,
  DrawdownPoint,
  TradeMarker,
  MonthlyReturn,
  AnalyticsStats,
} from '../api/types'

const DATE_RANGES = [
  { value: '7d', label: '7 Days' },
  { value: '30d', label: '30 Days' },
  { value: '90d', label: '90 Days' },
  { value: 'all', label: 'All Time' },
]

function formatTimestamp(ts: string): string {
  if (!ts) return ''
  const d = new Date(ts)
  return `${d.getMonth() + 1}/${d.getDate()} ${d.getHours()}:${String(d.getMinutes()).padStart(2, '0')}`
}

function formatMonth(m: string): string {
  if (!m) return ''
  const [year, month] = m.split('-')
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
  return `${months[parseInt(month, 10) - 1] || month} ${year}`
}

function Analytics() {
  const [equityCurve, setEquityCurve] = useState<EquityPoint[]>([])
  const [drawdown, setDrawdown] = useState<DrawdownPoint[]>([])
  const [tradeMarkers, setTradeMarkers] = useState<TradeMarker[]>([])
  const [monthlyReturns, setMonthlyReturns] = useState<MonthlyReturn[]>([])
  const [stats, setStats] = useState<AnalyticsStats | null>(null)
  const [range, setRange] = useState('all')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchAnalytics = useCallback(async (selectedRange: string) => {
    try {
      setLoading(true)
      const res = await apiClient.get<AnalyticsResponse>(`/analytics?range=${selectedRange}`)
      setEquityCurve(res.data.equity_curve)
      setDrawdown(res.data.drawdown)
      setTradeMarkers(res.data.trade_markers)
      setMonthlyReturns(res.data.monthly_returns)
      setStats(res.data.stats)
      setError(null)
    } catch {
      setError('Failed to load analytics data')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchAnalytics(range)
  }, [range, fetchAnalytics])

  const handleRangeChange = useCallback((newRange: string) => {
    setRange(newRange)
  }, [])

  // Separate buy/sell markers for overlay
  const buyMarkers = tradeMarkers.filter((m) => m.side === 'BUY')
  const sellMarkers = tradeMarkers.filter((m) => m.side === 'SELL')

  if (loading) {
    return (
      <div>
        <h2 className="text-2xl font-bold mb-6">Performance Analytics</h2>
        <div className="space-y-4">
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6 animate-pulse h-80" />
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6 animate-pulse h-40" />
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="bg-gray-800 rounded-lg border border-gray-700 p-4 animate-pulse h-20" />
            ))}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold">Performance Analytics</h2>
        {/* Date Range Selector */}
        <div className="flex gap-1 bg-gray-800 rounded-lg border border-gray-700 p-1">
          {DATE_RANGES.map((r) => (
            <button
              key={r.value}
              onClick={() => handleRangeChange(r.value)}
              className={`px-3 py-1.5 rounded text-sm transition-colors ${
                range === r.value
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-700'
              }`}
            >
              {r.label}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="bg-red-900/30 border border-red-700 rounded-lg p-3 mb-4 text-red-300 text-sm">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Main charts column */}
        <div className="lg:col-span-3 space-y-4">
          {/* Equity Curve */}
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
            <h3 className="text-lg font-semibold mb-4">Equity Curve</h3>
            {equityCurve.length > 0 ? (
              <ResponsiveContainer width="100%" height={350}>
                <LineChart data={equityCurve}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis
                    dataKey="timestamp"
                    tick={{ fill: '#9ca3af', fontSize: 11 }}
                    axisLine={{ stroke: '#4b5563' }}
                    tickFormatter={formatTimestamp}
                    minTickGap={50}
                  />
                  <YAxis
                    tick={{ fill: '#9ca3af', fontSize: 11 }}
                    axisLine={{ stroke: '#4b5563' }}
                    tickFormatter={(v: number) => `$${v.toLocaleString()}`}
                    domain={['auto', 'auto']}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1f2937',
                      border: '1px solid #4b5563',
                      borderRadius: '8px',
                    }}
                    labelStyle={{ color: '#e5e7eb' }}
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    labelFormatter={(label: any) => formatTimestamp(String(label))}
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    formatter={(value: any) => [`$${Number(value).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`, 'Portfolio Value']}
                  />
                  <Line
                    type="monotone"
                    dataKey="total_value"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4 }}
                  />
                  {/* Buy markers */}
                  {buyMarkers.map((m, i) => (
                    <ReferenceDot
                      key={`buy-${i}`}
                      x={m.timestamp}
                      y={m.value}
                      r={5}
                      fill="#22c55e"
                      stroke="#22c55e"
                    />
                  ))}
                  {/* Sell markers */}
                  {sellMarkers.map((m, i) => (
                    <ReferenceDot
                      key={`sell-${i}`}
                      x={m.timestamp}
                      y={m.value}
                      r={5}
                      fill="#ef4444"
                      stroke="#ef4444"
                    />
                  ))}
                  {equityCurve.length > 20 && (
                    <Brush
                      dataKey="timestamp"
                      height={30}
                      stroke="#4b5563"
                      fill="#1f2937"
                      tickFormatter={formatTimestamp}
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-64 flex items-center justify-center text-gray-500">
                No equity curve data available
              </div>
            )}
            {tradeMarkers.length > 0 && (
              <div className="flex gap-4 mt-2 text-xs text-gray-400">
                <span className="flex items-center gap-1">
                  <span className="w-3 h-3 rounded-full bg-green-500 inline-block" /> BUY
                </span>
                <span className="flex items-center gap-1">
                  <span className="w-3 h-3 rounded-full bg-red-500 inline-block" /> SELL
                </span>
              </div>
            )}
          </div>

          {/* Drawdown Chart */}
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
            <h3 className="text-lg font-semibold mb-4">Drawdown</h3>
            {drawdown.length > 0 ? (
              <ResponsiveContainer width="100%" height={180}>
                <AreaChart data={drawdown}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis
                    dataKey="timestamp"
                    tick={{ fill: '#9ca3af', fontSize: 11 }}
                    axisLine={{ stroke: '#4b5563' }}
                    tickFormatter={formatTimestamp}
                    minTickGap={50}
                  />
                  <YAxis
                    tick={{ fill: '#9ca3af', fontSize: 11 }}
                    axisLine={{ stroke: '#4b5563' }}
                    tickFormatter={(v: number) => `${v.toFixed(1)}%`}
                    reversed
                    domain={['auto', 0]}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1f2937',
                      border: '1px solid #4b5563',
                      borderRadius: '8px',
                    }}
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    labelFormatter={(label: any) => formatTimestamp(String(label))}
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    formatter={(value: any) => [`${Number(value).toFixed(2)}%`, 'Drawdown']}
                  />
                  <Area
                    type="monotone"
                    dataKey="drawdown_pct"
                    stroke="#ef4444"
                    fill="rgba(239, 68, 68, 0.2)"
                    strokeWidth={1.5}
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-32 flex items-center justify-center text-gray-500">
                No drawdown data available
              </div>
            )}
          </div>

          {/* Monthly Returns Heatmap */}
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
            <h3 className="text-lg font-semibold mb-4">Monthly Returns</h3>
            {monthlyReturns.length > 0 ? (
              <MonthlyHeatmap data={monthlyReturns} />
            ) : (
              <div className="h-20 flex items-center justify-center text-gray-500">
                Not enough data for monthly returns
              </div>
            )}
          </div>
        </div>

        {/* Stats Sidebar */}
        <div className="space-y-4">
          {stats && (
            <>
              <StatsCard
                label="Total Return"
                value={`${stats.total_return_pct >= 0 ? '+' : ''}${stats.total_return_pct.toFixed(2)}%`}
                color={stats.total_return_pct >= 0 ? 'text-green-400' : 'text-red-400'}
              />
              <StatsCard
                label="Sharpe Ratio"
                value={stats.sharpe_ratio.toFixed(4)}
                color={stats.sharpe_ratio > 0 ? 'text-green-400' : stats.sharpe_ratio < 0 ? 'text-red-400' : undefined}
              />
              <StatsCard
                label="Sortino Ratio"
                value={stats.sortino_ratio.toFixed(4)}
                color={stats.sortino_ratio > 0 ? 'text-green-400' : stats.sortino_ratio < 0 ? 'text-red-400' : undefined}
              />
              <StatsCard
                label="Max Drawdown"
                value={`${stats.max_drawdown_pct.toFixed(2)}%`}
                color="text-red-400"
              />
              <StatsCard
                label="Profit Factor"
                value={stats.profit_factor.toFixed(2)}
                color={stats.profit_factor > 1 ? 'text-green-400' : 'text-red-400'}
              />
              <StatsCard
                label="Win Rate"
                value={`${stats.win_rate.toFixed(1)}%`}
                color={stats.win_rate >= 50 ? 'text-green-400' : 'text-red-400'}
              />
              <StatsCard
                label="Total Trades"
                value={`${stats.total_trades} (${stats.winning_trades}W / ${stats.losing_trades}L)`}
              />
              <StatsCard
                label="Avg Trade PnL"
                value={`$${stats.avg_trade_pnl.toFixed(2)}`}
                color={stats.avg_trade_pnl >= 0 ? 'text-green-400' : 'text-red-400'}
              />
              <StatsCard
                label="Best Trade"
                value={`$${stats.best_trade.toFixed(2)}`}
                color="text-green-400"
              />
              <StatsCard
                label="Worst Trade"
                value={`$${stats.worst_trade.toFixed(2)}`}
                color="text-red-400"
              />
            </>
          )}
        </div>
      </div>
    </div>
  )
}

function StatsCard({
  label,
  value,
  color,
}: {
  label: string
  value: string
  color?: string
}) {
  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
      <p className="text-xs text-gray-500 mb-1">{label}</p>
      <p className={`text-lg font-bold ${color || 'text-gray-200'}`}>{value}</p>
    </div>
  )
}

function MonthlyHeatmap({ data }: { data: MonthlyReturn[] }) {
  // Group by year
  const years: Record<string, MonthlyReturn[]> = {}
  for (const item of data) {
    const year = item.month.split('-')[0]
    if (!years[year]) years[year] = []
    years[year].push(item)
  }

  const maxAbs = Math.max(...data.map((d) => Math.abs(d.return_pct)), 1)

  return (
    <div className="space-y-3">
      {Object.entries(years)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([year, months]) => (
          <div key={year}>
            <p className="text-xs text-gray-500 mb-1">{year}</p>
            <div className="flex gap-1 flex-wrap">
              {months.map((m) => {
                const intensity = Math.min(Math.abs(m.return_pct) / maxAbs, 1)
                const bg =
                  m.return_pct >= 0
                    ? `rgba(34, 197, 94, ${0.15 + intensity * 0.65})`
                    : `rgba(239, 68, 68, ${0.15 + intensity * 0.65})`
                return (
                  <div
                    key={m.month}
                    className="rounded px-2 py-1 text-xs font-mono cursor-default"
                    style={{ backgroundColor: bg }}
                    title={`${formatMonth(m.month)}: ${m.return_pct >= 0 ? '+' : ''}${m.return_pct.toFixed(2)}%`}
                  >
                    <span className="text-gray-300">{m.month.split('-')[1]}</span>
                    <span className={`ml-1 ${m.return_pct >= 0 ? 'text-green-300' : 'text-red-300'}`}>
                      {m.return_pct >= 0 ? '+' : ''}{m.return_pct.toFixed(1)}%
                    </span>
                  </div>
                )
              })}
            </div>
          </div>
        ))}
    </div>
  )
}

export default Analytics
