import { useState, useEffect, useCallback } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  PieChart,
  Pie,
  Cell,
  Legend,
  LineChart,
  Line,
} from 'recharts'
import apiClient from '../api/client'
import type {
  EngineMetrics,
  PerformanceSummary,
  DailySummaryEntry,
  MetricsHistoryResponse,
  MetricsCompareResponse,
} from '../api/types'

const ENGINE_COLORS: Record<string, string> = {
  funding_rate_arb: '#3b82f6',
  grid_trading: '#22c55e',
  cross_exchange_arb: '#f59e0b',
  stat_arb: '#8b5cf6',
}
const LINE_COLORS = ['#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6', '#ec4899']

function getColor(name: string): string {
  return ENGINE_COLORS[name] || '#6b7280'
}

function MetricCard({
  label,
  value,
  color,
}: {
  label: string
  value: string
  color?: string
}) {
  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <p className="text-xs text-gray-500 mb-1">{label}</p>
      <p className={`text-lg font-mono font-semibold ${color || 'text-gray-100'}`}>
        {value}
      </p>
    </div>
  )
}

function EngineMetricsCard({
  name,
  metrics,
}: {
  name: string
  metrics: EngineMetrics
}) {
  const color = getColor(name)
  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <div className="flex items-center gap-2 mb-3">
        <span
          className="w-3 h-3 rounded-full"
          style={{ backgroundColor: color }}
        />
        <h4 className="font-semibold">{name}</h4>
      </div>
      <div className="grid grid-cols-3 gap-3 text-sm">
        <div>
          <p className="text-xs text-gray-500">Win Rate</p>
          <p className="font-mono">{(metrics.win_rate * 100).toFixed(1)}%</p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Sharpe</p>
          <p className="font-mono">{metrics.sharpe_ratio.toFixed(2)}</p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Max DD</p>
          <p className="font-mono text-red-400">
            {(metrics.max_drawdown * 100).toFixed(1)}%
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Profit Factor</p>
          <p className="font-mono">{metrics.profit_factor.toFixed(2)}</p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Trades</p>
          <p className="font-mono">{metrics.total_trades}</p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Total Cost</p>
          <p className="font-mono">${metrics.total_cost.toFixed(2)}</p>
        </div>
      </div>
    </div>
  )
}

const RANGE_OPTIONS = [
  { label: '7D', days: 7 },
  { label: '30D', days: 30 },
  { label: '90D', days: 90 },
]

const COMPARE_METRICS = [
  { label: 'Sharpe', value: 'sharpe' },
  { label: 'Win Rate', value: 'win_rate' },
  { label: 'Total PnL', value: 'total_pnl' },
  { label: 'Max DD', value: 'max_drawdown' },
]

function Performance() {
  const [data, setData] = useState<PerformanceSummary | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [engineStatus, setEngineStatus] = useState<
    Record<string, { allocated_capital: number }>
  >({})

  // Historical state
  const [tab, setTab] = useState<'current' | 'historical'>('current')
  const [rangeDays, setRangeDays] = useState(30)
  const [dailySummary, setDailySummary] = useState<DailySummaryEntry[]>([])
  const [historyEngine, setHistoryEngine] = useState<string>('')
  const [historyData, setHistoryData] = useState<MetricsHistoryResponse | null>(null)
  const [compareMetric, setCompareMetric] = useState('sharpe')
  const [compareData, setCompareData] = useState<MetricsCompareResponse | null>(null)
  const [histLoading, setHistLoading] = useState(false)

  const fetchData = useCallback(async () => {
    try {
      const [summaryRes, enginesRes] = await Promise.all([
        apiClient.get<PerformanceSummary>('/performance/summary'),
        apiClient.get('/engines'),
      ])
      setData(summaryRes.data)

      const capitalMap: Record<string, { allocated_capital: number }> = {}
      if (enginesRes.data) {
        for (const [name, info] of Object.entries(enginesRes.data)) {
          const eng = info as { allocated_capital?: number }
          capitalMap[name] = {
            allocated_capital: eng.allocated_capital || 0,
          }
        }
      }
      setEngineStatus(capitalMap)

      // Set default history engine from first engine
      if (!historyEngine && summaryRes.data?.engines) {
        const engines = Object.keys(summaryRes.data.engines)
        if (engines.length > 0) setHistoryEngine(engines[0])
      }

      setError(null)
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'response' in err) {
        const axiosErr = err as { response?: { status?: number } }
        if (axiosErr.response?.status === 503) {
          setError('Engine mode is not enabled.')
        } else {
          setError('Failed to fetch performance data')
        }
      } else {
        setError('Failed to fetch performance data')
      }
    } finally {
      setLoading(false)
    }
  }, [historyEngine])

  const fetchHistorical = useCallback(async () => {
    if (!data) return
    setHistLoading(true)
    try {
      const engineNames = Object.keys(data.engines).join(',')
      const results = await Promise.allSettled([
        apiClient.get<{ daily: DailySummaryEntry[] }>('/metrics/daily-summary', {
          params: { days: rangeDays },
        }),
        historyEngine
          ? apiClient.get<MetricsHistoryResponse>('/metrics/history', {
              params: { engine: historyEngine, days: rangeDays },
            })
          : Promise.resolve(null),
        engineNames
          ? apiClient.get<MetricsCompareResponse>('/metrics/compare', {
              params: { engines: engineNames, metric: compareMetric, days: rangeDays },
            })
          : Promise.resolve(null),
      ])

      if (results[0].status === 'fulfilled' && results[0].value) {
        const val = results[0].value
        const d = 'data' in val ? (val as { data: { daily: DailySummaryEntry[] } }).data : null
        setDailySummary(d?.daily || [])
      }
      if (results[1].status === 'fulfilled' && results[1].value) {
        const val = results[1].value
        const d = 'data' in val ? (val as { data: MetricsHistoryResponse }).data : null
        setHistoryData(d)
      }
      if (results[2].status === 'fulfilled' && results[2].value) {
        const val = results[2].value
        const d = 'data' in val ? (val as { data: MetricsCompareResponse }).data : null
        setCompareData(d)
      }
    } catch {
      // Graceful degradation
    } finally {
      setHistLoading(false)
    }
  }, [data, rangeDays, historyEngine, compareMetric])

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 10000)
    return () => clearInterval(interval)
  }, [fetchData])

  useEffect(() => {
    if (tab === 'historical') {
      fetchHistorical()
    }
  }, [tab, fetchHistorical])

  if (loading) {
    return (
      <div className="space-y-6">
        <h2 className="text-2xl font-bold">Performance</h2>
        <div className="grid grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <div
              key={i}
              className="bg-gray-800 rounded-lg p-4 border border-gray-700 animate-pulse h-20"
            />
          ))}
        </div>
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700 animate-pulse h-64" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="space-y-6">
        <h2 className="text-2xl font-bold">Performance</h2>
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <p className="text-yellow-400">{error}</p>
        </div>
      </div>
    )
  }

  if (!data) {
    return (
      <div className="space-y-6">
        <h2 className="text-2xl font-bold">Performance</h2>
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700 text-center text-gray-400">
          No performance data available yet.
        </div>
      </div>
    )
  }

  const { engines, totals } = data
  const engineNames = Object.keys(engines)

  // Build cost analysis chart data
  const costChartData = Object.entries(engines).map(([name, m]) => ({
    name,
    gross: m.total_pnl + m.total_cost,
    cost: m.total_cost,
    net: m.total_pnl,
  }))

  // Build capital allocation pie data
  const capitalData = Object.entries(engineStatus)
    .filter(([, v]) => v.allocated_capital > 0)
    .map(([name, v]) => ({
      name,
      value: v.allocated_capital,
    }))

  // Build metrics trend chart data from history
  const trendData = historyData
    ? historyData.timestamps.map((ts, i) => ({
        time: ts.slice(5, 16),
        sharpe: historyData.sharpe[i] ?? 0,
        win_rate: (historyData.win_rate[i] ?? 0) * 100,
        total_pnl: historyData.total_pnl[i] ?? 0,
        max_drawdown: (historyData.max_drawdown[i] ?? 0) * 100,
      }))
    : []

  // Build compare chart data
  const compareChartData: Record<string, unknown>[] = []
  if (compareData?.engines) {
    // Collect all unique timestamps
    const tsSet = new Set<string>()
    for (const eng of Object.values(compareData.engines)) {
      eng.timestamps.forEach((t) => tsSet.add(t))
    }
    const sortedTs = Array.from(tsSet).sort()
    for (const ts of sortedTs) {
      const point: Record<string, unknown> = { time: ts.slice(5, 16) }
      for (const [eng, d] of Object.entries(compareData.engines)) {
        const idx = d.timestamps.indexOf(ts)
        point[eng] = idx >= 0 ? d.values[idx] : null
      }
      compareChartData.push(point)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Performance</h2>
        <div className="flex gap-2">
          <button
            onClick={() => setTab('current')}
            className={`px-3 py-1.5 rounded text-sm ${
              tab === 'current'
                ? 'bg-blue-600 text-white'
                : 'text-gray-300 hover:bg-gray-700'
            }`}
          >
            Current
          </button>
          <button
            onClick={() => setTab('historical')}
            className={`px-3 py-1.5 rounded text-sm ${
              tab === 'historical'
                ? 'bg-blue-600 text-white'
                : 'text-gray-300 hover:bg-gray-700'
            }`}
          >
            Historical
          </button>
        </div>
      </div>

      {tab === 'current' && (
        <>
          {/* Overall summary */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <MetricCard
              label="Total PnL"
              value={`$${totals.total_pnl >= 0 ? '+' : ''}${totals.total_pnl.toFixed(2)}`}
              color={totals.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}
            />
            <MetricCard
              label="Overall Win Rate"
              value={`${(totals.overall_win_rate * 100).toFixed(1)}%`}
            />
            <MetricCard
              label="Overall Sharpe"
              value={totals.overall_sharpe.toFixed(2)}
            />
            <MetricCard
              label="Total Trades"
              value={String(totals.total_trades)}
            />
            <MetricCard
              label="Total Cost"
              value={`$${totals.total_cost.toFixed(2)}`}
              color="text-yellow-400"
            />
          </div>

          {/* Per-engine metrics cards */}
          <div>
            <h3 className="text-lg font-semibold mb-3">Engine Metrics</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.entries(engines).map(([name, metrics]) => (
                <EngineMetricsCard key={name} name={name} metrics={metrics} />
              ))}
            </div>
          </div>

          {/* Charts row */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Cost analysis bar chart */}
            <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <h3 className="text-sm font-semibold mb-3 text-gray-300">
                Gross PnL vs Costs vs Net PnL
              </h3>
              {costChartData.length > 0 ? (
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={costChartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="name" tick={{ fill: '#9ca3af', fontSize: 11 }} />
                    <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1f2937',
                        border: '1px solid #374151',
                        borderRadius: '8px',
                      }}
                    />
                    <Bar dataKey="gross" name="Gross PnL" fill="#3b82f6" />
                    <Bar dataKey="cost" name="Cost" fill="#ef4444" />
                    <Bar dataKey="net" name="Net PnL" fill="#22c55e" />
                    <Legend />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex items-center justify-center h-[250px] text-gray-500 text-sm">
                  No cost data available
                </div>
              )}
            </div>

            {/* Capital allocation pie chart */}
            <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <h3 className="text-sm font-semibold mb-3 text-gray-300">
                Capital Allocation
              </h3>
              {capitalData.length > 0 ? (
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={capitalData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={90}
                      dataKey="value"
                      nameKey="name"
                      label={({ name, value }) =>
                        `${name}: $${value.toLocaleString()}`
                      }
                    >
                      {capitalData.map((entry) => (
                        <Cell
                          key={entry.name}
                          fill={getColor(entry.name)}
                        />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1f2937',
                        border: '1px solid #374151',
                        borderRadius: '8px',
                      }}
                      formatter={(value: number | undefined) =>
                        value != null ? `$${value.toLocaleString()}` : ''
                      }
                    />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex items-center justify-center h-[250px] text-gray-500 text-sm">
                  No capital allocation data
                </div>
              )}
            </div>
          </div>
        </>
      )}

      {tab === 'historical' && (
        <>
          {/* Range selector */}
          <div className="flex items-center gap-3">
            <span className="text-sm text-gray-400">Range:</span>
            {RANGE_OPTIONS.map((opt) => (
              <button
                key={opt.days}
                onClick={() => setRangeDays(opt.days)}
                className={`px-3 py-1 rounded text-sm ${
                  rangeDays === opt.days
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-300 hover:bg-gray-700 border border-gray-600'
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>

          {histLoading && (
            <div className="text-center text-gray-400 py-8">Loading historical data...</div>
          )}

          {/* Daily PnL bar chart */}
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <h3 className="text-sm font-semibold mb-3 text-gray-300">
              Daily PnL
            </h3>
            {dailySummary.length > 0 ? (
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={dailySummary}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis
                    dataKey="date"
                    tick={{ fill: '#9ca3af', fontSize: 10 }}
                    tickFormatter={(v: string) => v.slice(5)}
                  />
                  <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1f2937',
                      border: '1px solid #374151',
                      borderRadius: '8px',
                    }}
                    formatter={(value: number | undefined) =>
                      value != null ? `$${value.toFixed(2)}` : ''
                    }
                    labelFormatter={(label: unknown) => `Date: ${label}`}
                  />
                  <Bar dataKey="total_pnl" name="Net PnL">
                    {dailySummary.map((entry, idx) => (
                      <Cell
                        key={idx}
                        fill={entry.total_pnl >= 0 ? '#22c55e' : '#ef4444'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-[280px] text-gray-500 text-sm">
                No daily PnL data available yet.
              </div>
            )}
          </div>

          {/* Metrics trend line chart */}
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-gray-300">
                Metrics Trend
              </h3>
              <select
                value={historyEngine}
                onChange={(e) => setHistoryEngine(e.target.value)}
                className="bg-gray-700 text-gray-200 text-sm rounded px-2 py-1 border border-gray-600"
              >
                {engineNames.map((n) => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
            </div>
            {trendData.length > 0 ? (
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={trendData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="time" tick={{ fill: '#9ca3af', fontSize: 10 }} />
                  <YAxis yAxisId="left" tick={{ fill: '#9ca3af', fontSize: 11 }} />
                  <YAxis yAxisId="right" orientation="right" tick={{ fill: '#9ca3af', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1f2937',
                      border: '1px solid #374151',
                      borderRadius: '8px',
                    }}
                  />
                  <Line yAxisId="left" type="monotone" dataKey="sharpe" name="Sharpe" stroke="#3b82f6" dot={false} strokeWidth={2} />
                  <Line yAxisId="right" type="monotone" dataKey="win_rate" name="Win Rate %" stroke="#22c55e" dot={false} strokeWidth={2} />
                  <Legend />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-[280px] text-gray-500 text-sm">
                No metrics history available yet.
              </div>
            )}
          </div>

          {/* Engine comparison chart */}
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-gray-300">
                Engine Comparison
              </h3>
              <select
                value={compareMetric}
                onChange={(e) => setCompareMetric(e.target.value)}
                className="bg-gray-700 text-gray-200 text-sm rounded px-2 py-1 border border-gray-600"
              >
                {COMPARE_METRICS.map((m) => (
                  <option key={m.value} value={m.value}>{m.label}</option>
                ))}
              </select>
            </div>
            {compareChartData.length > 0 ? (
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={compareChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="time" tick={{ fill: '#9ca3af', fontSize: 10 }} />
                  <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1f2937',
                      border: '1px solid #374151',
                      borderRadius: '8px',
                    }}
                  />
                  {engineNames.map((eng, i) => (
                    <Line
                      key={eng}
                      type="monotone"
                      dataKey={eng}
                      name={eng}
                      stroke={LINE_COLORS[i % LINE_COLORS.length]}
                      dot={false}
                      strokeWidth={2}
                      connectNulls
                    />
                  ))}
                  <Legend />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-[280px] text-gray-500 text-sm">
                No comparison data available yet.
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}

export default Performance
