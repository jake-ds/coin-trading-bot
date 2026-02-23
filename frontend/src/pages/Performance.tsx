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
} from 'recharts'
import apiClient from '../api/client'
import type { EngineMetrics, PerformanceSummary } from '../api/types'

const ENGINE_COLORS: Record<string, string> = {
  funding_rate_arb: '#3b82f6',
  grid_trading: '#22c55e',
  cross_exchange_arb: '#f59e0b',
  stat_arb: '#8b5cf6',
}

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

function Performance() {
  const [data, setData] = useState<PerformanceSummary | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [engineStatus, setEngineStatus] = useState<
    Record<string, { allocated_capital: number }>
  >({})

  const fetchData = useCallback(async () => {
    try {
      const [summaryRes, enginesRes] = await Promise.all([
        apiClient.get<PerformanceSummary>('/performance/summary'),
        apiClient.get('/engines'),
      ])
      setData(summaryRes.data)

      // Extract capital allocations from engines response
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
  }, [])

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 10000)
    return () => clearInterval(interval)
  }, [fetchData])

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

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Performance</h2>

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
    </div>
  )
}

export default Performance
